#!/usr/bin/env python3
"""
Evaluation of Multiple Models and Ensembles

Tests four trained models (2 per backbone) in normal and TTA modes,
then ensembles each pair of models (simple ensemble and ensemble+TTA),
saving all original metrics without any modification.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

import cv2
from PIL import Image
import albumentations as albu
import timm
from torchvision import transforms
from nltk.tree import Tree


# =============================================================================
# GENERAL CONFIGURATION
# =============================================================================
TOPK_TO_CONSIDER = (1, 5)
TEST_CSV        = "data/test_mbm.csv"
BATCH_SIZE      = 64
NUM_CLASSES     = 23
IMAGE_SIZE      = (384, 384)
NUM_RUNS        = 5
TTA_MODES       = [None, 'hflip', 'vflip', 'hvflip']

MODEL_CONFIGS = [
    {
        'name':     'effnetv2_s_flat',
        'backbone': 'tf_efficientnetv2_s.in21k_ft_in1k',
        'ckpt':     'checkpoints/effnetv2_s_flat/checkpoint.pth.tar',
    },
    {
        'name':     'effnetv2_s_hier',
        'backbone': 'tf_efficientnetv2_s.in21k_ft_in1k',
        'ckpt':     'checkpoints/effnetv2_s_hier/checkpoint.pth.tar',
    },
    {
        'name':     'caformer_s36_flat',
        'backbone': 'caformer_s36.sail_in22k_ft_in1k_384',
        'ckpt':     'checkpoints/caformer_s36_flat/checkpoint.pth.tar',
    },
    {
        'name':     'caformer_s36_2_hier',
        'backbone': 'caformer_s36.sail_in22k_ft_in1k_384',
        'ckpt':     'checkpoints/caformer_s36_2_hier/checkpoint.pth.tar',
    },
]

ENS_BUNDLES = [
    {
        'name':    'ensemble_hier',
        'members': ['effnetv2_s_hier', 'caformer_s36_2_hier'],
    }
]


# =============================================================================
# HELPERS
# =============================================================================
class DistanceDict(dict):
    def __init__(self, distances):
        super().__init__()
        self.distances = {tuple(sorted(k)): v for k, v in distances.items()}
    def __getitem__(self, idx):
        if idx[0] == idx[1]:
            return 0.0
        key = tuple(sorted(idx))
        return self.distances[key]
    
def prf_from_conf(conf_mat):
    """
    Dada uma confusion matrix square, retorna:
      - precisão média (macro) por classe
      - recall médio (macro) por classe
      - F1 médio (macro) por classe
    """
    K = conf_mat.shape[0]
    ps, rs, fs = [], [], []
    for c in range(K):
        TP = conf_mat[c, c]
        FP = conf_mat[:, c].sum() - TP
        FN = conf_mat[c, :].sum() - TP
        p = TP / (TP + FP) if TP + FP > 0 else 0.0
        r = TP / (TP + FN) if TP + FN > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        ps.append(p); rs.append(r); fs.append(f)
    return np.mean(ps), np.mean(rs), np.mean(fs)

def accuracy(output, target, ks=(1,)):
    with torch.no_grad():
        maxk = max(ks)
        batch_size = target.size(0)
        _, pred_ = output.topk(maxk, 1, True, True)
        pred = pred_.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in ks:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res, pred


def get_order_family_target(targets, trees):
    order = []; family = []; species = []
    for t in targets.tolist():
        o, f, s = trees[int(t)]
        order.append(o); family.append(f); species.append(s)
    dev = targets.device
    return (torch.tensor(order, device=dev),
            torch.tensor(family, device=dev),
            torch.tensor(species, device=dev))


# =============================================================================
# HIERARCHY AND LOSS
# =============================================================================
class HierarchicalLLLoss(nn.Module):
    def __init__(self, hierarchy, classes, weights):
        super().__init__()
        # mapeia labels para posições na árvore
        pos_leaves = { get_label(hierarchy[p]) : p
                       for p in hierarchy.treepositions('leaves') }
        leaves = [pos_leaves[c] for c in classes]
        edges = hierarchy.treepositions()[1:]
        idx_leaf = {leaves[i]: i for i in range(len(leaves))}
        idx_edge = {edges[i]: i for i in range(len(edges))}
        paths = [[ idx_edge[p[:j]]
                   for j in range(len(p),0,-1) ]
                 for p in leaves]
        num_e = max(len(p) for p in paths)

        self.onehot_num = nn.Parameter(
            torch.zeros((len(classes), len(classes), num_e)),
            requires_grad=False
        )
        self.onehot_den = nn.Parameter(
            torch.zeros((len(classes), len(classes), num_e)),
            requires_grad=False
        )
        self.weights = nn.Parameter(
            torch.zeros((len(classes), num_e)),
            requires_grad=False
        )

        for i, path in enumerate(paths):
            for j, eidx in enumerate(path):
                self.onehot_num[i, idx_leaf[tuple(leaves[j])], j] = 1.0
                self.onehot_den[i, :, :j+1] = 1.0
                self.weights[i, j] = get_label(weights[edges[eidx]])

    def forward(self, inputs, target):
        probs = F.softmax(inputs, dim=1).unsqueeze(1)
        num   = (probs @ self.onehot_num[target]).squeeze(1)
        den   = (probs @ self.onehot_den[target]).squeeze(1)
        mask  = num != 0
        num[mask] = -torch.log(num[mask] / den[mask])
        loss = (self.weights[target] * num).sum(dim=1)
        return loss.mean()


class HierarchicalCrossEntropyLoss(HierarchicalLLLoss):
    def forward(self, inputs, target):
        return super().forward(inputs, target)


# helper for HierarchicalLLLoss
def get_label(node):
    return node.label() if isinstance(node, Tree) else node


# =============================================================================
# DATASET AND MODEL
# =============================================================================
class InputDataset(torch.utils.data.Dataset):
    def __init__(self, csv, transform=None, alb=None):
        df = pd.read_csv(csv)
        self.data = [(r['image_path'], int(r['class_id']))
                     for _, r in df.iterrows()]
        self.transform = transform
        self.alb = alb

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        p, t = self.data[i]
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        if self.alb:
            img = self.alb(image=img)['image']
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, t


class CustomModel(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        d = self.backbone.num_features
        self.norm = nn.LayerNorm(d, eps=1e-6)
        self.fc   = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f = self.backbone.forward_features(x)
        x = self.pool(f).flatten(1)
        x = self.norm(x)
        return self.fc(x)


# =============================================================================
# METRICS CALCULATION
# =============================================================================
def calculate_metrics(outputs, targets, distances, classes, best_sim, max_dist):
    batch = {'hdist_avg': [], 'hdist_top': [], 'hprec': [], 'hmAP': []}
    _, preds = accuracy(outputs, targets, TOPK_TO_CONSIDER)
    bs = targets.size(0)

    for k in TOPK_TO_CONSIDER:
        avg_d = top_d = prec = mapv = 0.0
        for i in range(bs):
            gt = targets[i].item()
            pk = preds[:k, i].tolist()
            dlist = [distances[(classes[p], classes[gt])] for p in pk]
            avg_d += np.mean(dlist)
            top_d += np.min(dlist)
            sims = 1 - np.array(dlist) / max_dist
            prec += np.sum(sims) / np.sum(best_sim[gt, :k])
            ap = 0.0
            for j in range(1, k + 1):
                ap += np.sum(sims[:j]) / np.sum(best_sim[gt, :j])
            mapv += ap / k

        batch['hdist_avg'].append(avg_d / bs)
        batch['hdist_top'].append(top_d / bs)
        batch['hprec'].append(prec / bs)
        batch['hmAP'].append(mapv / bs)

    return batch


def _make_best_hier_similarities(classes, distances, max_dist):
    n = len(classes)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_mat[i, j] = distances[(classes[i], classes[j])]
    return 1 - np.sort(dist_mat, axis=1) / max_dist


# =============================================================================
# AVALIAÇÃO INDIVIDUAL
# =============================================================================
def evaluate_model(model, loader, loss_fn, distances, classes, trees, device, tta=False):
    model.eval()
    max_dist = max(distances.distances.values())
    best_sim = _make_best_hier_similarities(classes, distances, max_dist)

    # general counters
    tot = 0
    m = defaultdict(float)
    hd_avg = np.zeros(len(TOPK_TO_CONSIDER))
    hd_top = np.zeros(len(TOPK_TO_CONSIDER))
    hprec  = np.zeros(len(TOPK_TO_CONSIDER))
    hmAP   = np.zeros(len(TOPK_TO_CONSIDER))
    acc1 = acc2 = acc3 = 0

    # global and per-level confusion matrices
    conf  = np.zeros((len(classes), len(classes)), int)
    n1 = len(set(t[0] for t in trees.values()))
    n2 = len(set(t[1] for t in trees.values()))
    n3 = len(set(t[2] for t in trees.values()))
    conf1 = np.zeros((n1, n1), int)
    conf2 = np.zeros((n2, n2), int)
    conf3 = np.zeros((n3, n3), int)

    desc = "Eval TTA" if tta else "Eval"
    with torch.no_grad():
        for imgs, tgts in tqdm(loader, desc):
            imgs, tgts = imgs.to(device), tgts.to(device)

            if not tta:
                outs = model(imgs)
                loss = loss_fn(outs, tgts)
                probs = F.softmax(outs, dim=1)
            else:
                probs_sum = torch.zeros((imgs.size(0), len(classes)), device=device)
                loss_sum = 0.0
                for mode in TTA_MODES:
                    if mode is None:
                        im = imgs
                    elif mode == 'hflip':
                        im = torch.flip(imgs, dims=[3])
                    elif mode == 'vflip':
                        im = torch.flip(imgs, dims=[2])
                    else:  # 'hvflip'
                        im = torch.flip(torch.flip(imgs, dims=[3]), dims=[2])
                    outk = model(im)
                    loss_sum += loss_fn(outk, tgts).item()
                    probs_sum += F.softmax(outk, dim=1)
                loss = loss_sum / len(TTA_MODES)
                probs = probs_sum / len(TTA_MODES)
                outs = torch.log(probs)

            # accumulate existing metrics
            m['loss']     += float(loss) * imgs.size(0)
            accs, preds   = accuracy(outs, tgts, TOPK_TO_CONSIDER)
            m['top1_acc'] += accs[0].item() * imgs.size(0)
            m['top5_acc'] += accs[1].item() * imgs.size(0)

            batch = calculate_metrics(outs, tgts, distances, classes, best_sim, max_dist)
            for i in range(len(TOPK_TO_CONSIDER)):
                hd_avg[i] += batch['hdist_avg'][i] * imgs.size(0)
                hd_top[i] += batch['hdist_top'][i] * imgs.size(0)
                hprec[i]  += batch['hprec'][i]     * imgs.size(0)
                hmAP[i]   += batch['hmAP'][i]      * imgs.size(0)

            l1t, l2t, l3t = get_order_family_target(tgts, trees)
            t1            = preds[0]
            l1p, l2p, l3p = get_order_family_target(t1, trees)
            acc1 += l1p.eq(l1t).sum().item()
            acc2 += l2p.eq(l2t).sum().item()
            acc3 += l3p.eq(l3t).sum().item()

            # fill global and per-level confusion matrices
            for i in range(imgs.size(0)):
                gt, pr = tgts[i].item(), t1[i].item()
                conf[gt, pr] += 1
                o_gt, f_gt, s_gt = trees[gt]
                o_pr, f_pr, s_pr = trees[pr]
                conf1[o_gt, o_pr] += 1
                conf2[f_gt, f_pr] += 1
                conf3[s_gt, s_pr] += 1

            tot += imgs.size(0)

    # consolidate results into a dict
    res = {
        'loss':      m['loss']     / tot,
        'accuracy':  m['top1_acc'] / tot,
        'top5_acc':  m['top5_acc'] / tot,
        'acc_level1': acc1 / tot,
        'acc_level2': acc2 / tot,
        'acc_level3': acc3 / tot,
        'hdist_avg': (hd_avg / tot).tolist(),
        'hdist_top': (hd_top / tot).tolist(),
        'hprec':     (hprec / tot).tolist(),
        'hmAP':      (hmAP / tot).tolist()
    }

    # Global and per-level Precision/Recall/F1
    prec_g, rec_g, f1_g = prf_from_conf(conf)
    prec_l1, rec_l1, f1_l1 = prf_from_conf(conf1)
    prec_l2, rec_l2, f1_l2 = prf_from_conf(conf2)
    prec_l3, rec_l3, f1_l3 = prf_from_conf(conf3)

    res.update({
        # globais (originais)
        'precision':    prec_g,
        'recall':       rec_g,
        'f1_score':     f1_g,
        # por nível
        'precision_l1': prec_l1,
        'recall_l1':    rec_l1,
        'f1_l1':        f1_l1,
        'precision_l2': prec_l2,
        'recall_l2':    rec_l2,
        'f1_l2':        f1_l2,
        'precision_l3': prec_l3,
        'recall_l3':    rec_l3,
        'f1_l3':        f1_l3,
    })

    return res

# =============================================================================
# ENSEMBLE EVALUATION
# =============================================================================
def evaluate_ensemble(models, loader, loss_fn, distances, classes, trees, device, tta=False):
    for mdl in models:
        mdl.eval()
    max_dist = max(distances.distances.values())
    best_sim = _make_best_hier_similarities(classes, distances, max_dist)

    tot = 0
    stats  = defaultdict(float)
    hd_avg = np.zeros(len(TOPK_TO_CONSIDER))
    hd_top = np.zeros(len(TOPK_TO_CONSIDER))
    hprec  = np.zeros(len(TOPK_TO_CONSIDER))
    hmAP   = np.zeros(len(TOPK_TO_CONSIDER))
    acc1 = acc2 = acc3 = 0

    # confusion matrices
    conf  = np.zeros((len(classes), len(classes)), int)
    n1 = len(set(t[0] for t in trees.values()))
    n2 = len(set(t[1] for t in trees.values()))
    n3 = len(set(t[2] for t in trees.values()))
    conf1 = np.zeros((n1, n1), int)
    conf2 = np.zeros((n2, n2), int)
    conf3 = np.zeros((n3, n3), int)

    desc = "Ensemble TTA" if tta else "Ensemble"
    with torch.no_grad():
        for imgs, tgts in tqdm(loader, desc):
            imgs, tgts = imgs.to(device), tgts.to(device)

            probs_sum = torch.zeros((imgs.size(0), len(classes)), device=device)
            loss_sum  = 0.0

            for mdl in models:
                if not tta:
                    out = mdl(imgs)
                    loss_sum  += loss_fn(out, tgts).item()
                    probs_sum += F.softmax(out, dim=1)
                else:
                    for mode in TTA_MODES:
                        if mode is None:
                            im = imgs
                        elif mode == 'hflip':
                            im = torch.flip(imgs, dims=[3])
                        elif mode == 'vflip':
                            im = torch.flip(imgs, dims=[2])
                        else:
                            im = torch.flip(torch.flip(imgs, dims=[3]), dims=[2])
                        out = mdl(im)
                        loss_sum  += loss_fn(out, tgts).item()
                        probs_sum += F.softmax(out, dim=1)

            n_votes = len(models) * (len(TTA_MODES) if tta else 1)
            loss = loss_sum / n_votes
            probs = probs_sum / n_votes
            outs = torch.log(probs)

            stats['loss']      += float(loss)       * imgs.size(0)
            accs, preds         = accuracy(outs, tgts, TOPK_TO_CONSIDER)
            stats['top1_acc']  += accs[0].item()    * imgs.size(0)
            stats['top5_acc']  += accs[1].item()    * imgs.size(0)

            batch = calculate_metrics(outs, tgts, distances, classes, best_sim, max_dist)
            for i in range(len(TOPK_TO_CONSIDER)):
                hd_avg[i] += batch['hdist_avg'][i] * imgs.size(0)
                hd_top[i] += batch['hdist_top'][i] * imgs.size(0)
                hprec[i]  += batch['hprec'][i]     * imgs.size(0)
                hmAP[i]   += batch['hmAP'][i]      * imgs.size(0)

            l1t, l2t, l3t = get_order_family_target(tgts, trees)
            t1            = preds[0]
            l1p, l2p, l3p = get_order_family_target(t1, trees)
            acc1 += l1p.eq(l1t).sum().item()
            acc2 += l2p.eq(l2t).sum().item()
            acc3 += l3p.eq(l3t).sum().item()

            for i in range(imgs.size(0)):
                gt, pr = tgts[i].item(), t1[i].item()
                conf[gt, pr] += 1
                o_gt, f_gt, s_gt = trees[gt]
                o_pr, f_pr, s_pr = trees[pr]
                conf1[o_gt, o_pr] += 1
                conf2[f_gt, f_pr] += 1
                conf3[s_gt, s_pr] += 1

            tot += imgs.size(0)

    res = {
        'loss':      stats['loss']     / tot,
        'accuracy':  stats['top1_acc'] / tot,
        'top5_acc':  stats['top5_acc'] / tot,
        'acc_level1': acc1 / tot,
        'acc_level2': acc2 / tot,
        'acc_level3': acc3 / tot,
        'hdist_avg': (hd_avg / tot).tolist(),
        'hdist_top': (hd_top / tot).tolist(),
        'hprec':     (hprec / tot).tolist(),
        'hmAP':      (hmAP   / tot).tolist()
    }

    prec_g, rec_g, f1_g = prf_from_conf(conf)
    prec_l1, rec_l1, f1_l1 = prf_from_conf(conf1)
    prec_l2, rec_l2, f1_l2 = prf_from_conf(conf2)
    prec_l3, rec_l3, f1_l3 = prf_from_conf(conf3)

    res.update({
        # globais (originais)
        'precision':    prec_g,
        'recall':       rec_g,
        'f1_score':     f1_g,
        # por nível
        'precision_l1': prec_l1,
        'recall_l1':    rec_l1,
        'f1_l1':        f1_l1,
        'precision_l2': prec_l2,
        'recall_l2':    rec_l2,
        'f1_l2':        f1_l2,
        'precision_l3': prec_l3,
        'recall_l3':    rec_l3,
        'f1_l3':        f1_l3,
    })

    return res


def get_exponential_weighting(hierarchy,value,normalize=True):
    w=deepcopy(hierarchy);all_w=[]
    for pos in w.treepositions():
        wt=np.exp(-value*len(pos));all_w.append(wt)
        if hasattr(w[pos],'set_label'): w[pos].set_label(wt)
        else: w[pos]=wt
    if normalize:
        tot=sum(all_w)
        for pos in w.treepositions():
            if hasattr(w[pos],'set_label'): w[pos].set_label(w[pos].label()/tot)
            else: w[pos]/=tot
    return w


# =============================================================================
# MAIN
# =============================================================================
def main():
    import os, csv, json, pickle
    import torch
    import albumentations as albu
    from collections import defaultdict
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load distances and hierarchy
    with open('tct_distances.pkl', 'rb') as f:
        dist = DistanceDict(pickle.load(f))
    with open('tct_tree.pkl', 'rb') as f:
        hier = pickle.load(f)

    # 'flat' hierarchy for per-level accuracy
    trees = {
        0:[0,0,0], 1:[0,1,1], 2:[0,2,2], 3:[0,3,3], 4:[0,4,4],
        5:[0,5,5], 6:[0,6,6], 7:[0,7,7], 8:[1,8,8], 9:[1,9,9],
       10:[1,10,10],11:[1,11,11],12:[1,12,12],13:[2,14,13],14:[2,13,14],
       15:[2,13,15],16:[2,15,16],17:[2,15,17],18:[3,16,18],19:[3,17,19],
       20:[3,18,20],21:[3,19,21],22:[3,20,22]
    }

    classes = [
        'Normal','ECC','RPC','MPC','PG','Atrophy','EMC','HCG','ASC-US',
        'LSIL','ASC-H','HSIL','SCC','AGC-FN','AGC-ECC-NOS','AGC-EMC-NOS',
        'ADC-ECC','ADC-EMC','FUNGI','ACTINO','TRI','HSV','CC'
    ]

    # DataLoader
    norm = transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    tx   = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                               transforms.ToTensor(), norm])
    ax   = albu.Compose([albu.Resize(*IMAGE_SIZE)])
    ds   = InputDataset(TEST_CSV, transform=tx, alb=ax)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # Load the 4 models
    models = {}
    for cfg in MODEL_CONFIGS:
        m = CustomModel(cfg['backbone'], NUM_CLASSES).to(device)
        ckpt = torch.load(cfg['ckpt'], map_location=device, weights_only=False)
        state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        m.load_state_dict(state_dict)
        models[cfg['name']] = m

    # Loss function
    weights = get_exponential_weighting(hier, value=0.8)
    loss_fn = HierarchicalCrossEntropyLoss(hier, classes, weights).to(device)

    # Output files
    json_path = "evaluation_all_models.json"
    csv_path  = "evaluation_all_models.csv"

    # Collect already executed combos
    seen = set()
    existing = []
    if os.path.exists(json_path):
        with open(json_path) as f:
            existing = json.load(f)
        for r in existing:
            seen.add((r['model'], r['run'], bool(r['tta']), bool(r['ensemble'])))

    # Prepare CSV for appending
    csv_file = open(csv_path, "a", newline="")
    fieldnames = [
    'model','run','tta','ensemble',
    'loss','accuracy','top5_acc',
    'acc_level1','acc_level2','acc_level3',
    'hdist_avg','hdist_top','hprec','hmAP',
    'precision','recall','f1_score',
    'precision_l1','recall_l1','f1_l1',
    'precision_l2','recall_l2','f1_l2',
    'precision_l3','recall_l3','f1_l3',
    ]

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    if os.stat(csv_path).st_size == 0:
        writer.writeheader()

    def save_record(rec):
        writer.writerow({k: rec[k] for k in fieldnames})
        csv_file.flush()
        existing.append(rec)
        with open(json_path, "w") as jf:
            json.dump(existing, jf, indent=2)

    # List of tasks: individual + ensembles
    tasks = []
    for name in models:
        for run in range(1, NUM_RUNS+1):
            for tta in (False, True):
                tasks.append((False, name, run, tta))
    for ens in ENS_BUNDLES:
        for run in range(1, NUM_RUNS+1):
            for tta in (False, True):
                tasks.append((True, ens['name'], run, tta))

    # Execute (or resume) all tasks
    for is_ens, key, run, tta in tasks:
        combo = (key, run, tta, is_ens)
        if combo in seen:
            continue

        if not is_ens:
            m   = models[key]
            res = evaluate_model(m, loader, loss_fn,
                                 dist, classes, trees, device, tta=tta)
        else:
            names = next(e for e in ENS_BUNDLES if e['name']==key)['members']
            members = [models[n] for n in names]
            res = evaluate_ensemble(members, loader, loss_fn,
                                    dist, classes, trees, device, tta=tta)

        rec = {
            'model':    key,
            'run':      run,
            'tta':      tta,
            'ensemble': is_ens,
            **res
        }
        save_record(rec)
        seen.add(combo)

    csv_file.close()


if __name__ == "__main__":
    main()


