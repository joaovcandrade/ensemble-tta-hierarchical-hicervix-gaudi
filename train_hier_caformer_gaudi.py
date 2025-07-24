#!/usr/bin/env python

# =============================================================================
# IMPORTS
# =============================================================================

# --- Standard library imports ---
import os
import os.path as osp
import json
import shutil
import random
import warnings
import pickle
import lzma
import time
from datetime import datetime
from math import exp, fsum
from copy import deepcopy
from collections import defaultdict
from typing import List
from distutils.util import strtobool as boolean
from pprint import PrettyPrinter
from tqdm import tqdm
from timm.data.mixup import Mixup
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import math


# --- Third-party libraries ---
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.autograd import Variable
import torch.nn.functional as F
# to ensure the Habana plugin is loaded
import habana_frameworks.torch.core as htcore  


import torchvision.transforms as transforms

import albumentations as albu
import timm
import tensorboardX

from nltk import Tree
from nltk.tree import Tree as NltkTree

from conditional import conditional

# =============================================================================
# GLOBAL CONSTANTS AND CONFIGURATIONS
# =============================================================================
TOPK_TO_CONSIDER = (1, 5)
ACCURACY_IDS = ["accuracy_top/%02d" % k for k in TOPK_TO_CONSIDER]
DIST_AVG_IDS = ["_avg/%02d" % k for k in TOPK_TO_CONSIDER]
DIST_TOP_IDS = ["_top/%02d" % k for k in TOPK_TO_CONSIDER]
DIST_AVG_MISTAKES_IDS = ["_mistakes/avg%02d" % k for k in TOPK_TO_CONSIDER]
HPREC_IDS = ["_precision/%02d" % k for k in TOPK_TO_CONSIDER]
HMAP_IDS = ["_mAP/%02d" % k for k in TOPK_TO_CONSIDER]

# =============================================================================
# UTILITY FUNCTIONS AND CLASSES
# =============================================================================
def make_weighted_sampler(dataset, num_classes=23):
    """
    Creates a sampler that assigns inverse weight to the frequency of each class.
    Uses the 'class_id' field of the dataset (corresponding to the level_3_id, ranging from 0 to 22).
    """
    class_counts = np.zeros(num_classes, dtype=int)
    for _, target in dataset.data:
        if 0 <= target < num_classes:
            class_counts[target] += 1
    weights = 1. / np.maximum(class_counts, 1)
    sample_weights = [weights[target] if 0 <= target < num_classes else 0.0 for _, target in dataset.data]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

def get_expm_folder(script_path, output_folder='out', expm_id=''):
    """
    Generates the experiment folder path based on the script name and the current date.
    """
    script_name = os.path.basename(script_path)
    folder_path = os.path.splitext(script_name)[0]
    now = datetime.today().strftime('%Y-%m-%d_%H%M')
    bottom_folder = now + ('_' + expm_id if expm_id else '')
    return os.path.join(output_folder, folder_path, bottom_folder)

class DistanceDict(dict):
    """
    Symmetric dictionary to store distances between classes.
    Keys are ordered pairs of classes.
    """
    def __init__(self, distances):
        self.distances = {tuple(sorted(t)): v for t, v in distances.items()}
    
    def __getitem__(self, indices):
        if indices[0] == indices[1]:
            return 0
        key = (indices[0], indices[1]) if indices[0] < indices[1] else (indices[1], indices[0])
        return self.distances[key]
    
    def __setitem__(self, key, value):
        raise NotImplementedError("Setting items is not supported in DistanceDict.")

def get_label(node):
    """
    Returns the label of a node (if it is an nltk tree) or the node itself.
    """
    return node.label() if isinstance(node, Tree) else node

def get_uniform_weighting(hierarchy: Tree, value):
    """
    Constructs a tree of uniform weights based on a hierarchy.
    """
    weights = deepcopy(hierarchy)
    for p in weights.treepositions():
        node = weights[p]
        if isinstance(node, Tree):
            node.set_label(value)
        else:
            weights[p] = value
    return weights

def get_exponential_weighting(hierarchy: Tree, value, normalize=True):
    """
    Constructs a tree with exponentially decayed weights.
    """
    weights = deepcopy(hierarchy)
    all_weights = []
    for p in weights.treepositions():
        node = weights[p]
        weight = exp(-value * len(p))
        all_weights.append(weight)
        if isinstance(node, Tree):
            node.set_label(weight)
        else:
            weights[p] = weight
    total = fsum(all_weights)
    if normalize:
        for p in weights.treepositions():
            node = weights[p]
            if isinstance(node, Tree):
                node.set_label(node.label() / total)
            else:
                weights[p] /= total
    return weights

def get_weighting(hierarchy: Tree, weighting="uniform", **kwargs):
    """
    Returns a tree of weights according to the specified type.
    """
    if weighting == "exponential":
        return get_exponential_weighting(hierarchy, **kwargs)
    elif weighting == "uniform":
        return get_uniform_weighting(hierarchy, kwargs.get("value", 1))
    else:
        raise ValueError("Unsupported weighting type: {}".format(weighting))

def make_all_soft_labels(distances, classes, hardness):
    """
    Creates soft labels for all classes from the distance matrix.
    """
    distance_matrix = torch.Tensor([[distances[c1, c2] for c1 in classes] for c2 in classes])
    max_distance = torch.max(distance_matrix)
    distance_matrix /= max_distance
    soft_labels = torch.exp(-hardness * distance_matrix)
    soft_labels = soft_labels / torch.sum(torch.exp(-hardness * distance_matrix), dim=0)
    return soft_labels

def make_weighted_soft_labels(distances, classes, hardness, class_freq):
    """
    Creates weighted soft labels based on class frequencies.
    """
    distance_matrix = torch.Tensor([[distances[c1, c2] for c1 in classes] for c2 in classes])
    max_distance = torch.max(distance_matrix)
    distance_matrix /= max_distance
    soft_labels = torch.exp(-hardness * distance_matrix)
    weights = torch.tensor([1.0 / class_freq.get(c, 1) for c in classes], dtype=torch.float32)
    soft_labels = soft_labels * weights.unsqueeze(0)
    soft_labels = soft_labels / torch.sum(soft_labels, dim=0)
    return soft_labels

def compute_class_frequency(dataset, num_classes=23):
    """
    Calculates the frequency of each class in the dataset.
    """
    freq = {i: 0 for i in range(num_classes)}
    for _, target in dataset.data:
        if 0 <= target < num_classes:
            freq[target] += 1
    return freq

def accuracy(output, target, ks=(1,)):
    """
    Calculates top-k accuracy.
    Returns a list of accuracies and the transposed predictions (shape: [maxk, batch_size]).
    """
    with torch.no_grad():
        maxk = max(ks)
        batch_size = target.size(0)
        _, pred_ = output.topk(maxk, 1, True, True)  # pred_ has shape (batch_size, maxk)
        pred = pred_.t()  # transpose: (maxk, batch_size)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in ks:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res, pred

def _make_best_hier_similarities(classes, distances, max_dist):
    """
    Creates the "optimal" hierarchical similarity matrix for each class.
    """
    num_classes = len(classes)
    distance_matrix = np.zeros((num_classes, num_classes))
    best_hier_similarities = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            distance_matrix[i, j] = distances[(classes[i], classes[j])]
    for i in range(num_classes):
        best_hier_similarities[i, :] = 1 - np.sort(distance_matrix[i, :]) / max_dist
    return best_hier_similarities

def _generate_summary(loss_accum, flat_accuracy_accums, hdist_accums, hdist_top_accums,
                      hdist_mistakes_accums, hprecision_accums, hmAP_accums, num_logged,
                      norm_mistakes_accum, loss_id, dist_id):
    """
    Generates a dictionary with the summary of metrics for an epoch.
    """
    summary = dict()
    summary[loss_id] = loss_accum / num_logged
    summary.update({ACCURACY_IDS[i]: flat_accuracy_accums[i] / num_logged for i in range(len(TOPK_TO_CONSIDER))})
    summary.update({dist_id + DIST_AVG_IDS[i]: hdist_accums[i] / num_logged for i in range(len(TOPK_TO_CONSIDER))})
    summary.update({dist_id + DIST_TOP_IDS[i]: hdist_top_accums[i] / num_logged for i in range(len(TOPK_TO_CONSIDER))})
    summary.update({dist_id + DIST_AVG_MISTAKES_IDS[i]: hdist_mistakes_accums[i] / (norm_mistakes_accum * TOPK_TO_CONSIDER[i]) for i in range(len(TOPK_TO_CONSIDER))})
    summary.update({dist_id + HPREC_IDS[i]: hprecision_accums[i] / num_logged for i in range(len(TOPK_TO_CONSIDER))})
    summary.update({dist_id + HMAP_IDS[i]: hmAP_accums[i] / num_logged for i in range(len(TOPK_TO_CONSIDER))})
    return summary

def _update_tb_from_summary(summary, writer, steps, loss_id, dist_id):
    """
    Updates TensorBoard with metric values.
    """
    writer.add_scalar(loss_id, summary[loss_id], steps)
    for i in range(len(TOPK_TO_CONSIDER)):
        writer.add_scalar(ACCURACY_IDS[i], summary[ACCURACY_IDS[i]] * 100, steps)
        writer.add_scalar(dist_id + DIST_AVG_IDS[i], summary[dist_id + DIST_AVG_IDS[i]], steps)
        writer.add_scalar(dist_id + DIST_TOP_IDS[i], summary[dist_id + DIST_TOP_IDS[i]], steps)
        writer.add_scalar(dist_id + DIST_AVG_MISTAKES_IDS[i], summary[dist_id + DIST_AVG_MISTAKES_IDS[i]], steps)
        writer.add_scalar(dist_id + HPREC_IDS[i], summary[dist_id + HPREC_IDS[i]] * 100, steps)
        writer.add_scalar(dist_id + HMAP_IDS[i], summary[dist_id + HMAP_IDS[i]] * 100, steps)

def get_order_family_target(targets):
    """
    Separates targets into three levels (order, family, species) based on the global tree.
    """
    order_target_list = []
    family_target_list = []
    species_target_list = []
    for i in range(targets.size(0)):
        target_id = int(targets[i])
        order, family, species = trees[target_id]
        if -1 in (order, family, species):
            raise ValueError(f"Target {targets[i]} has -1 at some level: {[order, family, species]}")
        order_target_list.append(order)
        family_target_list.append(family)
        species_target_list.append(species)
    order_target_list = Variable(torch.from_numpy(np.array(order_target_list)))
    family_target_list = Variable(torch.from_numpy(np.array(family_target_list)))
    species_target_list = Variable(torch.from_numpy(np.array(species_target_list)))
    return order_target_list, family_target_list, species_target_list

def run(loader, model, loss_function, distances, all_soft_labels, classes, epoch, prev_steps,
        optimizer=None, scheduler=None, is_inference=True, corrector=lambda x: x):

    """
    Executes training or inference and accumulates metrics.
    """
    max_dist = max(distances.distances.values()) if distances.distances else 1.0
    best_hier_similarities = _make_best_hier_similarities(classes, distances, max_dist)
    log_freq = 1 if is_inference else 100
    descriptor = "VAL" if is_inference else "TRAIN"
    loss_id = "loss/HXE"
    dist_id = "ilsvrc_dist"
    with_tb = True

    if with_tb:
        tb_writer = tensorboardX.SummaryWriter(os.path.join("output", "tb", descriptor.lower()))

    num_logged = 0
    loss_accum = 0.0
    time_accum = 0.0
    norm_mistakes_accum = 0.0
    flat_accuracy_accums = np.zeros(len(TOPK_TO_CONSIDER), dtype=float)
    hdist_accums = np.zeros(len(TOPK_TO_CONSIDER))
    hdist_top_accums = np.zeros(len(TOPK_TO_CONSIDER))
    hdist_mistakes_accums = np.zeros(len(TOPK_TO_CONSIDER))
    hprecision_accums = np.zeros(len(TOPK_TO_CONSIDER))
    hmAP_accums = np.zeros(len(TOPK_TO_CONSIDER))
    species_probs = []

    if is_inference:
        model.eval()
    else:
        model.train()

    with conditional(is_inference, torch.no_grad()):
        time_load0 = time.time()
        for batch_idx, (embeddings, target) in enumerate(
            tqdm(loader, desc=f"Epoch {epoch} {'Inference' if is_inference else 'Training'}")
        ):
            this_load_time = time.time() - time_load0
            this_rest0 = time.time()
            embeddings = embeddings.to("hpu", non_blocking=True)
            _, _, target = get_order_family_target(target)
            target = target.to("hpu", non_blocking=True)
            with torch.autocast(device_type="hpu", dtype=torch.bfloat16):
                output = model(embeddings)
                species_probs.extend(F.softmax(output, dim=1).tolist())
                loss = loss_function(output, target)
            if not is_inference:
                optimizer.zero_grad()
                loss.backward()
                htcore.mark_step()
                optimizer.step()
                htcore.mark_step()
                scheduler.step()
            this_rest_time = time.time() - this_rest0
            time_accum += this_load_time + this_rest_time
            time_load0 = time.time()
            tot_steps = prev_steps if is_inference else prev_steps + batch_idx
            output_corrected = corrector(output)
            if batch_idx % log_freq == 0:
                num_logged += 1
                loss_accum += loss.item()
                topK_accuracies, topK_predicted_classes = accuracy(output_corrected, target, ks=TOPK_TO_CONSIDER)
                bs = target.size(0)
                topK_hdist = np.empty((bs, TOPK_TO_CONSIDER[-1]))
                for b in range(bs):
                    for k in range(TOPK_TO_CONSIDER[-1]):
                        class_idx_ground_truth = target[b].item()
                        class_idx_predicted = topK_predicted_classes[k, b].item()
                        topK_hdist[b, k] = distances[(classes[class_idx_predicted], classes[class_idx_ground_truth])]
                mistakes_ids = np.where(topK_hdist[:, 0] != 0)[0]
                norm_mistakes_accum += len(mistakes_ids)
                topK_hdist_mistakes = topK_hdist[mistakes_ids, :]
                topK_hsimilarity = 1 - topK_hdist / max_dist
                topK_AP = [
                    np.sum(topK_hsimilarity[:, :k]) / np.sum(best_hier_similarities[:, :k])
                    for k in range(1, TOPK_TO_CONSIDER[-1] + 1)
                ]
                for i in range(len(TOPK_TO_CONSIDER)):
                    flat_accuracy_accums[i] += topK_accuracies[i].item()
                    hdist_accums[i] += np.mean(topK_hdist[:, :TOPK_TO_CONSIDER[i]])
                    hdist_top_accums[i] += np.mean([np.min(topK_hdist[b, :TOPK_TO_CONSIDER[i]]) for b in range(bs)])
                    hdist_mistakes_accums[i] += np.sum(topK_hdist_mistakes[:, :TOPK_TO_CONSIDER[i]])
                    hprecision_accums[i] += topK_AP[TOPK_TO_CONSIDER[i] - 1]
                    hmAP_accums[i] += np.mean(topK_AP[:TOPK_TO_CONSIDER[i]])
                if with_tb and not is_inference:
                    summary = _generate_summary(
                        loss_accum, flat_accuracy_accums, hdist_accums,
                        hdist_top_accums, hdist_mistakes_accums, hprecision_accums,
                        hmAP_accums, num_logged, norm_mistakes_accum, loss_id, dist_id
                    )
                    _update_tb_from_summary(summary, tb_writer, tot_steps, loss_id, dist_id)
        summary = _generate_summary(
            loss_accum, flat_accuracy_accums, hdist_accums,
            hdist_top_accums, hdist_mistakes_accums, hprecision_accums,
            hmAP_accums, num_logged, norm_mistakes_accum, loss_id, dist_id
        )
        if with_tb:
            _update_tb_from_summary(summary, tb_writer, tot_steps, loss_id, dist_id)
            tb_writer.close()
        return summary, tot_steps, species_probs


def create_scheduler(optimizer, warmup_epochs=5, max_epochs=100):
    # Combine linear warmup with cosine annealing
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda=lr_lambda)

# =============================================================================
# MODEL DEFINITION
# =============================================================================
class CustomCaFormer(nn.Module):
    """
    CustomCaFormer Model with modified head.

    Uses the pre-trained 'caformer_s36.sail_in22k_ft_in1k_384' from timm with the head removed.
    After feature extraction, applies global pooling, LayerNorm, and a custom fully-connected network.
    
    The feature dimension is automatically detected from a forward pass with a dummy input.

    """
    def __init__(self, num_classes, input_size=(384, 384)):
        """
        Args:
            num_classes: Number of output classes (23).
            input_size: Input image size (height, width).
        """
        super().__init__()
        # Create the backbone model with head removed.
        self.backbone = timm.create_model(
            'caformer_s36.sail_in22k_ft_in1k_384',
            pretrained=True,
            num_classes=0
        )
        self._freeze_backbone()
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Automatically detects the feature dimension from a dummy forward pass.

        device = next(self.backbone.parameters()).device  
        dummy = torch.randn(1, 3, *input_size, device=device)
        with torch.no_grad():
            features = self.backbone.forward_features(dummy)
            pooled = self.pool(features)  # Expects [1, C, 1, 1]
        feature_dim = pooled.flatten(1).shape[1]
        print(f"Detected feature_dim = {feature_dim}")

        self.norm = nn.LayerNorm(feature_dim, eps=1e-6)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), 
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Performs the forward pass.
        """
        x = self.backbone.forward_features(x)  # Feature extraction
        x = self.pool(x)                       # Global pooling: output [B, C, 1, 1]
        x = x.flatten(1)                       # Reduces to [B, C]
        x = self.norm(x)
        x = self.fc(x)
        return x

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_layers(self, num_layers):
        """Unfreezes the last `num_layers` layers of the backbone."""
        total_layers = len(list(self.backbone.children()))
        layers_to_unfreeze = list(self.backbone.children())[-num_layers:]
        
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
                
    def unfreeze_components(self, component_names: list):
        """Unfreezes specific components by name"""
        for name, param in self.backbone.named_parameters():
            if any(comp in name for comp in component_names):
                param.requires_grad = True
                print(f"✅ Unfroze: {name}")


def init_model_on_gpu():
    """
    Initializes the CustomCaFormer model on the available device.
    """
    print("=> Using CaFormer with custom head")
    model = CustomCaFormer(num_classes=23)
    device = torch.device("hpu")
    return model.to(device)

# =============================================================================
# DEFINIÇÃO DAS FUNÇÕES DE PERDA
# =============================================================================
class HierarchicalLLLoss(nn.Module):
    """
    Hierarchical Log Likelihood Loss based on an nltk tree to define the weights.
    """
    def __init__(self, hierarchy: Tree, classes: List[str], weights: Tree):
        super(HierarchicalLLLoss, self).__init__()
        assert hierarchy.treepositions() == weights.treepositions()
        positions_leaves = {get_label(hierarchy[p]): p for p in hierarchy.treepositions("leaves")}
        print("Leaf positions:", list(positions_leaves.keys()))
        num_classes = len(positions_leaves)
        positions_leaves = [positions_leaves[c] for c in classes]
        positions_edges = hierarchy.treepositions()[1:]
        index_map_leaves = {positions_leaves[i]: i for i in range(len(positions_leaves))}
        index_map_edges = {positions_edges[i]: i for i in range(len(positions_edges))}
        edges_from_leaf = [[index_map_edges[position[:i]] for i in range(len(position), 0, -1)] for position in positions_leaves]
        num_edges = max(len(p) for p in edges_from_leaf)
        def get_leaf_positions(position):
            node = hierarchy[position]
            return node.treepositions("leaves") if isinstance(node, Tree) else [()]
        leaf_indices = [[index_map_leaves[position + leaf] for leaf in get_leaf_positions(position)] for position in positions_edges]
        self.onehot_den = nn.Parameter(torch.zeros((num_classes, num_classes, num_edges)), requires_grad=False)
        self.onehot_num = nn.Parameter(torch.zeros((num_classes, num_classes, num_edges), dtype=torch.float32), requires_grad=False)
        self.weights = nn.Parameter(torch.zeros((num_classes, num_edges)), requires_grad=False)
        for i in range(num_classes):
            for j, k in enumerate(edges_from_leaf[i]):
                self.onehot_num[i, leaf_indices[k], j] = 1.0
                self.weights[i, j] = get_label(weights[positions_edges[k]])
            for j, k in enumerate(edges_from_leaf[i][1:]):
                self.onehot_den[i, leaf_indices[k], j] = 1.0
            self.onehot_den[i, :, j + 1] = 1.0
    
    def forward(self, inputs, target):
        """
        Performs the forward pass calculating the loss.
        """
        inputs = torch.unsqueeze(inputs, 1)
        num = torch.squeeze(torch.bmm(inputs, self.onehot_num[target].contiguous()))
        den = torch.squeeze(torch.bmm(inputs, self.onehot_den[target]))
        idx = num != 0
        num[idx] = -torch.log(num[idx] / den[idx]).to(num.dtype)
        num = torch.sum(torch.flip(self.weights[target] * num, dims=[1]), dim=1)
        return torch.mean(num)

class HierarchicalCrossEntropyLoss(HierarchicalLLLoss):
    """
    Combines softmax with HierarchicalLLLoss to calculate hierarchical cross entropy.
    """
    def __init__(self, hierarchy: Tree, classes: List[str], weights: Tree):
        super(HierarchicalCrossEntropyLoss, self).__init__(hierarchy, classes, weights)
    
    def forward(self, inputs, index):
        return super(HierarchicalCrossEntropyLoss, self).forward(F.softmax(inputs, 1), index)

# =============================================================================
# DEFINIÇÃO DO DATASET E AUGMENTAÇÕES
# =============================================================================
class InputDataset(Dataset):
    """
    Custom dataset that loads images and labels from a CSV,
    with support for transformations via torchvision and albumentations.
    """
    def __init__(self, data_csv_file, train=True, transform=None,
                 target_transform=None, albu_transform=None,
                 rare_albu_transform=None, rare_prob=None):
        """
        Args:
            data_csv_file: CSV containing [image_path, class_id].
            train: Flag if the dataset is for training.
            transform: Transformations from torchvision.
            albu_transform: Augmentation pipeline with albumentations.
            rare_albu_transform: Special pipeline for rare classes.
            rare_prob: Dictionary with probability of extra augmentation.
        """
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.albu_transform = albu_transform
        self.rare_albu_transform = rare_albu_transform
        self.rare_prob = rare_prob if rare_prob is not None else {}
        df = pd.read_csv(data_csv_file)
        self.data = []
        for n in range(len(df)):
            row = df.iloc[n]
            image_path = row["image_path"]
            class_id = int(row["class_id"])
            self.data.append((image_path, class_id))
        if self.train:
            random.shuffle(self.data)
    
    def __getitem__(self, index):
        img_path, target = self.data[index]
        img_path = os.path.expanduser(img_path)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: could not read {img_path}. Returning blank image.")
            img = np.zeros((384, 384, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.albu_transform is not None:
            p_aug = self.rare_prob.get(target, 0)
            if self.rare_albu_transform is not None and random.random() < p_aug:
                img = self.rare_albu_transform(image=img)["image"]
            else:
                img = self.albu_transform(image=img)["image"]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.data)

def detect_rare_classes(dataset, num_classes=23, threshold=200):
    """
    Detects rare classes based on frequency.
    """
    freq = compute_class_frequency(dataset, num_classes=num_classes)
    rare = [cls for cls, count in freq.items() if count < threshold]
    print("Frequency of classes (0 to {}):".format(num_classes - 1), freq)
    print("Detected rare classes (freq < {}): {}".format(threshold, rare))
    return rare

def compute_rare_probabilities(dataset, num_classes=23, alpha=0.5):
    freq = compute_class_frequency(dataset, num_classes)
    max_freq = max(freq.values())
    rare_prob = {cls: (max_freq - count + alpha) / (max_freq + alpha) for cls, count in freq.items()}
    return rare_prob

# =============================================================================
# GLOBAL VARIABLE: CLASS TREE (Hierarchy)
# =============================================================================
# Levels: level_1, level_2, level_3 (23 classes)
trees =  {0: [0, 0, 0],
 1: [0, 1, 1],
 2: [0, 2, 2],
 3: [0, 3, 3],
 4: [0, 4, 4],
 5: [0, 5, 5],
 6: [0, 6, 6],
 7: [0, 7, 7],
 8: [1, 8, 8],
 9: [1, 9, 9],
 10: [1, 10, 10],
 11: [1, 11, 11],
 12: [1, 12, 12],
 13: [2, 14, 13],
 14: [2, 13, 14],
 15: [2, 13, 15],
 16: [2, 15, 16],
 17: [2, 15, 17],
 18: [3, 16, 18],
 19: [3, 17, 19],
 20: [3, 18, 20],
 21: [3, 19, 21],
 22: [3, 20, 22]}

def get_order_family_target(targets):
    """
    Separates targets into three levels (order, family, species) based on the global tree.
    """
    order_target_list = []
    family_target_list = []
    species_target_list = []
    for i in range(targets.size(0)):
        target_id = int(targets[i])
        order, family, species = trees[target_id]
        if -1 in (order, family, species):
            raise ValueError(f"Target {targets[i]} has -1 at some level: {[order, family, species]}")
        order_target_list.append(order)
        family_target_list.append(family)
        species_target_list.append(species)
    order_target_list = Variable(torch.from_numpy(np.array(order_target_list)))
    family_target_list = Variable(torch.from_numpy(np.array(family_target_list)))
    species_target_list = Variable(torch.from_numpy(np.array(species_target_list)))
    return order_target_list, family_target_list, species_target_list

# Function to recreate the optimizer
def create_optimizer(model, lr_mult=1.0):
    params = [
        {'params': model.fc.parameters(), 'lr': 3e-4},
        {'params': (p for p in model.backbone.parameters() if p.requires_grad), 'lr': 3e-5 * lr_mult}
    ]
    return torch.optim.AdamW(params, weight_decay=1e-3)


# =============================================================================
# CHECKPOINTS E FUNÇÕES AUXILIARES DE AVALIAÇÃO
# =============================================================================
def _load_checkpoint(model, optimizer, scheduler):
    checkpoint_path = os.path.join("output", "checkpoint.pth.tar")
    start_epoch = 0
    steps = 0
    unfreeze_schedule = {}
    current_lr_mult = 1.0
    if os.path.isfile(checkpoint_path):
        print("=> Loading checkpoint '{}'".format("output"))
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint["epoch"]
        steps = checkpoint.get("steps", 0)
        unfreeze_schedule = checkpoint.get("unfreeze_schedule", {})
        current_lr_mult = checkpoint.get("current_lr_mult", 1.0)
        
        # Load model
        model.load_state_dict(checkpoint["state_dict"])
        
        # Load scheduler
        if "scheduler" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        
        # Load optimizer
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except (ValueError, KeyError) as e:
            print(f"=> Error loading optimizer: {e}. Optimizer reset.")
        
        # Apply unfreezing policies from the checkpoint
        print("Applying unfreezing policies up to epoch", start_epoch)
        for epoch_key in sorted(unfreeze_schedule.keys()):
            if epoch_key <= start_epoch:
                schedule = unfreeze_schedule[epoch_key]
                model.unfreeze_components(schedule["components"])
                print(f"✅ Applied unfreezing from epoch {epoch_key}: {schedule['components']}")
        
        # Recreate optimizer and scheduler with checkpoint parameters
        optimizer = create_optimizer(model, lr_mult=current_lr_mult)
        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"], strict=False)
            except:
                print("Could not load the optimizer, initializing from scratch.")
        
        scheduler = create_scheduler(optimizer, warmup_epochs=5, max_epochs=100)
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
    
    return start_epoch, steps, optimizer, scheduler, unfreeze_schedule, current_lr_mult

def _save_checkpoint(state, do_validate, epoch, out_folder, unfreeze_schedule, current_lr_mult):
    state.update({
        "unfreeze_schedule": unfreeze_schedule,
        "current_lr_mult": current_lr_mult
    })
    filename = os.path.join(out_folder, "checkpoint.pth.tar")
    torch.save(state, filename)
    if do_validate:
        snapshot_name = "checkpoint.epoch%04d" % epoch + ".pth.tar"
        shutil.copy(filename, os.path.join(out_folder, "model_snapshots", snapshot_name))



def load_data(train_csv, val_csv, distributed):
    """
    Loads the training and validation data with transformations for 256x256.
    """
    print("Loading data and defining transformations...")

    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # torchvision Transformations: ToTensor and Normalize
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Albumentations augmentation pipeline for training
    train_albu_transform = albu.Compose([
        # Using the new API: RandomResizedCrop expects the "size" parameter
        albu.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0), ratio=(0.75, 1.33), always_apply=True),
        albu.OneOf([
            albu.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            albu.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        ], p=0.7),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ])

    # Aggressive pipeline for rare classes
    rare_albu_transform = albu.Compose([
        albu.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0), ratio=(0.75, 1.33), always_apply=True),
        albu.OneOf([
            albu.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            albu.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        ], p=0.7),
        albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.9),
        albu.GaussianBlur(blur_limit=(3, 7), p=0.5),
        albu.CoarseDropout(max_holes=8, max_height=50, max_width=50, p=0.3),
    ])

    # Augmentation pipeline for validation: only resize
    test_albu_transform = albu.Compose([
        albu.Resize(height=384, width=384, interpolation=cv2.INTER_LINEAR, always_apply=True),
    ])

    dataset_basic = InputDataset(train_csv, train=True, transform=None, albu_transform=None)
    rare_prob = compute_rare_probabilities(dataset_basic, num_classes=23)

    print("Loading training data")
    st = time.time()
    dataset = InputDataset(train_csv, True, transform=train_transform,
                           albu_transform=train_albu_transform,
                           rare_albu_transform=rare_albu_transform,
                           rare_prob=rare_prob)
    print("Took", time.time() - st)

    print("Loading validation data")
    dataset_test = InputDataset(val_csv, False, test_transform, albu_transform=test_albu_transform)

    print("Creating data loaders")
    train_sampler = make_weighted_sampler(dataset, num_classes=23)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler



# =============================================================================
# MAIN WORKER (TREINAMENTO/VALIDAÇÃO)
# =============================================================================
def main_worker():
    """
    Coordinates the training and validation of the model.
    """
    cudnn.benchmark = True
    pp = PrettyPrinter(indent=4)
    batch_size = 98
    dataset, dataset_test, train_sampler, test_sampler = load_data("data/train_mbm.csv", "data/val_mbm.csv", distributed=False)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=16,
        sampler=test_sampler, num_workers=4, pin_memory=True, drop_last=True)

    print('Number of epochs: {}'.format(100))
    print(os.getcwd())

    with open('tct_distances.pkl', "rb") as f:
        distances = DistanceDict(pickle.load(f))
    with open('tct_tree.pkl', "rb") as f:
        hierarchy = pickle.load(f)

    classes = ['Normal', 'ECC', 'RPC', 'MPC', 'PG', 'Atrophy', 'EMC', 'HCG', 'ASC-US',
               'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-FN', 'AGC-ECC-NOS', 'AGC-EMC-NOS',
               'ADC-ECC', 'ADC-EMC', 'FUNGI', 'ACTINO', 'TRI', 'HSV', 'CC']


    model = init_model_on_gpu()
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer, warmup_epochs=5, max_epochs=100)

    start_epoch, steps, optimizer, scheduler, unfreeze_schedule, current_lr_mult = _load_checkpoint(model, optimizer, scheduler)
    
    # If no checkpoint, use default schedule
    if not unfreeze_schedule:        
        unfreeze_schedule = {
             5: {"components": ["stages.3"], "lr_mult": 0.2},  # the last stage
            10: {"components": ["stages.2"], "lr_mult": 0.3},  # 
            15: {"components": ["stages.1"], "lr_mult": 0.4},
            20: {"components": ["stages.0"], "lr_mult": 0.5},
            25: {"components": ["stem"],     "lr_mult": 0.6},  # the stem
        }








    print("Using hierarchical-cross-entropy...")
    weights = get_weighting(hierarchy, "exponential", value=0.8)
    loss_function = HierarchicalCrossEntropyLoss(hierarchy, classes, weights).to("hpu")
    corrector = lambda x: x

    class_freq = compute_class_frequency(dataset, num_classes=23)
    soft_labels = make_weighted_soft_labels(distances, classes, 2.0, class_freq)

    for epoch in range(start_epoch, 100):  # Single epoch loop
        json_name = f"epoch.{epoch:04d}.json"  # Define JSON name
        do_validate = (epoch % 1 == 0)  # Validate every epoch

        # Apply progressive unfreezing policy
        if epoch in unfreeze_schedule:
            schedule = unfreeze_schedule[epoch]
            model.unfreeze_components(schedule["components"])  
            optimizer = create_optimizer(model, lr_mult=schedule["lr_mult"])
            new_scheduler = create_scheduler(optimizer, warmup_epochs=5, max_epochs=100)
            if scheduler is not None:
                new_scheduler.load_state_dict(scheduler.state_dict())
            scheduler = new_scheduler



        # --- Training ---
        summary_train, steps, _ = run(
            train_loader, model, loss_function, distances, soft_labels, classes, 
            epoch, steps, optimizer=optimizer, scheduler=scheduler, 
            is_inference=False, corrector=corrector
        )

        # --- Validation and Checkpoints ---
        with open(os.path.join("output", "json/train", f"epoch.{epoch:04d}.json"), "w") as fp:
            json.dump(summary_train, fp)

        state = {
            "epoch": epoch + 1,
            "steps": steps,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), 
        }

        applicable_epochs = [k for k in unfreeze_schedule if k <= epoch]
        current_lr_mult = unfreeze_schedule[max(applicable_epochs)]["lr_mult"] if applicable_epochs else 1.0
        _save_checkpoint(state, do_validate, epoch, "output", unfreeze_schedule, current_lr_mult)

        if do_validate:
            summary_val, steps, _ = run(
                val_loader, model, loss_function, distances, soft_labels, classes, epoch, steps,
                is_inference=True, corrector=corrector,
            )
            print("\nSummary for epoch %04d (for val set):" % epoch)
            pp.pprint(summary_val)
            print("\n\n")
            with open(os.path.join("output", "json/val", json_name), "w") as fp:
                json.dump(summary_val, fp)

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import traceback
    import gc

    out_folder = "output"
    print("Ensuring all output folders exist...")
    os.makedirs(os.path.join(out_folder, "json/train"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "json/val"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "model_snapshots"), exist_ok=True)


    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            print(f"\n[INFO] Starting attempt {retry_count + 1} of execution...\n")
            main_worker()
            break  # Success: exit the loop
        except RuntimeError as e:
            if "UR_RESULT_ERROR_OUT_OF_RESOURCES" in str(e):
                retry_count += 1
                print(f"[WARNING] Insufficient resources on hpu. Restarting attempt {retry_count}/{max_retries}...\n")

                # Free GPU cache and collect garbage
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if hasattr(torch, "hpu") and torch.hpu.is_available():
                        torch.hpu.empty_cache()
                except Exception as cache_e:
                    print(f"[WARNING] Failed to free cache: {cache_e}")

                gc.collect()
                time.sleep(5)
            else:
                print("[ERROR] Unexpected error:")
                traceback.print_exc()
                with open("error_log.txt", "a") as f:
                    f.write(traceback.format_exc() + "\n")
                break  # Exit the loop on different error
    else:
        print("[FATAL] Maximum number of attempts exceeded. Ending execution.")