# 🔬 Ensemble TTA HiCervix HPU Training & Evaluation – Reproducible Pipeline

This repository provides a full and reproducible pipeline for training and evaluating **hierarchical cervical cell classification** models, based on the open-source **HiCervix** project. Our extended version supports:

- Hierarchical training with **Hierarchical Cross-Entropy (HXE)**
- Flat training with CrossEntropyLoss
- Multiple backbones (EfficientNetV2-S, CaFormer-S36)
- **Test-Time Augmentation (TTA)** and ensemble inference
- Execution on **HPU (Habana Gaudi)** hardware

> 🔗 **Reference repo**:  
> All methods and preprocessing steps are adapted from the official HiCervix repository:  
> https://github.com/Scu-sen/HiCervix

---

## ⚙️ 1. Prerequisites & Hardware (Gaudi HPU)

### ✅ Required:
- **Conda (Miniconda or Anaconda)** installed and available in your terminal
- Python **≥ 3.9**
- HPU-compatible machine (e.g., Intel Tiber Cloud, AWS DL1)
- Internet access to install Python packages


This project **requires** a runtime environment with support for **Gaudi HPUs (Habana AI processors)**.

You can obtain free or paid access to Gaudi infrastructure using:

- ✅ **Intel Tiber Developer Cloud**:  

  Offers **free credits** (no credit card required). Select a template with **HPU PyTorch** pre-installed for a ready-to-use environment.  
  > _As of the latest update to this project, these options are current; please check the provider's website for any changes since the time of writing._
- ✅ **Amazon EC2 DL1 Instances (Gaudi)**:  
  Available through AWS Marketplace for large-scale experiments.

Make sure the machine you're using:
- Has access to **HPU drivers and runtime libraries** (Habana SynapseAI)
- Uses **Python ≥ 3.9** and **Conda** 
- Has connectivity to install required packages via `pip`/`conda`
- Can run `torch.hpu` operations (check with a small HPU script)

---

## 📦 2. Environment Setup (with HPU support)

First, create the environment using the provided YAML:

```bash
conda env create -f environment.yml
conda activate hicervix-env
```

> ⚠️ **Important:**  
> You must have the **Habana PyTorch bridge packages** installed, such as:
> - `optimum-habana`
> - `lightning-habana`
> - `habana_frameworks`
>  
> If you're not using an official Gaudi image, follow setup instructions from:  
> https://docs.habana.ai/en/latest/PyTorch/PyTorch_Overview.html

Test with a minimal PyTorch HPU script:

```python
import torch
x = torch.ones((3, 3), device='hpu')
print(x * 2)
```

---

## 🧬 3. Dataset Download & CSV Format

The Hicervix dataset is publicly available on **Zenodo**, under DOI:

📁 **Zenodo Hicervix Record**: https://zenodo.org/records/11087263

Download and extract the dataset into a folder like `data/`.

You must create three CSV files with **no headers**, two columns each:

| File           | Purpose           |
|----------------|-------------------|
| `train_mbm.csv` | Training split    |
| `val_mbm.csv`   | Validation split  |
| `test_mbm.csv`  | Final evaluation  |

Each line must follow this format:

```
relative_path/to/image.jpg,class_id
```

## 📄 Expected CSV Format

The CSV file must contain the **hierarchical labels** for each image across **three levels** (Level 1, Level 2, and Level 3), along with their corresponding **numeric class IDs** and the **image path**.

This structure is **mandatory** for the following files:
- `train.csv`
- `test.csv`
- `eval.csv`

### ✅ Required CSV Columns

| Column Name     | Description                                                              |
|-----------------|--------------------------------------------------------------------------|
| `image_path`    | Relative or absolute path to the image                                   |
| `level_1`       | Class name at level 1 (most general category)                            |
| `level_2`       | Class name at level 2 (intermediate category)                            |
| `level_3`       | Class name at level 3 (most specific, also known as the leaf class)      |
| `level_1_id`    | Integer ID for the `level_1` class                                       |
| `level_2_id`    | Integer ID for the `level_2` class                                       |
| `level_3_id`    | Integer ID for the `level_3` class                                       |

> 💡 Class IDs must be consistent with the hierarchy files: `level_names_dict.pkl`, `tct_tree.pkl`, and `tct_distances.pkl`.

---

## 🧪 CSV Example train.csv (First 3 Rows)

```csv
image_path,level_1,level_2,level_3,level_1_id,level_2_id,level_3_id
images/B001.png,Epithelial cells,Superficial cells,Superficial squamous cells,0,3,12
images/B002.png,Epithelial cells,Intermediate cells,Intermediate squamous cells,0,4,13
images/B003.png,Epithelial cells,Basal cells,Basal squamous cells,0,5,14
```

---

## 🌳 4. Generate Hierarchical Artifacts (.pkl)

You need to create three key pickle files required for hierarchical training:

| File                    | Description                                      |
|-------------------------|--------------------------------------------------|
| `level_names_dict.pkl` | Maps class IDs to label names at each level      |
| `tct_distances.pkl`    | Pairwise semantic distances between classes      |
| `tct_tree.pkl`         | Hierarchical tree as an `nltk.Tree` object       |

To generate these:

1. Clone the original repository:  
   https://github.com/Scu-sen/HiCervix

2. Open and execute the notebook:  
   `HiCervix_pre-processing.ipynb`

This will generate the three `.pkl` files. Place them in the root of this project.

You may adapt the tree-related variables inside the project’s .py files to match the generated tree.pkl file."

---

## 🗂️ 5. Recommended Folder Structure

```
├── data/                    # images and CSV files
├── checkpoints/             # training checkpoints
├── output/
│   ├── json/train/          # training metrics
│   ├── json/val/            # validation metrics
│   ├── tb/                  # TensorBoard logs (not used)
│   └── model_snapshots/     # saved model weights
├── tct_distances.pkl
├── tct_tree.pkl
├── level_names_dict.pkl
├── environment.yml
├── train_hier_efficientnet_gaudi.py
├── train_hier_caformer_gaudi.py
├── train_flat_efficientnet.py
├── train_flat_caformer.py
└── evaluate_ensemble.py
```

---

## 🏋️‍♂️ 6. Training Execution

You may choose between hierarchical and flat training:

| Script                          | Model           | Training Type | Loss |
|--------------------------------|------------------|----------------|------|
| `train_hier_efficientnet_gaudi.py` | EfficientNetV2-S | Hierarchical   | HXE  |
| `train_hier_caformer_gaudi.py`     | CaFormer-S36     | Hierarchical   | HXE  |
| `train_flat_efficientnet.py`       | EfficientNetV2-S | Flat           | CE   |
| `train_flat_caformer.py`           | CaFormer-S36     | Flat           | CE   |

Each script has a block of configurable parameters at the top:
- `BATCH_SIZE`, `IMAGE_SIZE`, `NUM_EPOCHS`, etc.
- Paths to data, model checkpointing, etc.

Run training:

```bash
python train_hier_efficientnet_gaudi.py
```

You will find:
- JSON logs in `output/json/`
- Checkpoints in `output/model_snapshots/`
- TensorBoard logs in `output/tb/`

---

## 🧪 7. Evaluation and Ensemble Inference

To evaluate one or multiple models, use:

```bash
python evaluate_ensemble.py
```

You may configure:

- `TEST_CSV`: Path to the test CSV
- `BATCH_SIZE`: Test-time batch size (e.g., 64)
- `NUM_RUNS`: Number of TTA repetitions
- `TTA_MODES`: Test-time augmentations (e.g., `['hflip', 'vflip']`)
- `MODEL_CONFIGS`: List of (name, architecture, checkpoint)
- `ENS_BUNDLES`: Ensemble configurations (name + model list)

Outputs:
- JSON summary: `evaluation_all_models.json`
- CSV table: `evaluation_all_models.csv`

These are saved in the current working directory unless reconfigured.

---

## 📚 8. References and Dependencies

- **HiCervix** (Original): https://github.com/Scu-sen/HiCervix  
- **Dataset (Zenodo)**: https://zenodo.org/records/11087263  
- **Habana / Gaudi Docs**: https://docs.habana.ai  
- **optimum-habana**: [PyPI](https://pypi.org/project/optimum-habana/)  
- **lightning-habana**: [PyPI](https://pypi.org/project/lightning-habana/)  
- **timm**: PyTorch Image Models ([PyPI](https://pypi.org/project/timm/))  
- **albumentations**: Augmentations ([PyPI](https://pypi.org/project/albumentations/))  
- **NLTK punkt**: Required corpus ([nltk.org](https://www.nltk.org/nltk_data/))  

---
