# Chest X-ray Multi-Label Classification Project

A multi-label classification project for chest X-ray images using DenseNet-121. The system takes a chest X-ray image as input and answers 14 binary questions (whether the image exhibits a specific pathology or not).

## Overview of the Processing Pipeline
- **Input:** A chest X-ray image ($224 \times 224$ pixels).
- **Feature Extraction:** DenseNet-121 outputs 14 raw logits ($z$).
- **Temperature Scaling:** Divide by the learned temperature parameter ($T$) for each specific pathology (handled by Block E).
- **Probability Computation:** Compute probabilities $p = \text{sigmoid}(z / T)$.
- **Thresholding:** Compare against the optimized decision threshold ($t$) for each pathology (handled by Block D).
- **Output:** 14 binary decisions along with 14 independent probabilities (each pathology has its own probability; they do not sum to 1).
- *Note:* The temperature scaling ($T$) and decision thresholds ($t$) are not intrinsic parts of the neural network architecture; they are optimized post-training using the validation set (implemented in Blocks D and E).

## Pipeline Architecture across 5 Blocks

- **Block A (Data Preprocessing):**
  - **Inputs:** Kaggle dataset images, `Data_Entry_2017.csv`, and `test_list_NIH.txt`.
  - **Tasks:** Parse and transform labels into 14 binary columns ($0/1$), tag official test images, and record image paths.
  - **Output:** `meta.csv` (112,120 rows containing image file names, patient IDs, official split flags, file paths, and 14 binary labels).

- **Block B (Data Splitting):**
  - **Input:** `meta.csv`.
  - **Task:** Perform a strict patient-level split (*Patient-wise split*) to prevent data leakage.
  - **Outputs:** `train.csv` (75,860 images), `val.csv` (10,664 images), and `test.csv` (25,596 images).

- **Block C (Model Training):**
  - **Inputs:** `train.csv`, `val.csv`, and raw image files.
  - **Task:** Train DenseNet-121 and select the best epoch based on validation AUROC.
  - **Outputs:** `best.pt` model weights and 4 numpy arrays (`logits_val.npy`, `labels_val.npy`, `logits_test.npy`, `labels_test.npy`).

- **Block D (Decision Threshold Tuning):**
  - **Inputs:** Validation logits and labels (subsequently evaluated on the test set).
  - **Tasks:** Optimize individual decision thresholds ($t$) per pathology, calculate AUROC, AP, F1-scores, and bootstrap confidence intervals.
  - **Output:** Excel spreadsheet summarizing performance metrics per pathology.

- **Block E (Probability Calibration):**
  - **Inputs:** The 4 logits/labels arrays generated from Block C.
  - **Tasks:** Learn temperature scaling coefficients ($T$) per pathology, compute Expected Calibration Error (ECE), Brier scores, and plot reliability diagrams.
  - **Outputs:** Excel spreadsheets and calibration PNG plots.

    # Training & Evaluation Instructions

## 1. Environment Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Data Preparation

Follow the instructions in `DATA.md` to download and prepare the NIH ChestX-ray14 dataset (224×224 version) and create the official patient-wise splits.

### Expected Directory Structure

```text
data/
├── images/                 # 224×224 images
├── Data_Entry_2017.csv
├── train_val_list.txt
├── test_list.txt
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

---

## 3. Training

We use **DenseNet-121** pretrained on ImageNet as the backbone model.

Run training with:

```bash
python train.py \
  --data_dir data/ \
  --output_dir results/ \
  --model densenet121 \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --optimizer adamw \
  --loss bce_with_logits
```

### Training Configuration

- **Backbone:** DenseNet-121 (ImageNet pretrained)
- **Loss Function:** BCEWithLogitsLoss
- **Optimizer:** AdamW
- **Learning Rate:** 1e-4
- **Epochs:** 20
- **Batch Size:** 32

The best-performing model checkpoint will be automatically saved in:

```text
results/
```

---

## 4. Post-hoc Calibration (Main Method)

After training, two inexpensive post-hoc calibration steps are applied. Both are fitted **only on the validation set**:

1. **Per-disease F1-optimal threshold selection**
2. **Per-disease temperature scaling**

Run:

```bash
python calibrate.py \
  --checkpoint results/best_model.pth \
  --val_csv data/splits/val.csv \
  --output_dir results/calibration/
```

### Generated Outputs

The calibration procedure produces:

- Optimal thresholds $\tau_c$ for each of the 14 diseases
- Temperature parameters $T_c$ for each disease

Output files are saved in:

```text
results/calibration/
```

---

## 5. Evaluation

Evaluate on the official test set using the validation-tuned thresholds and temperature parameters:

```bash
python evaluate.py \
  --checkpoint results/best_model.pth \
  --test_csv data/splits/test.csv \
  --thresholds results/calibration/thresholds.json \
  --temperatures results/calibration/temperatures.json \
  --output_dir results/evaluation/
```

Evaluation results and analysis will be saved to:

```text
results/evaluation/
```

---

# Main Results

| Method | Macro-F1 | Micro-F1 |
|----------|----------:|----------:|
| Baseline (threshold = 0.5) | 0.144 | 0.250 |
| + F1-optimal thresholds | **0.320** | **0.381** |

Temperature scaling improves calibration quality by reducing the mean Expected Calibration Error (ECE):

```text
0.0204 → 0.0176
```

while leaving AUROC unchanged.

---

# Additional Analysis

The `results/` directory contains:

- Detailed per-disease performance metrics
- Reliability diagrams
- Calibration analysis
- Error analysis reports
- Saved model checkpoints

---
