# DATA.md

## 1. Dataset Overview

We use the **NIH ChestX-ray14** dataset.

- **Total images**: 112,120 frontal-view chest X-ray images
- **Number of patients**: 30,805 unique patients
- **Labels**: Multi-label classification with 14 thoracic diseases + “No Finding”
- Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule,
Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema,
Fibrosis, Pleural Thickening, Hernia
- Any image without positive disease labels is tagged as **“No Finding”**.

## 2. Official Dataset Source

- **Official URL**: [https://nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- **Version**: 15/12/2017 (official release containing `train_val_list.txt` and `test_list.txt`)
- **Paper**: Wang et al., “ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases”, CVPR 2017.

## 3. Dataset Version Used in This Project

Because the original images are large (1024×1024), we use a publicly available **pre-resized version** (224×224) for computational efficiency:

- **Kaggle Dataset**: [NIH Chest X-ray 14 (224×224 resized)](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized)
- Images have already been resized to **224 × 224** pixels.
- Labels remain identical to the official `Data_Entry_2017.csv`.

## 4. Data Split

We follow the **official patient-wise split** provided by NIH to avoid data leakage (no patient appears in more than one split).

```
| Split          | Number of Images | Source                                     | Notes                                    |
| -------------- | ---------------- | ------------------------------------------ | ---------------------------------------- |
| **Test**       | 25,596           | Official `test_list.txt`                   | Held-out test set                        |
| **Validation** | 10,664           | Sampled from official `train_val_list.txt` | Used for threshold & temperature scaling |
| **Train**      | 75,860           | Remaining images from `train_val_list.txt` | Used for model training                  |
```

- Thresholds and temperature scaling parameters are fitted **only on the validation set**.
- The test set is never used during training or hyperparameter tuning.

## 5. Preprocessing Procedure

Since we use the already resized version, the preprocessing steps applied are:

1. **Image resizing**: Already performed by the Kaggle dataset provider (original 1024×1024 → 224×224).
2. **Label processing**: Multi-label binary encoding from `Data_Entry_2017.csv`.
3. **Patient-wise splitting**: Using official `train_val_list.txt` and `test_list.txt`.
4. **Normalization**: Standard ImageNet mean/std normalization (applied during training).

No additional geometric resizing or cropping is performed in our pipeline.

## 6. How to Reproduce the Data

1. Download the pre-resized dataset (224×224) from Kaggle:
   https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized

2. Use the official NIH files (`Data_Entry_2017.csv`, `train_val_list.txt`, `test_list.txt`) to create the patient-wise train / validation / test splits as described in Section 4.

