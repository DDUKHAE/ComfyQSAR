# ComfyQSAR

<div align="center">

![ComfyQSAR Logo](https://img.shields.io/badge/ComfyQSAR-QSAR%20Modeling-blue?style=for-the-badge)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=flat-square)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![ComfyUI 0.4.0](https://img.shields.io/badge/ComfyUI-0.4.0-green?style=flat-square)](https://github.com/comfyanonymous/ComfyUI)

**A Visual Node-Based QSAR Modeling Platform for ComfyUI**

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Node Reference](#node-reference) • [Examples](#examples)

</div>

---

## Overview

**ComfyQSAR** is a custom node extension for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that brings **Quantitative Structure-Activity Relationship (QSAR)** modeling into a visual, node-based workflow environment. Build reproducible machine learning pipelines for drug discovery, toxicity prediction, and molecular property modeling through an intuitive drag-and-drop interface.

---

## Features

### Complete QSAR Pipeline (8 Steps)

Both Classification and Regression workflows follow the same 8-step pipeline:

| Step | Node                                   | Description                                    |
| ---- | -------------------------------------- | ---------------------------------------------- |
| 1    | Data Load & Standardization            | Load molecules and remove invalid structures   |
| 2    | Descriptor Calculation                 | PaDEL-based 2D/3D descriptor computation       |
| 3    | Descriptor Preprocessing               | NaN/Inf handling and imputation                |
| 4    | Data Split                             | Train/test stratified split                    |
| 5    | Descriptor Optimization                | Filter-based and model-based feature selection |
| 6    | Descriptor Combination                 | Multi-descriptor set combination search        |
| 7    | Hyperparameter Tuning & Model Training | Grid search with cross-validation              |
| 8    | Model Validation                       | External test set evaluation                   |

### Supported Algorithms

#### Classification

| Algorithm           | Parameters                                                               |
| ------------------- | ------------------------------------------------------------------------ |
| Random Forest       | n_estimators, max_depth, min_samples_split                               |
| Decision Tree       | max_depth, min_samples_split, criterion                                  |
| Logistic Regression | C, penalty                                                               |
| LASSO (L1 Logistic) | C                                                                        |
| SVM                 | C, kernel, gamma                                                         |
| XGBoost             | n_estimators, learning_rate, max_depth                                   |
| LightGBM            | n_estimators, learning_rate, max_depth, subsample, reg_alpha, reg_lambda |

**Metrics**: Accuracy, F1-Score, Precision, Recall, ROC-AUC, Specificity

#### Regression

| Algorithm     | Parameters                                                                |
| ------------- | ------------------------------------------------------------------------- |
| Random Forest | n_estimators, max_depth, min_samples_split, bootstrap                     |
| Decision Tree | max_depth, min_samples_split, criterion                                   |
| Lasso         | alpha                                                                     |
| Ridge         | alpha                                                                     |
| ElasticNet    | alpha, l1_ratio                                                           |
| SVR           | C, kernel, gamma, epsilon                                                 |
| XGBoost       | n_estimators, learning_rate, max_depth, subsample, reg_alpha, reg_lambda  |
| LightGBM      | n_estimators, learning_rate, max_depth, num_leaves, reg_alpha, reg_lambda |

**Metrics**: R², MSE, MAE, RMSE

### Virtual Screening

Two screening modes are available:

- **QSAR DB Screener**: Screen 7 pre-computed compound databases instantly (no descriptor calculation needed)
- **Custom User Screener**: All-in-one node — standardize, calculate descriptors, preprocess, and screen your own SDF file

---

## Installation

### Prerequisites

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed
- Python 3.10 or higher
- Java Runtime Environment (JRE 11+) for PaDEL-Descriptor

### Method : Git Clone

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YourUsername/ComfyQSAR.git
cd ComfyQSAR
pip install -r requirements.txt
```

Restart ComfyUI.


### Verify Installation

After restart, look for these node categories in the node browser:

- `QSAR/CLASSIFICATION`
- `QSAR/CLASSIFICATION/5. Descriptor Optimization/5.1 Filter-based Selection`
- `QSAR/CLASSIFICATION/5. Descriptor Optimization/5.2 Model-based Selection`
- `QSAR/REGRESSION`
- `QSAR/SCREENER`

---

## Quick Start

### Classification Workflow

```
1. Data Load & Standardization  →  2. Descriptor Calculation  →  3. Descriptor Preprocessing
→  4. Data Split  →  5. Descriptor Optimization  →  6. Descriptor Combination
→  7. Hyperparameter Tuning & Model Training  →  8. Model Validation
```

1. Add **"1. Data Load & Standardization"** node — set paths to positive and negative compound files (`.sdf`, `.smi`, or `.csv`)
2. Add **"2. Descriptor Calculation"** node — choose 2D or 3D descriptors
3. Add **"3. Descriptor Preprocessing"** node — configure NaN threshold and imputation method
4. Add **"4. Data Split"** node — set test size (default: 20%) and stratify option
5. Add descriptor optimization nodes — combine filter-based and model-based feature selection
6. Add **"6. Descriptor Combination"** node — search over descriptor combinations
7. Add **"7. Hyperparameter Tuning & Model Training"** node — select algorithm and parameter ranges
8. Add **"8. Model Validation"** node — evaluate on external test set

### Regression Workflow

Same 8-step structure with regression-specific nodes under `QSAR/REGRESSION`.

### Virtual Screening

See [README_CustomScreening.md](README_CustomScreening.md) for detailed instructions on both screening modes.

---

## Node Reference

### QSAR/CLASSIFICATION

#### 1. Data Load & Standardization

**Combined node (recommended):**

| Node                                  | Display Name                   | Description                    |
| ------------------------------------- | ------------------------------ | ------------------------------ |
| `Load_and_Standardize_Classification` | 1. Data Load & Standardization | Load + standardize in one step |

**Individual nodes (QSAR/CLASSIFICATION/OTHERS):**

| Node                             | Display Name    | Description                                      |
| -------------------------------- | --------------- | ------------------------------------------------ |
| `Data_Loader_Classification`     | Data Loader     | Load positive/negative files                     |
| `Standardization_Classification` | Standardization | Remove metals, multi-fragment, invalid molecules |

**Standardization filters:**

- Metal-only molecules (Li, Na, Fe, Cu, Zn, ... 63 elements)
- Multi-fragment molecules
- RDKit-invalid structures

**Input formats**: `.sdf`, `.smi`, `.csv`

---

#### 2. Descriptor Calculation

| Node                                     | Display Name              |
| ---------------------------------------- | ------------------------- |
| `Descriptor_Calculations_Classification` | 2. Descriptor Calculation |

Calculates 1400+ molecular descriptors using PaDEL-Descriptor.

**Key parameters:**

- `descriptor_type`: 2D or 3D
- `threads`: CPU cores (-1 for all)
- `max_runtime`: Per-molecule timeout (ms)
- `remove_salt`, `detect_aromaticity`, `standardize_nitro`

---

#### 3. Descriptor Preprocessing

**Combined node (recommended):**

| Node                                      | Display Name                | Description                   |
| ----------------------------------------- | --------------------------- | ----------------------------- |
| `Descriptor_preprocessing_Classification` | 3. Descriptor Preprocessing | All preprocessing in one step |

**Individual nodes (QSAR/CLASSIFICATION/OTHERS):**

| Node                                         | Display Name                | Description                                    |
| -------------------------------------------- | --------------------------- | ---------------------------------------------- |
| `Replace_inf_with_nan_Classification`        | Replace Inf with NaN        | Convert infinite values                        |
| `Remove_high_nan_compounds_Classification`   | Remove High NaN Compounds   | Filter by compound NaN ratio                   |
| `Remove_high_nan_descriptors_Classification` | Remove High NaN Descriptors | Filter by descriptor NaN ratio                 |
| `Impute_missing_values_Classification`       | Impute Missing Values       | Fill remaining NaN (mean/median/most_frequent) |

---

#### 4. Data Split

| Node                           | Display Name  |
| ------------------------------ | ------------- |
| `QSARDataSplit_Classification` | 4. Data Split |

**Parameters:**

- `test_size`: Fraction for test set (0.05–0.5, default: 0.2)
- `random_state`: Seed for reproducibility
- `stratify`: Stratified split (default: True)

**Outputs**: `train_path`, `X_test_path`, `y_test_path`

---

#### 5. Descriptor Optimization

##### 5.1 Filter-based Selection (`QSAR/CLASSIFICATION/5. Descriptor Optimization/5.1 Filter-based Selection`)

| Node                                              | Display Name                | Description                          |
| ------------------------------------------------- | --------------------------- | ------------------------------------ |
| `Remove_Low_Variance_Descriptors_Classification`  | 5.1 Remove Low Variance     | Remove near-zero variance features   |
| `Remove_High_Correlation_Features_Classification` | 5.1 Remove High Correlation | Remove highly correlated descriptors |

##### 5.2 Model-based Selection (`QSAR/CLASSIFICATION/5. Descriptor Optimization/5.2 Model-based Selection`)

| Node               | Display Name                  | Method                        |
| ------------------ | ----------------------------- | ----------------------------- |
| `lasso_CL`         | 5.2 LASSO Selection           | L1 Logistic Regression        |
| `decision_tree_CL` | 5.2 Decision Tree Selection   | Feature importances           |
| `random_forest_CL` | 5.2 Random Forest Selection   | Feature importances           |
| `xgboost_CL`       | 5.2 XGBoost Selection         | Feature importances           |
| `lightgbm_CL`      | 5.2 LightGBM Selection        | Feature importances           |
| `rfe_CL`           | 5.2 RFE Selection             | Recursive Feature Elimination |
| `sfm_CL`           | 5.2 SelectFromModel Selection | Threshold-based selection     |

---

#### 6. Descriptor Combination

| Node                                        | Display Name              |
| ------------------------------------------- | ------------------------- |
| `Feature_Combination_Search_Classification` | 6. Descriptor Combination |

Searches combinations of multiple descriptor sets to find the optimal subset.

---

#### 7. Hyperparameter Tuning & Model Training

| Node                                        | Display Name                              |
| ------------------------------------------- | ----------------------------------------- |
| `Hyperparameter_Grid_Search_Classification` | 7. Hyperparameter Tuning & Model Training |

**Parameters:** Select algorithm and provide parameter lists as Python list strings (e.g., `[100, 200, 300]`).

**Output files** (saved to `ComfyUI/output/QSAR_GridSearch/`):

- `Best_Classifier_<model>.pkl` — trained model
- `Final_Selected_Descriptors.txt` — selected feature names
- `Best_Hyperparameters_<model>.txt` — best parameters found
- `X_test.csv`, `y_test.csv` — held-out test set

---

#### 8. Model Validation

| Node                              | Display Name        |
| --------------------------------- | ------------------- |
| `Model_Validation_Classification` | 8. Model Validation |

**Inputs**: model `.pkl`, `selected_descriptors.txt`, `X_test.csv`, `y_test.csv`

**Output files**:

- `Evaluation_Results_ExternalTestSet.csv` — Accuracy, F1, ROC-AUC, Precision, Recall, Specificity
- `Actual_vs_Predicted.csv` — per-compound predictions

---

### QSAR/REGRESSION

Mirrors the Classification structure with regression-specific implementations.

**Step 7 algorithms**: Random Forest, Decision Tree, Lasso, Ridge, ElasticNet, SVR, XGBoost, LightGBM

**Step 8 metrics**: R², MSE, MAE, RMSE

---

### QSAR/SCREENER

| Node                     | Display Name         | Use Case                              |
| ------------------------ | -------------------- | ------------------------------------- |
| `QSARDBScreener`         | QSAR DB Screener     | Screen pre-computed databases         |
| `QSARCustomUserScreener` | Custom User Screener | Screen a custom SDF file (all-in-one) |

See [README_CustomScreening.md](README_CustomScreening.md) for details.

---

## Output File Structure

```
ComfyUI/output/
├── QSAR_GridSearch/               # Classification training outputs
│   ├── Best_Classifier_RF.pkl
│   ├── Final_Selected_Descriptors.txt
│   ├── Best_Hyperparameters_RF.txt
│   ├── X_test.csv
│   └── y_test.csv
├── QSAR_GridSearch_Regression/    # Regression training outputs
│   └── (same structure)
└── QSAR_ModelValidation/          # Validation results
    ├── Evaluation_Results_ExternalTestSet.csv
    └── Actual_vs_Predicted.csv

ComfyQSAR/screening_results_DB/    # DB screener results
ComfyQSAR/py/Screener/Custom_DB_Screening_Results/  # Custom screener results
```

---

See [requirements.txt](requirements.txt) for the complete list.

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.

---

</div>
