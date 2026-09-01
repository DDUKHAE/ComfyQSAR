# ComfyQSAR

<div align="center">

![ComfyQSAR Logo](https://img.shields.io/badge/ComfyQSAR-QSAR%20Modeling-blue?style=for-the-badge)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=flat-square)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![ComfyUI v0.4.0](https://img.shields.io/badge/ComfyUI-v0.4.0-green?style=flat-square)](https://github.com/comfyanonymous/ComfyUI)

**A Visual Node-Based QSAR Modeling Platform for ComfyUI**

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Node Reference](#node-reference) • [Examples](#examples)

</div>

---

## Overview

**ComfyQSAR** is a custom node extension for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that brings **Quantitative Structure-Activity Relationship (QSAR)** modeling into a visual, node-based workflow environment. Build reproducible machine learning pipelines for drug discovery, toxicity prediction, and molecular property modeling through an intuitive drag-and-drop interface.

---

## Features

### Complete QSAR Pipeline

Both Classification and Regression workflows follow the same pipeline:

| Step | Node                                   | Description                                    |
| ---- | -------------------------------------- | ---------------------------------------------- |
| 1    | Data Load & Standardization            | Load molecules and remove invalid structures   |
| 2    | Descriptor Calculation                 | PaDEL-based 2D/3D descriptor computation, then inf/invalid-numeric cleanup and Name/target validation (descriptor quality control always runs as part of this node) |
| 3    | Data Split                             | Train/hold-out split (before any missingness handling) |
| 4    | Descriptor Preprocessing               | NaN column/row filtering + imputation, fit on training only and applied to hold-out |
| 5    | Descriptor Optimization                | Filter-based and model-based feature selection |
| 6    | Descriptor Combination                 | Multi-descriptor set combination search        |
| 7    | Hyperparameter Tuning & Model Training | Grid search with cross-validation              |
| 8    | Model Evaluation                       | Three parallel checks on the trained model — see below |
| 8.1  | ↳ Hold-out & External Performance      | Predictive performance on a fixed hold-out/external set |
| 8.2  | ↳ Chance-Correlation Test (Y-Scrambling) | Does the model beat a target-permuted baseline?     |
| 8.3  | ↳ Resampling Stability Assessment      | Does performance hold up across repeated/alternative splits? |
| 9    | Applicability Domain Assessment        | Is a given hold-out or external evaluation compound within the training domain? |

> Step 8's three sub-nodes (8.1/8.2/8.3) all answer variations of "how good/
> robust is this model?" and can run in any order, in parallel, off the same
> trained model from step 7 — they are not a numbered sequence. Step 9 asks a
> different question (is *this compound* inside the domain the model was
> trained on?), not a model-performance question, which is why it's a
> separate top-level step rather than a fourth 8.x sibling. Step 9 assesses
> hold-out/external evaluation compounds; assessing screening candidates
> instead is the separate Screening-Candidate AD node under `QSAR/3.
> SCREENER` -- see [README_CustomScreening.md](README_CustomScreening.md).

> Steps 3 and 4 run in this order deliberately: descriptor missingness
> thresholds and the imputer are fit on the training split only (step 4),
> never on the combined train+hold-out file, so no hold-out information
> leaks into the training statistics. If your workflow has no train/hold-out
> split at all (e.g. a whole-dataset CV-only study), use the standalone
> "Descriptor Preprocessing (Whole-Dataset / No Split)" node instead of
> steps 3+4 -- see [Node Reference](#4-descriptor-preprocessing).

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

**Metrics**: Accuracy, F1-Score, Precision, Recall, Specificity, ROC-AUC, Balanced Accuracy, MCC

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

**Metrics**: Test-set R² (Q2F2), Pearson r², MSE, MAE, RMSE, CCC, Q2F1, Q2F3, r2m

### Virtual Screening

Two screening modes are available:

- **External Screening (Database)**: Screen 7 pre-computed compound databases instantly
- **External Screening (Custom Compounds)**: All-in-one node — standardize, calculate descriptors, preprocess, and screen your own SDF file

---

## Installation

### Prerequisites

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed
- Python 3.10
- Java Runtime Environment (JRE 11+) for PaDEL-Descriptor

### Method : Git Clone + Conda Environment

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DDUKHAE/ComfyQSAR.git
cd ComfyQSAR
conda create -n comfyqsar python=3.10 -y
conda activate comfyqsar
pip install -r requirements.txt
```

If you run ComfyUI in a separate environment, install the same packages there instead of creating a new environment.

Restart ComfyUI after installation.


### Verify Installation

After restart, look for these node categories in the node browser:

- `QSAR/1. CLASSIFICATION`
- `QSAR/1. CLASSIFICATION/5. Descriptor Optimization/5.1 Filter-based Selection`
- `QSAR/1. CLASSIFICATION/5. Descriptor Optimization/5.2 Model-based Selection`
- `QSAR/1. CLASSIFICATION/8. Model Evaluation` (Hold-out & External Performance, Chance-Correlation Test, and Resampling Stability Assessment share this one category)
- `QSAR/2. REGRESSION`
- `QSAR/3. SCREENER`

---

## Quick Start

### Classification Workflow

![Classification workflow example](https://github.com/DDUKHAE/ComfyQSAR/blob/main/example/Classification_example/Classification_workflow.png)

```
1. Data Load & Standardization  →  2. Descriptor Calculation  →  3. Data Split
→  4. Descriptor Preprocessing  →  5. Descriptor Optimization  →  6. Descriptor Combination
→  7. Hyperparameter Tuning & Model Training  →  8.1/8.2/8.3 Model Evaluation (parallel)
→  9. Applicability Domain Assessment (optional)
```

1. Add **"1. Data Load & Standardization"** node — set paths to positive and negative compound files (`.sdf`, `.smi`, or `.csv`)
2. Add **"2. Descriptor Calculation"** node — choose 2D or 3D descriptors; also cleans inf/invalid-numeric values and validates Name/target (descriptor quality control always runs as part of this node, so it can never be skipped)
3. Add **"3. Data Split"** node — set test size (default: 20%); the split always stratifies on the target
4. Add **"4. Descriptor Preprocessing"** node — takes the split's training AND hold-out output together; fits NaN-column threshold and imputer on training only, applies the same transform to hold-out
5. Add descriptor optimization nodes — combine filter-based and model-based feature selection
6. Add **"6. Descriptor Combination"** node — search over descriptor combinations
7. Add **"7. Hyperparameter Tuning & Model Training"** node — select algorithm and parameter ranges
8. Add any/all of the three **Model Evaluation** nodes off the trained model — they're independent, not a sequence:
   - **"8.1 Hold-out & External Performance"** — evaluate on the external test set
   - **"8.2 Chance-Correlation Test (Y-Scrambling)"** — does the model beat a target-permuted baseline?
   - **"8.3 Resampling Stability Assessment"** — does performance hold up across repeated/alternative splits?
9. Optionally add **"9. Applicability Domain Assessment"** — checks whether hold-out/external evaluation compounds actually fall inside the training domain (a different question from steps 8.1-8.3, so it's its own step). For screening candidates, use the separate Screening-Candidate AD node instead (see [README_CustomScreening.md](README_CustomScreening.md))

### Regression Workflow

![Regression workflow example](https://github.com/DDUKHAE/ComfyQSAR/blob/main/example/Regression_example/Regression_workflow.png)

Same step structure (1-9, with step 8 split into 8.1/8.2/8.3) with regression-specific nodes under `QSAR/2. REGRESSION`.

### Virtual Screening

See [README_CustomScreening.md](README_CustomScreening.md) for detailed instructions on both screening modes.

---

## Node Reference

### QSAR/1. CLASSIFICATION

#### 1. Data Load & Standardization

**Combined node (recommended):**

| Node                                  | Display Name                   | Description                    |
| ------------------------------------- | ------------------------------ | ------------------------------ |
| `Load_and_Standardize_Classification` | 1. Data Load & Standardization | Load + standardize in one step |

**Individual nodes (QSAR/1. CLASSIFICATION/OTHERS):**

| Node                             | Display Name    | Description                                      |
| -------------------------------- | --------------- | ------------------------------------------------ |
| `Data_Loader_Classification`     | Data Loader     | Load positive/negative files                     |
| `Standardization_Classification` | Standardization | Clean up, strip salts, uncharge, canonicalize tautomers; reject unparseable/metal-only/empty molecules |

**Standardization:**

Each molecule is cleaned up, then reduced to its largest fragment (salts and counter-ions are stripped — e.g. `CCO.[Na+]` becomes `CCO`, not rejected for having more than one fragment), uncharged, and canonicalized to a single tautomer.

Rejected outright: unparseable structures, metal-only molecules (Li, Na, Fe, Cu, Zn, ... 63 elements), and structures that become empty after standardization.

**Input formats**: `.sdf`, `.smi`, `.csv`

---

#### 2. Descriptor Calculation

| Node                                     | Display Name              |
| ---------------------------------------- | ------------------------- |
| `Descriptor_Calculations_Classification` | 2. Descriptor Calculation |

Calculates 1400+ molecular descriptors using PaDEL-Descriptor, then always runs a descriptor quality control step on the result before any train/hold-out split exists. This step converts +inf/-inf to NaN, coerces any stray non-numeric descriptor cell to NaN (so it enters the normal missing-value pipeline instead of silently riding through as a string), and validates Name (blank/duplicate) and the target column (existence, NaN, and for Classification strictly 0/1) if a target column is present. A missing target column is not an error -- it's expected for screening-only inputs with no label yet.

**Key parameters:**

- `descriptor_type`: 2D or 3D
- `threads`: CPU cores (-1 for all)
- `max_runtime`: Per-molecule timeout (ms)
- `remove_salt`, `detect_aromaticity`, `standardize_nitro`
- `target_column`: only validated if present in the merged descriptors

**Output**: `DESCRIPTOR_MATRIX`

---

#### 3. Data Split

| Node                           | Display Name  |
| ------------------------------ | ------------- |
| `QSARDataSplit_Classification` | 3. Data Split |

**Parameters:**

- `test_size`: Fraction for hold-out set (0.05–0.5, default: 0.2)
- `random_state`: Seed for reproducibility

Splits by a fixed target column (Classification: `Label`, Regression: `value`), not a user-supplied `target_column`. Classification always stratifies on the target; there is no `stratify` toggle.

**Outputs**: `TRAINING_DATA`, `HOLDOUT_DATA`, `HOLDOUT_TARGETS`

Splits BEFORE any missingness handling -- descriptor columns can still contain NaN at this point. That's intentional: `4. Descriptor Preprocessing` fits its statistics on the training portion only, which requires the split to happen first.

---

#### 4. Descriptor Preprocessing

**Paired node (recommended when you have a train/hold-out split):**

| Node                                             | Display Name                | Description                                                                                |
| ------------------------------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------- |
| `Paired_Descriptor_Preprocessing_Classification` | 4. Descriptor Preprocessing | Fits column-retention threshold + imputer on training only, applies unchanged to hold-out |

**Inputs**: `training_data_path` (3's `TRAINING_DATA`), `holdout_data_path` (3's `HOLDOUT_DATA`), `holdout_targets_path` (3's `HOLDOUT_TARGETS`), `target_column`, `compound_nan_threshold`, `descriptor_nan_threshold`, `imputation_method`

**Order**: descriptor columns to retain are decided from training missingness only → both train and hold-out are restricted to that column set → per-row missingness is computed on the *restricted* columns (so a hold-out compound isn't penalized for missing values in a descriptor training already dropped) and filtered independently on each side → the imputer is fit on training (post-filtering) and applied to both.

**Outputs (graph sockets)**: `PREPROCESSED_TRAINING`, `PREPROCESSED_HOLDOUT`, `FILTERED_HOLDOUT_TARGETS` (`holdout_targets_path` with any hold-out compounds dropped by this node already removed -- use this, not 3's raw `Y_TEST`, as `8.1 Hold-out & External Performance`'s `holdout_targets_path`), `PREPROCESSING_RECIPE` (path to `preprocessing_recipe.json` -- training-fitted imputation statistics; feed to the Screener's `preprocessing_recipe_path`)

**Also written to disk (not a graph socket -- check the log text for its path)**: `excluded_compounds_report.csv` (which compounds were dropped for missingness, from which side, and why)

**Whole-dataset node (only when there is NO train/hold-out split, e.g. a whole-dataset CV-only study):**

| Node                                      | Display Name                                         | Description                                          |
| ------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------- |
| `Descriptor_preprocessing_Classification` | Descriptor Preprocessing (Whole-Dataset / No Split)   | All preprocessing in one step, fit on the whole file |

Fitting this node's thresholds/imputer on a file that mixes train and hold-out compounds leaks hold-out information into the training values -- use the paired node above instead whenever a real hold-out set exists.

**Individual sub-nodes (QSAR/1. CLASSIFICATION/OTHERS, building blocks for the whole-dataset node):**

| Node                                         | Display Name                | Description                                    |
| -------------------------------------------- | --------------------------- | ---------------------------------------------- |
| `Remove_high_nan_compounds_Classification`   | Remove High NaN Compounds   | Filter by compound NaN ratio                   |
| `Remove_high_nan_descriptors_Classification` | Remove High NaN Descriptors | Filter by descriptor NaN ratio                 |
| `Impute_missing_values_Classification`       | Impute Missing Values       | Fill remaining NaN (mean/median/most_frequent) |

---

#### 5. Descriptor Optimization

##### 5.1 Filter-based Selection (`QSAR/1. CLASSIFICATION/5. Descriptor Optimization/5.1 Filter-based Selection`)

| Node                                              | Display Name                | Description                          |
| ------------------------------------------------- | --------------------------- | ------------------------------------ |
| `Remove_Low_Variance_Descriptors_Classification`  | 5.1 Remove Low Variance     | Remove near-zero variance features   |
| `Remove_High_Correlation_Features_Classification` | 5.1 Remove High Correlation | Remove highly correlated descriptors |

##### 5.2 Model-based Selection (`QSAR/1. CLASSIFICATION/5. Descriptor Optimization/5.2 Model-based Selection`)

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

**Outputs (graph sockets)**: `TRAINED_MODEL`, `SELECTED_DESCRIPTOR_LIST`

**Output files** (saved to `ComfyUI/output/Classification/07_Model_Training/`):

- `Best_Classifier_<algorithm>.pkl` — trained model
- `Final_Selected_Descriptors_<algorithm>.txt` — selected feature names
- `Best_Hyperparameters_<algorithm>.txt` — best parameters found
- `Metric_Sensitivity_Report_<algorithm>.csv` — CV metric sensitivity across the parameter grid

---

#### 8. Model Evaluation

Three independent nodes, all consuming the same trained model from step 7 --
they can run in any order, in parallel, and answer different questions (see
the pipeline table above). None of them feed into each other.

##### 8.1 Hold-out & External Performance

| Node                              | Display Name                          |
| --------------------------------- | -------------------------------------- |
| `Model_Validation_Classification` | 8.1 Hold-out & External Performance   |

**Inputs**: `trained_model_path`, `descriptor_list_path` (both from step 7), `holdout_data_path` (4's `PREPROCESSED_HOLDOUT`), `holdout_targets_path` (4's `FILTERED_HOLDOUT_TARGETS`)

**Output files** (saved to `ComfyUI/output/Classification/08_Model_Evaluation/Holdout_External_Performance/`):

- `Evaluation_Results_ExternalTestSet.csv` — Accuracy, F1, ROC-AUC, Precision, Recall, Specificity, Balanced Accuracy, MCC
- `Actual_vs_Predicted.csv` — per-compound predictions
- `Confusion_Matrix.csv`
- `Bootstrap_CI_Results.csv` (if `compute_bootstrap_ci=True`, default)

##### 8.2 Chance-Correlation Test (Y-Scrambling)

| Node                                 | Display Name                              |
| ------------------------------------ | ------------------------------------------ |
| `YScramblingValidation_Classification` | 8.2 Chance-Correlation Test (Y-Scrambling) |

Refits the model's own pipeline structure/hyperparameters (via `sklearn.base.clone`, no new grid search) on repeated random permutations of the target column, on the same fixed descriptor set the model already uses -- reports whether real-target performance beats the permuted-target distribution.

**Output** (saved to `ComfyUI/output/Classification/08_Model_Evaluation/Chance_Correlation_Test/`): `Y_Scrambling_Results.csv`

##### 8.3 Resampling Stability Assessment

| Node                              | Display Name                         |
| ---------------------------------- | -------------------------------------- |
| `ResamplingValidation_Classification` | 8.3 Resampling Stability Assessment |

Repeated (stratified) k-fold / LOOCV / leave-N-out on the training data, reporting the mean and spread of performance across repeats/folds.

**Output** (saved to `ComfyUI/output/Classification/08_Model_Evaluation/Resampling_Stability/`): `Resampling_Validation_Results.csv`

---

#### 9. Applicability Domain Assessment

| Node                               | Display Name                               |
| ------------------------------------ | --------------------------------------------- |
| `ApplicabilityDomain_Classification` | 9. Structure-Similarity Applicability Domain |

A different question from step 8: is a given hold-out or external evaluation compound actually inside the domain the model was trained on? Classification uses kNN-mean Tanimoto similarity; Regression (below) uses Williams/leverage. For screening candidates, use the separate Screening-Candidate AD node instead (see [README_CustomScreening.md](README_CustomScreening.md)).

**Output** (saved to `ComfyUI/output/Classification/09_Applicability_Domain/`): AD report CSV.

---

### QSAR/2. REGRESSION

Mirrors the Classification structure with regression-specific implementations.

**Step 7 algorithms**: Random Forest, Decision Tree, Lasso, Ridge, ElasticNet, SVR, XGBoost, LightGBM

**Step 8.1 metrics**: Test-set R² (1-SSE/SST_test, identical to Q2F2), Pearson r² (squared correlation, reported separately -- not the same quantity as Test-set R²), MSE, MAE, RMSE, CCC, Q2F1-3, r2m. CSV columns use the names `Predictive_R2` (Test-set R²) and `Pearson_r2`.

**Step 9** (`ApplicabilityDomain_Regression`, display name "9. Descriptor-Space Applicability Domain"): Williams plot / leverage-based AD instead of Classification's structure-similarity approach.

---

### QSAR/3. SCREENER

| Node                                | Display Name                          | Use Case                                                        |
| ----------------------------------- | -------------------------------------- | ---------------------------------------------------------------- |
| `QSARDBScreener`                    | External Screening (Database)         | Screen pre-computed databases                                   |
| `QSARCustomUserScreener`            | External Screening (Custom Compounds) | Screen a custom SDF file (all-in-one)                            |
| `ScreeningCandidateAD_Classification` | Screening-Candidate AD (Classification) | AD assessment for screening candidates (kNN-mean Tanimoto)    |
| `ScreeningCandidateAD_Regression`   | Screening-Candidate AD (Regression)   | AD assessment for screening candidates (Williams/leverage)      |

The Screening-Candidate AD nodes are the screening-side counterpart to step
9 -- same method and training set as the evaluation-set AD node, applied to
a screening run's output instead of a hold-out/external set. They append AD
columns to `screening_results_path` rather than producing a separate
compound-id/SMILES-only report.

See [README_CustomScreening.md](README_CustomScreening.md) for details.

---

## Output File Structure

```
ComfyUI/output/
├── Classification/
│   ├── 01_Data_Load_and_Standardization/
│   ├── 02_Descriptor_Calculation/
│   ├── 03_Data_Split/
│   ├── 04_Descriptor_Preprocessing/
│   ├── 05_Descriptor_Optimization/
│   │   ├── Filter_Based/
│   │   └── Model_Based/
│   ├── 06_Descriptor_Combination/
│   ├── 07_Model_Training/
│   ├── 08_Model_Evaluation/
│   │   ├── Holdout_External_Performance/
│   │   ├── Chance_Correlation_Test/
│   │   └── Resampling_Stability/
│   └── 09_Applicability_Domain/
├── Regression/
│   ├── 01_Data_Load_and_Standardization/
│   ├── 02_Descriptor_Calculation/
│   ├── 03_Data_Split/
│   ├── 04_Descriptor_Preprocessing/
│   ├── 05_Descriptor_Optimization/
│   │   ├── Filter_Based/
│   │   └── Model_Based/
│   ├── 06_Descriptor_Combination/
│   ├── 07_Model_Training/
│   ├── 08_Model_Evaluation/
│   │   ├── Holdout_External_Performance/
│   │   ├── Chance_Correlation_Test/
│   │   └── Resampling_Stability/
│   └── 09_Applicability_Domain/
└── Screening/
    ├── Database_Screening/
    │   └── <DB_NAME>/
    │       ├── <DB_NAME>_Screening_Selected_Compounds.csv
    │       └── SDF/
    │           └── <DB_NAME>_Selected_Molecules.sdf
    └── Custom_Screening/
        ├── custom_db_prepared/
        └── custom_screening_results/
```

---

## Examples

`example/` contains complete, real pipeline output for two benchmark
datasets and a custom-screening walkthrough — every intermediate file a
full run produces, not just the final model.

| Folder | Contents |
|---|---|
| `Classification_example/` | QDB116 (bioconcentration factor) — steps 1-9, real intermediate files at every step |
| `Regression_example/` | QDB261 (antiproliferative activity) — steps 1-9, real intermediate files at every step |
| `CustomScreening_example/` | PTP1B custom-screening walkthrough files (see [README_CustomScreening.md](README_CustomScreening.md)) |

Each dataset folder has its own `README.md` with the source citation and a
step-by-step file index.

---

See [requirements.txt](requirements.txt) for the complete list. [requirements-tested.txt](requirements-tested.txt) pins the exact package versions this release was tested against (Python 3.10.13, ComfyUI v0.4.0, Java OpenJDK 11.0.27).

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.

---

</div>
