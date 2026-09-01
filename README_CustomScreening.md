# ComfyQSAR — Virtual Screening Guide

## Overview

ComfyQSAR provides four nodes under the `QSAR/3. SCREENER` category: two run screening, two assess whether the screened candidates fall inside the model's training domain.

| Node | Display Name | Use Case |
|------|-------------|----------|
| `QSARDBScreener` | External Screening (Database) | Screen 7 pre-computed compound databases |
| `QSARCustomUserScreener` | External Screening (Custom Compounds) | Screen a custom SDF file (all-in-one) |
| `ScreeningCandidateAD_Classification` | Screening-Candidate AD (Classification) | AD assessment for a Classification screening run's candidates |
| `ScreeningCandidateAD_Regression` | Screening-Candidate AD (Regression) | AD assessment for a Regression screening run's candidates |

The two screening nodes require a trained QSAR model (`.pkl`), a selected descriptor list (`.txt`) produced by Step 7 (Hyperparameter Tuning & Model Training), and a `preprocessing_recipe.json` produced by Step 4 (Descriptor Preprocessing). The recipe supplies training-fitted imputation statistics for the selected descriptors; both nodes fill missing values from these statistics and never re-fit imputation at screening time.

---

## Node 1: External Screening (Database)

**Category**: `QSAR/3. SCREENER`

Screens one of 7 pre-computed compound databases. Descriptor calculation is already done — results are instant.

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `trained_model_path` | STRING | Path to trained `.pkl` model file (from Step 7) |
| `descriptor_list_path` | STRING | Path to selected descriptors `.txt` file (from Step 7) |
| `preprocessing_recipe_path` | STRING | Path to `preprocessing_recipe.json` (from Step 4) |
| `db_name` | CHOICE | Database to screen (see table below) |
| `threshold` | FLOAT | Activity probability cutoff (0.0–1.0, default: 0.5) |

### Available Databases

| `db_name` | File | Compounds |
|-----------|------|-----------|
| `ASINEX` | `Asinex_10177.sdf` | 10,177 |
| `IBS_NP` | `IBS_NP_3678.sdf` | 3,678 |
| `IBS_SP1` | `IBS_SP1_5629.sdf` | 5,629 |
| `IBS_SP2` | `IBS_SP2_3424.sdf` | 3,424 |
| `IBS_SP3` | `IBS_SP3_9690.sdf` | 9,690 |
| `NCI` | `NCI_10283.sdf` | 10,283 |
| `ZINC_NP` | `ZINC_NP_9644.sdf` | 9,644 |

All database files are in `ComfyQSAR/Screening_DB/`.

### Outputs

| Output | Description |
|------|-------------|
| `SCREENING_RESULTS` | CSV containing compounds selected by the threshold |
| `SELECTED_MOLECULES` | SDF containing the selected molecular structures |

Results are saved to `ComfyUI/output/Screening/Database_Screening/<DB_NAME>/`.

### Example: PTP1B Screening with ASINEX

1. Add **External Screening (Database)** node
2. Set `trained_model_path`: `example/CustomScreening_example/PTP1B_prediction_QSAR_model.pkl`
3. Set `descriptor_list_path`: `example/CustomScreening_example/selected_features_list.txt`
4. Set `preprocessing_recipe_path`: `example/CustomScreening_example/preprocessing_recipe.json`
5. Set `db_name`: `ASINEX`
6. Set `threshold`: `0.5`
7. Run — selected compounds saved as CSV + SDF

---

## Node 2: External Screening (Custom Compounds)

**Category**: `QSAR/3. SCREENER`

An all-in-one node that processes a custom SDF file through the complete screening pipeline: standardization → descriptor calculation → preprocessing → screening.

### Inputs

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_sdf_path` | STRING | Empty | Input SDF file |
| `trained_model_path` | STRING | Empty | Trained QSAR model (from Step 7) |
| `descriptor_list_path` | STRING | Empty | Selected descriptor list (from Step 7) |
| `preprocessing_recipe_path` | STRING | Empty | `preprocessing_recipe.json` (from Step 4) |
| `threshold` | FLOAT | `0.5` | Prediction threshold |
| `max_missing_fraction` | FLOAT | `0.5` | Optional. A compound whose selected descriptors are missing more than this fraction is flagged `low_quality_input` in the output, not excluded. The fraction is computed over the model's selected descriptors only, not the full PaDEL descriptor set. |

The file path fields are intentionally empty by default.

### Outputs

| Output | Description |
|---|---|
| `SCREENING_RESULTS` | Screening prediction CSV |
| `SELECTED_MOLECULES` | Selected-compound SDF |

The standardized-molecule SDF, raw descriptor CSV, and preprocessed descriptor CSV are also written to disk (paths shown in the node log) but are not exposed as separate output sockets.

### Internal Pipeline

```
Input SDF
    ↓
1. Standardization
   - Cleanup → largest-fragment retention (salts/counter-ions stripped)
     → uncharge → canonical tautomer
   - Reject: unparseable structures, metal-only molecules,
     structures empty after standardization
    ↓
2. Descriptor Calculation (PaDEL, 2D)
    ↓
3. Preprocessing
   - Filter to the model's selected descriptors
   - Impute missing values from the recipe's training-fitted statistics
   - Flag (not exclude) compounds above max_missing_fraction as low_quality_input
    ↓
4. Screening
   - Predict with trained model
   - Select compounds above threshold
    ↓
Output CSV + SDF
```

### Output File Structure

```
ComfyUI/output/Screening/Custom_Screening/
├── custom_db_prepared/
│   ├── standardized_input.sdf
│   ├── molecular_descriptors.csv
│   └── preprocessed_data.csv
└── custom_screening_results/
    ├── User_Screening_Predictions.csv
    └── User_Screening_Selected_Molecules.sdf
```

### Example: Screening a Custom SDF

![Custom screening workflow example](https://github.com/DDUKHAE/ComfyQSAR/blob/main/example/CustomScreening_example/Custom_Screening_workflow.png)

**Included example file**: `example/CustomScreening_example/PTP1B_custom.sdf`

1. Add **External Screening (Custom Compounds)** node
2. Set `input_sdf_path`: `/path/to/ComfyQSAR/example/CustomScreening_example/PTP1B_custom.sdf`
3. Set `trained_model_path`: `/path/to/ComfyQSAR/example/CustomScreening_example/PTP1B_prediction_QSAR_model.pkl`
4. Set `descriptor_list_path`: `/path/to/ComfyQSAR/example/CustomScreening_example/selected_features_list.txt`
5. Set `preprocessing_recipe_path`: `/path/to/ComfyQSAR/example/CustomScreening_example/preprocessing_recipe.json`
6. Set `threshold`: `0.5`
7. Run

**Input**: 200 compounds. Standardization keeps every parseable, non-metal-only compound. Preprocessing imputes missing selected-descriptor values from the recipe and flags (rather than excludes) any compound above `max_missing_fraction`. Screening then applies `threshold` to the imputed descriptor matrix.

The screenshot's right-hand node (Screening-Candidate AD) connects to this node's `SCREENING_RESULTS` output but needs its own `training_data_path` -- see [Node 3/4](#node-34-screening-candidate-ad-classification--regression) below for why that file isn't included with this example.

---

## Node 3/4: Screening-Candidate AD (Classification / Regression)

**Category**: `QSAR/3. SCREENER`

Applies the same Applicability Domain method and training set as step 9's evaluation-set AD node, but to a screening run's candidates instead of a hold-out/external set. `in_domain` does not confirm a prediction is correct -- only that the candidate lies within the training chemical/descriptor space; `out_of_domain` means the prediction is an extrapolation.

Classification uses kNN-mean Tanimoto similarity (same as step 9). Regression uses leverage only -- no standardized-residual axis, since screening candidates have no observed endpoint to compute a residual against.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `training_data_path` | STRING | Exact training dataset used by Node 7, after preprocessing and any optional descriptor selection or combination -- the same file given to step 9's evaluation-set AD node. This is a full descriptor matrix (e.g. step 4's `PREPROCESSED_TRAINING` output, or step 6's combination output if that step was used), not the descriptor-name list from step 7 |
| `screening_results_path` | STRING | The Screener's `SCREENING_RESULTS` output |
| `mode` | CHOICE | `auto` (default): run if both files have a SMILES column, skip silently otherwise. `manual`: same computation, but error instead of skipping if SMILES is missing. `disabled`: do nothing |
| `fingerprint_radius`, `fingerprint_bits`, `k_neighbors`, `ad_percentile_threshold` (Classification) | — | Keep identical to the evaluation-set AD node |
| `target_column`, `warning_leverage_multiplier` (Regression) | — | Keep identical to the evaluation-set AD node |

### Outputs

| Output | Description |
|---|---|
| `SCREENING_AD_REPORT` | `screening_results_path` with AD columns (`in_domain`/`out_of_domain`, leverage for Regression) appended -- every original column is preserved |

Results are saved to `ComfyUI/output/Screening/Applicability_Domain/<Classification\|Regression>/Screening_Candidate_AD_Report.csv`.

The bundled PTP1B custom-screening example (Node 2 above) has no `training_data_path` to supply -- its original training dataset was never packaged with the pre-trained model, only the model itself. Screening-Candidate AD requires a `training_data_path` from a training run you performed yourself (see [Using Your Own Trained Model](#using-your-own-trained-model)); it cannot be run against the PTP1B example as-is.

---

## Using Your Own Trained Model

After running the training pipeline through Step 7 (Hyperparameter Tuning & Model Training):

Classification:

- `ComfyUI/output/Classification/07_Model_Training/Best_Classifier_<algorithm>.pkl`
- `ComfyUI/output/Classification/07_Model_Training/Final_Selected_Descriptors_<algorithm>.txt`
- `ComfyUI/output/Classification/04_Descriptor_Preprocessing/preprocessing_recipe.json`

Regression:

- `ComfyUI/output/Regression/07_Model_Training/Best_Regressor_<algorithm>.pkl`
- `ComfyUI/output/Regression/07_Model_Training/Final_Descriptors_<algorithm>.txt`
- `ComfyUI/output/Regression/04_Descriptor_Preprocessing/preprocessing_recipe.json`

Use these paths in either screening node.

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `FileNotFoundError: model file not found` | Use absolute paths or verify the file exists |
| `FileNotFoundError: ... file not found` (descriptor list / recipe) | Use absolute paths; ensure the `.txt` descriptor list and `preprocessing_recipe.json` are from the same training run as the model |
| `ValueError: Recipe has no stored imputer statistic for ...` | The recipe was built from a different descriptor selection than `descriptor_list_path`; regenerate one against the other |
| `PaDEL error` | Install Java JRE 11+: `sudo apt install default-jre` |
| `padelpy not found` | `pip install padelpy` |
| Many compounds flagged `low_quality_input` | Raise `max_missing_fraction`, or check that the input SDF's chemistry matches what the model was trained on |
| Memory error | Reduce input SDF size; use 2D descriptors |

---

## Notes

- **Path resolution**: Relative paths are resolved from `ComfyQSAR/py/Screener/`. Absolute paths are recommended.
- **Descriptor consistency**: The External Screening (Custom Compounds) node always calculates 2D descriptors. Ensure your model was also trained with 2D descriptors.
- **Threshold**: For classification models, threshold applies to the predicted probability of the positive class. For regression models, it is a minimum predicted value cutoff.
- **LOG_MESSAGE**: Both nodes print a detailed log to the ComfyUI console showing compound counts at each step.
