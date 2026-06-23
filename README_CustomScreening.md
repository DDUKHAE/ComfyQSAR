# ComfyQSAR — Virtual Screening Guide

## Overview

ComfyQSAR provides two screening nodes under the `QSAR/SCREENER` category:

| Node | Display Name | Use Case |
|------|-------------|----------|
| `QSARDBScreener` | QSAR DB Screener | Screen 7 pre-computed compound databases |
| `QSARCustomUserScreener` | Custom User Screener | Screen a custom SDF file (all-in-one) |

Both nodes require a trained QSAR model (`.pkl`) and a selected feature list (`.txt`) produced by the training pipeline (Step 7: Hyperparameter Tuning & Model Training).

---

## Node 1: QSAR DB Screener

**Category**: `QSAR/SCREENER`

Screens one of 7 pre-computed compound databases. Descriptor calculation is already done — results are instant.

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | STRING | Path to trained `.pkl` model file |
| `features_path` | STRING | Path to selected descriptors `.txt` file |
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

1. Add **QSAR DB Screener** node
2. Set `model_path`: `PTP1B_prediction_QSAR_model.pkl`
3. Set `features_path`: `selected_features_V3.txt`
4. Set `db_name`: `ASINEX`
5. Set `threshold`: `0.5`
6. Run — selected compounds saved as CSV + SDF

---

## Node 2: Custom User Screener

**Category**: `QSAR/SCREENER`

An all-in-one node that processes a custom SDF file through the complete screening pipeline: standardization → descriptor calculation → preprocessing → screening.

### Inputs

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_sdf_path` | STRING | Empty | Input SDF file |
| `model_path` | STRING | Empty | Trained QSAR model |
| `features_path` | STRING | Empty | Selected descriptor list |
| `threshold` | FLOAT | `0.5` | Prediction threshold |
| `nan_threshold` | FLOAT | `0.5` | Maximum allowed NaN fraction |
| `impute_method` | CHOICE | `mean` | Missing-value imputation method |

The file path fields are intentionally empty by default.

### Outputs

| Output | Description |
|---|---|
| `STANDARDIZED_MOLECULES` | Standardized SDF |
| `DESCRIPTORS` | Raw PaDEL descriptor CSV |
| `PREPROCESSED_DESCRIPTORS` | Preprocessed descriptor CSV |
| `SCREENING_RESULTS` | Screening prediction CSV |
| `SELECTED_MOLECULES` | Selected-compound SDF |

### Internal Pipeline

```
Input SDF
    ↓
1. Standardization
   - Remove metal-only molecules
   - Remove multi-fragment molecules
   - Remove RDKit-invalid structures
    ↓
2. Descriptor Calculation (PaDEL, 2D)
    ↓
3. Preprocessing
   - Replace Inf → NaN
   - Remove compounds with NaN > nan_threshold
   - Remove descriptors with NaN > nan_threshold
   - Impute remaining NaN (mean/median/most_frequent)
    ↓
4. Screening
   - Filter to selected features
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

**Included example file**: `example/PTP1B_custom.sdf`

1. Add **Custom User Screener** node
2. Set `input_sdf_path`: `/path/to/ComfyQSAR/example/PTP1B_custom.sdf`
3. Set `model_path`: `/path/to/ComfyQSAR/PTP1B_prediction_QSAR_model.pkl`
4. Set `features_path`: `/path/to/ComfyQSAR/selected_features_V3.txt`
5. Set `threshold`: `0.5`
6. Set `nan_threshold`: `0.5`, `impute_method`: `mean`
7. Run

**Expected results** (200-compound PTP1B example):
- Input: 200 compounds
- After standardization: ~200 (invalid structures removed)
- After preprocessing: ~198 (high-NaN compounds filtered)
- Selected (threshold 0.5): ~195

---

## Using Your Own Trained Model

After completing the 8-step training pipeline:

Classification:

- `ComfyUI/output/Classification/07_Model_Training/Best_Classifier_<model>.pkl`
- `ComfyUI/output/Classification/07_Model_Training/Final_Selected_Descriptors.txt`

Regression:

- `ComfyUI/output/Regression/07_Model_Training/Best_Regressor_<model>.pkl`
- `ComfyUI/output/Regression/07_Model_Training/Final_Descriptors.txt`

Use these paths in either screening node.

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `FileNotFoundError: model file not found` | Use absolute paths or verify the file exists |
| `FileNotFoundError: features file not found` | Use absolute paths; ensure the `.txt` file is from the same training run |
| `PaDEL error` | Install Java JRE 11+: `sudo apt install default-jre` |
| `padelpy not found` | `pip install padelpy` |
| Missing features after preprocessing | Check `nan_threshold` — lower values are more permissive; ensure the model and descriptor calculation settings are consistent |
| Memory error | Reduce input SDF size; use 2D descriptors |

---

## Notes

- **Path resolution**: Relative paths are resolved from `ComfyQSAR/py/Screener/`. Absolute paths are recommended.
- **Descriptor consistency**: The Custom User Screener always calculates 2D descriptors. Ensure your model was also trained with 2D descriptors.
- **Threshold**: For classification models, threshold applies to the predicted probability of the positive class. For regression models, it is a minimum predicted value cutoff.
- **LOG_MESSAGE**: Both nodes print a detailed log to the ComfyUI console showing compound counts at each step.
