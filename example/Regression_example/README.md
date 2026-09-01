# Regression Example — Antiproliferative Activity (QDB261)

**Source**: Zukić S, Osmanović A, Harej Hrkać A, Kraljević Pavelić S,
Špirtović-Halilović S, Veljović E, Roca S, Trifunović S, Završnik D, Maran U.
Data-Driven Modelling of Substituted Pyrimidine and Uracil-Based Derivatives
Validated with Newly Synthesized and Antiproliferative Evaluated Compounds.
*International Journal of Molecular Sciences*. 2024;25(17):9390.
DOI: 10.3390/ijms25179390

**QSARDB archive**: `10.15152/QDB.261`

**Task**: Regression — antiproliferative activity of substituted
pyrimidine/uracil derivatives. 39 compounds, QSARDB-provided split: training
31, validation 8.

This folder is the full ComfyQSAR pipeline output for this dataset, one file
set per step, matching the "ComfyQSAR default workflow" track reported for
QDB261 in the manuscript's benchmark (Linear Regression, SHCsats/minsssCH/
maxaaCH + combination search).

## Structure

| Folder | Step | Contents |
|---|---|---|
| `data/` | raw input | QSARDB-provided train/validation SMILES and target values |
| `01_Data_Load_and_Standardization/` | 1 | Standardized structures + report (train, validation) |
| `02_Descriptor_Calculation/` | 2 | PaDEL descriptors, sanitized (train, validation) |
| `04_Descriptor_Preprocessing/` | 4 | Paired train/hold-out preprocessing output + `preprocessing_recipe.json` |
| `05_Descriptor_Optimization/` | 5 | Filter-based and model-based descriptor selection |
| `06_Descriptor_Combination/` | 6 | Descriptor-combination search results |
| `07_Model_Training/` | 7 | Trained regressor + selected descriptors |
| `08_Model_Evaluation/` | 8.1-8.3 | Hold-out/external performance, Y-scrambling, resampling stability |
| `09_Applicability_Domain/` | 9 | AD assessment report |

`Regression_workflow.png` is the corresponding ComfyUI workflow screenshot.
