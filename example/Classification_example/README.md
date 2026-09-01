# Classification Example — Bioconcentration Factor (QDB116)

**Source**: Piir G, Sild S, Maran U. Comparative analysis of local and
consensus quantitative structure-activity relationship models for
bioconcentration factor. *SAR and QSAR in Environmental Research*.
2014;25(12):967-981. DOI: 10.1080/1062936X.2014.969310

**QSARDB archive**: `10.15152/QDB.116`

**Task**: Classification — bioaccumulative vs. non-bioaccumulative compounds.
1,007 compounds: training 673 (141 bioaccumulative/532 non), external 334
(68/266).

This folder is the full ComfyQSAR pipeline output for this dataset, one file
set per step, matching the "ComfyQSAR default workflow" track reported for
QDB116 in the manuscript's benchmark (Logistic Regression, ALogP/nHBAcc/
LipinskiFailures + combination search).

## Structure

| Folder | Step | Contents |
|---|---|---|
| `data/` | raw input | Training/external positive and negative SMILES |
| `01_Data_Load_and_Standardization/` | 1 | Standardized structures + report (train, external) |
| `02_Descriptor_Calculation/` | 2 | PaDEL descriptors, sanitized (train, external) |
| `04_Descriptor_Preprocessing/` | 4 | Paired train/hold-out preprocessing output + `preprocessing_recipe.json` |
| `05_Descriptor_Optimization/` | 5 | Filter-based and model-based descriptor selection |
| `06_Descriptor_Combination/` | 6 | Descriptor-combination search results |
| `07_Model_Training/` | 7 | Trained classifier + selected descriptors |
| `08_Model_Evaluation/` | 8.1-8.3 | Hold-out/external performance, Y-scrambling, resampling stability |
| `09_Applicability_Domain/` | 9 | AD assessment report |

`Classification_workflow.png` is the corresponding ComfyUI workflow
screenshot.
