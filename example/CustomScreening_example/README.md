# Custom Screening Example — PTP1B

Files for the "External Screening (Custom Compounds)" node walkthrough in
[README_CustomScreening.md](../../README_CustomScreening.md).

| File | Description |
|---|---|
| `PTP1B_custom.sdf` | 200-compound screening candidate set |
| `PTP1B_prediction_QSAR_model.pkl` | Trained XGBoost classifier (113 descriptors) |
| `selected_features_list.txt` | Selected descriptor names, matches the model |
| `preprocessing_recipe.json` | Training-fitted imputation statistics for the selected descriptors |
| `Custom_Screening_workflow.png` | ComfyUI node graph for this example |

## Recommended workflow

Add **External Screening (Custom Compounds)** and set:

| Input | Value |
|---|---|
| `input_sdf_path` | `PTP1B_custom.sdf` |
| `trained_model_path` | `PTP1B_prediction_QSAR_model.pkl` |
| `descriptor_list_path` | `selected_features_list.txt` |
| `preprocessing_recipe_path` | `preprocessing_recipe.json` |

This alone reproduces the screenshot's left-hand node and is sufficient to
run this example end to end.

The screenshot's right-hand node, **Screening-Candidate AD (Classification)**,
is wired by connecting the left node's `SCREENING_RESULTS` output to its
`screening_results_path` input. It also needs a `training_data_path`, which
this example does not include -- the PTP1B model's original training
dataset was never packaged, only the trained model itself. Adding the AD
node to this graph therefore requires a `training_data_path` from a
training run of your own (see README_CustomScreening.md's "Using Your Own
Trained Model"); it is not runnable against the files in this folder alone.
