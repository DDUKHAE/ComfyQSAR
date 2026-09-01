import os
import pandas as pd
import numpy as np
import joblib
import traceback
import folder_paths
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix,
)

def _clean_name_col(df):
    """Stripped string Name series if the column exists, else None. Blank
    strings and the literal text 'nan' (from a stray NaN read as text
    upstream) are normalized to '' so blank-detection catches both."""
    if "Name" not in df.columns:
        return None
    s = df["Name"].astype(str).str.strip()
    return s.where(s.str.lower() != "nan", "")

def load_classification_inputs(trained_model_path, x_path, y_path, features_path):
    model = joblib.load(trained_model_path)
    # dtype={"Name": str} forces the Name column to stay text even when every
    # value looks numeric (e.g. "007") -- otherwise pandas' own dtype
    # inference reads it as int64 (007 -> 7) before .astype(str) ever runs,
    # so the leading zero is already gone by the time we try to preserve it.
    # A dtype dict entry for a column that isn't present is a no-op, so this
    # is safe even when X/Y have no Name column at all.
    x_test_df = pd.read_csv(x_path, dtype={"Name": str})
    y_test_df = pd.read_csv(y_path, dtype={"Name": str})

    if "Label" in y_test_df.columns:
        target_col = "Label"
    elif y_test_df.shape[1] == 1:
        target_col = y_test_df.columns[0]
    else:
        raise ValueError(f"Could not determine target column in {os.path.basename(y_path)}.")

    x_name = _clean_name_col(x_test_df)
    y_name = _clean_name_col(y_test_df)

    # ID-based alignment when BOTH files carry a Name column -- never assume
    # row order matches just because two files happen to have the same
    # length (e.g. independent upstream filtering/sorting could silently
    # desynchronize them, combining a prediction with the wrong compound's
    # ground truth).
    if x_name is not None and y_name is not None:
        for label, path, names in (("X_test", x_path, x_name), ("Y_test", y_path, y_name)):
            n_blank = int((names == "").sum())
            if n_blank:
                raise ValueError(f"{label} ({os.path.basename(path)}) has {n_blank} blank/missing Name value(s).")
            dup = names[names.duplicated()]
            if len(dup):
                dup_list = sorted(set(dup.tolist()))
                preview = dup_list[:10]
                more = f" (+{len(dup_list) - 10} more)" if len(dup_list) > 10 else ""
                raise ValueError(f"{label} ({os.path.basename(path)}) has duplicate Name value(s): {preview}{more}")

        x_set, y_set = set(x_name.tolist()), set(y_name.tolist())
        if x_set != y_set:
            only_x = sorted(x_set - y_set)[:10]
            only_y = sorted(y_set - x_set)[:10]
            raise ValueError(
                f"X_test ({os.path.basename(x_path)}) and Y_test ({os.path.basename(y_path)}) "
                f"Name sets do not match. In X_test only (up to 10): {only_x}. "
                f"In Y_test only (up to 10): {only_y}."
            )

        y_lookup = y_test_df.copy()
        y_lookup["Name"] = y_name
        y_reordered = y_lookup.set_index("Name").loc[x_name.tolist()].reset_index()
        y_test = y_reordered[target_col].to_numpy()
        alignment_mode = "id_matched"
        names = x_name.tolist()
    else:
        if len(x_test_df) != len(y_test_df):
            raise ValueError(
                f"X_test ({os.path.basename(x_path)}, {len(x_test_df)} rows) and Y_test "
                f"({os.path.basename(y_path)}, {len(y_test_df)} rows) have different lengths, and "
                "at least one file has no 'Name' column to align by -- cannot safely combine them."
            )
        y_test = y_test_df[target_col].to_numpy()
        alignment_mode = "row_order_fallback"
        names = x_name.tolist() if x_name is not None else [str(i) for i in range(len(x_test_df))]

    nan_mask = pd.isna(y_test)
    if nan_mask.any():
        n_nan = int(nan_mask.sum())
        if alignment_mode == "id_matched":
            bad_names = x_name.tolist()
            bad_names = [n for n, is_nan in zip(bad_names, nan_mask) if is_nan][:10]
            raise ValueError(f"Target has {n_nan} missing (NaN) value(s), e.g. Name={bad_names}.")
        raise ValueError(f"Target has {n_nan} missing (NaN) value(s) (row_order_fallback -- no Name available to identify them).")

    unique_labels = set(pd.unique(y_test))
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Classification target in {os.path.basename(y_path)} must be binary-encoded as 0/1 "
            f"(found: {sorted(unique_labels, key=str)}). Encode binary labels as 0/1 before "
            "running this node. (A single-class external set, e.g. {0} or {1} only, is allowed.)"
        )

    smiles = x_test_df["SMILES"].astype(str).tolist() if "SMILES" in x_test_df.columns else [""] * len(x_test_df)

    with open(features_path, "r") as f:
        selected_features = [line.strip() for line in f if line.strip()]
    missing = [ft for ft in selected_features if ft not in x_test_df.columns]
    if missing:
        raise ValueError(f"The following features are missing from X_test: {', '.join(missing)}")
    return model, x_test_df[selected_features], y_test, names, smiles, alignment_mode

def _binary_metrics_with_domain_table(y_true, y_pred, y_proba):
    """Computes every headline binary-classification metric, but only where
    it is mathematically defined given y_true's actual class composition
    for THIS sample (whether the full external set or one bootstrap
    replicate) -- an undefined metric is None (N/A), never sklearn's
    zero_division=0 fallback masquerading as a real measurement.

      - both classes present: everything is well-defined and computed.
      - all-negative (y_true only {0}): Accuracy and Specificity are
        well-defined (their denominators are all-negative-safe).
        Sensitivity/Precision/F1/Balanced Accuracy/MCC/ROC-AUC all have a
        0/0 or single-class denominator -> None.
      - all-positive (y_true only {1}): Accuracy and Sensitivity are
        well-defined. Specificity/Balanced Accuracy/MCC/ROC-AUC -> None
        (same reasoning, mirrored). Precision/F1 are *conditionally*
        defined: false positives are impossible when every true label is
        positive, so Precision = TP/(TP+0) = 1.0 whenever TP>0, and is a
        true 0/0 (None) only when the model predicted zero positives.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes_present = set(np.unique(y_true))
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": None, "precision": None, "recall": None,
        "balanced_accuracy": None, "mcc": None, "roc_auc": None, "specificity": None,
    }
    if classes_present == {0, 1}:
        result["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
        result["precision"] = precision_score(y_true, y_pred, zero_division=0)
        result["recall"] = recall_score(y_true, y_pred, zero_division=0)
        result["specificity"] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        result["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        result["mcc"] = matthews_corrcoef(y_true, y_pred)
        if y_proba is not None:
            result["roc_auc"] = roc_auc_score(y_true, y_proba)
    elif classes_present == {0}:
        result["specificity"] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    elif classes_present == {1}:
        result["recall"] = recall_score(y_true, y_pred, zero_division=0)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        if tp > 0:
            result["precision"] = precision_score(y_true, y_pred, zero_division=0)
            result["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
    return result

def calculate_classification_metrics(model, x_test, y_test):
    if hasattr(model, "classes_") and len(model.classes_) > 2:
        raise ValueError(
            f"Model has {len(model.classes_)} classes ({list(model.classes_)}) -- this node "
            "only supports binary 0/1 classification models."
        )
    y_pred = model.predict(x_test)
    y_proba = None
    try:
        proba_matrix = model.predict_proba(x_test)
        if hasattr(model, "classes_") and 1 in list(model.classes_):
            pos_idx = list(model.classes_).index(1)
        else:
            pos_idx = 1 if proba_matrix.shape[1] > 1 else 0
        y_proba = proba_matrix[:, pos_idx]
    except Exception:
        pass
    metrics = _binary_metrics_with_domain_table(y_test, y_pred, y_proba)
    return metrics, y_pred, y_proba

def bootstrap_classification_ci(y_test, y_pred, y_proba, metrics, n_bootstrap=2000, seed=42):
    """Bootstrap 95% CI for the headline classification metrics. Resamples
    compound indices once per replicate and reuses that draw for every
    metric (mirrors thesis_metric_verification/calculate_manuscript_uncertainty.py's
    methodology so results are directly comparable across analyses). Each
    replicate is scored through the same domain-table logic as the point
    estimate -- a replicate that happens to draw only one class invalidates
    only the metrics that are actually undefined for it (e.g. an
    all-negative replicate still contributes a valid Accuracy/Specificity
    draw, just not Sensitivity/Precision/F1/Balanced Accuracy/MCC/ROC-AUC)."""
    metric_keys = ["accuracy", "f1_score", "precision", "recall", "specificity",
                   "balanced_accuracy", "mcc", "roc_auc"]
    labels = {
        "accuracy": "Accuracy", "f1_score": "F1-Score", "precision": "Precision",
        "recall": "Recall (Sensitivity)", "specificity": "Specificity",
        "balanced_accuracy": "Balanced Accuracy", "mcc": "MCC", "roc_auc": "ROC-AUC",
    }
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    y_proba_arr = np.asarray(y_proba) if y_proba is not None else None
    n = len(y_test)
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in metric_keys}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_test[idx], y_pred[idx]
        yproba_i = y_proba_arr[idx] if y_proba_arr is not None else None
        rep_metrics = _binary_metrics_with_domain_table(yt, yp, yproba_i)
        for k in metric_keys:
            if rep_metrics[k] is not None:
                vals[k].append(rep_metrics[k])

    rows = []
    for k in metric_keys:
        if metrics.get(k) is None:
            continue
        v = np.asarray(vals[k], dtype=float)
        n_valid = len(v)
        if n_valid > 0:
            ci_lo, ci_hi = float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))
        else:
            ci_lo, ci_hi = float("nan"), float("nan")
        rows.append({
            "Metric": labels[k], "Point_Estimate": metrics[k],
            "CI_Lower": ci_lo, "CI_Upper": ci_hi,
            "N_Bootstrap": n_bootstrap, "Bootstrap_Seed": seed,
            "N_Valid_Replicates": n_valid, "N_Invalid_Replicates": n_bootstrap - n_valid,
        })
    return pd.DataFrame(rows)

def save_classification_results(output_dir, y_test, y_pred, y_proba, metrics, names, smiles, alignment_mode,
                                 compute_bootstrap_ci=True, n_bootstrap=2000, bootstrap_seed=42):
    pred_df = pd.DataFrame({
        "Name": names,
        "SMILES": smiles,
        "Actual": y_test,
        "Predicted": y_pred,
        "Probability": y_proba if y_proba is not None else [np.nan] * len(y_test),
        "Alignment_Mode": alignment_mode,
    })
    pred_path = os.path.join(output_dir, "Actual_vs_Predicted.csv")
    pred_df.to_csv(pred_path, index=False)
    eval_data = {
        "Metric": [
            "Accuracy", "F1-Score", "ROC-AUC", "Precision", "Recall (Sensitivity)",
            "Specificity", "Balanced Accuracy", "MCC", "Alignment_Mode",
        ],
        "Value": [
            metrics["accuracy"], metrics["f1_score"], metrics["roc_auc"],
            metrics["precision"], metrics["recall"], metrics["specificity"],
            metrics["balanced_accuracy"], metrics["mcc"], alignment_mode,
        ]
    }
    eval_df = pd.DataFrame(eval_data)
    eval_path = os.path.join(output_dir, "Evaluation_Results_ExternalTestSet.csv")
    eval_df.to_csv(eval_path, index=False)

    labels = sorted(set(np.unique(y_test)) | set(np.unique(y_pred)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{l}" for l in labels],
        columns=[f"Predicted_{l}" for l in labels],
    )
    cm_path = os.path.join(output_dir, "Confusion_Matrix.csv")
    cm_df.to_csv(cm_path)

    ci_path = None
    ci_df = None
    stale_ci_path = os.path.join(output_dir, "Bootstrap_CI_Results.csv")
    if compute_bootstrap_ci:
        ci_df = bootstrap_classification_ci(y_test, y_pred, y_proba, metrics, n_bootstrap, bootstrap_seed)
        ci_path = stale_ci_path
        ci_df.to_csv(ci_path, index=False)
    elif os.path.exists(stale_ci_path):
        # A prior run with compute_bootstrap_ci=True left this file behind --
        # remove it so it can't be mistaken for this run's (nonexistent) CI.
        os.remove(stale_ci_path)

    return eval_path, pred_path, cm_path, ci_path, ci_df

class Model_Validation_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trained_model_path": ("STRING", {"tooltip": "From 7."}),
                "descriptor_list_path": ("STRING", {"tooltip": "From 7. The .txt name list, not a CSV."}),
                "holdout_data_path": ("STRING", {"tooltip": "Use 4's PREPROCESSED_HOLDOUT -- not 3's HOLDOUT_DATA (no train-fitted imputation yet)."}),
                "holdout_targets_path": ("STRING", {"tooltip": "Use 4's FILTERED_HOLDOUT_TARGETS -- not 3's HOLDOUT_TARGETS (4 may drop compounds, so the row sets must match)."}),
            },
            "optional": {
                "compute_bootstrap_ci": ("BOOLEAN", {"default": True}),
                "n_bootstrap": ("INT", {"default": 2000, "min": 100, "max": 100000, "step": 100}),
                "bootstrap_seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("TRAINED_MODEL", "SELECTED_DESCRIPTOR_LIST")
    OUTPUT_TOOLTIPS = (
        "Same value as this node's own trained_model_path input -- routed through here so "
        "downstream nodes (e.g. External Data Screening) only run after this validation.",
        "Same value as this node's own descriptor_list_path input -- routed through for the same reason.",
    )
    FUNCTION = "validate_model"
    CATEGORY = "QSAR/1. CLASSIFICATION/8. Model Evaluation"
    OUTPUT_NODE = True

    def validate_model(self, trained_model_path, descriptor_list_path, holdout_data_path, holdout_targets_path,
                        compute_bootstrap_ci=True, n_bootstrap=2000, bootstrap_seed=42):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Classification", "08_Model_Evaluation", "Holdout_External_Performance")
            os.makedirs(output_dir, exist_ok=True)
            model, x_test_filtered, y_test, names, smiles, alignment_mode = load_classification_inputs(
                trained_model_path, holdout_data_path, holdout_targets_path, descriptor_list_path
            )
            metrics, y_pred, y_proba = calculate_classification_metrics(model, x_test_filtered, y_test)
            eval_path, pred_path, cm_path, ci_path, ci_df = save_classification_results(
                output_dir, y_test, y_pred, y_proba, metrics, names, smiles, alignment_mode,
                compute_bootstrap_ci, n_bootstrap, bootstrap_seed,
            )
            def _fmt_metric(label, key):
                v = metrics[key]
                return f"📊 {label}: {v:.4f}" if v is not None else f"📊 {label}: N/A (undefined for this class composition)"

            unique_test_classes = sorted(set(np.unique(y_test)))
            log_lines = [
                "========================================",
                "🔹 8.1 Hold-out & External Performance Done! 🔹",
                "========================================",
                f"📌 Model: {os.path.basename(trained_model_path)}",
                f"📌 External set classes present: {unique_test_classes}"
                + (" (single-class -- some metrics are N/A, see below)" if len(unique_test_classes) < 2 else ""),
                f"🏆 Accuracy: {metrics['accuracy']:.4f}",
                _fmt_metric("F1 Score", "f1_score"),
                _fmt_metric("ROC-AUC", "roc_auc"),
                _fmt_metric("Precision", "precision"),
                _fmt_metric("Recall (Sensitivity)", "recall"),
                _fmt_metric("Specificity", "specificity"),
                _fmt_metric("Balanced Accuracy", "balanced_accuracy"),
                _fmt_metric("MCC", "mcc"),
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}",
                f"💾 Evaluation Results: {os.path.basename(eval_path)}",
                f"💾 Predictions: {os.path.basename(pred_path)}",
                f"💾 Confusion Matrix: {os.path.basename(cm_path)}",
            ]
            log_lines.append(f"📌 Alignment mode: {alignment_mode}")
            if alignment_mode == "row_order_fallback":
                log_lines.append(
                    "⚠️ X_test/Y_test could not be aligned by 'Name' (missing on at least one side) -- "
                    "fell back to row order. Verify both files came from the same, unreordered pipeline run."
                )
            if compute_bootstrap_ci and ci_df is not None:
                log_lines.append(f"📈 Bootstrap 95% CI (n_bootstrap={n_bootstrap}, seed={bootstrap_seed}):")
                for _, row in ci_df.iterrows():
                    log_lines.append(
                        f"    {row['Metric']}: {row['Point_Estimate']:.4f} "
                        f"[{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}]"
                    )
                log_lines.append(f"💾 Bootstrap CI: {os.path.basename(ci_path)}")
            log_lines.append("========================================")
            return {
                "ui": {"text": "\n".join(log_lines)},
                "result": (
                    str(trained_model_path),
                    str(descriptor_list_path),
                )
            }
        except Exception as e:
            return {
                "ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"},
                "result": ("", "")
            }

NODE_CLASS_MAPPINGS = {
    "Model_Validation_Classification": Model_Validation_Classification,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Model_Validation_Classification": "8.1 Hold-out & External Performance",
}
