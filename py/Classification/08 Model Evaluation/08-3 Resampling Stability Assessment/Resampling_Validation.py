import os
import multiprocessing
import traceback
import numpy as np
import pandas as pd
import joblib
import folder_paths
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import make_scorer, f1_score, accuracy_score, matthews_corrcoef, balanced_accuracy_score


class ResamplingValidation_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "training_data_path": ("STRING", {
                    "tooltip": "Exact training dataset used by Node 7, after preprocessing and any optional descriptor selection or combination.",
                }),
                "trained_model_path": ("STRING", {
                    "tooltip": "Same as 8.2 -- hyperparameters only, refit each fold.",
                }),
                "target_column": ("STRING", {"default": "Label"}),
                "method": (["repeated_stratified_kfold", "repeated_kfold", "loocv"], {"default": "repeated_stratified_kfold"}),
            },
            "optional": {
                "n_repeats": ("INT", {"default": 10, "min": 1, "max": 100, "tooltip": "Ignored for loocv."}),
                "cv_splits": ("INT", {"default": 5, "min": 2, "max": 10, "tooltip": "Ignored for loocv."}),
                "random_state": ("INT", {"default": 42}),
                "num_cores": ("INT", {"default": 1, "min": -1, "max": multiprocessing.cpu_count(),
                                       "tooltip": "1 (default) runs in-process with no worker-pool startup cost -- "
                                                  "recommended for typical QSAR-sized datasets/repeat counts, "
                                                  "where loky's per-worker boot/unpickle overhead outweighs any "
                                                  "speedup. Only raise this for genuinely large workloads."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESAMPLING_REPORT",)
    FUNCTION = "run"
    CATEGORY = "QSAR/1. CLASSIFICATION/8. Model Evaluation"
    OUTPUT_NODE = True

    def run(self, training_data_path, trained_model_path, target_column, method,
            n_repeats=10, cv_splits=5, random_state=42, num_cores=1):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Classification", "08_Model_Evaluation", "Resampling_Stability")
            os.makedirs(output_dir, exist_ok=True)

            data = pd.read_csv(training_data_path)
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found.")
            metadata_cols = [c for c in ("Name", "SMILES", target_column) if c in data.columns]
            X = data.drop(columns=metadata_cols)
            y = data[target_column].values
            unique_labels = set(pd.unique(y))
            if unique_labels != {0, 1}:
                raise ValueError(
                    f"Classification target column '{target_column}' must be binary-encoded as 0/1 "
                    f"(found: {sorted(unique_labels, key=str)}). Encode binary labels as 0/1 before "
                    "running this node."
                )
            n_train = len(y)

            base_model = joblib.load(trained_model_path)
            cpu_count = multiprocessing.cpu_count()
            n_jobs = cpu_count if num_cores == -1 else max(1, min(num_cores, cpu_count))
            total_fits = n_train if method == "loocv" else n_repeats * cv_splits
            cores_warning = ""
            if num_cores == -1 and cpu_count > 8:
                cores_warning = (
                    f"⚠️ num_cores=-1 requests all {cpu_count} CPU cores -- for typical "
                    "QSAR-sized datasets/repeat counts, per-worker startup overhead can "
                    "make this slower than a small explicit core count (e.g. 2-4). Consider "
                    "setting num_cores explicitly if this run feels slow.\n"
                )

            if method == "loocv":
                # Per-fold F1 is unreliable here: with a single test sample
                # per fold, a correctly-predicted true-negative fold has NO
                # positive-class instance at all, so F1 (which is defined
                # w.r.t. the positive class) falls back to 0 via
                # zero_division=0 despite the prediction being right --
                # averaging these across many folds systematically deflates
                # F1. Instead, pool every fold's single out-of-fold
                # prediction and compute accuracy/F1/MCC ONCE over the full
                # training set (mirrors the Regression counterpart's pooled
                # LOO Q2/RMSE, for the same reason).
                y_pred_loo = cross_val_predict(base_model, X, y, cv=LeaveOneOut(), n_jobs=n_jobs)
                acc = accuracy_score(y, y_pred_loo)
                bacc = balanced_accuracy_score(y, y_pred_loo)
                f1 = f1_score(y, y_pred_loo, zero_division=0)
                mcc = matthews_corrcoef(y, y_pred_loo)

                summary_df = pd.DataFrame([{
                    "test_scope": "fixed_descriptor_set",
                    "method": method,
                    "n_train": n_train,
                    "n_splits_total": n_train,
                    "mean_accuracy": float(acc), "std_accuracy": None,
                    "pooled_balanced_accuracy": float(bacc),
                    "mean_f1": float(f1), "std_f1": None,
                    "pooled_mcc": float(mcc),
                }])
                raw_df = pd.DataFrame({
                    "split_index": np.arange(n_train),
                    "y_true": y,
                    "y_pred_loo": y_pred_loo,
                })
                acc_headline, f1_headline = float(acc), float(f1)
                n_splits_reported = n_train
                extra_metrics_note = f"\n📊 Pooled Balanced Accuracy: {bacc:.4f}\n📊 Pooled MCC: {mcc:.4f}"
            else:
                # Repeat-level pooled metrics: for each repeat, pool that
                # repeat's out-of-fold predictions across all cv_splits folds
                # into ONE prediction per training compound, then compute
                # accuracy/F1/MCC ONCE on that pooled vector -- giving one
                # score per repeat (n_repeats values total), rather than
                # scoring each individual fold separately (n_repeats*cv_splits
                # values). This avoids the same per-fold small-sample metric
                # distortion the loocv branch above already guards against
                # (a fold with few/no positive-class samples can give an
                # unstable per-fold F1 even though the pooled prediction
                # across the whole repeat is well-defined), and it directly
                # answers "how much does the pooled estimate move if the
                # split itself changes?" rather than "how much does any one
                # fold's score vary?".
                if method == "repeated_stratified_kfold":
                    def make_cv(i):
                        return StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state + i)
                elif method == "repeated_kfold":
                    def make_cv(i):
                        return KFold(n_splits=cv_splits, shuffle=True, random_state=random_state + i)
                else:
                    raise ValueError(f"Unknown method: {method}")

                acc_per_repeat = np.empty(n_repeats, dtype=float)
                bacc_per_repeat = np.empty(n_repeats, dtype=float)
                f1_per_repeat = np.empty(n_repeats, dtype=float)
                mcc_per_repeat = np.empty(n_repeats, dtype=float)
                for i in range(n_repeats):
                    y_pred_oof = cross_val_predict(base_model, X, y, cv=make_cv(i), n_jobs=n_jobs)
                    acc_per_repeat[i] = accuracy_score(y, y_pred_oof)
                    bacc_per_repeat[i] = balanced_accuracy_score(y, y_pred_oof)
                    f1_per_repeat[i] = f1_score(y, y_pred_oof, zero_division=0)
                    mcc_per_repeat[i] = matthews_corrcoef(y, y_pred_oof)

                summary_df = pd.DataFrame([{
                    "test_scope": "fixed_descriptor_set",
                    "method": method,
                    "n_train": n_train,
                    "n_repeats": n_repeats,
                    "cv_splits_per_repeat": cv_splits,
                    "mean_accuracy": float(np.mean(acc_per_repeat)), "std_accuracy": float(np.std(acc_per_repeat)),
                    "mean_balanced_accuracy": float(np.mean(bacc_per_repeat)), "std_balanced_accuracy": float(np.std(bacc_per_repeat)),
                    "mean_f1": float(np.mean(f1_per_repeat)), "std_f1": float(np.std(f1_per_repeat)),
                    "mean_mcc": float(np.mean(mcc_per_repeat)), "std_mcc": float(np.std(mcc_per_repeat)),
                }])
                raw_df = pd.DataFrame({
                    "repeat_index": np.arange(n_repeats),
                    "pooled_accuracy": acc_per_repeat,
                    "pooled_balanced_accuracy": bacc_per_repeat,
                    "pooled_f1": f1_per_repeat,
                    "pooled_mcc": mcc_per_repeat,
                })
                acc_headline, f1_headline = float(np.mean(acc_per_repeat)), float(np.mean(f1_per_repeat))
                n_splits_reported = n_repeats
                extra_metrics_note = (
                    f"\n📊 Balanced Accuracy: {np.mean(bacc_per_repeat):.4f} ± {np.std(bacc_per_repeat):.4f}"
                    f"\n📊 MCC: {np.mean(mcc_per_repeat):.4f} ± {np.std(mcc_per_repeat):.4f}"
                )

            summary_path = os.path.join(output_dir, "Resampling_Validation_Results.csv")
            summary_df.to_csv(summary_path, index=False)
            raw_path = os.path.join(output_dir, "Resampling_Validation_Raw_Scores.csv")
            raw_df.to_csv(raw_path, index=False)

            if method == "loocv":
                acc_label, f1_label = "Pooled LOO Accuracy", "Pooled LOO F1"
                splits_note = f"n_train: {n_train}, total splits: {n_splits_reported}"
            else:
                acc_label, f1_label = "Mean Accuracy (pooled per repeat)", "Mean F1 (pooled per repeat)"
                splits_note = f"n_train: {n_train}, n_repeats: {n_splits_reported}, cv_splits_per_repeat: {cv_splits}"
            log_message = (
                "========================================\n"
                "🔹 8.3 Resampling Stability Assessment Complete! 🔹\n"
                "========================================\n"
                f"🖥️ Requested cores: {num_cores}, Effective cores: {n_jobs} (CPU count: {cpu_count})\n"
                f"📊 Total fits: {total_fits}\n"
                f"{cores_warning}"
                f"📌 Method: {method}\n"
                f"📌 {splits_note}\n"
                f"📊 {acc_label}: {acc_headline:.4f}" + ("" if method == "loocv" else f" ± {np.std(acc_per_repeat):.4f}") + "\n"
                f"📊 {f1_label}: {f1_headline:.4f}" + ("" if method == "loocv" else f" ± {np.std(f1_per_repeat):.4f}") + f"{extra_metrics_note}\n"
                "----------------------------------------\n"
                "ℹ️ Scope: descriptor set (training_data_path) is fixed -- upstream feature selection/\n"
                "   combination (05/06) is NOT re-run per resampling split. This measures\n"
                "   robustness of the fitted model+descriptors to how the data is split, not\n"
                "   the stability of the descriptor-selection workflow itself.\n"
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
                f"💾 Results: {os.path.basename(summary_path)}\n"
                f"💾 Raw Scores: {os.path.basename(raw_path)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(summary_path),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}


NODE_CLASS_MAPPINGS = {
    "ResamplingValidation_Classification": ResamplingValidation_Classification,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResamplingValidation_Classification": "8.3 Resampling Stability Assessment",
}
