import os
import multiprocessing
import traceback
import numpy as np
import pandas as pd
import joblib
import folder_paths
from sklearn.model_selection import KFold, LeaveOneOut, ShuffleSplit, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import clone


class ResamplingValidation_Regression:
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
                "target_column": ("STRING", {"default": "value"}),
                "method": (["repeated_kfold", "loocv", "repeated_leave_n_out"], {"default": "repeated_kfold"}),
            },
            "optional": {
                "n_repeats": ("INT", {"default": 10, "min": 1, "max": 1000, "tooltip": "Ignored for loocv; number of random leave-N-out repeats otherwise."}),
                "cv_splits": ("INT", {"default": 5, "min": 2, "max": 10, "tooltip": "Only used for repeated_kfold."}),
                "leave_n": ("INT", {"default": 2, "min": 1, "max": 1000, "tooltip": "Only used for repeated_leave_n_out. leave_n=1 is allowed but prefer 'loocv' instead -- it is the exhaustive, deterministic version of leave-1-out (every compound held out exactly once) rather than a random, possibly-incomplete sample of it."}),
                "random_state": ("INT", {"default": 42}),
                "num_cores": ("INT", {"default": 1, "min": -1, "max": multiprocessing.cpu_count(),
                                       "tooltip": "1 (default) runs in-process with no worker-pool startup cost -- "
                                                  "recommended for typical QSAR-sized datasets/repeat counts, "
                                                  "where loky's per-worker boot/unpickle overhead outweighs any "
                                                  "speedup. Only raise this for genuinely large workloads. "
                                                  "Ignored for repeated_leave_n_out (not parallelized)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESAMPLING_REPORT",)
    FUNCTION = "run"
    CATEGORY = "QSAR/2. REGRESSION/8. Model Evaluation"
    OUTPUT_NODE = True

    def run(self, training_data_path, trained_model_path, target_column, method,
            n_repeats=10, cv_splits=5, leave_n=2, random_state=42, num_cores=1):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Regression", "08_Model_Evaluation", "Resampling_Stability")
            os.makedirs(output_dir, exist_ok=True)

            data = pd.read_csv(training_data_path)
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found.")
            metadata_cols = [c for c in ("Name", "SMILES", target_column) if c in data.columns]
            X = data.drop(columns=metadata_cols)
            y = data[target_column].values
            n_train = len(y)
            names = data["Name"].astype(str).to_numpy() if "Name" in data.columns else None

            base_model = joblib.load(trained_model_path)
            cpu_count = multiprocessing.cpu_count()
            n_jobs = cpu_count if num_cores == -1 else max(1, min(num_cores, cpu_count))
            if method == "loocv":
                total_fits = n_train
            elif method == "repeated_kfold":
                total_fits = n_repeats * cv_splits
            else:
                total_fits = n_repeats  # repeated_leave_n_out: sequential, not parallelized
            cores_warning = ""
            if num_cores == -1 and cpu_count > 8 and method != "repeated_leave_n_out":
                cores_warning = (
                    f"⚠️ num_cores=-1 requests all {cpu_count} CPU cores -- for typical "
                    "QSAR-sized datasets/repeat counts, per-worker startup overhead can "
                    "make this slower than a small explicit core count (e.g. 2-4). Consider "
                    "setting num_cores explicitly if this run feels slow.\n"
                )

            if method == "loocv":
                # R2/RMSE are not meaningfully defined on a single-point test
                # fold (R2's denominator needs a variance, which a 1-sample
                # set doesn't have) -- averaging naive per-fold "scores"
                # here would silently compute something closer to MAE than
                # true RMSE, and an undefined/degenerate R2. Standard QSAR
                # practice ("LOO Q2") instead pools every fold's single
                # out-of-fold prediction and computes ONE aggregate metric
                # over the whole training set at the end.
                y_pred_loo = cross_val_predict(base_model, X, y, cv=LeaveOneOut(), n_jobs=n_jobs)
                sse = float(np.sum((y - y_pred_loo) ** 2))
                sst = float(np.sum((y - y.mean()) ** 2))
                q2_loo = 1 - sse / sst if sst > 0 else float("nan")
                rmse_loo = float(np.sqrt(sse / n_train))

                summary_df = pd.DataFrame([{
                    "test_scope": "fixed_descriptor_set",
                    "method": method,
                    "n_train": n_train,
                    "n_splits_total": n_train,
                    "mean_r2": q2_loo, "std_r2": None,
                    "mean_rmse": rmse_loo, "std_rmse": None,
                }])
                raw_df = pd.DataFrame({
                    "split_index": np.arange(n_train),
                    "y_true": y,
                    "y_pred_loo": y_pred_loo,
                    "squared_error": (y - y_pred_loo) ** 2,
                })
                r2_headline, rmse_headline = q2_loo, rmse_loo
                n_splits_reported = n_train
            elif method == "repeated_kfold":
                # Repeat-level pooled Q2/RMSE: for each repeat, pool that
                # repeat's out-of-fold predictions across all cv_splits folds
                # into ONE prediction per training compound, then compute
                # R2/RMSE ONCE on that pooled vector -- one score per repeat
                # (n_repeats values total) rather than scoring each fold
                # separately. Mirrors the Classification counterpart's
                # repeat-level pooling and avoids the same small-fold metric
                # instability the loocv branch above already guards against.
                r2_per_repeat = np.empty(n_repeats, dtype=float)
                rmse_per_repeat = np.empty(n_repeats, dtype=float)
                for i in range(n_repeats):
                    cv_i = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state + i)
                    y_pred_oof = cross_val_predict(base_model, X, y, cv=cv_i, n_jobs=n_jobs)
                    sse_i = float(np.sum((y - y_pred_oof) ** 2))
                    sst_i = float(np.sum((y - y.mean()) ** 2))
                    r2_per_repeat[i] = 1 - sse_i / sst_i if sst_i > 0 else float("nan")
                    rmse_per_repeat[i] = float(np.sqrt(mean_squared_error(y, y_pred_oof)))

                summary_df = pd.DataFrame([{
                    "test_scope": "fixed_descriptor_set",
                    "method": method,
                    "n_train": n_train,
                    "n_repeats": n_repeats,
                    "cv_splits_per_repeat": cv_splits,
                    "mean_r2": float(np.nanmean(r2_per_repeat)), "std_r2": float(np.nanstd(r2_per_repeat)),
                    "mean_rmse": float(np.mean(rmse_per_repeat)), "std_rmse": float(np.std(rmse_per_repeat)),
                }])
                raw_df = pd.DataFrame({
                    "repeat_index": np.arange(n_repeats),
                    "pooled_r2": r2_per_repeat,
                    "pooled_rmse": rmse_per_repeat,
                })
                r2_headline, rmse_headline = float(np.nanmean(r2_per_repeat)), float(np.mean(rmse_per_repeat))
                n_splits_reported = n_repeats
            else:
                if method != "repeated_leave_n_out":
                    raise ValueError(f"Unknown method: {method}")
                if leave_n >= n_train:
                    raise ValueError(f"leave_n ({leave_n}) must be smaller than n_train ({n_train}).")
                # A single leave-N-out repeat's own R2 is not a safe unit to
                # average (SS_tot for a 1-2 point test set is either exactly
                # 0 or a razor-thin denominator) -- mirroring loocv above,
                # pool every repeat's held-out (actual, predicted) pairs into
                # one combined set (a compound drawn in multiple repeats
                # contributes multiple times, weighted by draw frequency,
                # standard Monte-Carlo-CV aggregate-reporting practice) and
                # compute ONE Q2/RMSE/MAE over that pool.
                #
                # SST must be summed over the SAME pooled index set as SSE,
                # not over the original n_train-length y -- R2 = 1 -
                # sum(err^2)/sum(dev^2) is only meaningful when both sums
                # range over identical observations. Using the training
                # mean as the centering *value* is still correct/desired
                # (keeps this comparable to loocv/repeated_kfold's pooled
                # Q2, which also center on the training mean), but the SUM
                # itself must run over all n_repeats*leave_n pooled terms,
                # not just n_train of them. Getting this wrong makes SSE
                # grow with n_repeats while SST stays fixed, so Q2 degrades
                # purely as a function of n_repeats regardless of model
                # quality -- confirmed empirically (QDB261, leave_n=2):
                # n_repeats=20/40/100 gave Q2=0.304/-0.949/-2.898 with the
                # bug, and stable ~0.26-0.34 once SST is pooled correctly.
                cv = ShuffleSplit(n_splits=n_repeats, test_size=leave_n, random_state=random_state)
                train_mean = float(y.mean())
                repeat_idx_col, orig_idx_col, name_col = [], [], []
                y_true_pooled, y_pred_pooled, covered_counts = [], [], np.zeros(n_train, dtype=int)
                for rep_i, (train_idx, test_idx) in enumerate(cv.split(X)):
                    model_i = clone(base_model)
                    model_i.fit(X.iloc[train_idx], y[train_idx])
                    y_pred_i = np.asarray(model_i.predict(X.iloc[test_idx]))
                    y_true_pooled.extend(y[test_idx].tolist())
                    y_pred_pooled.extend(y_pred_i.tolist())
                    repeat_idx_col.extend([rep_i] * len(test_idx))
                    orig_idx_col.extend(test_idx.tolist())
                    if names is not None:
                        name_col.extend(names[test_idx].tolist())
                    covered_counts[test_idx] += 1
                y_true_pooled = np.asarray(y_true_pooled, dtype=float)
                y_pred_pooled = np.asarray(y_pred_pooled, dtype=float)

                sse = float(np.sum((y_true_pooled - y_pred_pooled) ** 2))
                sst_pooled = float(np.sum((y_true_pooled - train_mean) ** 2))
                q2_pooled = 1 - sse / sst_pooled if sst_pooled > 0 else float("nan")
                rmse_pooled = float(np.sqrt(mean_squared_error(y_true_pooled, y_pred_pooled)))
                mae_pooled = float(mean_absolute_error(y_true_pooled, y_pred_pooled))
                n_unique_covered = int(np.sum(covered_counts > 0))
                coverage_fraction = n_unique_covered / n_train
                coverage_warning = ""
                if coverage_fraction < 0.5:
                    coverage_warning = (
                        f"\n⚠️ Only {coverage_fraction:.0%} of training compounds were ever held out "
                        f"across {n_repeats} repeats -- increase n_repeats for a more complete picture, "
                        "or this pooled Q2 mostly reflects whichever compounds happened to be sampled."
                    )
                leave1_note = (
                    "\nℹ️ leave_n=1: consider 'loocv' instead -- it holds out every compound exactly "
                    "once (exhaustive, deterministic) rather than a random, possibly-incomplete sample."
                ) if leave_n == 1 else ""

                summary_df = pd.DataFrame([{
                    "test_scope": "fixed_descriptor_set",
                    "method": method,
                    "n_train": n_train,
                    "n_repeats": n_repeats,
                    "leave_n": leave_n,
                    "n_pooled_predictions": len(y_true_pooled),
                    "n_unique_compounds_covered": n_unique_covered,
                    "coverage_fraction": coverage_fraction,
                    "pooled_q2": q2_pooled,
                    "pooled_rmse": rmse_pooled,
                    "pooled_mae": mae_pooled,
                }])
                raw_df = pd.DataFrame({
                    "repeat_index": repeat_idx_col,
                    "original_row_index": orig_idx_col,
                    **({"Name": name_col} if names is not None else {}),
                    "y_true": y_true_pooled,
                    "y_pred": y_pred_pooled,
                    "residual": y_true_pooled - y_pred_pooled,
                })
                # Per-compound held-out count, for spotting sampling
                # imbalance (some compounds tested many times, others never).
                coverage_df = pd.DataFrame({
                    "original_row_index": np.arange(n_train),
                    **({"Name": names} if names is not None else {}),
                    "times_held_out": covered_counts,
                })
                coverage_path = os.path.join(output_dir, "Resampling_LeaveNOut_Coverage.csv")
                coverage_df.to_csv(coverage_path, index=False)
                r2_headline, rmse_headline = q2_pooled, rmse_pooled
                n_splits_reported = n_repeats

            summary_path = os.path.join(output_dir, "Resampling_Validation_Results.csv")
            summary_df.to_csv(summary_path, index=False)
            raw_path = os.path.join(output_dir, "Resampling_Validation_Raw_Scores.csv")
            raw_df.to_csv(raw_path, index=False)

            if method == "loocv":
                r2_label, rmse_label = "LOO Q² (pooled out-of-fold; 1-PRESS/SST_input)", "Pooled LOO RMSE"
                splits_note = f"n_train: {n_train}, total splits: {n_splits_reported}"
                r2_std_str = rmse_std_str = ""
            elif method == "repeated_kfold":
                r2_label, rmse_label = "Repeated k-fold Q² (pooled out-of-fold per repeat)", "Mean RMSE (pooled per repeat)"
                splits_note = f"n_train: {n_train}, n_repeats: {n_splits_reported}, cv_splits_per_repeat: {cv_splits}"
                r2_std_str, rmse_std_str = f" ± {np.nanstd(r2_per_repeat):.4f}", f" ± {np.std(rmse_per_repeat):.4f}"
            else:
                r2_label, rmse_label = "Leave-N-out Q² (pooled held-out predictions; centered on full input-target mean)", "Pooled RMSE (leave-N-out)"
                splits_note = (
                    f"n_train: {n_train}, n_repeats: {n_splits_reported}, leave_n: {leave_n}, "
                    f"pooled predictions: {len(y_true_pooled)}, "
                    f"unique compounds covered: {n_unique_covered}/{n_train} ({coverage_fraction:.0%})"
                )
                r2_std_str = rmse_std_str = ""
            mae_line = f"\n📊 Pooled MAE (leave-N-out): {mae_pooled:.4f}" if method == "repeated_leave_n_out" else ""
            extra_notes = (coverage_warning + leave1_note) if method == "repeated_leave_n_out" else ""
            coverage_file_line = f"\n💾 Per-compound coverage: {os.path.basename(coverage_path)}" if method == "repeated_leave_n_out" else ""
            log_message = (
                "========================================\n"
                "🔹 8.3 Resampling Stability Assessment Complete! 🔹\n"
                "========================================\n"
                f"🖥️ Requested cores: {num_cores}, Effective cores: {n_jobs if method != 'repeated_leave_n_out' else 1} (CPU count: {cpu_count})\n"
                f"📊 Total fits: {total_fits}\n"
                f"{cores_warning}"
                f"📌 Method: {method}\n"
                f"📌 {splits_note}\n"
                f"📊 {r2_label}: {r2_headline:.4f}{r2_std_str}\n"
                f"📊 {rmse_label}: {rmse_headline:.4f}{rmse_std_str}{mae_line}\n"
                "----------------------------------------\n"
                "ℹ️ Scope: descriptor set (training_data_path) is fixed -- upstream feature selection/\n"
                "   combination (05/06) is NOT re-run per resampling split. This measures\n"
                "   robustness of the fitted model+descriptors to how the data is split, not\n"
                "   the stability of the descriptor-selection workflow itself.\n"
                f"{extra_notes}\n"
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
                f"💾 Results: {os.path.basename(summary_path)}\n"
                f"💾 Raw Scores: {os.path.basename(raw_path)}{coverage_file_line}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(summary_path),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}


NODE_CLASS_MAPPINGS = {
    "ResamplingValidation_Regression": ResamplingValidation_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResamplingValidation_Regression": "8.3 Resampling Stability Assessment",
}
