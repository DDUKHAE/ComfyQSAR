import os
import multiprocessing
import traceback
import numpy as np
import pandas as pd
import joblib
import folder_paths
from sklearn.model_selection import KFold, cross_val_score

# Mirrors the `scoring` dict in '7. Hyperparameter Tuning & Model Training' --
# all four are already valid sklearn scoring strings, so no make_scorer
# wrapping is needed (unlike the Classification counterpart).
SELECTION_METRICS = ["r2", "neg_mean_squared_error", "neg_root_mean_squared_error", "neg_mean_absolute_error"]
# neg_* scorers are negated by sklearn convention (higher = better); flipping
# the sign back for display matches how '7. Hyperparameter Tuning & Model
# Training' reports CV MSE/RMSE/MAE as positive error magnitudes.
METRIC_DISPLAY_LABELS = {
    "r2": "R²", "neg_mean_squared_error": "MSE",
    "neg_root_mean_squared_error": "RMSE", "neg_mean_absolute_error": "MAE",
}


class YScramblingValidation_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "training_data_path": ("STRING", {
                    "tooltip": "Exact training dataset used by Node 7, after preprocessing and any optional descriptor selection or combination.",
                }),
                "trained_model_path": ("STRING", {
                    "tooltip": "Only the hyperparameters are reused; the model is refit on every split.",
                }),
                "target_column": ("STRING", {"default": "value"}),
            },
            "optional": {
                "n_permutations": ("INT", {"default": 100, "min": 10, "max": 500}),
                "cv_splits": ("INT", {"default": 5, "min": 3, "max": 10}),
                "random_state": ("INT", {"default": 42}),
                "selection_metric": (
                    SELECTION_METRICS,
                    {"default": "r2",
                     "tooltip": "Match 7's selection_metric so the test scores the same quantity."}),
                "num_cores": ("INT", {"default": 1, "min": -1, "max": multiprocessing.cpu_count(),
                                       "tooltip": "1 (default) runs in-process with no worker-pool startup cost -- "
                                                  "recommended for typical QSAR-sized datasets/permutation counts, "
                                                  "where loky's per-worker boot/unpickle overhead outweighs any "
                                                  "speedup. Only raise this for genuinely large workloads."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Y_SCRAMBLING_REPORT",)
    FUNCTION = "run"
    CATEGORY = "QSAR/2. REGRESSION/8. Model Evaluation"
    OUTPUT_NODE = True

    def run(self, training_data_path, trained_model_path, target_column,
            n_permutations=100, cv_splits=5, random_state=42, selection_metric="r2", num_cores=1):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Regression", "08_Model_Evaluation", "Chance_Correlation_Test")
            os.makedirs(output_dir, exist_ok=True)

            if selection_metric not in SELECTION_METRICS:
                raise ValueError(
                    f"selection_metric='{selection_metric}' is not one of the supported metrics "
                    f"({sorted(SELECTION_METRICS)})."
                )
            metric_label = METRIC_DISPLAY_LABELS[selection_metric]
            sign = -1.0 if selection_metric.startswith("neg_") else 1.0

            data = pd.read_csv(training_data_path)
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found.")
            metadata_cols = [c for c in ("Name", "SMILES", target_column) if c in data.columns]
            X = data.drop(columns=metadata_cols)
            y = data[target_column].values
            n_train = len(y)

            base_model = joblib.load(trained_model_path)
            cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
            cpu_count = multiprocessing.cpu_count()
            n_jobs = cpu_count if num_cores == -1 else max(1, min(num_cores, cpu_count))
            total_fits = (n_permutations + 1) * cv_splits
            cores_warning = ""
            if num_cores == -1 and cpu_count > 8:
                cores_warning = (
                    f"⚠️ num_cores=-1 requests all {cpu_count} CPU cores -- for typical "
                    "QSAR-sized datasets/permutation counts, per-worker startup overhead can "
                    "make this slower than a small explicit core count (e.g. 2-4). Consider "
                    "setting num_cores explicitly if this run feels slow.\n"
                )

            # cross_val_score clones base_model internally for every fold
            # (standard sklearn behavior) and fits only the clone -- see the
            # Classification counterpart for the same reasoning.
            real_scores = cross_val_score(base_model, X, y, cv=cv, scoring=selection_metric, n_jobs=n_jobs)
            real_score = float(real_scores.mean())

            rng = np.random.RandomState(random_state)
            permuted_scores = np.empty(n_permutations, dtype=float)
            for i in range(n_permutations):
                y_perm = rng.permutation(y)
                scores = cross_val_score(base_model, X, y_perm, cv=cv, scoring=selection_metric, n_jobs=n_jobs)
                permuted_scores[i] = float(scores.mean())

            n_as_good_or_better = int(np.sum(permuted_scores >= real_score))
            p_value = (n_as_good_or_better + 1) / (n_permutations + 1)

            results_df = pd.DataFrame([{
                "test_scope": "fixed_descriptor_set",
                "selection_metric": selection_metric,
                "n_train": n_train,
                "real_cv_score": real_score,
                "mean_permuted_score": float(permuted_scores.mean()),
                "std_permuted_score": float(permuted_scores.std()),
                "n_permutations": n_permutations,
                "cv_splits": cv_splits,
                "random_state": random_state,
                "p_value": p_value,
            }])
            results_path = os.path.join(output_dir, "Y_Scrambling_Results.csv")
            results_df.to_csv(results_path, index=False)

            permuted_path = os.path.join(output_dir, "Y_Scrambling_Permuted_Scores.csv")
            pd.DataFrame({
                "permutation_index": np.arange(n_permutations),
                "cv_score": permuted_scores,
            }).to_csv(permuted_path, index=False)

            flag = " (p<0.05 -- real performance unlikely due to chance)" if p_value < 0.05 else ""
            log_message = (
                "========================================\n"
                "🔹 8.2 Chance-Correlation Test (Fixed-Descriptor Y-Scrambling) Complete! 🔹\n"
                "========================================\n"
                f"🖥️ Requested cores: {num_cores}, Effective cores: {n_jobs} (CPU count: {cpu_count})\n"
                f"📊 Total fits: {total_fits} ({n_permutations} permutation(s) + 1 real, {cv_splits}-fold each)\n"
                f"{cores_warning}"
                f"📌 n_train: {n_train}\n"
                f"📌 Scoring metric: {metric_label} (selection_metric='{selection_metric}')\n"
                f"📌 Real Mean CV {metric_label}: {sign * real_score:.4f}\n"
                f"📊 Permuted Mean CV {metric_label}: {sign * permuted_scores.mean():.4f} ± {permuted_scores.std():.4f} "
                f"(n={n_permutations}, seed={random_state})\n"
                f"📈 p-value: {p_value:.4f}{flag}\n"
                "----------------------------------------\n"
                "ℹ️ Scope: the descriptor set (training_data_path) is held fixed across all "
                "permutations -- only the target values are permuted and the final "
                "algorithm/hyperparameters are refit. This tests whether the chosen "
                "descriptor-set + algorithm combination could reach this performance "
                "by chance; it does not re-run descriptor selection or Descriptor "
                "Combination per permutation, so it is not a test of the full "
                "feature-selection workflow. Use alongside 08 hold-out evaluation, "
                "8c resampling, and 8d applicability domain -- not as the sole basis "
                "for model validity.\n"
                "----------------------------------------\n"
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
                f"💾 Results: {os.path.basename(results_path)}\n"
                f"💾 Permutation Distribution: {os.path.basename(permuted_path)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(results_path),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}


NODE_CLASS_MAPPINGS = {
    "YScramblingValidation_Regression": YScramblingValidation_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YScramblingValidation_Regression": "8.2 Chance-Correlation Test (Y-Scrambling)",
}
