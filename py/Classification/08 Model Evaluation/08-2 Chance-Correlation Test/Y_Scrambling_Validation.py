import os
import multiprocessing
import traceback
import numpy as np
import pandas as pd
import joblib
import folder_paths
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, make_scorer

# Mirrors the `scoring` dict in '7. Hyperparameter Tuning & Model Training' --
# kept as a separate copy (not a shared import) because the two nodes live in
# sibling directories with spaces in their names, not a shared package.
SELECTION_METRIC_SCORERS = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'f1': make_scorer(f1_score, zero_division=0),
    'roc_auc': 'roc_auc',
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'mcc': make_scorer(matthews_corrcoef),
}
METRIC_DISPLAY_LABELS = {
    'accuracy': 'Accuracy', 'balanced_accuracy': 'Balanced Accuracy', 'f1': 'F1 Score',
    'roc_auc': 'ROC AUC', 'precision': 'Precision', 'recall': 'Recall', 'mcc': 'MCC',
}


class YScramblingValidation_Classification:
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
                "target_column": ("STRING", {"default": "Label"}),
            },
            "optional": {
                "n_permutations": ("INT", {"default": 100, "min": 10, "max": 500}),
                "cv_splits": ("INT", {"default": 5, "min": 3, "max": 10}),
                "random_state": ("INT", {"default": 42}),
                "selection_metric": (
                    list(SELECTION_METRIC_SCORERS.keys()),
                    {"default": "accuracy",
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
    CATEGORY = "QSAR/1. CLASSIFICATION/8. Model Evaluation"
    OUTPUT_NODE = True

    def run(self, training_data_path, trained_model_path, target_column,
            n_permutations=100, cv_splits=5, random_state=42, selection_metric="accuracy", num_cores=1):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Classification", "08_Model_Evaluation", "Chance_Correlation_Test")
            os.makedirs(output_dir, exist_ok=True)

            if selection_metric not in SELECTION_METRIC_SCORERS:
                raise ValueError(
                    f"selection_metric='{selection_metric}' is not one of the supported metrics "
                    f"({sorted(SELECTION_METRIC_SCORERS)})."
                )
            scorer = SELECTION_METRIC_SCORERS[selection_metric]
            metric_label = METRIC_DISPLAY_LABELS[selection_metric]

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
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
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
            # (standard sklearn behavior) and fits only the clone -- the
            # loaded model's own fitted weights are never reused, only its
            # pipeline structure/hyperparameters, matching the "fixed
            # descriptor set + hyperparameters, refit each iteration"
            # strategy agreed for this node.
            real_scores = cross_val_score(base_model, X, y, cv=cv, scoring=scorer, n_jobs=n_jobs)
            real_score = float(real_scores.mean())

            rng = np.random.RandomState(random_state)
            permuted_scores = np.empty(n_permutations, dtype=float)
            for i in range(n_permutations):
                y_perm = rng.permutation(y)
                scores = cross_val_score(base_model, X, y_perm, cv=cv, scoring=scorer, n_jobs=n_jobs)
                permuted_scores[i] = float(scores.mean())

            # Permutation-test p-value: fraction of permuted (null) CV
            # scores that are at least as good as the real score, +1/+1
            # correction so p is never reported as exactly 0.
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
                f"📌 Real CV {metric_label}: {real_score:.4f}\n"
                f"📊 Permuted CV {metric_label}: {permuted_scores.mean():.4f} ± {permuted_scores.std():.4f} "
                f"(n={n_permutations}, seed={random_state})\n"
                f"📈 p-value: {p_value:.4f}{flag}\n"
                "----------------------------------------\n"
                "ℹ️ Scope: the descriptor set (training_data_path) is held fixed across all "
                "permutations -- only the target labels are permuted and the final "
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
    "YScramblingValidation_Classification": YScramblingValidation_Classification,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YScramblingValidation_Classification": "8.2 Chance-Correlation Test (Y-Scrambling)",
}
