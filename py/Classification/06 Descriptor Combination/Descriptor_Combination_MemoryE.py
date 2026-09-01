import os
import pandas as pd
import numpy as np
import itertools
import multiprocessing
import traceback
import folder_paths
import joblib
from math import comb

from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

try:
    from tqdm import tqdm
    TQDM_INSTALLED = True
except ImportError:
    TQDM_INSTALLED = False


def evaluate_classification_direct(X, y, feature_indices):
    try:
        X_subset = X[:, list(feature_indices)]
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver="liblinear"),
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(
            pipeline,
            X_subset,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        ).mean()
        return feature_indices, score
    except Exception:
        return feature_indices, float("-inf")


class Feature_Combination_Search:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "optimized_descriptor_path": ("STRING", {"default": "", "tooltip": "From 5. This node is optional -- 5 can connect straight to 7."}),
                "max_features": ("INT", {"default": 3, "min": 1, "step": 1}),
                "num_cores": ("INT", {"default": 1, "max": multiprocessing.cpu_count(), "min": -1, "step": 1,
                                       "tooltip": "1 (default) runs in-process with no worker-pool startup cost -- "
                                                  "recommended for typical QSAR-sized searches, where loky's "
                                                  "per-worker boot/unpickle overhead outweighs any speedup. Only "
                                                  "raise this for genuinely large combination counts."}),
                "top_n": ("INT", {"default": 3, "min": 1, "step": 1}),
                "chunk_size": ("INT", {"default": 2000, "min": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("OPTIMAL_DESCRIPTOR_DATA", "BASELINE_MODEL", "SELECTED_DESCRIPTOR_LIST")
    FUNCTION = "descriptor_combination_classification_MemoryE"
    CATEGORY = "QSAR/1. CLASSIFICATION"
    OUTPUT_NODE = True

    def descriptor_combination_classification_MemoryE(self, optimized_descriptor_path, max_features, num_cores, top_n, chunk_size):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Classification", "06_Descriptor_Combination")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(optimized_descriptor_path)

            if "Label" not in df.columns:
                raise ValueError("Target column 'Label' not found in the dataset.")

            metadata_cols = [c for c in ("Name", "SMILES", "Label") if c in df.columns]
            X = df.drop(columns=metadata_cols).to_numpy()
            y = df["Label"].to_numpy()
            feature_names = df.drop(columns=metadata_cols).columns.tolist()

            cpu_count = multiprocessing.cpu_count()
            total_combs = sum(
                comb(X.shape[1], nf)
                for nf in range(2, min(max_features + 1, len(feature_names) + 1))
            )
            # A handful of combinations finishes in well under a second running
            # in-process; a loky worker pool costs multiple seconds per worker
            # just to boot and unpickle the task, so parallelizing a small
            # search is a net slowdown, not a speedup -- force serial
            # execution regardless of the user's num_cores setting.
            force_serial = total_combs < 16
            if force_serial:
                cores = 1
            elif num_cores == -1:
                cores = cpu_count
            else:
                cores = max(1, min(num_cores, cpu_count))

            cores_warning = ""
            if num_cores == -1 and cpu_count > 8 and not force_serial:
                cores_warning = (
                    f"⚠️ num_cores=-1 requests all {cpu_count} CPU cores -- for typical "
                    "QSAR-sized searches, per-worker startup overhead can make this "
                    "slower than a small explicit core count (e.g. 2-4). Consider "
                    "setting num_cores explicitly if this run feels slow.\n"
                )

            top_results = []
            best_per_feature_count = {}

            for n_features in range(2, min(max_features + 1, len(feature_names) + 1)):
                num_combs = comb(X.shape[1], n_features)
                combinations_iter = itertools.combinations(range(X.shape[1]), n_features)

                raw_results = Parallel(
                    n_jobs=cores,
                    backend="loky",
                    return_as="generator",
                    batch_size=chunk_size,
                    pre_dispatch="2*n_jobs",
                )(
                    delayed(evaluate_classification_direct)(X, y, indices)
                    for indices in combinations_iter
                )

                if TQDM_INSTALLED:
                    raw_results = tqdm(raw_results, total=num_combs, desc=f"Features: {n_features}")

                for feature_indices, acc in raw_results:
                    result = {
                        "Num_Features": len(feature_indices),
                        "Feature_Indices": feature_indices,
                        "Best_Features": [feature_names[i] for i in feature_indices],
                        "Accuracy": acc,
                    }

                    if n_features not in best_per_feature_count or acc > best_per_feature_count[n_features]["Accuracy"]:
                        best_per_feature_count[n_features] = result

                    if len(top_results) < top_n:
                        top_results.append(result)
                        top_results.sort(key=lambda x: x["Accuracy"], reverse=True)
                    elif acc > top_results[-1]["Accuracy"]:
                        top_results[-1] = result
                        top_results.sort(key=lambda x: x["Accuracy"], reverse=True)

            if not top_results:
                return {"ui": {"text": "❌ No combinations were evaluated."}, "result": ("", "", "")}

            best_per_size_df = pd.DataFrame(best_per_feature_count.values())
            best_per_size_path = os.path.join(output_dir, "Best_combination_per_size_MemoryE.csv")
            best_per_size_df.to_csv(best_per_size_path, index=False)

            best_overall_result = top_results[0]
            output_file = ""

            for i, result in enumerate(top_results, start=1):
                df_selected = df[result["Best_Features"] + ["Label"]]
                output_path = os.path.join(output_dir, f"Optimal_Feature_Set_rank{i}_acc{result['Accuracy']:.4f}_MemoryE.csv")
                df_selected.to_csv(output_path, index=False)
                if i == 1:
                    output_file = output_path

            selected_features = best_overall_result["Best_Features"]
            final_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, solver="liblinear"),
            )
            final_model.fit(df[selected_features], y)

            model_path = os.path.join(output_dir, "Combination_Classifier_LogisticRegression.pkl")
            joblib.dump(final_model, model_path)

            descriptors_path = os.path.join(output_dir, "Combination_Selected_Descriptors.txt")
            with open(descriptors_path, "w") as f:
                f.write("\n".join(selected_features))

            log_message = (
                "========================================\n"
                "🔹 Feature Combination Search (Memory Efficient) Completed! 🔹\n"
                "========================================\n"
                f"🖥️ Requested cores: {num_cores}, Effective cores: {cores} (CPU count: {cpu_count})\n"
                f"📊 Total combinations evaluated: {total_combs}\n"
                f"{cores_warning}"
                f"🏆 Best CV Accuracy: {best_overall_result['Accuracy']:.4f}\n"
                f"✨ Optimal Features ({best_overall_result['Num_Features']}): {best_overall_result['Best_Features']}\n"
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
                f"💾 Top Ranked Set: {os.path.basename(output_file)}\n"
                f"💾 Model: {os.path.basename(model_path)}\n"
                f"💾 Selected Descriptors: {os.path.basename(descriptors_path)}\n"
                "========================================"
            )
            return {
                "ui": {"text": log_message},
                "result": (str(output_file), str(model_path), str(descriptors_path))
            }

        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("", "", "")}


NODE_CLASS_MAPPINGS = {
    "Feature_Combination_Search_Classification": Feature_Combination_Search,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Feature_Combination_Search_Classification": "6. Descriptor Combination",
}
