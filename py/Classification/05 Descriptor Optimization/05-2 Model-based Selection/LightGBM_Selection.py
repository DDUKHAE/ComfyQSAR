import os
import numpy as np
import pandas as pd
import folder_paths
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier

def train_lightgbm_classification(args):
    X, y, feature_names, i, n_estimators, max_depth, learning_rate, min_data_in_leaf, min_split_gain = args
    model = LGBMClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_data_in_leaf=min_data_in_leaf,
        min_split_gain=min_split_gain, learning_rate=learning_rate, random_state=i, n_jobs=1, verbosity=-1
    )
    model.fit(X, y)
    return np.array(model.feature_importances_)

class lgb_CL:
    @staticmethod
    def _train_lightgbm(args):
        return train_lightgbm_classification(args)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {}),
                "target_column": ("STRING", {"default": "Label"}),
                "n_estimators": ("INT", {"default": 100, "min": 10, "max": 1000}),
                "max_depth": ("INT", {"default": -1, "min": -1, "max": 1000}),
                "learning_rate": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01}),
                "threshold_mode": ("BOOLEAN", {"default": False, "forceInput": False, "label_on": "importance cutoff (%)", "label_off": "percentile"}),
                "threshold": ("INT", {"default": 90, "min": 1, "max": 100, "step": 1}),
                "n_iterations": ("INT", {"default": 100, "min": 10, "max": 1000}),
                "min_data_in_leaf": ("INT", {"default": 1, "min": 1, "max": 100}),
                "min_split_gain": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "num_cores": ("INT", {"default": -1, "min": -1, "max": cpu_count(), "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("SELECTED_DESCRIPTORS",)
    FUNCTION = "lightgbm_feature_selection"
    CATEGORY = "QSAR/CLASSIFICATION/5. Descriptor Optimization/5.2 Model-based Selection"
    OUTPUT_NODE = True

    def lightgbm_feature_selection(self, input_file, target_column, n_estimators, max_depth,
                                    learning_rate, threshold_mode, threshold, n_iterations,
                                    min_data_in_leaf, min_split_gain, num_cores):
        output_dir = os.path.join(folder_paths.get_output_directory(), "Classification", "05_Descriptor_Optimization", "Model_Based")
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(input_file)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        feature_names = list(X.columns)
        initial_feature_count = len(feature_names)
        available_cores = -1 if num_cores == -1 else max(1, min(num_cores, cpu_count()))
        cores_label = "-1 (all available cores)" if available_cores == -1 else str(available_cores)
        print(f"🖥️ Using {cores_label} for parallel LightGBM training...")
        args_list = [(X, y, feature_names, i, n_estimators, max_depth, learning_rate, min_data_in_leaf, min_split_gain)
                     for i in range(n_iterations)]
        results = Parallel(
            n_jobs=available_cores,
            backend="loky",
            pre_dispatch="2*n_jobs",
        )(
            delayed(train_lightgbm_classification)(args)
            for args in args_list
        )
        feature_importance_matrix = np.stack(results)
        feature_importances = np.mean(feature_importance_matrix, axis=0)
        total_importance = feature_importances.sum()
        if total_importance > 0:
            feature_importances = feature_importances / total_importance
        if threshold_mode:
            importance_cutoff = threshold / 100.0
            log_threshold_type = f"Importance cutoff - {threshold}%"
        else:
            importance_cutoff = np.percentile(feature_importances, threshold)
            log_threshold_type = f"Percentile - {threshold}%"
        important_indices = np.where(feature_importances >= importance_cutoff)[0]
        selected_columns = X.columns[important_indices].tolist()
        if not selected_columns:
            selected_columns = [X.columns[np.argmax(feature_importances)]]
        X_new = X[selected_columns]
        final_feature_count = len(selected_columns)
        removed_features = initial_feature_count - final_feature_count
        selected_features = X_new.copy()
        selected_features[target_column] = y.reset_index(drop=True)
        filename = f"features_lightgbm_LGBM_{initial_feature_count}_{final_feature_count}.csv"
        output_file = os.path.join(output_dir, filename)
        selected_features.to_csv(output_file, index=False)
        log_message = (
            "========================================\n"
            "🔹 LightGBM Feature Selection Completed! 🔹\n"
            "========================================\n"
            f"📌 Method: LightGBM (parallel)\n"
            f"📌 Threshold: {log_threshold_type}\n"
            f"📊 Initial Features: {initial_feature_count}\n"
            f"📉 Selected Features: {final_feature_count}\n"
            f"🗑️ Removed: {removed_features}\n"
            f"💾 Output File: {os.path.basename(output_file)}\n"
            f"⚙️ min_data_in_leaf={min_data_in_leaf}, min_split_gain={min_split_gain}, "
            f"max_depth={max_depth}, lr={learning_rate}, n_estimators={n_estimators}\n"
            f"🖥️ Parallel Cores: {cores_label}\n"
            "========================================"
        )
        return {"ui": {"text": log_message}, "result": (str(output_file),)}

NODE_CLASS_MAPPINGS = {
    "lgb_CL": lgb_CL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "lgb_CL": "5.2 LightGBM Selection",
}
