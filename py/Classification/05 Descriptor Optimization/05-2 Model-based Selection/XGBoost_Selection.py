import os
import numpy as np
import pandas as pd
import folder_paths
from multiprocessing import Pool, cpu_count
from xgboost import XGBClassifier

def train_xgb_classification(args):
    X, y, feature_names, i, params = args
    model = XGBClassifier(
        n_estimators=params["n_estimators"], max_depth=params["max_depth"],
        learning_rate=params["learning_rate"], use_label_encoder=False,
        eval_metric="logloss", random_state=i, verbosity=0
    )
    model.fit(X, y)
    return model.feature_importances_

class xgb_CL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {}),
                "target_column": ("STRING", {"default": "Label"}),
                "n_estimators": ("INT", {"default": 100, "min": 10, "max": 1000}),
                "max_depth": ("INT", {"default": 5, "min": 1, "max": 100}),
                "learning_rate": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01}),
                "threshold_mode": ("BOOLEAN", {"default": False, "forceInput": False, "label_on": "absolute", "label_off": "percentile"}),
                "threshold": ("INT", {"default": 90, "min": 1, "max": 100, "step": 1}),
                "n_iterations": ("INT", {"default": 30, "min": 1, "max": 200}),
                "num_cores": ("INT", {"default": 4, "min": 1, "max": cpu_count(), "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_file",)
    FUNCTION = "xgboost_feature_selection"
    CATEGORY = "QSAR/CLASSIFICATION/5. Descriptor Optimization/5.2 Model-based Selection"
    OUTPUT_NODE = True

    def xgboost_feature_selection(self, input_file, target_column, n_estimators, max_depth,
                                   learning_rate, threshold_mode, threshold, n_iterations, num_cores):
        output_dir = os.path.join(folder_paths.get_output_directory(), "selected_descriptors_output")
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(input_file)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        feature_names = list(X.columns)
        initial_feature_count = len(feature_names)
        available_cores = min(num_cores, cpu_count())
        params = {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": learning_rate}
        args_list = [(X, y, feature_names, i, params) for i in range(n_iterations)]
        with Pool(available_cores) as pool:
            importance_matrix = pool.map(train_xgb_classification, args_list)
        feature_importances = np.mean(np.vstack(importance_matrix), axis=0)
        if threshold_mode:
            importance_cutoff = threshold
            log_threshold_type = f"Absolute - {importance_cutoff}"
        else:
            importance_cutoff = np.percentile(feature_importances, threshold)
            log_threshold_type = f"Percentile - {threshold}%"
        important_indices = np.where(feature_importances >= importance_cutoff)[0]
        selected_columns = [feature_names[i] for i in important_indices]
        if not selected_columns:
            selected_columns = [feature_names[np.argmax(feature_importances)]]
        X_new = X[selected_columns]
        final_feature_count = len(selected_columns)
        removed_features = initial_feature_count - final_feature_count
        selected_features = X_new.copy()
        selected_features[target_column] = y.reset_index(drop=True)
        filename = f"features_xgboost_XGB_{initial_feature_count}_{final_feature_count}.csv"
        output_file = os.path.join(output_dir, filename)
        selected_features.to_csv(output_file, index=False)
        log_message = (
            "========================================\n"
            "🔹 XGBoost Feature Selection Completed! 🔹\n"
            "========================================\n"
            f"📌 Method: XGBoost\n"
            f"📌 Threshold: {log_threshold_type}\n"
            f"📊 Initial Features: {initial_feature_count}\n"
            f"📉 Selected Features: {final_feature_count}\n"
            f"🗑️ Removed: {removed_features}\n"
            f"💾 Output File: {os.path.basename(output_file)}\n"
            f"🖥️ Parallel Cores: {available_cores}\n"
            "========================================"
        )
        return {"ui": {"text": log_message}, "result": (str(output_file),)}

NODE_CLASS_MAPPINGS = {
    "xgb_CL": xgb_CL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "xgb_CL": "5.2 XGBoost Selection",
}
