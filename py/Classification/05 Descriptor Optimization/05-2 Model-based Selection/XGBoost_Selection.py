import os
import numpy as np
import pandas as pd
import folder_paths
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from xgboost import XGBClassifier

def train_xgb_classification(args):
    X, y, feature_names, i, params = args
    model = XGBClassifier(
        n_estimators=params["n_estimators"], max_depth=params["max_depth"],
        learning_rate=params["learning_rate"], use_label_encoder=False,
        eval_metric="logloss", random_state=i, verbosity=0, n_jobs=1
    )
    model.fit(X, y)
    return model.feature_importances_

class xgb_CL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "descriptor_data_path": ("STRING", {"tooltip": "Training data only -- from 4 directly, or from another 5.1/5.2 step already applied to it (05.1/05.2 can be combined in any order). The full dataset leaks the hold-out set into selection."}),
                "target_column": ("STRING", {"default": "Label"}),
                "n_estimators": ("INT", {"default": 100, "min": 10, "max": 1000}),
                "max_depth": ("INT", {"default": 5, "min": 1, "max": 100}),
                "learning_rate": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01}),
                "threshold_mode": ("BOOLEAN", {"default": False, "forceInput": False, "label_on": "importance cutoff (%)", "label_off": "percentile"}),
                "threshold": ("INT", {"default": 90, "min": 1, "max": 100, "step": 1}),
                "n_iterations": ("INT", {"default": 30, "min": 1, "max": 200}),
                "num_cores": ("INT", {"default": -1, "min": -1, "max": cpu_count(), "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("SELECTED_DESCRIPTOR_DATA",)
    FUNCTION = "xgboost_feature_selection"
    CATEGORY = "QSAR/1. CLASSIFICATION/5. Descriptor Optimization/5.2 Model-based Selection"
    OUTPUT_NODE = True

    def xgboost_feature_selection(self, descriptor_data_path, target_column, n_estimators, max_depth,
                                   learning_rate, threshold_mode, threshold, n_iterations, num_cores):
        output_dir = os.path.join(folder_paths.get_output_directory(), "Classification", "05_Descriptor_Optimization", "Model_Based")
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(descriptor_data_path)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")
        X = df.drop(columns=[c for c in ("Name", "SMILES", target_column) if c in df.columns])
        y = df[target_column]
        feature_names = list(X.columns)
        initial_feature_count = len(feature_names)
        available_cores = -1 if num_cores == -1 else max(1, min(num_cores, cpu_count()))
        cores_label = "-1 (all available cores)" if available_cores == -1 else str(available_cores)
        params = {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": learning_rate}
        args_list = [(X, y, feature_names, i, params) for i in range(n_iterations)]
        importance_matrix = Parallel(
            n_jobs=available_cores,
            backend="loky",
            pre_dispatch="2*n_jobs",
        )(
            delayed(train_xgb_classification)(args)
            for args in args_list
        )
        feature_importances = np.mean(np.vstack(importance_matrix), axis=0)
        if threshold_mode:
            importance_cutoff = threshold / 100.0
            log_threshold_type = f"Importance cutoff - {threshold}%"
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
            f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
            f"💾 Output: {os.path.basename(output_file)}\n"
            f"🖥️ Parallel Cores: {cores_label}\n"
            "========================================"
        )
        return {"ui": {"text": log_message}, "result": (str(output_file),)}

NODE_CLASS_MAPPINGS = {
    "xgb_CL": xgb_CL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "xgb_CL": "5.2 XGBoost Selection",
}
