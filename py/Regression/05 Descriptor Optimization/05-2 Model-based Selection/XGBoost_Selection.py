import os
import numpy as np
import pandas as pd
import traceback
import folder_paths
from multiprocessing import Pool, cpu_count
from xgboost import XGBRegressor

def train_xgb_importance_regression(args):
    X, y, feature_names, i, params = args
    model = XGBRegressor(
        n_estimators=params["n_estimators"], max_depth=params["max_depth"],
        learning_rate=params["learning_rate"], importance_type=params["importance_type"],
        random_state=i, verbosity=0
    )
    model.fit(X, y)
    return model.feature_importances_

class XGBoostFeatureSelectionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "input_file_path": ("STRING", {"forceInput": True}),
            "target_column": ("STRING", {"default": "value"}),
            "threshold_percentile": ("INT", {"default": 90, "min": 1, "max": 99, "step": 1}),
            "n_estimators": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10}),
            "learning_rate": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01}),
            "max_depth": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}),
            "n_iterations": ("INT", {"default": 30, "min": 1, "max": 200, "step": 1}),
            "importance_type": (["gain", "weight", "cover", "total_gain", "total_cover"],),
            "num_cores": ("INT", {"default": 4, "min": 1, "max": cpu_count(), "step": 1}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_file_path",)
    FUNCTION = "select_features"
    CATEGORY = "QSAR/REGRESSION/5. Descriptor Optimization/5.2 Model-based Selection"
    OUTPUT_NODE = True

    def select_features(self, input_file_path, target_column, threshold_percentile, n_estimators,
                        learning_rate, max_depth, n_iterations, importance_type, num_cores):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "feature_selection_results/XGBoost")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(input_file_path)
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found.")
            X = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
            y = df[target_column]
            initial_feature_count = X.shape[1]
            feature_names = X.columns.tolist()
            params = {"n_estimators": n_estimators, "max_depth": max_depth,
                      "learning_rate": learning_rate, "importance_type": importance_type}
            args_list = [(X, y, feature_names, i, params) for i in range(n_iterations)]
            with Pool(num_cores) as pool:
                importance_matrix = pool.map(train_xgb_importance_regression, args_list)
            feature_importances = np.mean(np.vstack(importance_matrix), axis=0)
            threshold_value = np.percentile(feature_importances, threshold_percentile)
            selected_indices = np.where(feature_importances >= threshold_value)[0]
            selected_columns = [feature_names[i] for i in selected_indices]
            if not selected_columns:
                selected_columns = [feature_names[np.argmax(feature_importances)]]
            X_new = X[selected_columns]
            final_feature_count = len(selected_columns)
            selected_features_df = X_new.copy()
            selected_features_df[target_column] = y.reset_index(drop=True)
            output_file = os.path.join(output_dir, f"features_xgb_{initial_feature_count}_to_{final_feature_count}.csv")
            selected_features_df.to_csv(output_file, index=False)
            log_message = (
                "========================================\n"
                "🔹 XGBoost Feature Selection Completed! 🔹\n"
                "========================================\n"
                f"📌 Method: XGBoost (Regression)\n"
                f"📊 Initial Features: {initial_feature_count}\n"
                f"📉 Selected Features: {final_feature_count}\n"
                f"🗑️ Removed Features: {initial_feature_count - final_feature_count}\n"
                f"🖥️ Parallel Cores: {num_cores}\n"
                f"💾 Output File: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}

NODE_CLASS_MAPPINGS = {
    "XGBoostFeatureSelection": XGBoostFeatureSelectionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XGBoostFeatureSelection": "5.2 XGBoost Selection",
}
