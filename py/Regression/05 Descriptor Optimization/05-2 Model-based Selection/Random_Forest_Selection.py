import os
import numpy as np
import pandas as pd
import traceback
import folder_paths
from sklearn.ensemble import RandomForestRegressor

class RandomForestFeatureSelectionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "input_file_path": ("STRING", {"forceInput": True}),
            "target_column": ("STRING", {"default": "value"}),
            "threshold_percentile": ("INT", {"default": 90, "min": 1, "max": 99, "step": 1}),
            "n_estimators": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10}),
            "n_iterations": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
            "max_depth": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_file_path",)
    FUNCTION = "select_features"
    CATEGORY = "QSAR/REGRESSION/5. Descriptor Optimization/5.2 Model-based Selection"
    OUTPUT_NODE = True

    def select_features(self, input_file_path, target_column, threshold_percentile, n_estimators, n_iterations, max_depth):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "feature_selection_results/RandomForest")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(input_file_path)
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found.")
            X = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
            y = df[target_column]
            initial_feature_count = X.shape[1]
            max_depth_val = None if max_depth == 0 else max_depth
            feature_importance_matrix = np.zeros((n_iterations, initial_feature_count))
            for i in range(n_iterations):
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth_val, random_state=i, n_jobs=-1)
                model.fit(X, y)
                feature_importance_matrix[i] = model.feature_importances_
            feature_importances = np.mean(feature_importance_matrix, axis=0)
            threshold_value = np.percentile(feature_importances, threshold_percentile)
            selected_indices = np.where(feature_importances >= threshold_value)[0]
            selected_columns = X.columns[selected_indices].tolist()
            if not selected_columns:
                selected_columns = [X.columns[np.argmax(feature_importances)]]
            X_new = X[selected_columns]
            final_feature_count = len(selected_columns)
            selected_features_df = X_new.copy()
            selected_features_df[target_column] = y.reset_index(drop=True)
            output_file = os.path.join(output_dir, f"features_rf_{initial_feature_count}_to_{final_feature_count}.csv")
            selected_features_df.to_csv(output_file, index=False)
            log_message = (
                "========================================\n"
                "🔹 Random Forest Feature Selection Completed! 🔹\n"
                "========================================\n"
                f"📌 Method: Random Forest (Regression)\n"
                f"📊 Initial Features: {initial_feature_count}\n"
                f"📉 Selected Features: {final_feature_count}\n"
                f"🗑️ Removed Features: {initial_feature_count - final_feature_count}\n"
                f"💾 Output File: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}

NODE_CLASS_MAPPINGS = {
    "RandomForestFeatureSelection": RandomForestFeatureSelectionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomForestFeatureSelection": "5.2 Random Forest Selection",
}
