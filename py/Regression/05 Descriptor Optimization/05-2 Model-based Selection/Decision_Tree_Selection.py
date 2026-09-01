import os
import numpy as np
import pandas as pd
import traceback
import folder_paths
from sklearn.tree import DecisionTreeRegressor

class DecisionTreeFeatureSelectionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "descriptor_data_path": ("STRING", {"forceInput": True, "tooltip": "Training data only -- from 4 directly, or from another 5.1/5.2 step already applied to it (05.1/05.2 can be combined in any order). The full dataset leaks the hold-out set into selection."}),
            "target_column": ("STRING", {"default": "value"}),
            "threshold_percentile": ("INT", {"default": 90, "min": 1, "max": 99, "step": 1}),
            "max_depth": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            "min_samples_split": ("INT", {"default": 2, "min": 2, "max": 100, "step": 1}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("SELECTED_DESCRIPTOR_DATA",)
    FUNCTION = "select_features"
    CATEGORY = "QSAR/2. REGRESSION/5. Descriptor Optimization/5.2 Model-based Selection"
    OUTPUT_NODE = True

    def select_features(self, descriptor_data_path, target_column, threshold_percentile, max_depth, min_samples_split):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Regression", "05_Descriptor_Optimization", "Model_Based")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(descriptor_data_path)
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found.")
            X = df.drop(columns=[c for c in (target_column, "Name", "SMILES") if c in df.columns]).select_dtypes(include=[np.number])
            y = df[target_column]
            initial_feature_count = X.shape[1]
            max_depth_val = None if max_depth == 0 else max_depth
            model = DecisionTreeRegressor(max_depth=max_depth_val, min_samples_split=min_samples_split, random_state=42)
            model.fit(X, y)
            importances = model.feature_importances_
            threshold_value = np.percentile(importances, threshold_percentile)
            selected_indices = np.where(importances >= threshold_value)[0]
            selected_columns = X.columns[selected_indices].tolist()
            if not selected_columns:
                selected_columns = [X.columns[np.argmax(importances)]]
            X_new = X[selected_columns]
            final_feature_count = len(selected_columns)
            selected_features_df = X_new.copy()
            selected_features_df[target_column] = y.reset_index(drop=True)
            output_file = os.path.join(output_dir, f"features_dtree_{initial_feature_count}_to_{final_feature_count}.csv")
            selected_features_df.to_csv(output_file, index=False)
            log_message = (
                "========================================\n"
                "🔹 Decision Tree Feature Selection Completed! 🔹\n"
                "========================================\n"
                f"📌 Method: Decision Tree (Regression)\n"
                f"📊 Initial Features: {initial_feature_count}\n"
                f"📉 Selected Features: {final_feature_count}\n"
                f"🗑️ Removed Features: {initial_feature_count - final_feature_count}\n"
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
                f"💾 Output: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}

NODE_CLASS_MAPPINGS = {
    "DecisionTreeFeatureSelection": DecisionTreeFeatureSelectionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DecisionTreeFeatureSelection": "5.2 Decision Tree Selection",
}
