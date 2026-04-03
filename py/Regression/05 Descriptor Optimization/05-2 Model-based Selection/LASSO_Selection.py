import os
import numpy as np
import pandas as pd
import traceback
import folder_paths
from sklearn.linear_model import Lasso

class LassoFeatureSelectionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "input_file_path": ("STRING", {"forceInput": True}),
            "target_column": ("STRING", {"default": "value"}),
            "alpha": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 10.0, "step": 0.001}),
            "max_iter": ("INT", {"default": 1000, "min": 100, "max": 10000, "step": 100}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_file_path",)
    FUNCTION = "select_features"
    CATEGORY = "QSAR/REGRESSION/5. Descriptor Optimization/5.2 Model-based Selection"
    OUTPUT_NODE = True

    def select_features(self, input_file_path, target_column, alpha, max_iter):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "feature_selection_results/LASSO")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(input_file_path)
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found.")
            X = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
            y = df[target_column]
            initial_feature_count = X.shape[1]
            model = Lasso(alpha=alpha, max_iter=max_iter)
            model.fit(X, y)
            selected_mask = model.coef_ != 0
            selected_columns = X.columns[selected_mask].tolist()
            if not selected_columns:
                top_idx = np.argmax(np.abs(model.coef_))
                selected_columns = [X.columns[top_idx]]
            X_new = X[selected_columns]
            final_feature_count = len(selected_columns)
            selected_features_df = X_new.copy()
            selected_features_df[target_column] = y.reset_index(drop=True)
            output_file = os.path.join(output_dir, f"features_lasso_{initial_feature_count}_to_{final_feature_count}.csv")
            selected_features_df.to_csv(output_file, index=False)
            log_message = (
                "========================================\n"
                "🔹 LASSO Feature Selection Completed! 🔹\n"
                "========================================\n"
                f"📌 Method: LASSO (Regression)\n"
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
    "LassoFeatureSelection": LassoFeatureSelectionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LassoFeatureSelection": "5.2 LASSO Selection",
}
