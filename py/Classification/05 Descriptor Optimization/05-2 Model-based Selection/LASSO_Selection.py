import os
import numpy as np
import pandas as pd
import folder_paths
from sklearn.linear_model import LogisticRegression

class lasso_CL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {}),
                "target_column": ("STRING", {"default": "Label"}),
                "alpha": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 1.0, "step": 0.0001}),
                "max_iter": ("INT", {"default": 1000, "min": 100, "max": 10000}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_file",)
    FUNCTION = "lasso_feature_selection"
    CATEGORY = "QSAR/CLASSIFICATION/5. Descriptor Optimization/5.2 Model-based Selection"
    OUTPUT_NODE = True

    def lasso_feature_selection(self, input_file, target_column, alpha, max_iter):
        output_dir = os.path.join(folder_paths.get_output_directory(), "selected_descriptors_output")
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(input_file)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        initial_feature_count = X.shape[1]
        model = LogisticRegression(penalty='l1', solver='saga', C=1/alpha, max_iter=max_iter, random_state=42)
        model.fit(X, y)
        selected_columns = X.columns[model.coef_[0] != 0]
        X_new = X[selected_columns]
        final_feature_count = len(selected_columns)
        removed_features = initial_feature_count - final_feature_count
        selected_features = X_new.copy()
        selected_features[target_column] = y.reset_index(drop=True)
        filename = f"features_lasso_LASSO_{initial_feature_count}_{final_feature_count}.csv"
        output_file = os.path.join(output_dir, filename)
        selected_features.to_csv(output_file, index=False)
        log_message = (
            "========================================\n"
            "🔹 LASSO Feature Selection Completed! 🔹\n"
            "========================================\n"
            f"📌 Method: LASSO (Logistic Regression L1)\n"
            f"📊 Initial Features: {initial_feature_count}\n"
            f"📉 Selected Features: {final_feature_count}\n"
            f"🗑️ Removed: {removed_features}\n"
            f"💾 Output File: {os.path.basename(output_file)}\n"
            "========================================"
        )
        return {"ui": {"text": log_message}, "result": (str(output_file),)}

NODE_CLASS_MAPPINGS = {
    "lasso_CL": lasso_CL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "lasso_CL": "5.2 LASSO Selection",
}
