import os
import numpy as np
import pandas as pd
import traceback
import folder_paths
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def get_regression_model(name, n_estimators, max_depth, learning_rate, alpha):
    effective_max_depth = None if max_depth == 0 else max_depth
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=n_estimators, max_depth=effective_max_depth, random_state=42)
    elif name == "decision_tree":
        return DecisionTreeRegressor(max_depth=effective_max_depth, random_state=42)
    elif name == "xgboost":
        return XGBRegressor(n_estimators=n_estimators, max_depth=max_depth or 6,
                            learning_rate=learning_rate, random_state=42, verbosity=0)
    elif name == "lightgbm":
        return LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth or -1,
                             learning_rate=learning_rate, random_state=42)
    elif name == "lasso":
        return Lasso(alpha=alpha, max_iter=1000)
    else:
        raise ValueError(f"Unsupported model: {name}")

class RFEFeatureSelectionNode:
    MODELS = ["random_forest", "xgboost", "lightgbm", "lasso", "decision_tree"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "input_file_path": ("STRING", {"forceInput": True}),
            "target_column": ("STRING", {"default": "value"}),
            "model_name": (s.MODELS, {"default": "random_forest"}),
            "n_features_to_select": ("INT", {"default": 50, "min": 1, "max": 10000, "step": 1}),
            "n_estimators": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10}),
            "max_depth": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            "learning_rate": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01}),
            "alpha": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 10.0, "step": 0.001}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_file_path",)
    FUNCTION = "select_features"
    CATEGORY = "QSAR/REGRESSION/5. Descriptor Optimization/5.2 Model-based Selection"
    OUTPUT_NODE = True

    def select_features(self, input_file_path, target_column, model_name, n_features_to_select,
                        n_estimators, max_depth, learning_rate, alpha):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), f"feature_selection_results/RFE_{model_name}")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(input_file_path)
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found.")
            X = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
            y = df[target_column]
            initial_feature_count = X.shape[1]
            if n_features_to_select > initial_feature_count:
                n_features_to_select = initial_feature_count
            model = get_regression_model(model_name, n_estimators, max_depth, learning_rate, alpha)
            selector = RFE(estimator=model, n_features_to_select=n_features_to_select)
            selector.fit(X, y)
            selected_columns = X.columns[selector.get_support()].tolist()
            X_new = X[selected_columns]
            final_feature_count = len(selected_columns)
            selected_features_df = X_new.copy()
            selected_features_df[target_column] = y.reset_index(drop=True)
            output_file = os.path.join(output_dir, f"features_rfe_{model_name}_{initial_feature_count}_to_{final_feature_count}.csv")
            selected_features_df.to_csv(output_file, index=False)
            log_message = (
                "========================================\n"
                "🔹 RFE Feature Selection Completed! 🔹\n"
                "========================================\n"
                f"📌 Method: RFE - {model_name} (Regression)\n"
                f"🎯 Target Features: {n_features_to_select}\n"
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
    "RFEFeatureSelection": RFEFeatureSelectionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RFEFeatureSelection": "5.2 RFE Selection",
}
