import os
import numpy as np
import pandas as pd
import traceback
import folder_paths
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def get_regression_model(name, n_estimators, max_depth, learning_rate, alpha, random_state):
    effective_max_depth = None if max_depth == 0 else max_depth
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=n_estimators, max_depth=effective_max_depth, random_state=random_state)
    elif name == "decision_tree":
        return DecisionTreeRegressor(max_depth=effective_max_depth, random_state=random_state)
    elif name == "xgboost":
        return XGBRegressor(n_estimators=n_estimators, max_depth=max_depth or 6,
                            learning_rate=learning_rate, random_state=random_state, verbosity=0)
    elif name == "lightgbm":
        return LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth or -1,
                             learning_rate=learning_rate, random_state=random_state)
    elif name == "lasso":
        return Lasso(alpha=alpha, max_iter=1000, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model: {name}")

class SelectFromModelFeatureSelectionNode:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "input_file_path": ("STRING", {"forceInput": True}),
            "target_column": ("STRING", {"default": "value"}),
            "random_state": ("INT", {"default": 42, "min": 0, "max": 1000, "step": 1}),
            "model_name": (["random_forest", "xgboost", "lightgbm", "lasso", "decision_tree"], {"default": "random_forest"}),
            "threshold": ("STRING", {"default": "mean"}),
            "alpha": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 10.0, "step": 0.001}),
            "n_estimators": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10}),
            "max_depth": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            "learning_rate": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01}),
            
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_file_path",)
    FUNCTION = "select_features"
    CATEGORY = "QSAR/REGRESSION/5. Descriptor Optimization/5.2 Model-based Selection"
    OUTPUT_NODE = True

    def select_features(self, input_file_path, target_column, random_state, model_name, threshold,
                        n_estimators, max_depth, learning_rate, alpha):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), f"feature_selection_results/SFM_{model_name}")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(input_file_path)
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found.")
            X = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
            y = df[target_column]
            initial_feature_count = X.shape[1]
            model = get_regression_model(model_name, n_estimators, max_depth, learning_rate, alpha, random_state)
            model.fit(X, y)
            selector = SelectFromModel(estimator=model, threshold=threshold, prefit=True)
            selected_columns = X.columns[selector.get_support()].tolist()
            if not selected_columns:
                importances = getattr(model, "feature_importances_", getattr(model, "coef_", None))
                if importances is not None:
                    selected_columns = [X.columns[np.argmax(np.abs(importances))]]
                else:
                    return {"ui": {"text": "No features selected."}, "result": ("",)}
            X_new = X[selected_columns]
            final_feature_count = len(selected_columns)
            selected_features_df = X_new.copy()
            selected_features_df[target_column] = y.reset_index(drop=True)
            output_file = os.path.join(output_dir, f"features_sfm_{model_name}_{initial_feature_count}_to_{final_feature_count}.csv")
            selected_features_df.to_csv(output_file, index=False)
            log_message = (
                "========================================\n"
                "🔹 SelectFromModel Feature Selection Completed! 🔹\n"
                "========================================\n"
                f"📌 Method: SelectFromModel - {model_name} (Regression)\n"
                f"📊 Initial Features: {initial_feature_count}\n"
                f"📉 Selected Features: {final_feature_count}\n"
                f"🗑️ Removed Features: {initial_feature_count - final_feature_count}\n"
                f"📌 Threshold: {threshold}\n"
                f"💾 Output File: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}

NODE_CLASS_MAPPINGS = {
    "SelectFromModelFeatureSelection": SelectFromModelFeatureSelectionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelectFromModelFeatureSelection": "5.2 Select From Model",
}
