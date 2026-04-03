import os
import numpy as np
import pandas as pd
import folder_paths
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_classification_model(name, n_estimators, max_depth, min_samples_split, learning_rate, alpha, max_iter):
    effective_max_depth = None if max_depth == 0 else max_depth
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=effective_max_depth,
                                      min_samples_split=min_samples_split, criterion="gini", random_state=42)
    elif name == "decision_tree":
        return DecisionTreeClassifier(max_depth=effective_max_depth, min_samples_split=min_samples_split,
                                      criterion="gini", random_state=42)
    elif name == "xgboost":
        return XGBClassifier(n_estimators=n_estimators, max_depth=effective_max_depth, learning_rate=learning_rate,
                             random_state=42, use_label_encoder=False, eval_metric="logloss", verbosity=0)
    elif name == "lightgbm":
        return LGBMClassifier(n_estimators=n_estimators, max_depth=effective_max_depth,
                              learning_rate=learning_rate, random_state=42)
    elif name == "lasso":
        return LogisticRegression(penalty='l1', solver='saga', C=1/alpha, max_iter=max_iter, random_state=42)
    else:
        raise ValueError(f"Unsupported model: {name}")

class rfe_CL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {}),
                "target_column": ("STRING", {"default": "Label"}),
                "model_name": (["random_forest", "decision_tree", "xgboost", "lightgbm", "lasso"], {"default": "random_forest"}),
                "n_features": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "alpha": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 1.0, "step": 0.0001}),
                "max_iter": ("INT", {"default": 10000, "min": 100, "max": 10000}),
                "n_estimators": ("INT", {"default": 100, "min": 10, "max": 1000}),
                "max_depth": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "min_samples_split": ("INT", {"default": 2, "min": 2, "max": 100}),
                "learning_rate": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_file",)
    FUNCTION = "rfe_feature_selection_node"
    CATEGORY = "QSAR/CLASSIFICATION/5. Descriptor Optimization/5.2 Model-based Selection"
    OUTPUT_NODE = True

    def rfe_feature_selection_node(self, input_file, target_column, model_name, n_features,
                                    alpha, max_iter, n_estimators, max_depth, min_samples_split, learning_rate):
        output_dir = os.path.join(folder_paths.get_output_directory(), "selected_descriptors_output")
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(input_file)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        initial_feature_count = X.shape[1]
        model = get_classification_model(model_name, n_estimators, max_depth, min_samples_split, learning_rate, alpha, max_iter)
        selector = RFE(estimator=model, n_features_to_select=n_features)
        selector.fit(X, y)
        selected_columns = X.columns[selector.get_support()]
        X_new = X[selected_columns]
        final_feature_count = len(selected_columns)
        removed_features = initial_feature_count - final_feature_count
        selected_features = X_new.copy()
        selected_features[target_column] = y.reset_index(drop=True)
        filename = f"features_rfe_{model_name}_{initial_feature_count}_{final_feature_count}.csv"
        output_file = os.path.join(output_dir, filename)
        selected_features.to_csv(output_file, index=False)
        log_message = (
            "========================================\n"
            "🔹 RFE Feature Selection Completed! 🔹\n"
            "========================================\n"
            f"📌 Method: RFE ({model_name})\n"
            f"📊 Target Features: {n_features}\n"
            f"📊 Initial Features: {initial_feature_count}\n"
            f"📉 Selected Features: {final_feature_count}\n"
            f"🗑️ Removed: {removed_features}\n"
            f"💾 Output File: {os.path.basename(output_file)}\n"
            "========================================"
        )
        return {"ui": {"text": log_message}, "result": (str(output_file),)}

NODE_CLASS_MAPPINGS = {
    "rfe_CL": rfe_CL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "rfe_CL": "5.2 RFE Selection",
}
