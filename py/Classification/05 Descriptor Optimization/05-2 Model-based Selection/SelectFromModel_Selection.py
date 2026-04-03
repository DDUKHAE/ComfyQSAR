import os
import numpy as np
import pandas as pd
import folder_paths
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_classification_model(name, n_estimators, max_depth, min_samples_split, learning_rate, alpha, max_iter, random_state):
    effective_max_depth = None if max_depth == 0 else max_depth
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=effective_max_depth,
                                      min_samples_split=min_samples_split, criterion="gini", random_state=random_state)
    elif name == "decision_tree":
        return DecisionTreeClassifier(max_depth=effective_max_depth, min_samples_split=min_samples_split,
                                      criterion="gini", random_state=random_state)
    elif name == "xgboost":
        return XGBClassifier(n_estimators=n_estimators, max_depth=effective_max_depth, learning_rate=learning_rate,
                             random_state=random_state, use_label_encoder=False, eval_metric="logloss", verbosity=0)
    elif name == "lightgbm":
        return LGBMClassifier(n_estimators=n_estimators, max_depth=effective_max_depth,
                              learning_rate=learning_rate, random_state=random_state)
    elif name == "lasso":
        return LogisticRegression(penalty='l1', solver='saga', C=1/alpha, max_iter=max_iter, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model: {name}")

class select_from_model_CL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {}),
                "target_column": ("STRING", {"default": "Label"}),
                "random_state": ("INT", {"default": 42, "min": 0, "max": 1000, "step": 1}),
                "model_name": (["random_forest", "decision_tree", "xgboost", "lightgbm", "lasso"], {"default": "random_forest"}),
                "threshold": (["mean", "median", "1.25*mean", "1.25*median", "2*mean", "%"], {"forceInput": False, "default": "mean"}),
                "threshold_num": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 1.0, "step": 0.001}),
                "max_iter": ("INT", {"default": 1000, "min": 100, "max": 10000, "step": 100}),
                "n_estimators": ("INT", {"default": 100, "min": 10, "max": 2000, "step": 10}),
                "max_depth": ("INT", {"default": 0, "min": 0, "max": 100}),
                "min_samples_split": ("INT", {"default": 2, "min": 2, "max": 100}),
                "learning_rate": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_file",)
    FUNCTION = "select_from_model_feature_selection"
    CATEGORY = "QSAR/CLASSIFICATION/5. Descriptor Optimization/5.2 Model-based Selection"
    OUTPUT_NODE = True

    def select_from_model_feature_selection(self, input_file, target_column, random_state,
                                            model_name, threshold, threshold_num, alpha, max_iter,
                                            n_estimators, max_depth, min_samples_split, learning_rate):
        output_dir = os.path.join(folder_paths.get_output_directory(), "selected_descriptors_output")
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(input_file)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        initial_feature_count = X.shape[1]
        model = get_classification_model(model_name, n_estimators, max_depth, min_samples_split, learning_rate, alpha, max_iter, random_state)
        model.fit(X, y)
        if threshold == "%":
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                importances = np.abs(model.coef_).flatten()
            effective_threshold = float(np.percentile(importances, (1 - threshold_num) * 100))
        else:
            effective_threshold = threshold
        selector = SelectFromModel(estimator=model, threshold=effective_threshold, prefit=True)
        selected_columns = X.columns[selector.get_support()]
        X_new = X[selected_columns]
        final_feature_count = len(selected_columns)
        removed_features = initial_feature_count - final_feature_count
        selected_features = X_new.copy()
        selected_features[target_column] = y.reset_index(drop=True)
        filename = f"features_sfm_{model_name}_{initial_feature_count}_{final_feature_count}.csv"
        output_file = os.path.join(output_dir, filename)
        selected_features.to_csv(output_file, index=False)
        log_message = (
            "========================================\n"
            "🔹 SelectFromModel Feature Selection Completed! 🔹\n"
            "========================================\n"
            f"📌 Method: SelectFromModel ({model_name})\n"
            f"📊 Threshold: {effective_threshold}\n"
            f"📊 Initial Features: {initial_feature_count}\n"
            f"📉 Selected Features: {final_feature_count}\n"
            f"🗑️ Removed: {removed_features}\n"
            f"💾 Output File: {os.path.basename(output_file)}\n"
            "========================================"
        )
        return {"ui": {"text": log_message}, "result": (str(output_file),)}

NODE_CLASS_MAPPINGS = {
    "select_from_model_CL": select_from_model_CL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "select_from_model_CL": "5.2 Select From Model",
}
