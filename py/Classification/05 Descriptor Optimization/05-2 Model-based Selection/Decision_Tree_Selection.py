import os
import numpy as np
import pandas as pd
import folder_paths
from sklearn.tree import DecisionTreeClassifier

class decision_tree_CL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {}),
                "target_column": ("STRING", {"default": "Label"}),
                "max_depth": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "min_samples_split": ("INT", {"default": 2, "min": 2, "max": 100}),
                "criterion": (["gini", "entropy", "log_loss"], {"default": "gini"}),
                "threshold_mode": ("BOOLEAN", {"default": False, "forceInput": False, "label_on": "absolute", "label_off": "percentile"}),
                "threshold": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "n_iterations": ("INT", {"default": 100, "min": 10, "max": 1000}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_file",)
    FUNCTION = "decision_tree_feature_selection"
    CATEGORY = "QSAR/CLASSIFICATION/5. Descriptor Optimization/5.2 Model-based Selection"
    OUTPUT_NODE = True

    def decision_tree_feature_selection(self, input_file, target_column, max_depth, min_samples_split,
                                        criterion, threshold_mode, threshold, n_iterations):
        output_dir = os.path.join(folder_paths.get_output_directory(), "selected_descriptors_output")
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(input_file)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        initial_feature_count = X.shape[1]
        feature_importance_matrix = np.zeros((n_iterations, X.shape[1]))
        current_max_depth = None if max_depth == 0 else max_depth
        for i in range(n_iterations):
            model = DecisionTreeClassifier(max_depth=current_max_depth, min_samples_split=min_samples_split,
                                           criterion=criterion, random_state=i)
            model.fit(X, y)
            feature_importance_matrix[i] = model.feature_importances_
        feature_importances = np.mean(feature_importance_matrix, axis=0)
        if threshold_mode:
            importance_cutoff = threshold
            log_threshold_type = f"Absolute - {importance_cutoff}"
        else:
            importance_cutoff = np.percentile(feature_importances, threshold)
            log_threshold_type = f"Percentile - {threshold}%"
        important_indices = np.where(feature_importances >= importance_cutoff)[0]
        selected_columns = X.columns[important_indices]
        X_new = X[selected_columns]
        final_feature_count = len(selected_columns)
        removed_features = initial_feature_count - final_feature_count
        selected_features = X_new.copy()
        selected_features[target_column] = y
        filename = f"features_decision_tree_DT_{initial_feature_count}_{final_feature_count}.csv"
        output_file = os.path.join(output_dir, filename)
        selected_features.to_csv(output_file, index=False)
        log_message = (
            "========================================\n"
            "🔹 Decision Tree Feature Selection Completed! 🔹\n"
            "========================================\n"
            f"📌 Method: Decision Tree\n"
            f"📌 Threshold: {log_threshold_type}\n"
            f"📊 Initial Features: {initial_feature_count}\n"
            f"📉 Selected Features: {final_feature_count}\n"
            f"🗑️ Removed: {removed_features}\n"
            f"💾 Output File: {os.path.basename(output_file)}\n"
            "========================================"
        )
        return {"ui": {"text": log_message}, "result": (str(output_file),)}

NODE_CLASS_MAPPINGS = {
    "decision_tree_CL": decision_tree_CL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "decision_tree_CL": "5.2 Decision Tree Selection",
}
