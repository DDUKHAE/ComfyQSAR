import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import folder_paths

class Remove_Low_Variance_Descriptors_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "input_file": ("STRING", {"forceInput": False}),
            "threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("LOW_VAR_FILTERED_PATH",)
    FUNCTION = "run"
    CATEGORY = "QSAR/REGRESSION/5. Descriptor Optimization/5.1 Filter-based Selection"
    OUTPUT_NODE = True

    def run(self, input_file, threshold):
        output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Regression_Optimized")
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(input_file)
        if "value" not in df.columns:
            raise ValueError("The dataset must contain a 'value' column.")
        df = df.drop("Name", axis=1, errors='ignore')
        target_column = df["value"]
        feature_columns = df.drop(columns=["value"])
        selector = VarianceThreshold(threshold=threshold)
        selected = selector.fit_transform(feature_columns)
        retained_cols = feature_columns.columns[selector.get_support()]
        df_retained = pd.DataFrame(selected, columns=retained_cols)
        df_retained["value"] = target_column.values
        initial_count = feature_columns.shape[1]
        final_count = len(retained_cols)
        output_file = os.path.join(output_dir, f"low_variance_results_({initial_count}_{final_count}).csv")
        df_retained.to_csv(output_file, index=False)
        log_message = (
            "========================================\n"
            "🔹 Low Variance Feature Removal Done! 🔹\n"
            "========================================\n"
            f"📊 Initial Features: {initial_count}\n"
            f"📉 Remaining Features: {final_count}\n"
            f"🗑️ Removed: {initial_count - final_count}\n"
            f"💾 Output File: {os.path.basename(output_file)}\n"
            "========================================"
        )
        return {"ui": {"text": log_message}, "result": (str(output_file),)}

class Remove_High_Correlation_Features_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "input_file": ("STRING", {"forceInput": True}),
            "threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
            "correlation_mode": (["target_based", "upper", "lower"], {"default": "target_based"}),
            "importance_model": (["lasso", "random_forest"], {"default": "lasso"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OPTIMIZED_DATA_PATH",)
    FUNCTION = "run"
    CATEGORY = "QSAR/REGRESSION/5. Descriptor Optimization/5.1 Filter-based Selection"
    OUTPUT_NODE = True

    def run(self, input_file, threshold, correlation_mode, importance_model):
        output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Regression_Optimized")
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(input_file)
        if "value" not in df.columns:
            raise ValueError("The dataset must contain a 'value' column.")
        df = df.drop("Name", axis=1, errors='ignore')
        target_column = df["value"]
        feature_columns = df.drop(columns=["value"])
        correlation_matrix = feature_columns.corr()
        to_remove = set()
        if correlation_mode == "target_based":
            feature_target_corr = feature_columns.corrwith(target_column).abs()
            feature_importance = {}
            X, y = feature_columns, target_column
            if importance_model == "lasso":
                model = Lasso(alpha=0.01, max_iter=1000, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X, y)
            importance_values = np.abs(model.coef_) if importance_model == "lasso" else model.feature_importances_
            feature_importance = dict(zip(feature_columns.columns, importance_values))
            rows, cols = np.where(np.abs(np.triu(correlation_matrix, k=1)) > threshold)
            for row, col in zip(rows, cols):
                f1, f2 = correlation_matrix.columns[row], correlation_matrix.columns[col]
                if feature_target_corr[f1] > feature_target_corr[f2]:
                    weaker = f2
                elif feature_target_corr[f1] < feature_target_corr[f2]:
                    weaker = f1
                else:
                    weaker = f2 if feature_importance.get(f1, 0) > feature_importance.get(f2, 0) else f1
                to_remove.add(weaker)
        else:
            tri = np.triu(correlation_matrix, k=1) if correlation_mode == "upper" else np.tril(correlation_matrix, k=-1)
            rows, cols = np.where(np.abs(tri) > threshold)
            for row, col in zip(rows, cols):
                to_remove.add(correlation_matrix.columns[row])
        retained_cols = [c for c in feature_columns.columns if c not in to_remove]
        df_retained = feature_columns[retained_cols].copy()
        df_retained["value"] = target_column.values
        initial_count = feature_columns.shape[1]
        final_count = len(retained_cols)
        output_file = os.path.join(output_dir, f"high_corr_results_({initial_count}_{final_count}_{correlation_mode}).csv")
        df_retained.to_csv(output_file, index=False)
        log_message = (
            "========================================\n"
            "🔹 High Correlation Feature Removal Done! 🔹\n"
            "========================================\n"
            f"ℹ️ Mode: {correlation_mode}, Importance Model: {importance_model}\n"
            f"📊 Initial Features: {initial_count}\n"
            f"📉 Remaining Features: {final_count}\n"
            f"🗑️ Removed: {initial_count - final_count}\n"
            f"💾 Output File: {os.path.basename(output_file)}\n"
            "========================================"
        )
        return {"ui": {"text": log_message}, "result": (str(output_file),)}

class Descriptor_Optimization_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "input_file": ("STRING", {"forceInput": False}),
            "variance_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            "correlation_threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
            "correlation_mode": (["target_based", "upper", "lower"], {"default": "target_based"}),
            "importance_model": (["lasso", "random_forest"], {"default": "lasso"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OPTIMIZED_DATA_PATH",)
    FUNCTION = "run"
    CATEGORY = "QSAR/REGRESSION/OTHERS"
    OUTPUT_NODE = True

    def run(self, input_file, variance_threshold, correlation_threshold, correlation_mode, importance_model):
        output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Regression_Optimized")
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(input_file)
        if "value" not in df.columns:
            raise ValueError("The dataset must contain a 'value' column.")
        df = df.drop("Name", axis=1, errors='ignore')
        target_column = df["value"]
        feature_columns = df.drop(columns=["value"])
        initial_count = feature_columns.shape[1]
        selector = VarianceThreshold(threshold=variance_threshold)
        selected = selector.fit_transform(feature_columns)
        retained_cols_var = feature_columns.columns[selector.get_support()]
        features_after_var = pd.DataFrame(selected, columns=retained_cols_var)
        count_after_var = len(retained_cols_var)
        correlation_matrix = features_after_var.corr()
        to_remove = set()
        if correlation_mode == "target_based":
            feature_target_corr = features_after_var.corrwith(target_column).abs()
            X, y = features_after_var, target_column
            if importance_model == "lasso":
                model = Lasso(alpha=0.01, max_iter=1000, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X, y)
            importance_values = np.abs(model.coef_) if importance_model == "lasso" else model.feature_importances_
            feature_importance = dict(zip(features_after_var.columns, importance_values))
            rows, cols = np.where(np.abs(np.triu(correlation_matrix, k=1)) > correlation_threshold)
            for row, col in zip(rows, cols):
                f1, f2 = correlation_matrix.columns[row], correlation_matrix.columns[col]
                if feature_target_corr[f1] > feature_target_corr[f2]:
                    weaker = f2
                elif feature_target_corr[f1] < feature_target_corr[f2]:
                    weaker = f1
                else:
                    weaker = f2 if feature_importance.get(f1, 0) > feature_importance.get(f2, 0) else f1
                to_remove.add(weaker)
        else:
            tri = np.triu(correlation_matrix, k=1) if correlation_mode == "upper" else np.tril(correlation_matrix, k=-1)
            rows, cols = np.where(np.abs(tri) > correlation_threshold)
            for row, col in zip(rows, cols):
                to_remove.add(correlation_matrix.columns[row])
        retained_cols_final = [c for c in features_after_var.columns if c not in to_remove]
        df_retained = features_after_var[retained_cols_final].copy()
        df_retained["value"] = target_column.values
        final_count = len(retained_cols_final)
        output_file = os.path.join(output_dir, f"optimized_{initial_count}to{final_count}.csv")
        df_retained.to_csv(output_file, index=False)
        log_message = (
            "========================================\n"
            "🔹 Descriptor Optimization Complete! 🔹\n"
            "========================================\n"
            f"📊 Initial Features: {initial_count}\n"
            f"🗑️ Removed by Variance: {initial_count - count_after_var}\n"
            f"🗑️ Removed by Correlation: {count_after_var - final_count}\n"
            f"📉 Final Features: {final_count}\n"
            f"💾 Output File: {os.path.basename(output_file)}\n"
            "========================================"
        )
        return {"ui": {"text": log_message}, "result": (str(output_file),)}

NODE_CLASS_MAPPINGS = {
    "Remove_Low_Variance_Descriptors_Regression": Remove_Low_Variance_Descriptors_Regression,
    "Remove_High_Correlation_Features_Regression": Remove_High_Correlation_Features_Regression,
    "Descriptor_Optimization_Regression": Descriptor_Optimization_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Remove_Low_Variance_Descriptors_Regression": "5.1 Remove Low Variance",
    "Remove_High_Correlation_Features_Regression": "5.1 Remove High Correlation",
    "Descriptor_Optimization_Regression": "Descriptor Optimization",
}
