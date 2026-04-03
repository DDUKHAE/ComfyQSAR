import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
import folder_paths

class Remove_Low_Variance_Descriptors_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {"forceInput": False}),
                "threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("LOW_VAR_FILTERED_PATH",)
    FUNCTION = "run"
    CATEGORY = "QSAR/CLASSIFICATION/5. Descriptor Optimization/5.1 Filter-based Selection"
    OUTPUT_NODE = True

    @classmethod
    def run(cls, input_file, threshold):
        output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Classification_Optimized")
        os.makedirs(output_dir, exist_ok=True)

        df = pd.read_csv(input_file)
        if "Label" not in df.columns:
            raise ValueError("The dataset must contain a 'Label' column.")
        df = df.drop("Name", axis=1, errors='ignore')

        target_column = df["Label"]
        feature_columns = df.drop(columns=["Label"])

        selector = VarianceThreshold(threshold=threshold)
        selected_features = selector.fit_transform(feature_columns)
        retained_columns = feature_columns.columns[selector.get_support()]

        df_retained = pd.DataFrame(selected_features, columns=retained_columns)
        df_retained["Label"] = target_column.values

        initial_count = feature_columns.shape[1]
        final_count = len(retained_columns)
        output_file = os.path.join(output_dir, f"low_variance_results_({initial_count}_{final_count}).csv")
        df_retained.to_csv(output_file, index=False)

        log_message = (
            "========================================\n"
            "🔹 Low Variance Feature Removal Done! 🔹\n"
            "========================================\n"
            f"📊 Initial Features: {initial_count}\n"
            f"📉 Remaining Features: {final_count}\n"
            f"🗑️ Removed: {initial_count - final_count}\n"
            f"💾 Saved: {output_file}\n"
            "========================================"
        )
        return {"ui": {"text": log_message}, "result": (str(output_file),)}

class Remove_High_Correlation_Features_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {"forceInput": True}),
                "threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "correlation_mode": (["target_based", "upper", "lower"], {"default": "target_based"}),
                "importance_model": (["lasso", "random_forest"], {"default": "lasso"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OPTIMIZED_DATA_PATH",)
    FUNCTION = "run"
    CATEGORY = "QSAR/CLASSIFICATION/5. Descriptor Optimization/5.1 Filter-based Selection"
    OUTPUT_NODE = True

    @classmethod
    def run(cls, input_file, threshold, correlation_mode, importance_model):
        output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Classification_Optimized")
        os.makedirs(output_dir, exist_ok=True)

        df = pd.read_csv(input_file)
        if "Label" not in df.columns:
            raise ValueError("The dataset must contain a 'Label' column.")
        if "Name" in df.columns:
            df = df.drop("Name", axis=1, errors='ignore')

        target_column = df["Label"]
        feature_columns = df.drop(columns=["Label"])
        correlation_matrix = feature_columns.corr()
        to_remove = set()

        if correlation_mode == "target_based":
            feature_target_corr = feature_columns.corrwith(target_column).abs()
            feature_importance = {}
            if importance_model in ["lasso", "random_forest"]:
                X, y = feature_columns, target_column
                if importance_model == "lasso":
                    model = Lasso(alpha=0.01, max_iter=1000, random_state=42)
                else:
                    model = RandomForestClassifier(n_estimators=200, random_state=42)
                model.fit(X, y)
                importance_values = (
                    np.abs(model.coef_) if importance_model == "lasso" else model.feature_importances_
                )
                feature_importance = dict(zip(feature_columns.columns, importance_values))

            rows, cols = np.where(np.abs(np.triu(correlation_matrix, k=1)) > threshold)
            for row, col in zip(rows, cols):
                f1 = correlation_matrix.columns[row]
                f2 = correlation_matrix.columns[col]
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
                f1 = correlation_matrix.columns[row]
                to_remove.add(f1)

        retained_columns = [c for c in feature_columns.columns if c not in to_remove]
        df_retained = feature_columns[retained_columns]
        df_retained["Label"] = target_column

        initial_count = feature_columns.shape[1]
        final_count = len(retained_columns)

        if correlation_mode == "target_based":
            output_file = os.path.join(output_dir, f"high_correlation_results_({initial_count}_{final_count}_{importance_model}).csv")
        else:
            output_file = os.path.join(output_dir, f"high_correlation_results_({initial_count}_{final_count}_{correlation_mode}).csv")
        df_retained.to_csv(output_file, index=False)

        log_message = (
            "========================================\n"
            "🔹 High Correlation Feature Removal Done! 🔹\n"
            "========================================\n"
            f"ℹ️ Mode: {correlation_mode}, Importance Model: {importance_model}\n"
            f"📊 Initial Features: {initial_count}\n"
            f"📉 Remaining Features: {final_count}\n"
            f"🗑️ Removed: {initial_count - final_count}\n"
            f"💾 Saved: {output_file}\n"
            "========================================"
        )
        return {"ui": {"text": log_message}, "result": (str(output_file),)}

class Descriptor_Optimization_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {"forceInput": False}),
                "variance_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "correlation_threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "correlation_mode": (["target_based", "upper", "lower"], {"default": "target_based"}),
                "importance_model": (["lasso", "random_forest"], {"default": "lasso"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OPTIMIZED_DATA_PATH",)
    FUNCTION = "run"
    CATEGORY = "QSAR/CLASSIFICATION/OTHERS"
    OUTPUT_NODE = True

    @classmethod
    def run(cls, input_file, variance_threshold, correlation_threshold, correlation_mode, importance_model):
        output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Classification_Optimized")
        os.makedirs(output_dir, exist_ok=True)

        df = pd.read_csv(input_file)
        if "Label" not in df.columns:
            raise ValueError("The dataset must contain a 'Label' column.")
        df = df.drop("Name", axis=1, errors='ignore')

        target_column = df["Label"]
        feature_columns = df.drop(columns=["Label"])
        initial_count = feature_columns.shape[1]

        # Step 1: Remove low variance features
        selector = VarianceThreshold(threshold=variance_threshold)
        selected_features = selector.fit_transform(feature_columns)
        retained_columns_var = feature_columns.columns[selector.get_support()]
        features_after_var = pd.DataFrame(selected_features, columns=retained_columns_var)
        count_after_var = len(retained_columns_var)

        # Step 2: Remove high correlation features
        correlation_matrix = features_after_var.corr()
        to_remove = set()

        if correlation_mode == "target_based":
            feature_target_corr = features_after_var.corrwith(target_column).abs()
            feature_importance = {}
            if importance_model in ["lasso", "random_forest"]:
                X, y = features_after_var, target_column
                if importance_model == "lasso":
                    model = Lasso(alpha=0.01, max_iter=1000, random_state=42)
                else:
                    model = RandomForestClassifier(n_estimators=200, random_state=42)
                model.fit(X, y)
                importance_values = (
                    np.abs(model.coef_) if importance_model == "lasso" else model.feature_importances_
                )
                feature_importance = dict(zip(features_after_var.columns, importance_values))

            rows, cols = np.where(np.abs(np.triu(correlation_matrix, k=1)) > correlation_threshold)
            for row, col in zip(rows, cols):
                f1 = correlation_matrix.columns[row]
                f2 = correlation_matrix.columns[col]
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

        retained_columns_final = [c for c in features_after_var.columns if c not in to_remove]
        df_retained = features_after_var[retained_columns_final]
        df_retained["Label"] = target_column.values
        final_count = len(retained_columns_final)

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
            f"💾 Saved: {output_file}\n"
            "========================================"
        )
        return {"ui": {"text": log_message}, "result": (str(output_file),)}

NODE_CLASS_MAPPINGS = {
    "Remove_Low_Variance_Descriptors_Classification": Remove_Low_Variance_Descriptors_Classification,
    "Remove_High_Correlation_Features_Classification": Remove_High_Correlation_Features_Classification,
    "Descriptor_Optimization_Classification": Descriptor_Optimization_Classification,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Remove_Low_Variance_Descriptors_Classification": "5.1 Remove Low Variance",
    "Remove_High_Correlation_Features_Classification": "5.1 Remove High Correlation",
    "Descriptor_Optimization_Classification": "Descriptor Optimization",
}
