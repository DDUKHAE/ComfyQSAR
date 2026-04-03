import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import traceback
import folder_paths

def replace_inf_regression(df):
    numeric_df = df.select_dtypes(include=np.number)
    inf_mask = numeric_df.isin([np.inf, -np.inf])
    total_inf = inf_mask.sum().sum()

    if total_inf > 0:
        df_copy = df.copy()
        df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df_copy, total_inf

    return df.copy(), 0

def remove_high_nan_rows_regression(df, threshold):

    if df.shape[1] == 0:
        return df.copy(), 0

    initial_rows = len(df)
    nan_percentage = df.isna().sum(axis=1) / df.shape[1]

    filtered_df = df[nan_percentage <= threshold]
    removed_count = initial_rows - len(filtered_df)

    return filtered_df, removed_count

def remove_high_nan_cols_regression(df, threshold):
    initial_cols = df.shape[1]
    nan_percentage = df.isna().mean()
    retained_columns = nan_percentage[nan_percentage <= threshold].index.tolist()

    if "SMILES" not in retained_columns:
        retained_columns.append("SMILES")

    if "value" not in retained_columns:
        retained_columns.append("value")

    filtered_df = df[retained_columns]
    removed_count = initial_cols - len(filtered_df.columns)

    return filtered_df, removed_count

def impute_missing_values_regression(df, method):
    smiles_values = df[["SMILES", "value"]]
    descriptors = df.drop(columns=["SMILES", "value"])

    missing_count = descriptors.isna().sum().sum()

    imputer = SimpleImputer(strategy=method)
    imputed_descriptors = pd.DataFrame(imputer.fit_transform(descriptors), columns=descriptors.columns)

    final_df = pd.concat([smiles_values.reset_index(drop=True), imputed_descriptors.reset_index(drop=True)], axis=1)

    return final_df, missing_count

class Replace_inf_with_nan_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"descriptors": ("STRING", {})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DESCRIPTORS",)
    FUNCTION = "run"
    CATEGORY = "QSAR/REGRESSION/OTHERS"
    OUTPUT_NODE = True

    def run(self, descriptors):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Preprocessed_Regression")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(descriptors)
            df, inf_count = replace_inf_regression(df)
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(descriptors))[0]}_inf_replaced.csv")
            df.to_csv(output_file, index=False)
            log_message = (
                "========================================\n"
                "🔹 Inf->NaN Conversion Complete! 🔹\n"
                "========================================\n"
                f"✅ Replaced Values: {inf_count}\n"
                f"💾 Output File: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}

class Remove_high_nan_compounds_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "preprocessed_descriptors": ("STRING", {"forceInput": True}),
            "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DESCRIPTORS",)
    FUNCTION = "run"
    CATEGORY = "QSAR/REGRESSION/OTHERS"
    OUTPUT_NODE = True

    def run(self, preprocessed_descriptors, threshold):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Preprocessed_Regression")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(preprocessed_descriptors)
            initial = len(df)
            df, removed = remove_high_nan_rows_regression(df, threshold)
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(preprocessed_descriptors))[0]}_compounds_filtered.csv")
            df.to_csv(output_file, index=False)
            log_message = (
                "========================================\n"
                "🔹 High NaN Compound Removal Complete! 🔹\n"
                "========================================\n"
                f"✅ Initial Compounds: {initial}\n"
                f"✅ Removed Compounds: {removed}\n"
                f"✅ Final Compounds: {len(df)}\n"
                f"💾 Output File: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}

class Remove_high_nan_descriptors_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "preprocessed_descriptors": ("STRING", {"forceInput": True}),
            "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DESCRIPTORS",)
    FUNCTION = "run"
    CATEGORY = "QSAR/REGRESSION/OTHERS"
    OUTPUT_NODE = True

    def run(self, preprocessed_descriptors, threshold):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Preprocessed_Regression")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(preprocessed_descriptors)
            initial_cols = df.shape[1]
            df, removed = remove_high_nan_cols_regression(df, threshold)
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(preprocessed_descriptors))[0]}_descriptors_filtered.csv")
            df.to_csv(output_file, index=False)
            log_message = (
                "========================================\n"
                "🔹 High NaN Descriptor Removal Complete! 🔹\n"
                "========================================\n"
                f"✅ Initial Descriptors: {initial_cols}\n"
                f"✅ Removed Descriptors: {removed}\n"
                f"✅ Final Descriptors: {df.shape[1]}\n"
                f"💾 Output File: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}

class Impute_missing_values_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "preprocessed_descriptors": ("STRING", {"forceInput": True}),
            "method": (["mean", "median", "most_frequent"],),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DESCRIPTORS",)
    FUNCTION = "run"
    CATEGORY = "QSAR/REGRESSION/OTHERS"
    OUTPUT_NODE = True

    def run(self, preprocessed_descriptors, method):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Preprocessed_Regression")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(preprocessed_descriptors)
            df, count = impute_missing_values_regression(df, method)
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(preprocessed_descriptors))[0]}_imputed.csv")
            df.to_csv(output_file, index=False)
            log_message = (
                "========================================\n"
                "🔹 Imputation Complete! 🔹\n"
                "========================================\n"
                f"✅ Imputation Method: '{method}'\n"
                f"✅ Filled Missing Values: {count}\n"
                f"💾 Output File: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}

class Descriptor_preprocessing_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "descriptors": ("STRING", {}),
            "compounds_nan_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "descriptors_nan_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "imputation_method": (["mean", "median", "most_frequent"],),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DESCRIPTORS",)
    FUNCTION = "preprocess"
    CATEGORY = "QSAR/REGRESSION"
    OUTPUT_NODE = True

    def preprocess(self, descriptors, compounds_nan_threshold, descriptors_nan_threshold, imputation_method):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Preprocessed_Regression")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(descriptors)
            initial_shape = f"{df.shape[0]}x{df.shape[1]}"
            df, inf_count = replace_inf_regression(df)
            df, cpd_removed = remove_high_nan_rows_regression(df, compounds_nan_threshold)
            df, desc_removed = remove_high_nan_cols_regression(df, descriptors_nan_threshold)
            df, imputed = impute_missing_values_regression(df, imputation_method)
            if df.empty:
                return {"ui": {"text": "⚠️ Warning: Preprocessing resulted in an empty dataset."}, "result": ("",)}
            final_shape = f"{df.shape[0]}x{df.shape[1]}"
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(descriptors))[0]}_preprocessed.csv")
            df.to_csv(output_file, index=False)
            summary = (
                "========================================\n"
                "🔹 Preprocessing Pipeline Complete! 🔹\n"
                "========================================\n"
                f"📊 Initial Shape: {initial_shape}\n"
                f"✅ Inf Replaced: {inf_count} values\n"
                f"✅ Compounds Removed: {cpd_removed}\n"
                f"✅ Descriptors Removed: {desc_removed}\n"
                f"✅ Values Imputed: {imputed} ('{imputation_method}')\n"
                f"📊 Final Shape: {final_shape}\n"
                f"💾 Output File: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": summary}, "result": (str(output_file),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}

NODE_CLASS_MAPPINGS = {
    "Replace_inf_with_nan_Regression": Replace_inf_with_nan_Regression,
    "Remove_high_nan_compounds_Regression": Remove_high_nan_compounds_Regression,
    "Remove_high_nan_descriptors_Regression": Remove_high_nan_descriptors_Regression,
    "Impute_missing_values_Regression": Impute_missing_values_Regression,
    "Descriptor_preprocessing_Regression": Descriptor_preprocessing_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Replace_inf_with_nan_Regression": "Replace Inf with NaN",
    "Remove_high_nan_compounds_Regression": "Remove High NaN Compounds",
    "Remove_high_nan_descriptors_Regression": "Remove High NaN Descriptors",
    "Impute_missing_values_Regression": "Impute Missing Values",
    "Descriptor_preprocessing_Regression": "3. Descriptor Preprocessing",
}
