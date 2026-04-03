import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import traceback
import folder_paths

def replace_inf_classification(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=np.number)
    inf_mask = numeric_df.isin([np.inf, -np.inf])
    total_inf_count = inf_mask.sum().sum()
    if total_inf_count > 0:
        df_copy = df.copy()
        df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df_copy, total_inf_count
    return df.copy(), 0

def remove_high_nan_rows_classification(df: pd.DataFrame, threshold: float):
    if df.shape[1] == 0:
        return df.copy(), 0
    initial_rows = len(df)
    nan_percentage = df.isna().sum(axis=1) / df.shape[1]
    filtered_df = df[nan_percentage <= threshold].copy()
    removed_count = initial_rows - len(filtered_df)
    return filtered_df, removed_count

def remove_high_nan_cols_classification(df: pd.DataFrame, threshold: float, essential_cols=['SMILES', 'Name', 'Label']):
    initial_cols = df.shape[1]
    nan_percentage = df.isna().mean()
    retained_cols = nan_percentage[nan_percentage <= threshold].index.tolist()
    for col in essential_cols:
        if col in df.columns and col not in retained_cols:
            retained_cols.append(col)
    filtered_df = df[retained_cols].copy()
    removed_count = initial_cols - filtered_df.shape[1]
    return filtered_df, removed_count

def impute_missing_values_classification(df: pd.DataFrame, method: str):
    non_descriptor_cols = [col for col in ["Name", "SMILES", "Label"] if col in df.columns]
    descriptor_cols = [col for col in df.columns if col not in non_descriptor_cols]

    descriptors = df[descriptor_cols].copy()
    imputed_count = int(descriptors.isna().sum().sum())

    imputer = SimpleImputer(strategy=method)
    imputed_descriptors = pd.DataFrame(
        imputer.fit_transform(descriptors),
        columns=descriptor_cols,
        index=df.index
    )

    final_data = pd.concat([df[non_descriptor_cols].reset_index(drop=True), imputed_descriptors.reset_index(drop=True)], axis=1)

    return final_data, imputed_count

class Replace_inf_with_nan_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "descriptors": ("STRING", {"placeholder": "Path to descriptors CSV file"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DESCRIPTORS",)
    FUNCTION = "run"
    CATEGORY = "QSAR/CLASSIFICATION/OTHERS"
    OUTPUT_NODE = True

    def run(self, descriptors):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Preprocessed")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(descriptors)
            df, inf_count = replace_inf_classification(df)
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
            error_msg = f"❌ Error replacing infinity values: {e}\n\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": ("",)}

class Remove_high_nan_compounds_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preprocessed_descriptors": ("STRING", {"forceInput": True}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DESCRIPTORS",)
    FUNCTION = "run"
    CATEGORY = "QSAR/CLASSIFICATION/OTHERS"
    OUTPUT_NODE = True

    def run(self, preprocessed_descriptors, threshold):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Preprocessed")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(preprocessed_descriptors)
            initial_rows = len(df)
            df, removed_count = remove_high_nan_rows_classification(df, threshold)
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(preprocessed_descriptors))[0]}_compounds_filtered.csv")
            df.to_csv(output_file, index=False)
            log_message = (
                "========================================\n"
                "🔹 High NaN Compound Removal Complete! 🔹\n"
                "========================================\n"
                f"✅ Initial Compounds: {initial_rows}\n"
                f"✅ Removed Compounds: {removed_count}\n"
                f"✅ Final Compounds: {len(df)}\n"
                f"💾 Output File: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
        except Exception as e:
            error_msg = f"❌ Error removing high NaN compounds: {e}\n\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": ("",)}

class Remove_high_nan_descriptors_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preprocessed_descriptors": ("STRING", {"forceInput": True}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DESCRIPTORS",)
    FUNCTION = "run"
    CATEGORY = "QSAR/CLASSIFICATION/OTHERS"
    OUTPUT_NODE = True

    def run(self, preprocessed_descriptors, threshold):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Preprocessed")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(preprocessed_descriptors)
            initial_cols = df.shape[1]
            df, removed_count = remove_high_nan_cols_classification(df, threshold)
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(preprocessed_descriptors))[0]}_descriptors_filtered.csv")
            df.to_csv(output_file, index=False)
            log_message = (
                "========================================\n"
                "🔹 High NaN Descriptor Removal Complete! 🔹\n"
                "========================================\n"
                f"✅ Initial Descriptors: {initial_cols}\n"
                f"✅ Removed Descriptors: {removed_count}\n"
                f"✅ Final Descriptors: {df.shape[1]}\n"
                f"💾 Output File: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
        except Exception as e:
            error_msg = f"❌ Error removing high NaN descriptors: {e}\n\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": ("",)}

class Impute_missing_values_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preprocessed_descriptors": ("STRING", {"forceInput": True}),
                "method": (["mean", "median", "most_frequent"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DESCRIPTORS",)
    FUNCTION = "run"
    CATEGORY = "QSAR/CLASSIFICATION/OTHERS"
    OUTPUT_NODE = True

    def run(self, preprocessed_descriptors, method):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Preprocessed")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(preprocessed_descriptors)
            df, imputed_count = impute_missing_values_classification(df, method)
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(preprocessed_descriptors))[0]}_imputed.csv")
            df.to_csv(output_file, index=False)
            log_message = (
                "========================================\n"
                "🔹 Imputation Complete! 🔹\n"
                "========================================\n"
                f"✅ Imputation Method: '{method}'\n"
                f"✅ Filled Missing Values: {imputed_count}\n"
                f"💾 Output File: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
        except Exception as e:
            error_msg = f"❌ Error during imputation: {e}\n\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": ("",)}

class Descriptor_preprocessing_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "descriptors": ("STRING", {}),
                "compounds_nan_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "descriptors_nan_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "imputation_method": (["mean", "median", "most_frequent"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DESCRIPTORS",)
    FUNCTION = "preprocess"
    CATEGORY = "QSAR/CLASSIFICATION"
    OUTPUT_NODE = True

    def preprocess(self, descriptors, compounds_nan_threshold, descriptors_nan_threshold, imputation_method):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Preprocessed")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(descriptors)
            initial_shape = f"{df.shape[0]}x{df.shape[1]}"
            df, inf_count = replace_inf_classification(df)
            df, compounds_removed = remove_high_nan_rows_classification(df, compounds_nan_threshold)
            df, descriptors_removed = remove_high_nan_cols_classification(df, descriptors_nan_threshold)
            df, values_imputed = impute_missing_values_classification(df, imputation_method)
            if df.empty:
                warning_msg = "⚠️ Warning: Preprocessing resulted in an empty dataset. Check your thresholds."
                return {"ui": {"text": warning_msg}, "result": ("",)}
            final_shape = f"{df.shape[0]}x{df.shape[1]}"
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(descriptors))[0]}_preprocessed.csv")
            df.to_csv(output_file, index=False)
            summary = (
                "========================================\n"
                "🔹 Preprocessing Pipeline Complete! 🔹\n"
                "========================================\n"
                f"📊 Initial Shape: {initial_shape}\n"
                f"✅ Inf Replaced: {inf_count} values\n"
                f"✅ Compounds Removed: {compounds_removed}\n"
                f"✅ Descriptors Removed: {descriptors_removed}\n"
                f"✅ Values Imputed: {values_imputed} ('{imputation_method}')\n"
                f"📊 Final Shape: {final_shape}\n"
                f"💾 Output File: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": summary}, "result": (str(output_file),)}
        except Exception as e:
            error_msg = f"❌ Error in preprocessing pipeline: {e}\n\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": ("",)}

NODE_CLASS_MAPPINGS = {
    "Replace_inf_with_nan_Classification": Replace_inf_with_nan_Classification,
    "Remove_high_nan_compounds_Classification": Remove_high_nan_compounds_Classification,
    "Remove_high_nan_descriptors_Classification": Remove_high_nan_descriptors_Classification,
    "Impute_missing_values_Classification": Impute_missing_values_Classification,
    "Descriptor_preprocessing_Classification": Descriptor_preprocessing_Classification,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Replace_inf_with_nan_Classification": "Replace Inf with NaN",
    "Remove_high_nan_compounds_Classification": "Remove High NaN Compounds",
    "Remove_high_nan_descriptors_Classification": "Remove High NaN Descriptors",
    "Impute_missing_values_Classification": "Impute Missing Values",
    "Descriptor_preprocessing_Classification": "3. Descriptor Preprocessing",
}
