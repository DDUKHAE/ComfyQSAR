import os
import numpy as np
import pandas as pd
import traceback
import folder_paths
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor

def train_lightgbm_regression(args):
    X, y, i, n_estimators, max_depth, learning_rate, min_child_samples, min_split_gain = args
    model = LGBMRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        min_child_samples=min_child_samples, min_split_gain=min_split_gain,
        random_state=i, n_jobs=1, verbosity=-1
    )
    model.fit(X, y)
    return model.feature_importances_

class LightGBMFeatureSelectionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "descriptor_data_path": ("STRING", {"forceInput": True, "tooltip": "Training data only -- from 4 directly, or from another 5.1/5.2 step already applied to it (05.1/05.2 can be combined in any order). The full dataset leaks the hold-out set into selection."}),
            "target_column": ("STRING", {"default": "value"}),
            "threshold_percentile": ("INT", {"default": 90, "min": 1, "max": 99, "step": 1}),
            "n_estimators": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10}),
            "learning_rate": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01}),
            "max_depth": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1}),
            "n_iterations": ("INT", {"default": 100, "min": 1, "max": 200, "step": 1}),
            "min_child_samples": ("INT", {"default": 20, "min": 1, "max": 100}),
            "min_split_gain": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "num_cores": ("INT", {"default": -1, "min": -1, "max": cpu_count(), "step": 1}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("SELECTED_DESCRIPTOR_DATA",)
    FUNCTION = "select_features"
    CATEGORY = "QSAR/2. REGRESSION/5. Descriptor Optimization/5.2 Model-based Selection"
    OUTPUT_NODE = True

    def select_features(self, descriptor_data_path, target_column, threshold_percentile, n_estimators,
                        learning_rate, max_depth, n_iterations, min_child_samples, min_split_gain, num_cores):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Regression", "05_Descriptor_Optimization", "Model_Based")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(descriptor_data_path)
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found.")
            X = df.drop(columns=[c for c in (target_column, "Name", "SMILES") if c in df.columns]).select_dtypes(include=[np.number])
            y = df[target_column]
            initial_feature_count = X.shape[1]
            args_list = [(X, y, i, n_estimators, max_depth, learning_rate, min_child_samples, min_split_gain)
                         for i in range(n_iterations)]
            available_cores = -1 if num_cores == -1 else max(1, min(num_cores, cpu_count()))
            cores_label = "-1 (all available cores)" if available_cores == -1 else str(available_cores)
            results = Parallel(
                n_jobs=available_cores,
                backend="loky",
                pre_dispatch="2*n_jobs",
            )(
                delayed(train_lightgbm_regression)(args)
                for args in args_list
            )
            feature_importances = np.mean(np.stack(results), axis=0)
            threshold_value = np.percentile(feature_importances, threshold_percentile)
            selected_indices = np.where(feature_importances >= threshold_value)[0]
            selected_columns = X.columns[selected_indices].tolist()
            if not selected_columns:
                selected_columns = [X.columns[np.argmax(feature_importances)]]
            X_new = X[selected_columns]
            final_feature_count = len(selected_columns)
            selected_features_df = X_new.copy()
            selected_features_df[target_column] = y.reset_index(drop=True)
            output_file = os.path.join(output_dir, f"features_lgbm_{initial_feature_count}_to_{final_feature_count}.csv")
            selected_features_df.to_csv(output_file, index=False)
            log_message = (
                "========================================\n"
                "🔹 LightGBM Feature Selection Completed! 🔹\n"
                "========================================\n"
                f"📌 Method: LightGBM (Regression)\n"
                f"📊 Initial Features: {initial_feature_count}\n"
                f"📉 Selected Features: {final_feature_count}\n"
                f"🗑️ Removed Features: {initial_feature_count - final_feature_count}\n"
                f"⚙️ min_child_samples={min_child_samples}, min_split_gain={min_split_gain}\n"
                f"🖥️ Parallel Cores: {cores_label}\n"
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
                f"💾 Output: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}

NODE_CLASS_MAPPINGS = {
    "LightGBMFeatureSelection": LightGBMFeatureSelectionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LightGBMFeatureSelection": "5.2 LightGBM Selection",
}
