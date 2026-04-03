import os
import pandas as pd
import numpy as np
import itertools
import traceback
import multiprocessing
from multiprocessing import Pool
from math import comb
import folder_paths

try:
    from tqdm import tqdm
    TQDM_INSTALLED = True
except ImportError:
    TQDM_INSTALLED = False

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_Worker_X = None
_Worker_y = None

def worker_init(X_shared, y_shared):
    global _Worker_X, _Worker_y
    _Worker_X = X_shared
    _Worker_y = y_shared

def evaluate_combination_regression(feature_indices):
    try:
        if _Worker_X is None or _Worker_y is None:
            return feature_indices, -999.0
            
        X_subset = _Worker_X[:, list(feature_indices)]
        pipeline = make_pipeline(StandardScaler(), LinearRegression())
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X_subset, _Worker_y, cv=cv, scoring='r2')
        return feature_indices, scores.mean()
    except Exception:
        return feature_indices, -999.0


class Regression_Feature_Combination_Search:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {}),
                "max_features": ("INT", {"default": 3, "min": 2, "max": 15}),
                "num_cores": ("INT", {"default": -1, "min": -1, "max": multiprocessing.cpu_count()}),
                "top_n": ("INT", {"default": 5, "min": 1, "max": 100}),
                "chunk_size": ("INT", {"default": 2000, "min" : 1}),
                "target_column": ("STRING", {"default": "value"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("BEST_FEATURE_SET",)
    FUNCTION = "find_best_combinations"
    CATEGORY = "QSAR/REGRESSION"
    OUTPUT_NODE = True

    def find_best_combinations(self, input_file, max_features, num_cores, top_n, chunk_size, target_column):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Regression_CombinationSearch")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(input_file)
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found.")
                
            X = df.drop(columns=[target_column]).to_numpy()
            y = df[target_column].to_numpy()
            feature_names = df.drop(columns=[target_column]).columns.tolist()
            
            cores = multiprocessing.cpu_count() if num_cores == -1 else min(num_cores, multiprocessing.cpu_count())
            top_results = []
            best_per_feature_count = {}
            with Pool(cores, initializer=worker_init, initargs=(X, y)) as pool:
                for n_features in range(2, min(max_features + 1, len(feature_names) + 1)):
                    num_combs = comb(X.shape[1], n_features)
                    combinations_iter = itertools.combinations(range(X.shape[1]), n_features)
                    results_iterator = pool.imap_unordered(
                        evaluate_combination_regression, 
                        combinations_iter, 
                        chunksize=chunk_size
                    )
                    
                    if TQDM_INSTALLED:
                        results_iterator = tqdm(results_iterator, total=num_combs, desc=f"Features: {n_features}")
                        
                    for feature_indices, r2 in results_iterator:
                        result = {
                            "Num_Features": len(feature_indices), 
                            "Feature_Indices": feature_indices,
                            "Best_Features": [feature_names[i] for i in feature_indices], 
                            "R2": r2
                        }
                        
                        if n_features not in best_per_feature_count or r2 > best_per_feature_count[n_features]['R2']:
                            best_per_feature_count[n_features] = result
                            
                        if len(top_results) < top_n:
                            top_results.append(result)
                            top_results.sort(key=lambda x: x['R2'], reverse=True)
                        elif r2 > top_results[-1]['R2']:
                            top_results[-1] = result
                            top_results.sort(key=lambda x: x['R2'], reverse=True)

            if not top_results:
                return {"ui": {"text": "❌ No combinations were evaluated."}, "result": ("",)}
                
            best_per_size_df = pd.DataFrame(best_per_feature_count.values())
            best_per_size_path = os.path.join(output_dir, "Best_combination_per_size.csv")
            best_per_size_df.to_csv(best_per_size_path, index=False)
            
            best_overall_result = top_results[0]
            output_file = ""
            
            for i, result in enumerate(top_results, start=1):
                df_selected = df[result["Best_Features"] + [target_column]]
                output_path = os.path.join(output_dir, f"Optimal_Feature_Set_rank{i}_r2{result['R2']:.4f}.csv")
                df_selected.to_csv(output_path, index=False)
                if i == 1:
                    output_file = output_path
                    
            log_message = (
                "========================================\n"
                "🔹 Feature Combination Search Completed! 🔹\n"
                "========================================\n"
                f"🏆 Best CV R2: {best_overall_result['R2']:.4f}\n"
                f"✨ Optimal Features ({best_overall_result['Num_Features']}): {best_overall_result['Best_Features']}\n"
                f"💾 Output File: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
            
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}

NODE_CLASS_MAPPINGS = {
    "Regression_Feature_Combination_Search": Regression_Feature_Combination_Search,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Regression_Feature_Combination_Search": "6. Descriptor Combination",
}

