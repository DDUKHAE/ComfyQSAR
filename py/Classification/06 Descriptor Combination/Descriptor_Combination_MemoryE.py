import os
import pandas as pd
import numpy as np
import itertools
import multiprocessing
from multiprocessing import Pool
import traceback
import folder_paths
from math import comb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

try:
    from tqdm import tqdm
    TQDM_INSTALLED = True
except ImportError:
    TQDM_INSTALLED = False

_WORKER_X = None
_WORKER_Y = None

def worker_init(X_shared, y_shared):
    global _WORKER_X, _WORKER_Y
    _WORKER_X = X_shared
    _WORKER_Y = y_shared

def evaluate_classification(feature_indices):
    try:
        if _WORKER_X is None or _WORKER_Y is None:
            raise RuntimeError("Classification worker was not initialized.")

        X_subset = _WORKER_X[:, list(feature_indices)]
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver="liblinear"),
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(
            pipeline,
            X_subset,
            _WORKER_Y,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        ).mean()
        return feature_indices, score
    except Exception as e:
        return feature_indices, float("-inf")

class Feature_Combination_Search:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required" : {
                "input_file" : ("STRING", {"default" : ""}),
                "max_features" : ("INT", {"default" : 5, "min" : 1, "step": 1}),
                "num_cores" : ("INT", {"default" : multiprocessing.cpu_count(), "max" : multiprocessing.cpu_count(), "min" : 1, "step": 1}),
                "top_n" : ("INT", {"default" : 3, "min" : 1, "step" : 1}),
                "chunk_size" : ("INT", {"default" : 2000}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("BEST OPTIMAL FEATURES",)
    FUNCTION = "descriptor_combination_classification_MemoryE"
    CATEGORY = "QSAR/CLASSIFICATION"
    OUTPUT_NODE = True

    def descriptor_combination_classification_MemoryE(self, input_file, max_features, num_cores, top_n, chunk_size):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_CombinationSearch")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(input_file)
            
            if "Label" not in df.columns:
                raise ValueError("Target column 'Label' not found in the dataset.")
                
            X = df.drop(columns=["Label"]).to_numpy()
            y = df["Label"].to_numpy()
            feature_names = df.drop(columns=["Label"]).columns.tolist()
            
            top_results = []
            best_per_feature_count = {}

            pool = None

            if num_cores == 1:
                worker_init(X, y)
            else:
                context = multiprocessing.get_context("spawn")
                pool = context.Pool(
                    num_cores,
                    initializer=worker_init,
                    initargs=(X, y),
                )
            
            try:
                for n_features in range(2, min(max_features + 1, len(feature_names) + 1)):
                    num_combs = comb(X.shape[1], n_features)
                    combinations_iter = itertools.combinations(range(X.shape[1]), n_features)
                    if pool is None:
                        results_iterator = map(
                            evaluate_classification,
                            combinations_iter,
                        )
                    else:
                        results_iterator = pool.imap_unordered(
                            evaluate_classification,
                            combinations_iter,
                            chunksize=chunk_size,
                        )
                    
                    if TQDM_INSTALLED:
                        results_iterator = tqdm(results_iterator, total=num_combs, desc=f"Features: {n_features}")
                        
                    for feature_indices, acc in results_iterator:
                        result = {
                            "Num_Features": len(feature_indices), 
                            "Feature_Indices": feature_indices,
                            "Best_Features": [feature_names[i] for i in feature_indices], 
                            "Accuracy": acc
                        }
                        
                        if n_features not in best_per_feature_count or acc > best_per_feature_count[n_features]['Accuracy']:
                            best_per_feature_count[n_features] = result
                            
                        if len(top_results) < top_n:
                            top_results.append(result)
                            top_results.sort(key=lambda x: x['Accuracy'], reverse=True)
                        elif acc > top_results[-1]['Accuracy']:
                            top_results[-1] = result
                            top_results.sort(key=lambda x: x['Accuracy'], reverse=True)
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()
                
            if not top_results:
                return {"ui": {"text": "❌ No combinations were evaluated."}, "result": ("",)}
                
            best_per_size_df = pd.DataFrame(best_per_feature_count.values())
            best_per_size_path = os.path.join(output_dir, "Best_combination_per_size_MemoryE.csv")
            best_per_size_df.to_csv(best_per_size_path, index=False)
            
            best_overall_result = top_results[0]
            output_file = ""
            
            for i, result in enumerate(top_results, start=1):
                df_selected = df[result["Best_Features"] + ["Label"]]
                output_path = os.path.join(output_dir, f"Optimal_Feature_Set_rank{i}_acc{result['Accuracy']:.4f}_MemoryE.csv")
                df_selected.to_csv(output_path, index=False)
                if i == 1:
                    output_file = output_path
                    
            log_message = (
                "========================================\n"
                "🔹 Feature Combination Search (Memory Efficient) Completed! 🔹\n"
                "========================================\n"
                f"🏆 Best CV Accuracy: {best_overall_result['Accuracy']:.4f}\n"
                f"✨ Optimal Features ({best_overall_result['Num_Features']}): {best_overall_result['Best_Features']}\n"
                f"💾 Top Ranked Set: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
            
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}

NODE_CLASS_MAPPINGS = {
    "Feature_Combination_Search_Classification": Feature_Combination_Search,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Feature_Combination_Search_Classification": "6. Descriptor Combination",
}