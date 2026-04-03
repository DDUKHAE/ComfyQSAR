import os
import joblib
import numpy as np
import pandas as pd
import multiprocessing
import traceback
import ast
import folder_paths

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LGBMRegressor = None
    LIGHTGBM_AVAILABLE = False


REGRESSION_MODEL_CONFIG = {
    "random_forest": {"model": RandomForestRegressor, "params": ['rf_n_estimators', 'rf_max_depth', 'rf_min_samples_split', 'rf_min_samples_leaf', 'rf_bootstrap']},
    "decision_tree": {"model": DecisionTreeRegressor, "params": ['dt_max_depth', 'dt_min_samples_split', 'dt_min_samples_leaf', 'dt_criterion']},
    "lasso": {"model": Lasso, "params": ['lasso_alpha']},
    "ridge": {"model": Ridge, "params": ['ridge_alpha']},
    "elasticnet": {"model": ElasticNet, "params": ['elastic_alpha', 'elastic_l1_ratio']},
    "svr": {"model": SVR, "params": ['svm_C', 'svm_kernel', 'svm_gamma', 'svm_epsilon']},
}

if XGBOOST_AVAILABLE:
    REGRESSION_MODEL_CONFIG["xgboost"] = {
        "model": XGBRegressor, "params": ['xgb_n_estimators', 'xgb_learning_rate', 'xgb_max_depth', 'xgb_subsample', 'xgb_reg_alpha', 'xgb_reg_lambda'],
        "static_params": {"verbosity": 0}
    }
if LIGHTGBM_AVAILABLE:
    REGRESSION_MODEL_CONFIG["lightgbm"] = {
        "model": LGBMRegressor, "params": ['lgb_n_estimators', 'lgb_learning_rate', 'lgb_max_depth', 'lgb_num_leaves', 'lgb_reg_alpha', 'lgb_reg_lambda']
    }


def parse_param(param_str):
    try:
        parsed = ast.literal_eval(param_str)
        return parsed if isinstance(parsed, list) else [parsed]
    except (ValueError, SyntaxError):
        return []


def setup_regression_pipeline(algorithm, user_params, random_state):
    if algorithm not in REGRESSION_MODEL_CONFIG:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    config = REGRESSION_MODEL_CONFIG[algorithm]
    model_class = config["model"]
    static_params = config.get("static_params", {})
    try:
        model_instance = model_class(random_state=random_state, **static_params)
    except TypeError:
        model_instance = model_class(**static_params)
    pipeline = Pipeline([("scaler", StandardScaler()), ("reg", model_instance)])
    param_grid = {}
    for param_name in config["params"]:
        actual_param = param_name.split('_', 1)[1]
        if param_name in user_params:
            param_grid[f'reg__{actual_param}'] = user_params.get(param_name, [])
    return pipeline, param_grid


def save_regression_results(grid_search, X, output_dir, algorithm):
    best_model = grid_search.best_estimator_
    model_abbr = "".join([s[0].upper() for s in algorithm.split('_')])
    model_path = os.path.join(output_dir, f"Best_Regressor_{model_abbr}.pkl")
    joblib.dump(best_model, model_path)
    descriptors_path = os.path.join(output_dir, "Final_Descriptors.txt")
    with open(descriptors_path, "w") as f:
        f.write("\n".join(X.columns))
    best_params_path = os.path.join(output_dir, f"Best_Hyperparameters_{model_abbr}.txt")
    with open(best_params_path, "w") as f:
        for param, value in grid_search.best_params_.items():
            f.write(f"{param.split('__')[-1]}: {value}\n")
    return model_path, descriptors_path


class Hyperparameter_Grid_Search_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {}),
                "algorithm": (list(REGRESSION_MODEL_CONFIG.keys()),),
                "target_column": ("STRING", {"default": "value"}),
                "advanced": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "num_cores": ("INT", {"default": -1, "min": -1, "max": multiprocessing.cpu_count()}),
                "cv_splits": ("INT", {"default": 5, "min": 3, "max": 10}),
                "verbose": ("INT", {"default": 1, "min": 0, "max": 2}),
                "random_state": ("INT", {"default": 42}),
                #random_forest
                'rf_n_estimators': ("STRING", {"default": "[100, 200, 300]"}),
                'rf_max_depth': ("STRING", {"default": "[None, 10, 20]"}),
                'rf_min_samples_split': ("STRING", {"default": "[2, 5, 10]"}),
                'rf_min_samples_leaf': ("STRING", {"default": "[1, 2, 4]"}),
                'rf_bootstrap': ("STRING", {"default": "[True, False]"}),
                #decision_tree
                'dt_max_depth': ("STRING", {"default": "[None, 10, 20]"}),
                'dt_min_samples_split': ("STRING", {"default": "[2, 5, 10]"}),
                'dt_min_samples_leaf': ("STRING", {"default": "[1, 2, 4]"}),
                'dt_criterion': ("STRING", {"default": "['squared_error']"}),
                #xgboost
                'xgb_n_estimators': ("STRING", {"default": "[100, 200, 300]"}),
                'xgb_learning_rate': ("STRING", {"default": "[0.01, 0.05, 0.1]"}),
                'xgb_max_depth': ("STRING", {"default": "[3, 5, 7]"}),
                'xgb_subsample': ("STRING", {"default": "[0.6, 0.8, 1.0]"}),
                'xgb_reg_alpha': ("STRING", {"default": "[0.1, 1, 10]"}),
                'xgb_reg_lambda': ("STRING", {"default": "[1, 10, 100]"}),
                #lightgbm
                'lgb_n_estimators': ("STRING", {"default": "[100, 200, 300]"}),
                'lgb_learning_rate': ("STRING", {"default": "[0.01, 0.05, 0.1]"}),
                'lgb_num_leaves': ("STRING", {"default": "[20, 31, 40]"}),
                'lgb_max_depth': ("STRING", {"default": "[-1, 5, 10]"}),
                'lgb_reg_alpha': ("STRING", {"default": "[0.1, 1, 10]"}),
                'lgb_reg_lambda': ("STRING", {"default": "[1, 10, 100]"}),           
                #SVM
                'svm_C': ("STRING", {"default": "[0.1, 1, 10, 100]"}),
                'svm_kernel': ("STRING", {"default": "['linear', 'rbf', 'poly']"}),
                'svm_gamma': ("STRING", {"default": "['scale', 'auto']"}),
                'svm_epsilon': ("STRING", {"default": "[0.01, 0.1, 0.5]"}),
                #ridge
                'ridge_alpha': ("STRING", {"default": "[0.01, 0.1, 1, 10, 100]"}),
                #lasso
                'lasso_alpha': ("STRING", {"default": "[0.01, 0.1, 1, 10, 100]"}),
                #elasticnet
                'elastic_alpha': ("STRING", {"default": "[0.01, 0.1, 1, 10, 100]"}),
                'elastic_l1_ratio': ("STRING", {"default": "[0.1, 0.5, 0.9]"})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("MODEL_PATH", "DESCRIPTORS_PATH")
    FUNCTION = "perform_grid_search"
    CATEGORY = "QSAR/REGRESSION"
    OUTPUT_NODE = True

    def perform_grid_search(self, input_file, algorithm, target_column, **kwargs):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Regression_GridSearch")
            os.makedirs(output_dir, exist_ok=True)
            user_params = {k: parse_param(v) for k, v in kwargs.items() if isinstance(v, str)}
            data = pd.read_csv(input_file)
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found.")
            X = data.drop(columns=[target_column])
            y = data[target_column]
            pipeline, param_grid = setup_regression_pipeline(
                algorithm, user_params, random_state=kwargs.get("random_state", 42)
            )
            cv = KFold(n_splits=kwargs.get("cv_splits", 5), shuffle=True, random_state=kwargs.get("random_state", 42))
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring='r2', refit=True,
                verbose=kwargs.get("verbose", 1), n_jobs=kwargs.get("num_cores", -1)
            )
            grid_search.fit(X, y)
            model_path, desc_path = save_regression_results(grid_search, X, output_dir, algorithm)
            best_params_text = "\n".join([f"  - {k.split('__')[-1]}: {v}" for k, v in grid_search.best_params_.items()])
            log_message = (
                "========================================\n"
                "🔹 Grid Search Completed (Regression) 🔹\n"
                "========================================\n"
                f"📌 Method: {algorithm.replace('_', ' ').title()}\n"
                f"🏆 Best CV R2: {grid_search.best_score_:.4f}\n"
                "----------------------------------------\n"
                "⚙️ Best Hyperparameters:\n"
                f"{best_params_text}\n"
                "----------------------------------------\n"
                f"💾 Output File: {os.path.basename(model_path)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(model_path), str(desc_path))}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("", "")}


NODE_CLASS_MAPPINGS = {
    "Hyperparameter_Grid_Search_Regression": Hyperparameter_Grid_Search_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hyperparameter_Grid_Search_Regression": "7. Hyperparameter Tuning & Model Training",
}
