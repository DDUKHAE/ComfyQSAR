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
        # n_jobs=1: unlike sklearn's own estimators (RandomForest/DecisionTree/
        # LinearRegression/SVR), which default to n_jobs=None (single-threaded),
        # XGBoost defaults to using every available core per fit. GridSearchCV's
        # own n_jobs=num_cores already parallelizes across fold/candidate
        # combinations -- leaving this unset means every one of those workers
        # additionally tries to grab all cores for itself, oversubscribing the
        # machine by num_cores x cores instead of just num_cores.
        "static_params": {"verbosity": 0, "n_jobs": 1}
    }
if LIGHTGBM_AVAILABLE:
    REGRESSION_MODEL_CONFIG["lightgbm"] = {
        "model": LGBMRegressor, "params": ['lgb_n_estimators', 'lgb_learning_rate', 'lgb_max_depth', 'lgb_num_leaves', 'lgb_reg_alpha', 'lgb_reg_lambda'],
        # Same reasoning as xgboost above -- LightGBM also defaults to using
        # every available core per fit.
        "static_params": {"n_jobs": 1}
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
    model_path = os.path.join(output_dir, f"Best_Regressor_{algorithm}.pkl")
    joblib.dump(best_model, model_path)
    descriptors_path = os.path.join(output_dir, f"Final_Descriptors_{algorithm}.txt")
    with open(descriptors_path, "w") as f:
        f.write("\n".join(X.columns))
    best_params_path = os.path.join(output_dir, f"Best_Hyperparameters_{algorithm}.txt")
    with open(best_params_path, "w") as f:
        for param, value in grid_search.best_params_.items():
            f.write(f"{param.split('__')[-1]}: {value}\n")
    return model_path, descriptors_path


class Hyperparameter_Grid_Search_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "training_data_path": ("STRING", {"tooltip": "From 6 if descriptor combination was used, otherwise straight from 5."}),
                "algorithm": (list(REGRESSION_MODEL_CONFIG.keys()),),
                "target_column": ("STRING", {"default": "value"}),
                "advanced": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "num_cores": ("INT", {"default": 1, "min": -1, "max": multiprocessing.cpu_count(),
                                       "tooltip": "1 (default) runs in-process with no worker-pool startup cost -- "
                                                  "recommended for typical QSAR-sized grids/datasets, where loky's "
                                                  "per-worker boot/unpickle overhead outweighs any speedup. Only "
                                                  "raise this for genuinely large grids/datasets."}),
                "cv_splits": ("INT", {"default": 5, "min": 3, "max": 10}),
                "verbose": ("INT", {"default": 1, "min": 0, "max": 2}),
                "random_state": ("INT", {"default": 42}),
                "selection_metric": (
                    ["r2", "neg_mean_squared_error", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
                    {"default": "r2",
                     "tooltip": "Which CV metric decides the deployed model's hyperparameters "
                                "(GridSearchCV's refit=). Default 'r2' matches prior behavior. "
                                "R2 and the error-based metrics are not guaranteed to rank hyperparameter "
                                "candidates identically across folds -- check "
                                "Metric_Sensitivity_Report_{algorithm}.csv first (always generated "
                                "regardless of this setting) to see whether the choice of metric actually "
                                "changes which candidate wins."}),
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
    RETURN_NAMES = ("TRAINED_MODEL", "SELECTED_DESCRIPTOR_LIST")
    FUNCTION = "perform_grid_search"
    CATEGORY = "QSAR/2. REGRESSION"
    OUTPUT_NODE = True

    def perform_grid_search(self, training_data_path, algorithm, target_column, **kwargs):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Regression", "07_Model_Training")
            os.makedirs(output_dir, exist_ok=True)
            user_params = {k: parse_param(v) for k, v in kwargs.items() if isinstance(v, str)}
            data = pd.read_csv(training_data_path)
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found.")
            metadata_cols = [c for c in ("Name", "SMILES", target_column) if c in data.columns]
            X = data.drop(columns=metadata_cols)
            y = data[target_column]
            pipeline, param_grid = setup_regression_pipeline(
                algorithm, user_params, random_state=kwargs.get("random_state", 42)
            )
            cv_splits = kwargs.get("cv_splits", 5)
            cv = KFold(n_splits=cv_splits, shuffle=True, random_state=kwargs.get("random_state", 42))

            cpu_count = multiprocessing.cpu_count()
            requested_cores = kwargs.get("num_cores", 1)
            num_cores = cpu_count if requested_cores == -1 else max(1, min(requested_cores, cpu_count))
            cores_warning = ""
            if requested_cores == -1 and cpu_count > 8:
                cores_warning = (
                    f"⚠️ num_cores=-1 requests all {cpu_count} CPU cores -- for typical "
                    "QSAR-sized grids/datasets, per-worker startup overhead can make this "
                    "slower than a small explicit core count (e.g. 2-4). Consider "
                    "setting num_cores explicitly if this run feels slow.\n"
                )
            n_candidates = 1
            for v in param_grid.values():
                n_candidates *= max(1, len(v))
            total_fits = n_candidates * cv_splits

            scoring = {
                'r2': 'r2',
                'neg_mean_squared_error': 'neg_mean_squared_error',
                'neg_root_mean_squared_error': 'neg_root_mean_squared_error',
                'neg_mean_absolute_error': 'neg_mean_absolute_error',
            }
            selection_metric = kwargs.get("selection_metric", "r2")
            if selection_metric not in scoring:
                raise ValueError(
                    f"selection_metric='{selection_metric}' is not one of the tracked metrics "
                    f"({sorted(scoring)})."
                )
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring=scoring, refit=selection_metric,
                verbose=kwargs.get("verbose", 1), n_jobs=num_cores
            )
            grid_search.fit(X, y)
            model_path, desc_path = save_regression_results(grid_search, X, output_dir, algorithm)

            best_idx = grid_search.best_index_
            cv_results = grid_search.cv_results_
            metric_names = list(scoring.keys())

            def cv_mean_std(metric, idx=best_idx):
                return cv_results[f'mean_test_{metric}'][idx], cv_results[f'std_test_{metric}'][idx]

            r2_mean, r2_std = cv_mean_std('r2')
            mse_mean, mse_std = cv_mean_std('neg_mean_squared_error')
            rmse_mean, rmse_std = cv_mean_std('neg_root_mean_squared_error')
            mae_mean, mae_std = cv_mean_std('neg_mean_absolute_error')

            # Metric-sensitivity report: `selection_metric` (default "r2",
            # user-overridable above) decides which candidate is actually
            # deployed -- but GridSearchCV already scores every candidate on
            # every metric in `scoring` at no extra fitting cost, so it's
            # cheap to also show, for EACH tracked metric, which candidate
            # *that* metric alone would have picked and how that candidate
            # performs across all metrics. R2 and the error-based metrics are
            # not guaranteed to rank candidates identically across folds
            # (fold-specific SS_tot for R2 vs. plain residual magnitude for
            # MSE/RMSE/MAE), so this shows whether the choice of metric would
            # have mattered at all -- most informative on small or
            # heavy-tailed datasets where R2 is least stable across folds --
            # so a user can decide whether to override selection_metric away
            # from its default, and it costs nothing (reuses already-computed
            # CV scores).
            sensitivity_rows = []
            for metric in metric_names:
                metric_idx = int(np.argmax(cv_results[f'mean_test_{metric}']))
                row = {
                    "candidate_metric": metric,
                    "selected_candidate_index": metric_idx,
                    "selected_candidate_params": str(cv_results['params'][metric_idx]),
                    "same_candidate_as_deployed_model": bool(metric_idx == best_idx),
                }
                for m in metric_names:
                    row[f"{m}_mean"] = cv_results[f'mean_test_{m}'][metric_idx]
                    row[f"{m}_std"] = cv_results[f'std_test_{m}'][metric_idx]
                sensitivity_rows.append(row)
            sensitivity_df = pd.DataFrame(sensitivity_rows)
            sensitivity_path = os.path.join(output_dir, f"Metric_Sensitivity_Report_{algorithm}.csv")
            sensitivity_df.to_csv(sensitivity_path, index=False)
            n_metrics_disagreeing = int((~sensitivity_df["same_candidate_as_deployed_model"]).sum())

            disagreement_note = (
                f"✅ All {len(metric_names)} tracked metrics select the same hyperparameter "
                f"candidate as the deployed ({selection_metric}-refit) model.\n"
                if n_metrics_disagreeing == 0 else
                f"⚠️ {n_metrics_disagreeing}/{len(metric_names)} metrics would select a DIFFERENT "
                f"hyperparameter candidate than the deployed ({selection_metric}-refit) model -- "
                "especially worth checking on small or heavy-tailed datasets. See "
                f"{os.path.basename(sensitivity_path)} for what each metric would have chosen, and "
                "the selection_metric input if you want to deploy a different candidate.\n"
            )

            best_params_text = "\n".join([f"  - {k.split('__')[-1]}: {v}" for k, v in grid_search.best_params_.items()])
            log_message = (
                "========================================\n"
                "🔹 Grid Search Completed (Regression) 🔹\n"
                "========================================\n"
                f"🖥️ Requested cores: {requested_cores}, Effective cores: {num_cores} (CPU count: {cpu_count})\n"
                f"📊 Using {cv_splits}-Fold Cross Validation ({n_candidates} candidate(s), {total_fits} total fits)\n"
                f"{cores_warning}"
                f"📌 Method: {algorithm.replace('_', ' ').title()} (selected by: {selection_metric})\n"
                f"📊 CV R²: {r2_mean:.4f} ± {r2_std:.4f}\n"
                f"📊 CV MSE: {-mse_mean:.4f} ± {mse_std:.4f}\n"
                f"📊 CV RMSE: {-rmse_mean:.4f} ± {rmse_std:.4f}\n"
                f"📊 CV MAE: {-mae_mean:.4f} ± {mae_std:.4f}\n"
                f"{disagreement_note}"
                "----------------------------------------\n"
                "⚙️ Best Hyperparameters:\n"
                f"{best_params_text}\n"
                "----------------------------------------\n"
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
                f"💾 Model: {os.path.basename(model_path)}\n"
                f"💾 Metric Sensitivity Report: {os.path.basename(sensitivity_path)} (what each of the "
                f"{len(metric_names)} tracked metrics would have selected)\n"
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
