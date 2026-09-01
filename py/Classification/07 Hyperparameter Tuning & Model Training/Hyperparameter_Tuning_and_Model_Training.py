import os
import joblib
import numpy as np
import pandas as pd
import multiprocessing
import traceback
import ast
import folder_paths

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LGBMClassifier = None
    LIGHTGBM_AVAILABLE = False

CLASSIFICATION_MODEL_CONFIG = {
    "random_forest": {"model": RandomForestClassifier, "params": ['rf_n_estimators', 'rf_max_depth', 'rf_min_samples_split']},
    "decision_tree": {"model": DecisionTreeClassifier, "params": ['dt_max_depth', 'dt_min_samples_split', 'dt_min_samples_leaf', 'dt_criterion']},
    "logistic": {"model": LogisticRegression, "params": ['lr_C', 'lr_penalty'], "static_params": {"solver": "liblinear", "max_iter": 2000}},
    "lasso": {"model": LogisticRegression, "params": ['lasso_C'], "static_params": {"penalty": "l1", "solver": "liblinear", "max_iter": 2000}},
    "svm": {"model": SVC, "params": ['svm_C', 'svm_kernel', 'svm_gamma'], "static_params": {"probability": True}},
}

if XGBOOST_AVAILABLE:
    CLASSIFICATION_MODEL_CONFIG["xgboost"] = {
        "model": XGBClassifier, "params": ['xgb_n_estimators', 'xgb_learning_rate', 'xgb_max_depth'],
        # n_jobs=1: unlike sklearn's own estimators (RandomForest/DecisionTree/
        # LogisticRegression/SVC), which default to n_jobs=None (single-threaded),
        # XGBoost defaults to using every available core per fit. GridSearchCV's
        # own n_jobs=num_cores already parallelizes across fold/candidate
        # combinations -- leaving this unset means every one of those workers
        # additionally tries to grab all cores for itself, oversubscribing the
        # machine by num_cores x cores instead of just num_cores.
        "static_params": {"eval_metric": "logloss", "use_label_encoder": False, "n_jobs": 1}
    }
if LIGHTGBM_AVAILABLE:
    CLASSIFICATION_MODEL_CONFIG["lightgbm"] = {
        "model": LGBMClassifier, "params": ['lgb_n_estimators', 'lgb_learning_rate', 'lgb_max_depth', 'lgb_subsample', 'lgb_reg_alpha', 'lgb_reg_lambda'],
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

def setup_classification_pipeline(algorithm, user_params, random_state):
    if algorithm not in CLASSIFICATION_MODEL_CONFIG:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    config = CLASSIFICATION_MODEL_CONFIG[algorithm]
    model_class = config["model"]
    static_params = config.get("static_params", {})
    model_instance = model_class(random_state=random_state, **static_params)
    
    use_pipeline = algorithm in ["logistic", "lasso", "svm"]
    if use_pipeline:
        model = Pipeline([("scaler", StandardScaler()), ("clf", model_instance)])
        prefix = "clf__"
    else:
        model = model_instance
        prefix = ""

    param_grid = {}
    for param_name in config["params"]:
        actual_param = param_name.split('_', 1)[1]
        values = user_params.get(param_name, [])
        if algorithm == 'lightgbm' and actual_param == 'max_depth':
            values = [d if d is None or d > 0 else -1 for d in values]
        param_grid[f'{prefix}{actual_param}'] = values
    return model, param_grid

def save_classification_results(grid_search, X, output_dir, algorithm):
    best_model = grid_search.best_estimator_
    model_path = os.path.join(output_dir, f"Best_Classifier_{algorithm}.pkl")
    joblib.dump(best_model, model_path)

    descriptors_path = os.path.join(output_dir, f"Final_Selected_Descriptors_{algorithm}.txt")
    with open(descriptors_path, "w") as f:
        f.write("\n".join(X.columns))

    best_params_path = os.path.join(output_dir, f"Best_Hyperparameters_{algorithm}.txt")
    with open(best_params_path, "w") as f:
        for param, value in grid_search.best_params_.items():
            f.write(f"{param.split('__')[-1]}: {value}\n")

    return model_path, descriptors_path

class Hyperparameter_Grid_Search_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "training_data_path": ("STRING", {"tooltip": "From 6 if descriptor combination was used, otherwise straight from 5."}),
                "algorithm": (list(CLASSIFICATION_MODEL_CONFIG.keys()),),
                "target_column": ("STRING", {"default": "Label"}),
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
                    ["accuracy", "balanced_accuracy", "f1", "roc_auc", "precision", "recall", "mcc"],
                    {"default": "accuracy",
                     "tooltip": "Which CV metric decides the deployed model's hyperparameters "
                                "(GridSearchCV's refit=). Default 'accuracy' matches prior "
                                "behavior. On imbalanced data, accuracy can favor a majority-class "
                                "-biased candidate over one with better balanced_accuracy/F1/MCC -- "
                                "check Metric_Sensitivity_Report_{algorithm}.csv first (always "
                                "generated regardless of this setting) to see whether the choice "
                                "of metric actually changes which candidate wins."}),
                #random_forest
                "rf_n_estimators": ("STRING", {"default": "[50, 100, 300]"}),
                "rf_max_depth": ("STRING", {"default": "[None, 10, 20]"}),
                "rf_min_samples_split": ("STRING", {"default": "[2, 5, 10]"}),
                #decision_tree
                "dt_max_depth": ("STRING", {"default": "[None, 10, 20]"}),
                "dt_min_samples_split": ("STRING", {"default": "[2, 5, 10]"}),
                "dt_min_samples_leaf": ("STRING", {"default": "[1, 2, 4]"}),
                "dt_criterion": ("STRING", {"default": "['gini', 'entropy']"}),
                #logistic
                "lr_C": ("STRING", {"default": "[0.01, 0.1, 1, 10, 100]"}),
                "lr_penalty": ("STRING", {"default": "['l2']"}),
                #lasso
                "lasso_C": ("STRING", {"default": "[0.01, 0.1, 1, 10, 100]"}),
                #svm
                "svm_C": ("STRING", {"default": "[0.1, 1, 10, 100]"}),
                "svm_kernel": ("STRING", {"default": "['rbf', 'linear']"}),
                "svm_gamma": ("STRING", {"default": "['scale', 'auto']"}),
                #xgboost
                "xgb_n_estimators": ("STRING", {"default": "[100, 200, 300]"}),
                "xgb_learning_rate": ("STRING", {"default": "[0.01, 0.05, 0.1]"}),
                "xgb_max_depth": ("STRING", {"default": "[3, 5, 7, None]"}),
                #lightgbm
                "lgb_n_estimators": ("STRING", {"default": "[100, 200, 300]"}),
                "lgb_learning_rate": ("STRING", {"default": "[0.01, 0.05, 0.1]"}),
                "lgb_max_depth": ("STRING", {"default": "[3, 5, 7]"}),
                "lgb_subsample": ("STRING", {"default": "[0.6, 0.8, 1.0]"}),
                "lgb_reg_alpha": ("STRING", {"default": "[0.1, 1, 10]"}),
                "lgb_reg_lambda": ("STRING", {"default": "[1, 10, 100]"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("TRAINED_MODEL", "SELECTED_DESCRIPTOR_LIST")
    FUNCTION = "perform_grid_search"
    CATEGORY = "QSAR/1. CLASSIFICATION"
    OUTPUT_NODE = True

    def perform_grid_search(self, training_data_path, algorithm, target_column, **kwargs):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Classification", "07_Model_Training")
            os.makedirs(output_dir, exist_ok=True)
            user_params = {k: parse_param(v) for k, v in kwargs.items() if isinstance(v, str)}
            data = pd.read_csv(training_data_path)
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found.")
            metadata_cols = [c for c in ("Name", "SMILES", target_column) if c in data.columns]
            X = data.drop(columns=metadata_cols)
            y = data[target_column]
            unique_labels = set(pd.unique(y))
            if unique_labels != {0, 1}:
                raise ValueError(
                    f"Classification target column '{target_column}' must be binary-encoded as 0/1 "
                    f"(found: {sorted(unique_labels, key=str)}). Encode binary labels as 0/1 before "
                    "running this node."
                )

            random_state = kwargs.get("random_state", 42)
            requested_cores = kwargs.get("num_cores", 1)
            cv_splits = kwargs.get("cv_splits", 5)

            cpu_count = multiprocessing.cpu_count()
            num_cores = cpu_count if requested_cores == -1 else max(1, min(requested_cores, cpu_count))
            cores_warning = ""
            if requested_cores == -1 and cpu_count > 8:
                cores_warning = (
                    f"⚠️ num_cores=-1 requests all {cpu_count} CPU cores -- for typical "
                    "QSAR-sized grids/datasets, per-worker startup overhead can make this "
                    "slower than a small explicit core count (e.g. 2-4). Consider "
                    "setting num_cores explicitly if this run feels slow.\n"
                )

            # NOTE: training_data_path is expected to already be the training-only
            # split produced by "3. Data Split" -> "4. Descriptor Preprocessing"
            # -- this node no longer carves out its own held-out subset (that
            # duplicated/bypassed the Data Split node's real 20% test set).
            # Reporting is now purely via
            # cross-validation, matching the Regression Hyperparameter node.
            model, param_grid = setup_classification_pipeline(
                algorithm, user_params, random_state=random_state
            )
            n_candidates = 1
            for v in param_grid.values():
                n_candidates *= max(1, len(v))
            total_fits = n_candidates * cv_splits

            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
            scoring = {
                'accuracy': 'accuracy',
                'balanced_accuracy': 'balanced_accuracy',
                'f1': make_scorer(f1_score, zero_division=0),
                'roc_auc': 'roc_auc',
                'precision': make_scorer(precision_score, zero_division=0),
                'recall': make_scorer(recall_score, zero_division=0),
                'mcc': make_scorer(matthews_corrcoef),
            }
            selection_metric = kwargs.get("selection_metric", "accuracy")
            if selection_metric not in scoring:
                raise ValueError(
                    f"selection_metric='{selection_metric}' is not one of the tracked metrics "
                    f"({sorted(scoring)})."
                )
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring=scoring, refit=selection_metric,
                verbose=kwargs.get("verbose", 1), n_jobs=num_cores
            )
            grid_search.fit(X, y)

            best_idx = grid_search.best_index_
            cv_results = grid_search.cv_results_
            metric_names = list(scoring.keys())

            def cv_mean_std(metric, idx=best_idx):
                return cv_results[f'mean_test_{metric}'][idx], cv_results[f'std_test_{metric}'][idx]

            acc_mean, acc_std = cv_mean_std('accuracy')
            f1_mean, f1_std = cv_mean_std('f1')
            auc_mean, auc_std = cv_mean_std('roc_auc')
            prec_mean, prec_std = cv_mean_std('precision')
            rec_mean, rec_std = cv_mean_std('recall')
            bacc_mean, bacc_std = cv_mean_std('balanced_accuracy')
            mcc_mean, mcc_std = cv_mean_std('mcc')

            # Metric-sensitivity report: `selection_metric` (default
            # "accuracy", user-overridable above) decides which candidate is
            # actually deployed -- but GridSearchCV already scores every
            # candidate on every metric in `scoring` at no extra fitting
            # cost, so it's cheap to also show, for EACH tracked metric,
            # which candidate *that* metric alone would have picked and how
            # that candidate performs across all metrics. This complements
            # (not replaces) the selection_metric choice: the report shows
            # whether the choice of metric would have mattered at all --
            # most informative precisely on the imbalanced datasets where
            # accuracy-based selection is riskiest -- so a user can decide
            # whether to override selection_metric away from its default,
            # and costs nothing (reuses already-computed CV scores).
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

            n_class0, n_class1 = int((y == 0).sum()), int((y == 1).sum())
            class_balance_note = (
                f"📊 Class distribution: 0={n_class0} ({100 * n_class0 / len(y):.1f}%), "
                f"1={n_class1} ({100 * n_class1 / len(y):.1f}%)\n"
            )
            disagreement_note = (
                f"✅ All {len(metric_names)} tracked metrics select the same hyperparameter "
                f"candidate as the deployed ({selection_metric}-refit) model.\n"
                if n_metrics_disagreeing == 0 else
                f"⚠️ {n_metrics_disagreeing}/{len(metric_names)} metrics would select a DIFFERENT "
                f"hyperparameter candidate than the deployed ({selection_metric}-refit) model -- "
                "especially worth checking on imbalanced data. See "
                f"{os.path.basename(sensitivity_path)} for what each metric would have chosen, and "
                "the selection_metric input if you want to deploy a different candidate.\n"
            )

            model_path, desc_path = save_classification_results(
                grid_search, X, output_dir, algorithm
            )

            best_params_text = "\n".join([f"  - {k.split('__')[-1]}: {v}" for k, v in grid_search.best_params_.items()])
            log_message = (
                f"🖥️ Requested cores: {requested_cores}, Effective cores: {num_cores} (CPU count: {cpu_count})\n"
                f"📊 Using {cv_splits}-Fold Stratified Cross Validation ({n_candidates} candidate(s), {total_fits} total fits)\n"
                f"🔍 Starting GridSearchCV for {algorithm}...\n"
                f"{cores_warning}"
                "========================================\n"
                "🔹 Classification Model Training Complete 🔹\n"
                "========================================\n"
                f"📌 Best Algorithm: {algorithm} (selected by: {selection_metric})\n"
                f"{class_balance_note}"
                f"📊 CV Accuracy: {acc_mean:.4f} ± {acc_std:.4f}\n"
                f"📊 CV Balanced Accuracy: {bacc_mean:.4f} ± {bacc_std:.4f}\n"
                f"📊 CV F1 Score: {f1_mean:.4f} ± {f1_std:.4f}\n"
                f"📊 CV ROC AUC: {auc_mean:.4f} ± {auc_std:.4f}\n"
                f"📊 CV Precision: {prec_mean:.4f} ± {prec_std:.4f}\n"
                f"📊 CV Recall: {rec_mean:.4f} ± {rec_std:.4f}\n"
                f"📊 CV MCC: {mcc_mean:.4f} ± {mcc_std:.4f}\n"
                f"{disagreement_note}"
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
                f"💾 Model: {os.path.basename(model_path)}\n"
                f"💾 Metric Sensitivity Report: {os.path.basename(sensitivity_path)} (what each of the "
                f"{len(metric_names)} tracked metrics would have selected)\n"
                "========================================\n"
                "⚙️ Best Hyperparameters:\n"
                f"{best_params_text}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(model_path), str(desc_path))}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("", "")}

NODE_CLASS_MAPPINGS = {
    "Hyperparameter_Grid_Search_Classification": Hyperparameter_Grid_Search_Classification,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hyperparameter_Grid_Search_Classification": "7. Hyperparameter Tuning & Model Training",
}
