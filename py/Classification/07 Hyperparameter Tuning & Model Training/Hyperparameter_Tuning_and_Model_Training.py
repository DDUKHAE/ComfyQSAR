import os
import joblib
import numpy as np
import pandas as pd
import multiprocessing
import traceback
import ast
import folder_paths

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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
    "logistic": {"model": LogisticRegression, "params": ['lr_C', 'lr_penalty'], "static_params": {"max_iter": 2000}},
    "lasso": {"model": LogisticRegression, "params": ['lasso_C'], "static_params": {"penalty": "l1", "solver": "liblinear", "max_iter": 2000}},
    "svm": {"model": SVC, "params": ['svm_C', 'svm_kernel', 'svm_gamma'], "static_params": {"probability": True}},
}

if XGBOOST_AVAILABLE:
    CLASSIFICATION_MODEL_CONFIG["xgboost"] = {
        "model": XGBClassifier, "params": ['xgb_n_estimators', 'xgb_learning_rate', 'xgb_max_depth'],
        "static_params": {"eval_metric": "logloss", "use_label_encoder": False}
    }
if LIGHTGBM_AVAILABLE:
    CLASSIFICATION_MODEL_CONFIG["lightgbm"] = {
        "model": LGBMClassifier, "params": ['lgb_n_estimators', 'lgb_learning_rate', 'lgb_max_depth', 'lgb_subsample', 'lgb_reg_alpha', 'lgb_reg_lambda']
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

def save_classification_results(grid_search, X_train, X_test, y_test, output_dir, algorithm, target_column):
    best_model = grid_search.best_estimator_
    model_abbr = "".join([s[0].upper() for s in algorithm.split('_')])
    model_path = os.path.join(output_dir, f"Best_Classifier_{model_abbr}.pkl")
    joblib.dump(best_model, model_path)
    
    descriptors_path = os.path.join(output_dir, "Final_Selected_Descriptors.txt")
    with open(descriptors_path, "w") as f:
        f.write("\n".join(X_train.columns))
        
    best_params_path = os.path.join(output_dir, f"Best_Hyperparameters_{model_abbr}.txt")
    with open(best_params_path, "w") as f:
        for param, value in grid_search.best_params_.items():
            f.write(f"{param.split('__')[-1]}: {value}\n")
            
    x_test_path = os.path.join(output_dir, "X_test.csv")
    y_test_path = os.path.join(output_dir, "y_test.csv")
    pd.DataFrame(X_test).to_csv(x_test_path, index=False)
    pd.DataFrame(y_test, columns=[target_column]).to_csv(y_test_path, index=False)
    
    return model_path, descriptors_path

class Hyperparameter_Grid_Search_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {}),
                "algorithm": (list(CLASSIFICATION_MODEL_CONFIG.keys()),),
                "target_column": ("STRING", {"default": "Label"}),
                "advanced": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "num_cores": ("INT", {"default": -1, "min": -1, "max": multiprocessing.cpu_count()}),
                "cv_splits": ("INT", {"default": 5, "min": 3, "max": 10}),
                "verbose": ("INT", {"default": 1, "min": 0, "max": 2}),
                "random_state": ("INT", {"default": 42}),
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
    RETURN_NAMES = ("MODEL_PATH", "DESCRIPTORS_PATH")
    FUNCTION = "perform_grid_search"
    CATEGORY = "QSAR/CLASSIFICATION"
    OUTPUT_NODE = True

    def perform_grid_search(self, input_file, algorithm, target_column, test_size = 0.2, **kwargs):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_GridSearch")
            os.makedirs(output_dir, exist_ok=True)
            user_params = {k: parse_param(v) for k, v in kwargs.items() if isinstance(v, str)}
            data = pd.read_csv(input_file)
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found.")
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            random_state = kwargs.get("random_state", 42)
            num_cores = kwargs.get("num_cores", -1)
            cv_splits = kwargs.get("cv_splits", 5)
            
            if num_cores == -1:
                num_cores = max(1, multiprocessing.cpu_count())

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            model, param_grid = setup_classification_pipeline(
                algorithm, user_params, random_state=random_state
            )

            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
            scoring = {
                'accuracy': 'accuracy', 
                'f1': make_scorer(f1_score, zero_division=0), 
                'roc_auc': 'roc_auc',
                'precision': make_scorer(precision_score, zero_division=0),
                'recall': make_scorer(recall_score, zero_division=0)
            }
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring=scoring, refit='accuracy',
                verbose=kwargs.get("verbose", 1), n_jobs=num_cores
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            predictions = best_model.predict(X_test)
            eval_results = {
                "accuracy": accuracy_score(y_test, predictions),
                "f1_score": f1_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
            }

            model_path, desc_path = save_classification_results(
                grid_search, X_train, X_test, y_test, output_dir, algorithm, target_column
            )
            
            best_params_text = "\n".join([f"  - {k.split('__')[-1]}: {v}" for k, v in grid_search.best_params_.items()])
            log_message = (
                f"🖥️ Using {num_cores} CPU cores for parallel processing!\n"
                f"📊 Using {cv_splits}-Fold Cross Validation\n"
                f"🔍 Starting GridSearchCV for {algorithm}...\n"
                "========================================\n"
                "🔹 Classification Model Training Complete 🔹\n"
                "========================================\n"
                f"📌 Best Algorithm: {algorithm}\n"
                f"📊 Accuracy: {eval_results['accuracy']:.4f}\n"
                f"📊 F1 Score: {eval_results['f1_score']:.4f}\n"
                f"📊 ROC AUC: {eval_results['roc_auc']:.4f}\n"
                f"📊 Precision: {eval_results['precision']:.4f}\n"
                f"📊 Recall: {eval_results['recall']:.4f}\n"
                f"💾 Saved Model: {model_path}\n"
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
