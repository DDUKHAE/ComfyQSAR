import os
import pandas as pd
import numpy as np
import joblib
import traceback
import folder_paths
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

def load_classification_inputs(model_path, x_path, y_path, features_path):
    model = joblib.load(model_path)
    x_test_df = pd.read_csv(x_path)
    y_test_df = pd.read_csv(y_path)
    if "Label" in y_test_df.columns:
        y_test = y_test_df["Label"].values
    elif y_test_df.shape[1] == 1:
        y_test = y_test_df.iloc[:, 0].values
    else:
        raise ValueError(f"Could not determine target column in {os.path.basename(y_path)}.")
    with open(features_path, "r") as f:
        selected_features = [line.strip() for line in f if line.strip()]
    missing = [ft for ft in selected_features if ft not in x_test_df.columns]
    if missing:
        raise ValueError(f"The following features are missing from X_test: {', '.join(missing)}")
    return model, x_test_df[selected_features], y_test

def calculate_classification_metrics(model, x_test, y_test):
    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": None, "specificity": None
    }
    try:
        y_proba = model.predict_proba(x_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    except Exception:
        pass
    if len(set(y_test)) == 2:
        try:
            metrics["specificity"] = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        except Exception:
            pass
    return metrics, y_pred

def save_classification_results(output_dir, y_test, y_pred, metrics):
    pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    pred_path = os.path.join(output_dir, "Actual_vs_Predicted.csv")
    pred_df.to_csv(pred_path, index=False)
    eval_data = {
        "Metric": ["Accuracy", "F1-Score", "ROC-AUC", "Precision", "Recall (Sensitivity)", "Specificity"],
        "Value": [metrics["accuracy"], metrics["f1_score"], metrics["roc_auc"],
                  metrics["precision"], metrics["recall"], metrics["specificity"]]
    }
    eval_df = pd.DataFrame(eval_data)
    eval_path = os.path.join(output_dir, "Evaluation_Results_ExternalTestSet.csv")
    eval_df.to_csv(eval_path, index=False)
    return eval_path, pred_path

class Model_Validation_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {}),
                "selected_descriptors_path": ("STRING", {}),
                "X_test_path": ("STRING", {}),
                "y_test_path": ("STRING", {}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("EVALUATION_PATH", "PREDICTION_PATH")
    FUNCTION = "validate_model"
    CATEGORY = "QSAR/CLASSIFICATION"
    OUTPUT_NODE = True

    def validate_model(self, model_path, selected_descriptors_path, X_test_path, y_test_path):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Classification_Validation")
            os.makedirs(output_dir, exist_ok=True)
            model, x_test_filtered, y_test = load_classification_inputs(
                model_path, X_test_path, y_test_path, selected_descriptors_path
            )
            metrics, y_pred = calculate_classification_metrics(model, x_test_filtered, y_test)
            eval_path, pred_path = save_classification_results(output_dir, y_test, y_pred, metrics)
            log_lines = [
                "========================================",
                "🔹 Classification Model Validation Done! 🔹",
                "========================================",
                f"📌 Model: {os.path.basename(model_path)}",
                f"🏆 Accuracy: {metrics['accuracy']:.4f}",
                f"🏆 F1 Score: {metrics['f1_score']:.4f}",
            ]
            if metrics['roc_auc'] is not None:
                log_lines.append(f"📊 ROC-AUC: {metrics['roc_auc']:.4f}")
            else:
                log_lines.append("📊 ROC-AUC: Not Available")
            log_lines.extend([
                f"📊 Precision: {metrics['precision']:.4f}",
                f"📊 Recall (Sensitivity): {metrics['recall']:.4f}",
            ])
            if metrics['specificity'] is not None:
                log_lines.append(f"📊 Specificity: {metrics['specificity']:.4f}")
            else:
                log_lines.append("📊 Specificity: Not Available")
            log_lines.extend([
                f"💾 Output File: {os.path.basename(eval_path)}",
                f"💾 Prediction File: {os.path.basename(pred_path)}",
                "========================================"
            ])
            return {"ui": {"text": "\n".join(log_lines)}, "result": (str(eval_path), str(pred_path))}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("", "")}

NODE_CLASS_MAPPINGS = {
    "Model_Validation_Classification": Model_Validation_Classification,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Model_Validation_Classification": "8. Model Validation",
}
