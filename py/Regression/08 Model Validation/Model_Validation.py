import os
import pandas as pd
import numpy as np
import joblib
import traceback
import folder_paths
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def load_regression_inputs(model_path, x_path, y_path, features_path):
    model = joblib.load(model_path)
    x_test_df = pd.read_csv(x_path)
    y_test_df = pd.read_csv(y_path)
    if "value" in y_test_df.columns:
        y_test = y_test_df["value"].values
    elif y_test_df.shape[1] == 1:
        y_test = y_test_df.iloc[:, 0].values
    else:
        raise ValueError(f"Could not determine target column in {os.path.basename(y_path)}.")
    with open(features_path, "r") as f:
        selected_features = [line.strip() for line in f if line.strip()]
    missing = [ft for ft in selected_features if ft not in x_test_df.columns]
    if missing:
        raise ValueError(f"Missing features in X_test: {', '.join(missing)}")
    return model, x_test_df[selected_features], y_test

def calculate_regression_metrics(model, x_test, y_test):
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    return {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae}, y_pred

def save_regression_results(output_dir, y_test, y_pred, metrics):
    pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    pred_path = os.path.join(output_dir, "Actual_vs_Predicted.csv")
    pred_df.to_csv(pred_path, index=False)
    eval_df = pd.DataFrame({
        "Metric": ["R2", "MSE", "RMSE", "MAE"],
        "Value": [metrics["r2"], metrics["mse"], metrics["rmse"], metrics["mae"]]
    })
    eval_path = os.path.join(output_dir, "Evaluation_Results_ExternalTestSet.csv")
    eval_df.to_csv(eval_path, index=False)
    return eval_path, pred_path

class Model_Validation_Regression:
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
    CATEGORY = "QSAR/REGRESSION"
    OUTPUT_NODE = True

    def validate_model(self, model_path, selected_descriptors_path, X_test_path, y_test_path):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR_Regression_Validation")
            os.makedirs(output_dir, exist_ok=True)
            model, x_test_filtered, y_test = load_regression_inputs(
                model_path, X_test_path, y_test_path, selected_descriptors_path
            )
            metrics, y_pred = calculate_regression_metrics(model, x_test_filtered, y_test)
            eval_path, pred_path = save_regression_results(output_dir, y_test, y_pred, metrics)
            log_message = (
                "========================================\n"
                "🔹 Regression Model Validation Done! 🔹\n"
                "========================================\n"
                f"📌 Model: {os.path.basename(model_path)}\n"
                f"📊 R2 Score: {metrics['r2']:.4f}\n"
                f"📊 MSE: {metrics['mse']:.4f}\n"
                f"📊 RMSE: {metrics['rmse']:.4f}\n"
                f"📊 MAE: {metrics['mae']:.4f}\n"
                f"💾 Output File: {os.path.basename(eval_path)}\n"
                f"💾 Prediction File: {os.path.basename(pred_path)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(eval_path), str(pred_path))}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("", "")}


NODE_CLASS_MAPPINGS = {
    "Model_Validation_Regression": Model_Validation_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Model_Validation_Regression": "8. Model Validation",
}
