import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class QSARDataSplit_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {"default": ""}),
                "output_dir": ("STRING", {"default": "./split_output"}),
                "test_size": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.5, "step": 0.05}),
                "random_state": ("INT", {"default": 42, "min": 0, "max": 9999}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("train_path", "X_test_path", "y_test_path")
    FUNCTION = "execute"
    CATEGORY = "QSAR/REGRESSION"
    OUTPUT_NODE = True

    def split_dataset_regression(
        input_file,
        output_dir="./split_output",
        test_size=0.2,
        random_state=42,
        target_column="value"
    ):

        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(input_file)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        n_total = len(df)
        y_mean  = float(y.mean())
        y_std   = float(y.std())
        y_min   = float(y.min())
        y_max   = float(y.max())
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )
        train_df = X_train.copy()
        train_df[target_column] = y_train.values

        test_df = X_test.copy()
        test_df[target_column] = y_test.values

        output_train  = os.path.join(output_dir, "train_data.csv")
        output_test   = os.path.join(output_dir, "test_data.csv")
        output_x_test = os.path.join(output_dir, "X_test.csv")
        output_y_test = os.path.join(output_dir, "Y_test.csv")

        train_df.to_csv(output_train,  index=False)
        test_df.to_csv(output_test,   index=False)
        X_test.to_csv(output_x_test,  index=False)
        y_test.to_frame(target_column).to_csv(output_y_test, index=False)

        actual_test_ratio = len(X_test) / n_total

        log_message = (
            "========================================\n"
            "🔹 Data Split (Regression) Completed! 🔹\n"
            "========================================\n"
            f"📊 Total compounds    : {n_total}\n"
            f"🎯 Target column      : {target_column}\n"
            f"📈 Target stats       : mean={y_mean:.3f}, std={y_std:.3f}, "
            f"min={y_min:.3f}, max={y_max:.3f}\n"
            f"✂️  Split ratio        : {1-actual_test_ratio:.0%} train / {actual_test_ratio:.0%} test\n"
            f"🎲 Random state       : {random_state}\n"
            "----------------------------------------\n"
            f"🏋️  Train set          : {len(X_train)} compounds  "
            f"(mean={float(y_train.mean()):.3f}, std={float(y_train.std()):.3f})\n"
            f"🧪 Test set           : {len(X_test)} compounds   "
            f"(mean={float(y_test.mean()):.3f}, std={float(y_test.std()):.3f})\n"
            "----------------------------------------\n"
            f"💾 train_data.csv     : {output_train}\n"
            f"💾 test_data.csv      : {output_test}\n"
            f"💾 X_test.csv         : {output_x_test}\n"
            f"💾 Y_test.csv         : {output_y_test}\n"
            "========================================"
        )

        return {"ui": {"text": log_message}, "result": (output_train, output_x_test, output_y_test)}


NODE_CLASS_MAPPINGS = {
    "QSARDataSplit_Regression": QSARDataSplit_Regression
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QSARDataSplit_Regression": "4. Data Split"
}
