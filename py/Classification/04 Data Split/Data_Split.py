import os
import pandas as pd
from sklearn.model_selection import train_test_split

class QSARDataSplit_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {
                    "default": "./preprocessing_output/preprocessed_data.csv",
                    "multiline": False,
                }),
                "output_dir": ("STRING", {
                    "default": "./split_output",
                    "multiline": False,
                }),
                "test_size": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.05,
                    "max": 0.5,
                    "step": 0.05,
                    "display": "slider",
                }),
                "random_state": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                }),
                "stratify": ("BOOLEAN", {
                    "default": True,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("train_path", "X_test_path", "y_test_path")
    FUNCTION = "execute"
    CATEGORY = "QSAR/CLASSIFICATION"
    OUTPUT_NODE = True

    def execute(
        input_file,
        output_dir="./split_output",
        test_size=0.2,
        random_state=42,
        target_column="Label"
    ):

        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(input_file)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        X = df.drop(columns=[target_column])
        y = df[target_column]
        n_total   = len(df)
        n_class0  = int((y == 0).sum())
        n_class1  = int((y == 1).sum())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
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

        train_c0 = int((y_train == 0).sum())
        train_c1 = int((y_train == 1).sum())
        test_c0  = int((y_test  == 0).sum())
        test_c1  = int((y_test  == 1).sum())
        actual_test_ratio = len(X_test) / n_total

        log_message = (
            "========================================\n"
            "🔹 Data Split Completed! 🔹\n"
            "========================================\n"
            f"📊 Total compounds    : {n_total} (Class 0: {n_class0}, Class 1: {n_class1})\n"
            f"✂️  Split ratio        : {1-actual_test_ratio:.0%} train / {actual_test_ratio:.0%} test\n"
            f"🎲 Random state       : {random_state}\n"
            "----------------------------------------\n"
            f"🏋️  Train set          : {len(X_train)} compounds (Class 0: {train_c0}, Class 1: {train_c1})\n"
            f"🧪 Test set           : {len(X_test)} compounds  (Class 0: {test_c0},  Class 1: {test_c1})\n"
            "----------------------------------------\n"
            f"💾 train_data.csv     : {output_train}\n"
            f"💾 test_data.csv      : {output_test}\n"
            f"💾 X_test.csv         : {output_x_test}\n"
            f"💾 Y_test.csv         : {output_y_test}\n"
            "========================================"
        )

        return {"ui": {"text": log_message}, "result": (str(output_train), str(output_test), str(output_x_test), str(output_y_test))}

NODE_CLASS_MAPPINGS = {
    "QSARDataSplit_Classification": QSARDataSplit_Classification,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QSARDataSplit_Classification": "4. Data Split",
}
