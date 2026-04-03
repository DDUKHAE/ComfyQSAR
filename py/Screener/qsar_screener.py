import os
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = str(CURRENT_DIR)

COMFY_QSAR_ROOT = CURRENT_DIR.parent.parent

SCREENING_DB_DIR = COMFY_QSAR_ROOT / "Screening_DB"
RESULT_BASE_DIR = COMFY_QSAR_ROOT / "screening_results_DB"

SCREENING_DATABASES = {}
if SCREENING_DB_DIR.exists():
    for csv_file in SCREENING_DB_DIR.glob("Des_*.csv"):
        core_name = csv_file.stem.replace("Des_", "")
        
        parts = core_name.split("_")
        db_key = "_".join(parts[:-1]) if len(parts) > 1 else core_name
        
        sdf_file = None
        for f in SCREENING_DB_DIR.glob("*.sdf"):
            if f.stem.lower() == core_name.lower():
                sdf_file = f
                break
        
        if sdf_file:
            SCREENING_DATABASES[db_key] = {
                "csv": str(csv_file),
                "sdf": str(sdf_file),
            }

if not SCREENING_DATABASES:
    print(f"⚠️ [ComfyQSAR] No database files found in folder '{SCREENING_DB_DIR}'.")


class QSARDBScreener:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "PTP1B_prediction_QSAR_model.pkl",
                    "multiline": False,
                }),
                "features_path": ("STRING", {
                    "default": "./selected_features_V3.txt",
                    "multiline": False,
                }),
                "db_name": (list(SCREENING_DATABASES.keys()),),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("csv_path", "sdf_path")
    FUNCTION = "execute"
    CATEGORY = "QSAR/SCREENER"
    OUTPUT_NODE = True

    def execute(self, model_path, features_path, db_name, threshold):
        if not os.path.isabs(model_path):
            model_path = os.path.join(BASE_DIR, model_path)

        if not os.path.isabs(features_path):
            features_path = os.path.join(BASE_DIR, features_path)

        if db_name not in SCREENING_DATABASES:
            raise ValueError(
                f"Unknown DB: '{db_name}'. Available: {list(SCREENING_DATABASES.keys())}"
            )

        screening_csv = SCREENING_DATABASES[db_name]["csv"]
        screening_sdf = SCREENING_DATABASES[db_name]["sdf"]

        for path, label in [
            (model_path, "Model"),
            (features_path, "Features"),
            (screening_csv, f"DB CSV ({db_name})"),
            (screening_sdf, f"DB SDF ({db_name})"),
        ]:
            if not os.path.isfile(path):
                dir_path = os.path.dirname(path)
                print(f"\n[ERROR DEBUG] File not found for '{label}'.")
                print(f" - Search path: {path}")
                
                if os.path.exists(dir_path):
                    print(f" - Folder exists: {dir_path}")
                    print(f" - Files in folder: {os.listdir(dir_path)}")
                else:
                    print(f" - Folder itself does not exist: {dir_path}")
                    parent_of_dir = os.path.dirname(dir_path)
                    if os.path.exists(parent_of_dir):
                        print(f" - Parent folder ({os.path.basename(parent_of_dir)}) content: {os.listdir(parent_of_dir)}")
                
                raise FileNotFoundError(f"{label} file not found: {path}")

        screening_data = pd.read_csv(screening_csv)
        model = joblib.load(model_path)

        with open(features_path, "r") as f:
            selected_descriptors = [line.strip() for line in f if line.strip()]

        missing = [d for d in selected_descriptors if d not in screening_data.columns]
        if missing:
            raise ValueError(f"Missing descriptors in DB: {missing}")

        X_screen = screening_data[selected_descriptors]

        if X_screen.isnull().any().any():
            raise ValueError("The DB contains missing values (NaN). Please check the screening data.")

        if hasattr(model, "predict_proba"):
            predictions = model.predict_proba(X_screen)[:, 1]
            selected_indices = np.where(predictions >= threshold)[0]
        else:
            predictions = model.predict(X_screen)
            selected_indices = np.where(predictions >= threshold)[0]

        output_dir = os.path.join(str(RESULT_BASE_DIR), db_name)
        sdf_subdir = os.path.join(output_dir, "SDF")
        os.makedirs(sdf_subdir, exist_ok=True)

        csv_path = os.path.join(output_dir, f"{db_name}_Screening_Selected_Compounds.csv")
        sdf_path = os.path.join(sdf_subdir, f"{db_name}_Selected_Molecules.sdf")

        selected_df = screening_data.iloc[selected_indices].copy()
        selected_df["prediction_value"] = predictions[selected_indices]
        selected_df.to_csv(csv_path, index=False)

        sdf_supplier = Chem.SDMolSupplier(screening_sdf)
        with Chem.SDWriter(sdf_path) as sdf_writer:
            for idx in selected_indices:
                if idx < len(sdf_supplier):
                    mol = sdf_supplier[int(idx)]
                    if mol is not None:
                        sdf_writer.write(mol)

        log_message = (
            "========================================\n"
            "🔹 Virtual Screening Completed! 🔹\n"
            "========================================\n"
            f"📁 Model: {model_path}\n"
            f"📁 DB: {db_name}\n"
            f"🧬 Descriptors: {features_path}\n"
            f"⚙️  Threshold: {threshold}\n"
            f"🔍 Screened: {len(screening_data)} compounds\n"
            f"✅ Selected: {len(selected_indices)} compounds\n"
            f"📄 CSV: {csv_path}\n"
            f"🧪 SDF: {sdf_path}\n"
            "========================================"
        )

        return {"ui": {"text": log_message}, "result": (csv_path, sdf_path,)}

NODE_CLASS_MAPPINGS = {
    "QSARDBScreener": QSARDBScreener,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QSARDBScreener": "QSAR DB Screener",
}