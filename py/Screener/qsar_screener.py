import json
import os
import numpy as np
import pandas as pd
import joblib
import folder_paths
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
                "trained_model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "From 7.",
                }),
                "descriptor_list_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "From 7. The .txt name list, not a CSV.",
                }),
                "preprocessing_recipe_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "From 4. Imputation values are training-fitted and never re-fit here.",
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
    RETURN_NAMES = ("SCREENING_RESULTS", "SELECTED_MOLECULES")
    FUNCTION = "execute"
    CATEGORY = "QSAR/3. SCREENER"
    OUTPUT_NODE = True

    def execute(self, trained_model_path, descriptor_list_path, preprocessing_recipe_path, db_name, threshold):
        if not os.path.isabs(trained_model_path):
            trained_model_path = os.path.join(BASE_DIR, trained_model_path)

        if not os.path.isabs(descriptor_list_path):
            descriptor_list_path = os.path.join(BASE_DIR, descriptor_list_path)

        if not os.path.isabs(preprocessing_recipe_path):
            preprocessing_recipe_path = os.path.join(BASE_DIR, preprocessing_recipe_path)

        if db_name not in SCREENING_DATABASES:
            raise ValueError(
                f"Unknown DB: '{db_name}'. Available: {list(SCREENING_DATABASES.keys())}"
            )

        screening_csv = SCREENING_DATABASES[db_name]["csv"]
        screening_sdf = SCREENING_DATABASES[db_name]["sdf"]

        for path, label in [
            (trained_model_path, "Model"),
            (descriptor_list_path, "Descriptor list"),
            (preprocessing_recipe_path, "Preprocessing recipe"),
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
        model = joblib.load(trained_model_path)

        with open(descriptor_list_path, "r") as f:
            selected_descriptors = [line.strip() for line in f if line.strip()]

        with open(preprocessing_recipe_path, "r") as f:
            recipe = json.load(f)
        imputer_statistics = recipe["imputer_statistics"]

        missing = [d for d in selected_descriptors if d not in screening_data.columns]
        if missing:
            raise ValueError(f"Missing descriptors in DB: {missing}")

        missing_stats = [d for d in selected_descriptors if d not in imputer_statistics]
        if missing_stats:
            raise ValueError(
                f"Recipe has no stored imputer statistic for {len(missing_stats)} selected "
                f"descriptor(s): {missing_stats[:10]}. Was this recipe produced by the same "
                "training run as descriptor_list_path?"
            )

        X_screen = screening_data[selected_descriptors].copy()

        # A descriptor missing for most compounds in this DB indicates a
        # systemic descriptor-generation failure for that column (e.g. the
        # 2026-08 ASINEX corruption: 791/1444 PaDEL columns NaN for all
        # 10,177 compounds, from a PaDEL/CDK concurrency fault), not
        # ordinary sparse missingness -- imputing it would silently replace
        # an entire column with one constant. Hard-fail instead of
        # imputing when a column's missing fraction is this high; genuinely
        # sparse missingness (a handful of compounds per column) is imputed
        # below using the training-fitted statistic, the same principle
        # QSARCustomUserScreener already applies to user-uploaded SDFs.
        systemic_failure_threshold = 0.5
        missing_fraction_by_col = X_screen.isnull().mean(axis=0)
        systemic_cols = missing_fraction_by_col[missing_fraction_by_col > systemic_failure_threshold]
        if len(systemic_cols) > 0:
            detail = ", ".join(f"{c} ({frac:.1%} missing)" for c, frac in systemic_cols.items())
            raise ValueError(
                f"{len(systemic_cols)} selected descriptor(s) are missing for more than "
                f"{systemic_failure_threshold:.0%} of compounds in '{db_name}': {detail}. This "
                "usually indicates a systemic descriptor-generation failure for that column "
                "rather than ordinary sparse missingness, and should be investigated (check the "
                "PaDEL log for this database) rather than imputed over."
            )

        for col in selected_descriptors:
            X_screen[col] = X_screen[col].fillna(imputer_statistics[col])

        if hasattr(model, "predict_proba"):
            predictions = model.predict_proba(X_screen)[:, 1]
            selected_indices = np.where(predictions >= threshold)[0]
        else:
            predictions = model.predict(X_screen)
            selected_indices = np.where(predictions >= threshold)[0]

        output_dir = os.path.join(folder_paths.get_output_directory(), "Screening", "Database_Screening", db_name)
        sdf_subdir = os.path.join(output_dir, "SDF")
        os.makedirs(sdf_subdir, exist_ok=True)

        csv_path = os.path.join(output_dir, f"{db_name}_Screening_Selected_Compounds.csv")
        sdf_path = os.path.join(sdf_subdir, f"{db_name}_Selected_Molecules.sdf")

        # SMILES is reattached from screening_sdf by row position (Des_*.csv
        # and the paired .sdf were built from the same PaDEL run in the same
        # row order, the same alignment assumption the SDF-writer loop below
        # already relies on) -- Des_*.csv itself never carries a SMILES
        # column (PaDEL output is Name + descriptors only), but a Screening-
        # Candidate Applicability Domain (Tanimoto) node needs one to work
        # on these candidates at all. Mirrors the identical fix already
        # applied in QSARCustomUserScreener (_preprocess_descriptors).
        sdf_supplier = Chem.SDMolSupplier(screening_sdf)
        smiles_by_position = {
            int(i): (Chem.MolToSmiles(sdf_supplier[int(i)]) if i < len(sdf_supplier) and sdf_supplier[int(i)] is not None else None)
            for i in selected_indices
        }

        selected_df = screening_data.iloc[selected_indices].copy()
        selected_df.insert(1, "SMILES", [smiles_by_position[int(idx)] for idx in selected_indices])
        selected_df["prediction_value"] = predictions[selected_indices]
        selected_df.to_csv(csv_path, index=False)

        with Chem.SDWriter(sdf_path) as sdf_writer:
            for idx in selected_indices:
                if idx < len(sdf_supplier):
                    mol = sdf_supplier[int(idx)]
                    if mol is not None:
                        sdf_writer.write(mol)

        # trained_model_path/descriptor_list_path are user-supplied inputs that can live
        # anywhere on disk, so the full path is meaningful there; csv_path/
        # sdf_path are this node's OWN outputs under a fixed, predictable
        # structure, so only the relative-from-output-root part is shown.
        rel_dir = os.path.relpath(output_dir, folder_paths.get_output_directory())
        log_message = (
            "========================================\n"
            "🔹 Virtual Screening Completed! 🔹\n"
            "========================================\n"
            f"📁 Model: {trained_model_path}\n"
            f"📁 DB: {db_name}\n"
            f"🧬 Descriptors: {descriptor_list_path}\n"
            f"⚙️  Threshold: {threshold}\n"
            f"🔍 Screened: {len(screening_data)} compounds\n"
            f"✅ Selected: {len(selected_indices)} compounds\n"
            f"📁 Directory: {rel_dir}{os.sep}\n"
            f"💾 Selected Compounds: {os.path.basename(csv_path)}\n"
            f"💾 Molecules: SDF/{os.path.basename(sdf_path)}\n"
            "========================================"
        )

        return {"ui": {"text": log_message}, "result": (csv_path, sdf_path,)}

NODE_CLASS_MAPPINGS = {
    "QSARDBScreener": QSARDBScreener,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QSARDBScreener": "External Screening (Database)",
}
