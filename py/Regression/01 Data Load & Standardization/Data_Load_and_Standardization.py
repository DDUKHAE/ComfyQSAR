import os
import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDWriter
import folder_paths
import traceback
from typing import List, Tuple, Dict, Any, Optional

# standardize_mol lives in code/py/_shared/ -- the ONE module shared across
# tracks/nodes in this codebase, because training (here) and screening
# (Screener/custom_user_screener.py) must standardize molecules identically
# or the same physical compound could be represented differently (or
# dropped by one path and not the other) depending on which stage sees it.
_PY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SHARED_DIR = os.path.join(_PY_DIR, "_shared")
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)
from chem_standardize import standardize_mol  # noqa: E402

def validate_file_path(file_path: str, supported_extensions: Tuple[str, ...]) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.lower().endswith(supported_extensions):
        raise ValueError(f"Unsupported file format. Use one of {supported_extensions}.")

class Data_Loader_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "smiles_file_path": ("STRING", {"placeholder": "path/to/smiles.smi or .csv"}),
                "value_file_path": ("STRING", {"placeholder": "path/to/values.csv"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("MERGED_DATA",)
    FUNCTION = "load_data"
    CATEGORY = "QSAR/2. REGRESSION/OTHERS"
    OUTPUT_NODE = True

    def load_data(self, smiles_file_path: str, value_file_path: str) -> Dict[str, Any]:
        try:
            validate_file_path(smiles_file_path, ('.smi', '.csv'))
            validate_file_path(value_file_path, ('.csv',))

            output_dir = os.path.join(folder_paths.get_output_directory(), "Regression", "01_Data_Load_and_Standardization")
            os.makedirs(output_dir, exist_ok=True)

            smiles_df = pd.read_csv(smiles_file_path, header=None if smiles_file_path.endswith('.smi') else 'infer')
            value_df = pd.read_csv(value_file_path)

            if smiles_file_path.endswith('.smi'):
                smiles_df.columns = ['SMILES']
            else:
                smiles_col = next((c for c in smiles_df.columns if 'smiles' in c.lower()), smiles_df.columns[0])
                smiles_df = smiles_df.rename(columns={smiles_col: 'SMILES'})

            if len(smiles_df) != len(value_df):
                raise ValueError(f"SMILES count ({len(smiles_df)}) != value count ({len(value_df)}).")

            value_col = next((c for c in value_df.columns if c.lower() not in ['index', 'unnamed']), value_df.columns[0])
            merged_df = pd.DataFrame({'SMILES': smiles_df['SMILES'].values, 'value': value_df[value_col].values})
            output_file = os.path.join(output_dir, "merged_smiles_values.csv")
            merged_df.to_csv(output_file, index=False)

            log_message = (
                "========================================\n"
                "🔹 Regression Data Loaded! 🔹\n"
                "========================================\n"
                f"✅ Compounds: {len(merged_df)}\n"
                f"📊 Value Column: '{value_col}'\n"
                f"💾 Output File: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}"}, "result": ("",)}

class Standardization_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "merged_data": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STANDARDIZED_DATA",)
    FUNCTION = "standardize_data"
    CATEGORY = "QSAR/2. REGRESSION/OTHERS"
    OUTPUT_NODE = True

    def standardize_data(self, merged_data: str) -> Dict[str, Any]:
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Regression", "01_Data_Load_and_Standardization")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(merged_data)
            initial_count = len(df)

            report_rows = []
            kept_rows = []
            n_fragment_changed = n_charge_changed = n_tautomer_changed = 0
            n_rejected_unparseable = n_rejected_metal = n_rejected_empty = 0

            for _, row in df.iterrows():
                original_smiles = str(row['SMILES']) if pd.notna(row['SMILES']) else None
                mol = Chem.MolFromSmiles(original_smiles) if original_smiles else None
                std_mol, info = standardize_mol(mol)

                if info["status"] == "rejected":
                    if info["reason"] == "unparseable":
                        n_rejected_unparseable += 1
                    elif info["reason"] == "metal_only":
                        n_rejected_metal += 1
                    else:
                        n_rejected_empty += 1
                    report_rows.append({
                        "original_smiles": original_smiles, "standardized_smiles": "",
                        "status": "rejected", "reason": info["reason"], "changed": False,
                        "fragment_changed": False, "charge_changed": False, "tautomer_changed": False,
                    })
                    continue

                standardized_smiles = Chem.MolToSmiles(std_mol)
                n_fragment_changed += int(info["fragment_changed"])
                n_charge_changed += int(info["charge_changed"])
                n_tautomer_changed += int(info["tautomer_changed"])
                report_rows.append({
                    "original_smiles": original_smiles, "standardized_smiles": standardized_smiles,
                    "status": "ok", "reason": "",
                    "changed": info["fragment_changed"] or info["charge_changed"] or info["tautomer_changed"],
                    "fragment_changed": info["fragment_changed"],
                    "charge_changed": info["charge_changed"], "tautomer_changed": info["tautomer_changed"],
                })
                kept_rows.append({"SMILES": standardized_smiles, "value": row["value"]})

            filtered_df = pd.DataFrame(kept_rows).reset_index(drop=True)
            n_duplicates = int(filtered_df["SMILES"].duplicated().sum()) if len(filtered_df) else 0

            output_file = os.path.join(output_dir, "standardized_compounds.csv")
            filtered_df.to_csv(output_file, index=False)

            report_file = os.path.join(output_dir, "standardization_report.csv")
            pd.DataFrame(report_rows).to_csv(report_file, index=False)

            # Only surface a sub-line when that category actually has a
            # nonzero count -- a clean run should read as one short line,
            # not a wall of "0" breakdowns the user has to scan past.
            n_rejected_total = n_rejected_unparseable + n_rejected_metal + n_rejected_empty
            n_changed_total = n_fragment_changed + n_charge_changed + n_tautomer_changed
            body_lines = [f"✅ {len(filtered_df)}/{initial_count} kept"]
            if n_rejected_total:
                parts = [f"{name} {v}" for name, v in (
                    ("unparseable", n_rejected_unparseable),
                    ("metal-only", n_rejected_metal),
                    ("empty-after-standardization", n_rejected_empty),
                ) if v]
                body_lines.append(f"   ⚠️  Rejected {n_rejected_total}: {', '.join(parts)}")
            if n_changed_total:
                parts = [f"{name} {v}" for name, v in (
                    ("fragment/salt", n_fragment_changed),
                    ("charge", n_charge_changed),
                    ("tautomer", n_tautomer_changed),
                ) if v]
                body_lines.append(f"   🧪 Changed {n_changed_total}: {', '.join(parts)}")
            if n_duplicates:
                body_lines.append(f"   ⚠️  {n_duplicates} duplicate SMILES kept (not removed)")

            rel_dir = os.path.relpath(output_dir, folder_paths.get_output_directory())
            log_message = (
                "========================================\n"
                "🔹 Standardization Completed! 🔹\n"
                "========================================\n"
                + "\n".join(body_lines) + "\n"
                f"📁 Directory: {rel_dir}{os.sep}\n"
                f"💾 Output: {os.path.basename(output_file)}\n"
                f"📋 Report: {os.path.basename(report_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}

class Load_and_Standardize_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "smiles_file_path": ("STRING", {"placeholder": "path/to/smiles.smi or .csv"}),
                "value_file_path": ("STRING", {"placeholder": "path/to/values.csv"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STANDARDIZED_DATA",)
    FUNCTION = "run"
    CATEGORY = "QSAR/2. REGRESSION"
    OUTPUT_NODE = True

    def run(self, smiles_file_path: str, value_file_path: str):
        loader = Data_Loader_Regression()
        result = loader.load_data(smiles_file_path, value_file_path)
        if not result['result'][0]:
            return result
        std = Standardization_Regression()
        return std.standardize_data(result['result'][0])


NODE_CLASS_MAPPINGS = {
    "Data_Loader_Regression": Data_Loader_Regression,
    "Standardization_Regression": Standardization_Regression,
    "Load_and_Standardize_Regression": Load_and_Standardize_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Data_Loader_Regression": "Data Loader",
    "Standardization_Regression": "Standardization",
    "Load_and_Standardize_Regression": "1. Data Load & Standardization",
}
