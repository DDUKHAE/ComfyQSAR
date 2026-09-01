import os
import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDWriter, AllChem
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

CLASSIFICATION_SUPPORTED_EXTENSIONS = ('.smi', '.csv', '.sdf')

def validate_file_path(file_path: str, supported_extensions: Tuple[str, ...]) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.lower().endswith(supported_extensions):
        raise ValueError(f"Unsupported file format. Use one of {supported_extensions}.")


def read_molecules(file_path: str) -> List[Optional[Chem.Mol]]:
    if file_path.endswith('.sdf'):
        suppl = Chem.SDMolSupplier(file_path, removeHs=False, strictParsing=False)
        return [mol for mol in suppl if mol is not None]
    elif file_path.endswith(('.smi', '.csv')):
        try:
            if file_path.endswith('.smi'):
                # .smi files are headerless (one SMILES per line) -- reading
                # with the default header='infer' silently drops the first
                # compound by mistaking it for a column name.
                df = pd.read_csv(file_path, header=None, skip_blank_lines=True)
                df.columns = ['SMILES']
            else:
                df = pd.read_csv(file_path, skip_blank_lines=True)
                smiles_col = next((col for col in df.columns if 'smiles' in col.lower()), None)
                if smiles_col:
                    df.rename(columns={smiles_col: 'SMILES'}, inplace=True)
                else:
                    df.rename(columns={df.columns[0]: 'SMILES'}, inplace=True)
        except Exception as e:
            raise IOError(f"Failed to read CSV/SMI file {os.path.basename(file_path)}: {e}")
        mols = []
        for smiles in df['SMILES']:
            if pd.notna(smiles):
                mols.append(Chem.MolFromSmiles(str(smiles)))
        return mols
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def write_molecules(mols: List[Chem.Mol], output_path: str) -> None:
    if output_path.endswith('.sdf'):
        with SDWriter(output_path) as writer:
            for mol in mols:
                if mol is not None:
                    if mol.GetNumConformers() == 0:
                        AllChem.Compute2DCoords(mol)
                    writer.write(mol)
    elif output_path.endswith('.csv'):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        pd.DataFrame(smiles_list, columns=['SMILES']).to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format for {output_path}")


class Data_Loader_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_file_path": ("STRING", {"placeholder": "path/to/positive.sdf, .csv, or .smi"}),
                "negative_file_path": ("STRING", {"placeholder": "path/to/negative.sdf, .csv, or .smi"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("LOADED_POSITIVE", "LOADED_NEGATIVE")
    FUNCTION = "load_data"
    CATEGORY = "QSAR/1. CLASSIFICATION/OTHERS"
    OUTPUT_NODE = True

    def load_data(self, positive_file_path: str, negative_file_path: str) -> Dict[str, Any]:
        try:
            validate_file_path(positive_file_path, CLASSIFICATION_SUPPORTED_EXTENSIONS)
            validate_file_path(negative_file_path, CLASSIFICATION_SUPPORTED_EXTENSIONS)

            pos_count = len(read_molecules(positive_file_path))
            neg_count = len(read_molecules(negative_file_path))
            total_count = pos_count + neg_count

            log_message = (
                "========================================\n"
                "🔹 Classification Data Loaded! 🔹\n"
                "========================================\n"
                f"✅ Positive Compounds: {pos_count}\n"
                f"✅ Negative Compounds: {neg_count}\n"
                f"📊 Total: {total_count} molecules\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (positive_file_path, negative_file_path)}

        except (FileNotFoundError, ValueError, IOError) as e:
            error_msg = f"❌ Error checking input files: {e}"
            return {"ui": {"text": [error_msg]}, "result": ("", "")}


class Standardization_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_path": ("STRING", {"forceInput": True}),
                "negative_path": ("STRING", {"forceInput": True})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("STANDARDIZED_POSITIVE", "STANDARDIZED_NEGATIVE")
    FUNCTION = "standardize_data"
    CATEGORY = "QSAR/1. CLASSIFICATION/OTHERS"
    OUTPUT_NODE = True

    def standardize_data(self, positive_path: str, negative_path: str) -> Dict[str, Any]:
        output_dir = os.path.join(folder_paths.get_output_directory(), "Classification", "01_Data_Load_and_Standardization")
        os.makedirs(output_dir, exist_ok=True)

        def process_and_standardize(file_path: str, output_name: str) -> Dict[str, Any]:
            ext = '.sdf' if file_path.endswith('.sdf') else '.csv'
            output_file = os.path.join(output_dir, f"{output_name}_standardized{ext}")
            report_file = os.path.join(output_dir, f"{output_name}_standardization_report.csv")

            mols = read_molecules(file_path)
            stats = {"rejected_unparseable": 0, "rejected_metal": 0, "rejected_empty": 0,
                     "fragment_changed": 0, "charge_changed": 0, "tautomer_changed": 0}
            report_rows = []
            standardized_mols = []

            for mol in mols:
                original_smiles = Chem.MolToSmiles(mol) if mol is not None else None
                std_mol, info = standardize_mol(mol)

                if info["status"] == "rejected":
                    key = {"unparseable": "rejected_unparseable",
                           "metal_only": "rejected_metal"}.get(info["reason"], "rejected_empty")
                    stats[key] += 1
                    report_rows.append({
                        "original_smiles": original_smiles, "standardized_smiles": "",
                        "status": "rejected", "reason": info["reason"], "changed": False,
                        "fragment_changed": False, "charge_changed": False, "tautomer_changed": False,
                    })
                    continue

                stats["fragment_changed"] += int(info["fragment_changed"])
                stats["charge_changed"] += int(info["charge_changed"])
                stats["tautomer_changed"] += int(info["tautomer_changed"])
                standardized_smiles = Chem.MolToSmiles(std_mol)
                report_rows.append({
                    "original_smiles": original_smiles, "standardized_smiles": standardized_smiles,
                    "status": "ok", "reason": "",
                    "changed": info["fragment_changed"] or info["charge_changed"] or info["tautomer_changed"],
                    "fragment_changed": info["fragment_changed"],
                    "charge_changed": info["charge_changed"], "tautomer_changed": info["tautomer_changed"],
                })
                standardized_mols.append(std_mol)

            write_molecules(standardized_mols, output_file)
            pd.DataFrame(report_rows).to_csv(report_file, index=False)

            smiles_list = [Chem.MolToSmiles(m) for m in standardized_mols]
            n_duplicates = int(pd.Series(smiles_list).duplicated().sum()) if smiles_list else 0

            return {
                "output_file": output_file, "report_file": report_file,
                "kept_count": len(standardized_mols), "stats": stats, "n_duplicates": n_duplicates,
            }

        try:
            pos = process_and_standardize(positive_path, "positive")
            neg = process_and_standardize(negative_path, "negative")

            def fmt_block(label: str, r: Dict[str, Any]) -> str:
                # Only surface a sub-line when that category actually has a
                # nonzero count -- a clean run should read as one short line,
                # not a wall of "0" breakdowns the user has to scan past.
                s = r["stats"]
                n_rejected = s["rejected_unparseable"] + s["rejected_metal"] + s["rejected_empty"]
                n_changed = s["fragment_changed"] + s["charge_changed"] + s["tautomer_changed"]
                initial = r["kept_count"] + n_rejected
                lines = [f"{label}: {r['kept_count']}/{initial} kept"]
                if n_rejected:
                    parts = [f"{name} {v}" for name, v in (
                        ("unparseable", s["rejected_unparseable"]),
                        ("metal-only", s["rejected_metal"]),
                        ("empty-after-standardization", s["rejected_empty"]),
                    ) if v]
                    lines.append(f"   ⚠️  Rejected {n_rejected}: {', '.join(parts)}")
                if n_changed:
                    parts = [f"{name} {v}" for name, v in (
                        ("fragment/salt", s["fragment_changed"]),
                        ("charge", s["charge_changed"]),
                        ("tautomer", s["tautomer_changed"]),
                    ) if v]
                    lines.append(f"   🧪 Changed {n_changed}: {', '.join(parts)}")
                if r["n_duplicates"]:
                    lines.append(f"   ⚠️  {r['n_duplicates']} duplicate SMILES kept (not removed)")
                return "\n".join(lines) + "\n"

            rel_dir = os.path.relpath(output_dir, folder_paths.get_output_directory())
            log_message = (
                "========================================\n"
                "🔹 Standardization Completed! 🔹\n"
                "========================================\n"
                f"{fmt_block('✅ Positive', pos)}"
                f"{fmt_block('✅ Negative', neg)}"
                f"📁 Directory: {rel_dir}{os.sep}\n"
                f"💾 Outputs: {os.path.basename(pos['output_file'])}, {os.path.basename(neg['output_file'])}\n"
                f"📋 Reports: {os.path.basename(pos['report_file'])}, {os.path.basename(neg['report_file'])}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (pos["output_file"], neg["output_file"])}

        except Exception as e:
            error_msg = f"❌ Standardization Error: {e}\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": ("", "")}


class Load_and_Standardize_Classification:
    """Combined node for loading and standardizing classification data."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_file_path": ("STRING", {"placeholder": "path/to/positive.sdf, .csv, or .smi"}),
                "negative_file_path": ("STRING", {"placeholder": "path/to/negative.sdf, .csv, or .smi"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("STANDARDIZED_POSITIVE", "STANDARDIZED_NEGATIVE")
    FUNCTION = "run"
    CATEGORY = "QSAR/1. CLASSIFICATION"
    OUTPUT_NODE = True

    def run(self, positive_file_path: str, negative_file_path: str) -> Dict[str, Any]:
        loader = Data_Loader_Classification()
        loader_result = loader.load_data(positive_file_path, negative_file_path)
        if not all(loader_result['result']):
            return loader_result
        standardizer = Standardization_Classification()
        return standardizer.standardize_data(positive_file_path, negative_file_path)

NODE_CLASS_MAPPINGS = {
    "Data_Loader_Classification": Data_Loader_Classification,
    "Standardization_Classification": Standardization_Classification,
    "Load_and_Standardize_Classification": Load_and_Standardize_Classification,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Data_Loader_Classification": "Data Loader",
    "Standardization_Classification": "Standardization",
    "Load_and_Standardize_Classification": "1. Data Load & Standardization",
}
