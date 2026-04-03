import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDWriter
import folder_paths
import traceback
from typing import List, Tuple, Dict, Any, Optional

METAL_IONS = {
    'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
    'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
    'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Cs', 'Ba',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
    'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Th', 'Pa', 'U'
}

def validate_file_path(file_path: str, supported_extensions: Tuple[str, ...]) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.lower().endswith(supported_extensions):
        raise ValueError(f"Unsupported file format. Use one of {supported_extensions}.")

def filter_molecule(mol: Optional[Chem.Mol]) -> bool:
    if mol is None:
        return False
    atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if atom_symbols and atom_symbols.issubset(METAL_IONS):
        return False
    if len(Chem.GetMolFrags(mol)) > 1:
        return False
    return True

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
    CATEGORY = "QSAR/REGRESSION/OTHERS"
    OUTPUT_NODE = True

    def load_data(self, smiles_file_path: str, value_file_path: str) -> Dict[str, Any]:
        try:
            validate_file_path(smiles_file_path, ('.smi', '.csv'))
            validate_file_path(value_file_path, ('.csv',))

            output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR/DataLoad")
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
                f"✅ Compounds: {len(merged_df)+1}\n"
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
    CATEGORY = "QSAR/REGRESSION/OTHERS"
    OUTPUT_NODE = True

    def standardize_data(self, merged_data: str) -> Dict[str, Any]:
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR/Standardization")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(merged_data)
            initial_count = len(df)
            valid_rows = []
            for _, row in df.iterrows():
                mol = Chem.MolFromSmiles(str(row['SMILES'])) if pd.notna(row['SMILES']) else None
                if filter_molecule(mol):
                    valid_rows.append(row)
            filtered_df = pd.DataFrame(valid_rows).reset_index(drop=True)
            output_file = os.path.join(output_dir, "standardized_compounds.csv")
            filtered_df.to_csv(output_file, index=False)
            log_message = (
                "========================================\n"
                "🔹 Standardization Completed! 🔹\n"
                "========================================\n"
                f"✅ Initial: {initial_count+1}, Filtered: {len(filtered_df)+1}\n"
                f"🗑️ Removed: {(initial_count+1) - (len(filtered_df)+1)}\n"
                f"💾 Output File: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}"}, "result": ("",)}

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
    CATEGORY = "QSAR/REGRESSION"
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
