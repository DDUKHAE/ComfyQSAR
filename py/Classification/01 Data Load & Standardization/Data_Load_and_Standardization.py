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
                    writer.write(mol)
    elif output_path.endswith('.csv'):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        pd.DataFrame(smiles_list, columns=['SMILES']).to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format for {output_path}")


def filter_molecule(mol: Optional[Chem.Mol]) -> bool:
    if mol is None:
        return False
    atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if atom_symbols and atom_symbols.issubset(METAL_IONS):
        return False
    if len(Chem.GetMolFrags(mol)) > 1:
        return False
    return True

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
    RETURN_NAMES = ("POSITIVE", "NEGATIVE")
    FUNCTION = "load_data"
    CATEGORY = "QSAR/CLASSIFICATION/OTHERS"
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
                f"✅ Positive Compounds: {pos_count+1}\n"
                f"✅ Negative Compounds: {neg_count+1}\n"
                f"📊 Total: {total_count+2} molecules\n"
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
    RETURN_NAMES = ("POSITIVE_STD", "NEGATIVE_STD")
    FUNCTION = "standardize_data"
    CATEGORY = "QSAR/CLASSIFICATION/OTHERS"
    OUTPUT_NODE = True

    def standardize_data(self, positive_path: str, negative_path: str) -> Dict[str, Any]:
        output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR/Standardization")
        os.makedirs(output_dir, exist_ok=True)

        def process_and_standardize(file_path: str, output_name: str) -> Tuple[str, int]:
            ext = '.sdf' if file_path.endswith('.sdf') else '.csv'
            output_file = os.path.join(output_dir, f"{output_name}_standardized{ext}")
            mols = read_molecules(file_path)
            filtered_mols = [mol for mol in mols if filter_molecule(mol)]
            write_molecules(filtered_mols, output_file)
            return output_file, len(filtered_mols)

        try:
            positive_output, pos_count = process_and_standardize(positive_path, "positive")
            negative_output, neg_count = process_and_standardize(negative_path, "negative")

            log_message = (
                "========================================\n"
                "🔹 Standardization Completed! 🔹\n"
                "========================================\n"
                f"✅ Positive Standardized: {pos_count+1}\n"
                f"✅ Negative Standardized: {neg_count+1}\n"
                f"💾 Output Dir: `{os.path.abspath(output_dir)}`\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (positive_output, negative_output)}

        except Exception as e:
            error_msg = f"❌ Standardization Error: {e}"
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
    RETURN_NAMES = ("POSITIVE_STD", "NEGATIVE_STD")
    FUNCTION = "run"
    CATEGORY = "QSAR/CLASSIFICATION"
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
