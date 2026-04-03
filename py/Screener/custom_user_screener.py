import os
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from sklearn.impute import SimpleImputer
from padelpy import padeldescriptor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_ROOT = os.path.join(BASE_DIR, "Custom_DB_Screening_Results")

class QSARCustomUserScreener:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_sdf_path": ("STRING", {
                    "default": "PTP1B_custom.sdf",
                    "multiline": False,
                }),
                "model_path": ("STRING", {
                    "default": "PTP1B_prediction_QSAR_model.pkl",
                    "multiline": False,
                }),
                "features_path": ("STRING", {
                    "default": "selected_features_V3.txt",
                    "multiline": False,
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                }),
                "nan_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                }),
                "impute_method": (["mean", "median", "most_frequent"],),
                "output_root_dir": ("STRING", {
                    "default": "Custom_DB_Screening_Results",
                    "multiline": False,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "standardized_sdf_path",
        "descriptor_csv_path",
        "preprocessed_csv_path",
        "prediction_csv_path",
        "selected_sdf_path",
    )
    FUNCTION = "execute"
    CATEGORY = "QSAR/SCREENER"
    OUTPUT_NODE = True

    @staticmethod
    def _resolve_path(path_value):
        if os.path.isabs(path_value):
            return path_value
        return os.path.join(BASE_DIR, path_value)

    @staticmethod
    def _validate_file(path_value, label):
        if not os.path.isfile(path_value):
            raise FileNotFoundError(f"{label} file not found: {path_value}")

    @staticmethod
    def _standardize_sdf(input_sdf_path, prepared_dir):
        metal_ions = {
            "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
            "Sb", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
            "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Th", "Pa", "U",
        }

        def keep_mol(mol):
            if mol is None:
                return False
            atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
            if atom_symbols and atom_symbols.issubset(metal_ions):
                return False
            if len(Chem.GetMolFrags(mol)) > 1:
                return False
            return True

        standardized_sdf = os.path.join(prepared_dir, "standardized_input.sdf")
        supplier = Chem.SDMolSupplier(input_sdf_path, removeHs=True)

        valid_count = 0
        with Chem.SDWriter(standardized_sdf) as writer:
            for mol in supplier:
                if keep_mol(mol):
                    writer.write(mol)
                    valid_count += 1

        return standardized_sdf, valid_count

    @staticmethod
    def _calculate_descriptors(standardized_sdf, prepared_dir):
        descriptors_file = os.path.join(prepared_dir, "molecular_descriptors.csv")
        padeldescriptor(
            mol_dir=standardized_sdf,
            d_file=descriptors_file,
            d_2d=True,
            d_3d=False,
            detectaromaticity=True,
            log=True,
            removesalt=True,
            standardizenitro=True,
            usefilenameasmolname=True,
            retainorder=True,
            threads=-1,
            waitingjobs=-1,
            maxruntime=10000,
            maxcpdperfile=0,
            headless=True,
        )
        return descriptors_file

    @staticmethod
    def _preprocess_descriptors(descriptor_csv, prepared_dir, nan_threshold, impute_method):
        df = pd.read_csv(descriptor_csv)
        df.insert(0, "__sdf_index__", np.arange(len(df)))

        name_col = df["Name"].copy() if "Name" in df.columns else None
        base_df = df.drop(columns=["Name"], errors="ignore")

        numeric_cols = [c for c in base_df.columns if c != "__sdf_index__"]
        numeric_df = base_df[numeric_cols].copy()
        inf_count = np.isinf(numeric_df.values).sum()
        numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        cleaned = pd.concat([base_df[["__sdf_index__"]], numeric_df], axis=1)
        if name_col is not None:
            cleaned.insert(1, "Name", name_col.reset_index(drop=True))
        cleaned_path = os.path.join(prepared_dir, "cleaned_data.csv")
        cleaned.to_csv(cleaned_path, index=False)

        row_nan_ratio = numeric_df.isna().sum(axis=1) / max(1, numeric_df.shape[1])
        keep_rows = row_nan_ratio <= nan_threshold
        filtered_rows = cleaned.loc[keep_rows].reset_index(drop=True)
        removed_rows = int((~keep_rows).sum())
        filtered_rows_path = os.path.join(
            prepared_dir,
            f"filtered_high_NAN_compound_({cleaned.shape[0]}_{filtered_rows.shape[0]}).csv",
        )
        filtered_rows.to_csv(filtered_rows_path, index=False)

        candidate = filtered_rows.drop(columns=["Name"], errors="ignore")
        descriptor_part = candidate.drop(columns=["__sdf_index__"], errors="ignore")
        col_nan_ratio = descriptor_part.isna().mean()
        keep_descriptor_cols = col_nan_ratio[col_nan_ratio <= nan_threshold].index.tolist()
        removed_cols = int(descriptor_part.shape[1] - len(keep_descriptor_cols))

        filtered_desc = pd.concat(
            [
                candidate[["__sdf_index__"]],
                descriptor_part[keep_descriptor_cols],
            ],
            axis=1,
        )
        if "Name" in filtered_rows.columns:
            filtered_desc.insert(1, "Name", filtered_rows["Name"])

        filtered_desc_path = os.path.join(
            prepared_dir,
            f"filtered_high_NAN_descriptors_({cleaned.shape[1]}_{filtered_desc.shape[1]}).csv",
        )
        filtered_desc.to_csv(filtered_desc_path, index=False)

        pre_impute = filtered_desc.copy()
        descriptor_data = pre_impute.drop(columns=["Name", "__sdf_index__"], errors="ignore")
        imputer = SimpleImputer(strategy=impute_method)
        imputed_values = imputer.fit_transform(descriptor_data)
        imputed_descriptors = pd.DataFrame(imputed_values, columns=descriptor_data.columns)

        preprocessed = pd.DataFrame({"__sdf_index__": pre_impute["__sdf_index__"].astype(int)})
        if "Name" in pre_impute.columns:
            preprocessed["Name"] = pre_impute["Name"].reset_index(drop=True)

        for col in imputed_descriptors.columns:
            preprocessed[col] = imputed_descriptors[col]

        preprocessed_path = os.path.join(prepared_dir, "preprocessed_data.csv")
        preprocessed.to_csv(preprocessed_path, index=False)

        return preprocessed_path, inf_count, removed_rows, removed_cols

    @staticmethod
    def _screen_compounds(model_path, features_path, preprocessed_csv, standardized_sdf, screening_dir, threshold):
        model = joblib.load(model_path)
        df = pd.read_csv(preprocessed_csv)

        with open(features_path, "r") as f:
            selected_descriptors = [line.strip() for line in f if line.strip()]

        missing = [d for d in selected_descriptors if d not in df.columns]
        if missing:
            raise ValueError(f"Missing descriptors in input data: {missing}")

        x_data = df[selected_descriptors]

        if hasattr(model, "predict_proba"):
            predictions = model.predict_proba(x_data)[:, 1]
            selected_mask = predictions >= threshold
        else:
            predictions = model.predict(x_data)
            selected_mask = predictions >= threshold

        prediction_df = df.copy()
        prediction_df["Prediction"] = predictions
        csv_path = os.path.join(screening_dir, "User_Screening_Predictions.csv")
        prediction_df.to_csv(csv_path, index=False)

        selected_rows = prediction_df[selected_mask].copy()
        sdf_indices = selected_rows["__sdf_index__"].astype(int).tolist()

        sdf_path = os.path.join(screening_dir, "User_Screening_Selected_Molecules.sdf")
        supplier = Chem.SDMolSupplier(standardized_sdf)
        with Chem.SDWriter(sdf_path) as writer:
            for sdf_idx in sdf_indices:
                if 0 <= sdf_idx < len(supplier):
                    mol = supplier[sdf_idx]
                    if mol is not None:
                        writer.write(mol)

        return csv_path, sdf_path, len(prediction_df), len(selected_rows)

    def execute(
        self,
        input_sdf_path,
        model_path,
        features_path,
        threshold,
        nan_threshold,
        impute_method,
        output_root_dir,
    ):
        input_sdf_path = self._resolve_path(input_sdf_path)
        model_path = self._resolve_path(model_path)
        features_path = self._resolve_path(features_path)
        output_root = self._resolve_path(output_root_dir)

        self._validate_file(input_sdf_path, "Input SDF")
        self._validate_file(model_path, "Model")
        self._validate_file(features_path, "Features")

        prepared_dir = os.path.join(output_root, "custom_db_prepared")
        screening_dir = os.path.join(output_root, "custom_screening_results")
        os.makedirs(prepared_dir, exist_ok=True)
        os.makedirs(screening_dir, exist_ok=True)

        standardized_sdf, valid_count = self._standardize_sdf(input_sdf_path, prepared_dir)
        descriptor_csv = self._calculate_descriptors(standardized_sdf, prepared_dir)
        preprocessed_csv, inf_count, removed_rows, removed_cols = self._preprocess_descriptors(
            descriptor_csv=descriptor_csv,
            prepared_dir=prepared_dir,
            nan_threshold=nan_threshold,
            impute_method=impute_method,
        )
        prediction_csv, selected_sdf, total_count, selected_count = self._screen_compounds(
            model_path=model_path,
            features_path=features_path,
            preprocessed_csv=preprocessed_csv,
            standardized_sdf=standardized_sdf,
            screening_dir=screening_dir,
            threshold=threshold,
        )

        log_message = (
            "========================================\n"
            "🔹 Custom User Screening Completed! 🔹\n"
            "========================================\n"
            f"📁 Input SDF      : {os.path.basename(input_sdf_path)}\n"
            f"📁 Model          : {os.path.basename(model_path)}\n"
            f"🧬 Features       : {os.path.basename(features_path)}\n"
            f"⚙️  Threshold      : {threshold}\n"
            f"📏 NaN Threshold  : {nan_threshold}\n"
            f"🩹 Impute Method  : {impute_method}\n"
            f"✅ Valid Molecules: {valid_count}\n"
            f"🧹 Inf Replaced   : {inf_count}\n"
            f"🗑️  Rows Removed   : {removed_rows}\n"
            f"🗑️  Cols Removed   : {removed_cols}\n"
            f"🔍 Screened       : {total_count}\n"
            f"✅ Selected       : {selected_count}\n"
            f"🧪 Standardized   : {os.path.basename(standardized_sdf)}\n"
            f"📊 Descriptors    : {os.path.basename(descriptor_csv)}\n"
            f"⚙️  Preprocessed   : {os.path.basename(preprocessed_csv)}\n"
            f"📄 Predictions    : {os.path.basename(prediction_csv)}\n"
            f"🧪 Selected SDF   : {os.path.basename(selected_sdf)}\n"
            "========================================"
        )

        return {"ui": {"text": log_message}, "result": (standardized_sdf, descriptor_csv, preprocessed_csv, prediction_csv, selected_sdf)}


NODE_CLASS_MAPPINGS = {
    "QSARCustomUserScreener": QSARCustomUserScreener,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QSARCustomUserScreener": "Custom User Screener",
}
