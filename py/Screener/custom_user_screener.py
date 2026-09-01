import json
import os
import sys
import numpy as np
import pandas as pd
import joblib
import folder_paths
from rdkit import Chem
from padelpy import padeldescriptor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_ROOT = os.path.join(folder_paths.get_output_directory(), "Screening", "Custom_Screening")

# standardize_mol lives in code/py/_shared/ -- the ONE module shared across
# tracks/nodes in this codebase. Screening candidates must be standardized
# by the EXACT SAME chemistry as training compounds (01 Data Load &
# Standardization): Cleanup -> FragmentParent (largest-fragment retention,
# e.g. "CCO.[Na+]" keeps "CCO") -> Uncharger -> tautomer canonicalization.
# The previous local keep_mol() instead rejected any multi-fragment
# molecule outright, which is a materially different (and stricter, in a
# way training never applied) filter -- confirmed to diverge on salts like
# "CCO.[Na+]".
_PY_DIR = os.path.dirname(BASE_DIR)
_SHARED_DIR = os.path.join(_PY_DIR, "_shared")
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)
from chem_standardize import standardize_mol  # noqa: E402

class QSARCustomUserScreener:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_sdf_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
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
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                }),
            },
            "optional": {
                "max_missing_fraction": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "A compound whose selected descriptors are missing more than this fraction is "
                               "flagged low_quality_input in the output (not excluded) -- every value can "
                               "technically be imputed, but a prediction built on a mostly-fabricated row "
                               "shouldn't be trusted without review. Denominator is the model's selected "
                               "descriptor count only, not the full PaDEL descriptor set.",
                }),
            },
        }

    # STANDARDIZED_MOLECULES/DESCRIPTORS/PREPROCESSED_DESCRIPTORS are still
    # written to disk (standardized_sdf/descriptor_csv/preprocessed_csv
    # below, paths shown in the log) -- just not exposed as connectable
    # output sockets, since nothing downstream wires into them and having
    # 5 sockets made the two that actually get connected (SCREENING_RESULTS,
    # SELECTED_MOLECULES) harder to find.
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = (
        "SCREENING_RESULTS",
        "SELECTED_MOLECULES",
    )
    FUNCTION = "execute"
    CATEGORY = "QSAR/3. SCREENER"
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
        standardized_sdf = os.path.join(prepared_dir, "standardized_input.sdf")
        report_path = os.path.join(prepared_dir, "standardization_report.csv")
        supplier = Chem.SDMolSupplier(input_sdf_path, removeHs=True)

        valid_count = 0
        report_rows = []
        with Chem.SDWriter(standardized_sdf) as writer:
            for mol in supplier:
                original_smiles = Chem.MolToSmiles(mol) if mol is not None else None
                std_mol, info = standardize_mol(mol)
                if info["status"] == "rejected":
                    report_rows.append({
                        "original_smiles": original_smiles, "standardized_smiles": "",
                        "status": "rejected", "reason": info["reason"],
                    })
                    continue
                # Tag with this molecule's position in standardized_sdf (the
                # index space _screen_compounds later indexes back into).
                # PaDEL can silently drop molecules mid-batch, so this tag is
                # how survivors get matched back to their real position
                # instead of by output row order -- see the identical fix in
                # "Classification/02 Descriptor Calculation".
                std_mol.SetProp('_Name', str(valid_count))
                writer.write(std_mol)
                report_rows.append({
                    "original_smiles": original_smiles, "standardized_smiles": Chem.MolToSmiles(std_mol),
                    "status": "ok", "reason": "",
                })
                valid_count += 1

        pd.DataFrame(report_rows).to_csv(report_path, index=False)
        return standardized_sdf, valid_count

    @staticmethod
    def _padel_log_has_errors(log_file):
        if not os.path.isfile(log_file):
            return False
        with open(log_file, "r", errors="replace") as f:
            log_text = f.read()
        return ("ClassCastException" in log_text) or ("NullPointerException" in log_text)

    @staticmethod
    def _calculate_descriptors(standardized_sdf, prepared_dir):
        descriptors_file = os.path.join(prepared_dir, "molecular_descriptors.csv")
        log_file = descriptors_file + ".log"

        # PaDEL's Java worker (CDK) has shown thread-safety failures under
        # concurrent load: the 2026-08 ASINEX screening-DB regeneration
        # returned normally but silently produced 791/1444 descriptor
        # columns as NaN for every one of 10,177 compounds, with tens of
        # thousands of ClassCastException/NullPointerException entries in
        # the PaDEL log -- coincident with heavy CPU contention from an
        # unrelated process. A normal return from padeldescriptor() is
        # therefore not sufficient evidence of a correct run; the log and
        # the output are checked explicitly below. threads=-1 with
        # maxruntime raised to 100000 (the training-side node's own
        # declared ceiling for this parameter, and previously validated to
        # run cleanly) is tried first; if that still corrupts the output,
        # threads=1 removes concurrency entirely as a fallback.
        fully_missing = []
        for threads, waitingjobs in ((-1, -1), (1, 1)):
            padeldescriptor(
                mol_dir=standardized_sdf,
                d_file=descriptors_file,
                d_2d=True,
                d_3d=False,
                detectaromaticity=True,
                log=True,
                removesalt=True,
                standardizenitro=True,
                # False, not True: "use filename as molecule name" would
                # override the per-molecule index tag set in _standardize_sdf
                # with "<filename>_<output position>" -- a sequential *output*
                # counter that can no longer be trusted to match the molecule's
                # real position in standardized_sdf once PaDEL drops anything
                # mid-batch. Keeping our own _Name tag is what makes safe
                # re-matching in _preprocess_descriptors possible.
                usefilenameasmolname=False,
                retainorder=True,
                threads=threads,
                waitingjobs=waitingjobs,
                maxruntime=100000,
                maxcpdperfile=0,
                headless=True,
            )
            df = pd.read_csv(descriptors_file)
            fully_missing = [c for c in df.columns if c != "Name" and df[c].isna().all()]
            if not fully_missing and not QSARCustomUserScreener._padel_log_has_errors(log_file):
                return descriptors_file
            print(
                f"⚠️ [ComfyQSAR] PaDEL output failed validation at threads={threads} "
                f"({len(fully_missing)} fully-missing descriptor column(s), "
                f"log errors={QSARCustomUserScreener._padel_log_has_errors(log_file)}). Retrying."
            )

        raise RuntimeError(
            "PaDEL descriptor calculation produced a corrupted output even after retrying at "
            f"threads=1 -- {len(fully_missing)} descriptor column(s) are NaN for every compound "
            f"(e.g. {fully_missing[:10]}), and/or the PaDEL log at {log_file} contains "
            "ClassCastException/NullPointerException entries. This does not raise inside "
            "padeldescriptor() itself, so it is not otherwise caught."
        )

    @staticmethod
    def _preprocess_descriptors(descriptor_csv, prepared_dir, recipe, selected_descriptors, valid_count,
                                 standardized_sdf, max_missing_fraction):
        df = pd.read_csv(descriptor_csv)

        # Recover each survivor's real position in standardized_sdf from the
        # _Name tag set in _standardize_sdf, rather than assuming PaDEL's
        # output row order matches submission order. PaDEL can silently drop
        # molecules (and everything after them) mid-batch; a naive
        # np.arange(len(df)) here would misalign descriptors with the wrong
        # compound from that point on, and _screen_compounds would then
        # pull the wrong molecules into the "selected candidates" SDF.
        try:
            recovered_idx = df["Name"].astype(int).tolist()
        except (KeyError, ValueError, TypeError):
            recovered_idx = None

        valid = (
            recovered_idx is not None
            and len(set(recovered_idx)) == len(df)
            and all(0 <= i < valid_count for i in recovered_idx)
        )
        if not valid:
            raise ValueError(
                "Could not recover a safe per-molecule index from PaDEL's "
                "'Name' output column -- refusing to align descriptors with "
                "compounds by row position (PaDEL can silently drop "
                "molecules mid-batch, which would misassign descriptors, "
                "predictions, and selected candidates to the wrong "
                "compound)."
            )

        dropped_indices = sorted(set(range(valid_count)) - set(recovered_idx))
        # prepared_dir is reused across runs, so a stale
        # padel_dropped_compounds.csv from a previous (dropped) run must not
        # linger and be mistaken for this run's result.
        dropped_report_path = os.path.join(prepared_dir, "padel_dropped_compounds.csv")

        # Reattach SMILES via the same __sdf_index__ lookup used above --
        # mirrors the Chem.MolToSmiles() pattern already used in
        # "02 Descriptor Calculation" -- rather than a positional guess.
        # Needed downstream for traceability and for an applicability-domain
        # (Tanimoto) node to work on screening candidates at all, since this
        # pipeline never otherwise carries a SMILES column past PaDEL.
        supplier = Chem.SDMolSupplier(standardized_sdf)
        smiles_by_index = {}
        for i in range(valid_count):
            mol = supplier[i] if i < len(supplier) else None
            smiles_by_index[i] = Chem.MolToSmiles(mol) if mol is not None else None

        if dropped_indices:
            pd.DataFrame({
                "standardized_sdf_index": dropped_indices,
                "SMILES": [smiles_by_index.get(i) for i in dropped_indices],
            }).to_csv(dropped_report_path, index=False)
        elif os.path.exists(dropped_report_path):
            os.remove(dropped_report_path)

        df.insert(0, "__sdf_index__", recovered_idx)
        df.insert(1, "SMILES", [smiles_by_index.get(i) for i in recovered_idx])

        name_col = df["Name"].copy() if "Name" in df.columns else None
        base_df = df.drop(columns=["Name", "SMILES"], errors="ignore")

        # Restrict to the FINAL MODEL's selected descriptors (descriptor_list_path),
        # NOT 04's full retained set (recipe["retained_descriptors"], which
        # can be ~1400 wide) -- the model never reads any column outside
        # selected_descriptors, so requiring/imputing the wider retained set
        # only means an unrelated PaDEL descriptor_type/version mismatch on a
        # column the model doesn't use can block screening entirely. A
        # selected descriptor that's absent here (e.g. different PaDEL
        # descriptor_type/version) is still a hard error, since the model
        # genuinely cannot run without it.
        imputer_statistics = recipe["imputer_statistics"]
        missing_cols = [c for c in selected_descriptors if c not in base_df.columns]
        if missing_cols:
            preview = missing_cols[:10]
            more = f" (+{len(missing_cols) - 10} more)" if len(missing_cols) > 10 else ""
            raise ValueError(
                f"Screening library is missing {len(missing_cols)} descriptor(s) the model "
                f"requires: {preview}{more}. This usually means a different PaDEL "
                "descriptor_type (2D/3D) or version was used than at training time."
            )

        numeric_df = base_df[selected_descriptors].copy()
        inf_count = int(np.isinf(numeric_df.to_numpy(dtype=float)).sum())
        numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # A compound whose selected descriptors are mostly missing can be
        # imputed in full technically, but the resulting prediction shouldn't
        # be trusted -- flag rather than silently predict on a mostly-
        # fabricated row. Denominator is the selected descriptor count only
        # (not the full ~1400 PaDEL/retained-by-04 descriptor set), matching
        # what the model actually consumes.
        missing_fraction = numeric_df.isna().mean(axis=1)
        low_quality_input = missing_fraction > max_missing_fraction
        n_low_quality = int(low_quality_input.sum())

        cleaned = pd.concat([base_df[["__sdf_index__"]], numeric_df], axis=1)
        if name_col is not None:
            cleaned.insert(1, "Name", name_col.reset_index(drop=True))
        cleaned.insert(2, "SMILES", [smiles_by_index.get(i) for i in recovered_idx])
        cleaned_path = os.path.join(prepared_dir, "cleaned_data.csv")
        cleaned.to_csv(cleaned_path, index=False)

        # Fill missing values with the TRAINING recipe's stored per-column
        # values -- equivalent to sklearn's fitted-imputer .transform() (the
        # recipe stores exactly one scalar per column, which is all a
        # mean/median/most_frequent SimpleImputer needs to replay), never
        # re-fit here.
        missing_stats = [c for c in selected_descriptors if c not in imputer_statistics]
        if missing_stats:
            raise ValueError(
                f"Recipe has no stored imputer statistic for {len(missing_stats)} selected "
                f"descriptor(s): {missing_stats[:10]}. Was this recipe produced by the same "
                "training run as descriptor_list_path?"
            )
        imputed_descriptors = numeric_df.copy()
        for col in selected_descriptors:
            imputed_descriptors[col] = imputed_descriptors[col].fillna(imputer_statistics[col])

        preprocessed = pd.DataFrame({"__sdf_index__": cleaned["__sdf_index__"].astype(int)})
        if name_col is not None:
            preprocessed["Name"] = name_col.reset_index(drop=True)
        preprocessed["SMILES"] = [smiles_by_index.get(i) for i in recovered_idx]
        preprocessed["selected_feature_missing_fraction"] = missing_fraction.reset_index(drop=True).values
        preprocessed["low_quality_input"] = low_quality_input.reset_index(drop=True).values
        for col in selected_descriptors:
            preprocessed[col] = imputed_descriptors[col].reset_index(drop=True).values

        preprocessed_path = os.path.join(prepared_dir, "preprocessed_data.csv")
        preprocessed.to_csv(preprocessed_path, index=False)

        return preprocessed_path, inf_count, len(dropped_indices), n_low_quality

    @staticmethod
    def _screen_compounds(trained_model_path, selected_descriptors, preprocessed_csv, standardized_sdf, screening_dir, threshold):
        model = joblib.load(trained_model_path)
        df = pd.read_csv(preprocessed_csv)

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
        trained_model_path,
        descriptor_list_path,
        preprocessing_recipe_path,
        threshold,
        max_missing_fraction=0.5,
    ):
        input_sdf_path = self._resolve_path(input_sdf_path)
        trained_model_path = self._resolve_path(trained_model_path)
        descriptor_list_path = self._resolve_path(descriptor_list_path)
        preprocessing_recipe_path = self._resolve_path(preprocessing_recipe_path)
        output_root = DEFAULT_OUTPUT_ROOT

        self._validate_file(input_sdf_path, "Input SDF")
        self._validate_file(trained_model_path, "Model")
        self._validate_file(descriptor_list_path, "Descriptor list")
        self._validate_file(preprocessing_recipe_path, "Preprocessing recipe")
        with open(preprocessing_recipe_path, "r") as f:
            recipe = json.load(f)
        with open(descriptor_list_path, "r") as f:
            selected_descriptors = [line.strip() for line in f if line.strip()]

        prepared_dir = os.path.join(output_root, "custom_db_prepared")
        screening_dir = os.path.join(output_root, "custom_screening_results")
        os.makedirs(prepared_dir, exist_ok=True)
        os.makedirs(screening_dir, exist_ok=True)

        standardized_sdf, valid_count = self._standardize_sdf(input_sdf_path, prepared_dir)
        descriptor_csv = self._calculate_descriptors(standardized_sdf, prepared_dir)
        preprocessed_csv, inf_count, padel_dropped_count, n_low_quality = self._preprocess_descriptors(
            descriptor_csv=descriptor_csv,
            prepared_dir=prepared_dir,
            recipe=recipe,
            selected_descriptors=selected_descriptors,
            valid_count=valid_count,
            standardized_sdf=standardized_sdf,
            max_missing_fraction=max_missing_fraction,
        )
        prediction_csv, selected_sdf, total_count, selected_count = self._screen_compounds(
            trained_model_path=trained_model_path,
            selected_descriptors=selected_descriptors,
            preprocessed_csv=preprocessed_csv,
            standardized_sdf=standardized_sdf,
            screening_dir=screening_dir,
            threshold=threshold,
        )

        # Inputs (input_sdf_path/trained_model_path/descriptor_list_path/recipe) are
        # user-supplied and can live anywhere, so their basenames are shown
        # without a directory line; this node's OWN outputs land in two
        # distinct subfolders (prepared vs. screening results), so each gets
        # its own Directory: line rather than one that wouldn't cover both.
        log_message = (
            "========================================\n"
            "🔹 Custom User Screening Completed! 🔹\n"
            "========================================\n"
            f"📁 Input SDF      : {os.path.basename(input_sdf_path)}\n"
            f"📁 Model          : {os.path.basename(trained_model_path)}\n"
            f"🧬 Descriptor list: {os.path.basename(descriptor_list_path)} ({len(selected_descriptors)} selected descriptor(s))\n"
            f"📋 Recipe         : {os.path.basename(preprocessing_recipe_path)} "
            f"(training-fitted imputation values for the selected descriptors above)\n"
            f"⚙️  Threshold      : {threshold}\n"
            f"✅ Valid Molecules: {valid_count}\n"
            f"⚠️ PaDEL Dropped  : {padel_dropped_count}"
            + (" (see padel_dropped_compounds.csv)\n" if padel_dropped_count else "\n")
            + f"🧹 Inf Replaced   : {inf_count}\n"
            f"⚠️ Low-Quality Input: {n_low_quality} compound(s) had >{max_missing_fraction:.0%} of selected "
            f"descriptors missing (flagged low_quality_input, not excluded)\n"
            f"🔍 Screened       : {total_count}\n"
            f"✅ Selected       : {selected_count}\n"
            f"📁 Directory: {os.path.relpath(prepared_dir, folder_paths.get_output_directory())}{os.sep}\n"
            f"💾 Standardized: {os.path.basename(standardized_sdf)}\n"
            f"💾 Descriptors: {os.path.basename(descriptor_csv)}\n"
            f"💾 Preprocessed: {os.path.basename(preprocessed_csv)}\n"
            f"📁 Directory: {os.path.relpath(screening_dir, folder_paths.get_output_directory())}{os.sep}\n"
            f"💾 Predictions: {os.path.basename(prediction_csv)}\n"
            f"💾 Selected SDF: {os.path.basename(selected_sdf)}\n"
            "========================================"
        )

        return {"ui": {"text": log_message}, "result": (prediction_csv, selected_sdf)}


NODE_CLASS_MAPPINGS = {
    "QSARCustomUserScreener": QSARCustomUserScreener,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QSARCustomUserScreener": "External Screening (Custom Compounds)",
}
