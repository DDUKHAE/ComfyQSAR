import os
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import traceback
import folder_paths

REGRESSION_ID_COLS = ["Name", "SMILES", "value"]


def audit_missing_and_infinite_values(df, output_dir, id_cols):
    """
    Audits every descriptor cell that will need missing-value handling
    downstream (excluding id_cols, e.g. Name/SMILES/value -- metadata/target
    are not descriptors and out of scope): cells already NaN (typically
    PaDEL's own "can't compute this descriptor for this molecule" output)
    and cells that are +inf/-inf (converted to NaN below). Both feed the
    same row/column-removal and imputation pipeline, so they are reported
    together in one set of files -- a separate inf-only report would make
    it impossible to tell, from the final removed/imputed counts alone,
    whether a given cell started as inf or was already NaN. Writes a
    detailed per-cell report plus per-descriptor and per-compound
    summaries, then replaces inf with NaN. Returns (df_with_nan,
    summary_dict, report_paths_dict).
    """
    os.makedirs(output_dir, exist_ok=True)
    descriptor_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in id_cols]

    detail_rows = []
    by_descriptor = {}
    by_compound = {}

    if descriptor_cols:
        sub = df[descriptor_cols]
        inf_mask = sub.isin([np.inf, -np.inf]).values
        nan_mask = sub.isna().values
        rows, cols = np.where(inf_mask | nan_mask)
        for r, c in zip(rows, cols):
            descriptor = descriptor_cols[c]
            row_index = df.index[r]
            original_value = sub.iat[r, c]
            if inf_mask[r, c]:
                value_type = "pos_inf" if original_value == np.inf else "neg_inf"
            else:
                value_type = "missing"
            compound_id = df.iloc[r]["Name"] if "Name" in df.columns else None
            smiles = df.iloc[r]["SMILES"] if "SMILES" in df.columns else None

            detail_rows.append({
                "row_index": row_index, "compound_id": compound_id, "SMILES": smiles,
                "descriptor": descriptor, "value_type": value_type, "original_value": original_value,
            })

            d = by_descriptor.setdefault(
                descriptor, {"n_nan": 0, "n_pos_inf": 0, "n_neg_inf": 0, "affected_rows": set()}
            )
            d["n_nan"] += int(value_type == "missing")
            d["n_pos_inf"] += int(value_type == "pos_inf")
            d["n_neg_inf"] += int(value_type == "neg_inf")
            d["affected_rows"].add(row_index)

            c_info = by_compound.setdefault(
                row_index, {"compound_id": compound_id, "SMILES": smiles, "nan_descriptors": [], "inf_descriptors": []}
            )
            if value_type == "missing":
                c_info["nan_descriptors"].append(descriptor)
            else:
                c_info["inf_descriptors"].append(descriptor)

    total_nan = sum(1 for r in detail_rows if r["value_type"] == "missing")
    total_inf = sum(1 for r in detail_rows if r["value_type"] in ("pos_inf", "neg_inf"))
    total_missing_or_inf = len(detail_rows)
    n_rows_total = len(df)

    summary_desc_rows = sorted(
        (
            {
                "descriptor": descriptor,
                "n_nan": d["n_nan"],
                "n_pos_inf": d["n_pos_inf"],
                "n_neg_inf": d["n_neg_inf"],
                "n_total_missing_or_inf": d["n_nan"] + d["n_pos_inf"] + d["n_neg_inf"],
                "n_affected_compounds": len(d["affected_rows"]),
                "affected_fraction": (len(d["affected_rows"]) / n_rows_total) if n_rows_total else 0.0,
            }
            for descriptor, d in by_descriptor.items()
        ),
        key=lambda row: -row["n_total_missing_or_inf"],
    )
    summary_cpd_rows = [
        {
            "row_index": row_index, "compound_id": info["compound_id"], "SMILES": info["SMILES"],
            "n_nan_descriptors": len(info["nan_descriptors"]),
            "n_inf_descriptors": len(info["inf_descriptors"]),
            "n_total_affected_descriptors": len(info["nan_descriptors"]) + len(info["inf_descriptors"]),
            "affected_descriptors": ";".join(info["nan_descriptors"] + info["inf_descriptors"]),
        }
        for row_index, info in by_compound.items()
    ]

    detail_path = os.path.join(output_dir, "missing_value_detail.csv")
    by_desc_path = os.path.join(output_dir, "missing_value_by_descriptor.csv")
    by_cpd_path = os.path.join(output_dir, "missing_value_by_compound.csv")
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)
    pd.DataFrame(summary_desc_rows).to_csv(by_desc_path, index=False)
    pd.DataFrame(summary_cpd_rows).to_csv(by_cpd_path, index=False)

    df_copy = df.copy()
    if descriptor_cols:
        df_copy[descriptor_cols] = df_copy[descriptor_cols].replace([np.inf, -np.inf], np.nan)

    summary = {
        "total_nan": total_nan,
        "total_inf": total_inf,
        "total_missing_or_inf": total_missing_or_inf,
        "n_affected_descriptors": len(by_descriptor),
        "n_affected_compounds": len(by_compound),
        "top_descriptors": summary_desc_rows[:5],
    }
    report_paths = {"detail": detail_path, "by_descriptor": by_desc_path, "by_compound": by_cpd_path}
    return df_copy, summary, report_paths


def _format_missing_value_log(summary, report_paths):
    # "Missing values" (plain English) replaces "NaN"/"inf" jargon, and --
    # same principle as the 2b/node-1 log cleanup -- a clean/expected cell
    # (PaDEL simply couldn't compute a descriptor) collapses to one line;
    # infinite values only get their own sub-line when they actually
    # occurred (rare for QSAR descriptors).
    total = summary["total_missing_or_inf"]
    if total == 0:
        return "✅ No missing or invalid values found\n"

    lines = [
        f"✅ Missing values: {total} cell(s) across "
        f"{summary['n_affected_descriptors']} descriptor(s), "
        f"{summary['n_affected_compounds']} compound(s)"
    ]
    if summary["total_inf"]:
        lines.append(f"   ⚠️  Infinite values found (treated as missing): {summary['total_inf']} cell(s)")
    top = ", ".join(f"{r['descriptor']}({r['n_total_missing_or_inf']})" for r in summary["top_descriptors"])
    lines.append(f"📌 Most affected descriptors: {top}")
    lines.append(
        f"📋 Reports: {os.path.basename(report_paths['detail'])}, "
        f"{os.path.basename(report_paths['by_descriptor'])}, "
        f"{os.path.basename(report_paths['by_compound'])}"
    )
    return "\n".join(lines) + "\n"


def remove_high_nan_rows_regression(df, threshold, id_cols):
    descriptor_cols = [c for c in df.columns if c not in id_cols]
    if not descriptor_cols:
        return df.copy(), 0
    initial_rows = len(df)
    nan_percentage = df[descriptor_cols].isna().sum(axis=1) / len(descriptor_cols)
    filtered_df = df[nan_percentage <= threshold].copy()
    removed_count = initial_rows - len(filtered_df)
    return filtered_df, removed_count


def remove_high_nan_cols_regression(df, threshold, id_cols):
    descriptor_cols = [c for c in df.columns if c not in id_cols]
    initial_descriptor_cols = len(descriptor_cols)
    if descriptor_cols:
        nan_percentage = df[descriptor_cols].isna().mean()
        keep_descriptor_cols = set(nan_percentage[nan_percentage <= threshold].index)
    else:
        keep_descriptor_cols = set()
    retained_columns = [c for c in df.columns if c in id_cols or c in keep_descriptor_cols]
    filtered_df = df[retained_columns].copy()
    removed_count = initial_descriptor_cols - len(keep_descriptor_cols)
    return filtered_df, removed_count


def impute_missing_values_regression(df, method):
    non_descriptor_cols = [col for col in ["Name", "SMILES", "value"] if col in df.columns]
    descriptor_cols = [col for col in df.columns if col not in non_descriptor_cols]

    descriptors = df[descriptor_cols].copy()
    missing_count = int(descriptors.isna().sum().sum())

    imputer = SimpleImputer(strategy=method)
    imputed_descriptors = pd.DataFrame(
        imputer.fit_transform(descriptors),
        columns=descriptor_cols,
    )

    final_df = pd.concat(
        [df[non_descriptor_cols].reset_index(drop=True), imputed_descriptors.reset_index(drop=True)],
        axis=1,
    )

    return final_df, missing_count


def _row_nan_fraction_report(df, id_col_present, descriptor_cols, threshold, source_label):
    """Per-row NaN fraction computed ONLY over the given (already
    train-decided) descriptor_cols -- a test compound that looks bad against
    the ORIGINAL full descriptor set can still pass once the columns
    training already dropped are excluded from the denominator, so this
    must run after column restriction, not before."""
    if not descriptor_cols:
        return df.copy(), []
    row_nan_frac = df[descriptor_cols].isna().sum(axis=1) / len(descriptor_cols)
    bad_mask = row_nan_frac > threshold
    excluded = []
    for pos in np.where(bad_mask.values)[0]:
        idx = df.index[pos]
        excluded.append({
            "Name": df.at[idx, "Name"] if "Name" in df.columns else str(idx),
            "SMILES": df.at[idx, "SMILES"] if "SMILES" in df.columns else "",
            "source": source_label,
            "row_nan_fraction": float(row_nan_frac.loc[idx]),
            "threshold": threshold,
            "reason": "row_missingness_exceeds_threshold_after_column_restriction",
        })
    return df[~bad_mask].copy(), excluded


def paired_preprocess_regression(train_df, test_df, y_test_df, target_column, compound_nan_threshold, descriptor_nan_threshold, method):
    """Fits every population-level statistic (retained-descriptor set,
    imputer) on TRAIN only and applies it unchanged to TEST -- no test value
    is ever used to decide what gets kept or what a missing value gets
    filled with. Order matters and follows this specific sequence:
      1. inf -> NaN (cell-local, order-independent, done defensively even
         though 2b should already have done it)
      2. retained-descriptor decision from TRAIN's own missingness only,
         with an all-NaN-in-train column forced out regardless of threshold
         (SimpleImputer cannot fit a statistic for a column with zero
         observed values, and silently drops it rather than raising --
         doing it explicitly here keeps 'retained' truthful)
      3. restrict BOTH train and test to that column set
      4. per-row missingness computed on the RESTRICTED columns, applied
         independently to each side (row-local, not a fit -- safe either
         side, but must come after step 3 so a test compound isn't
         penalized for missingness in a descriptor training already
         dropped)
      5. imputer fit on retained TRAIN rows only
      6. transform both TRAIN and TEST with that one fitted imputer, with
         postcondition checks that no NaN/inf survives and that the
         imputer's statistics line up 1:1 with the retained columns
      7. Y_TEST is filtered to match whichever test compounds survived step
         4 -- test row exclusion is independent of Y_TEST, so without this
         a downstream node requiring X_test/Y_test Name sets to match
         exactly (as '8. Model Validation' does) would fail on any dataset
         where even one test compound is dropped for missingness.
    """
    train_id_cols = [c for c in ("Name", "SMILES", target_column) if c in train_df.columns]
    test_id_cols = [c for c in ("Name", "SMILES") if c in test_df.columns]

    train_descriptor_cols = [c for c in train_df.columns if c not in train_id_cols]
    test_descriptor_cols = [c for c in test_df.columns if c not in test_id_cols]

    missing_in_test = sorted(set(train_descriptor_cols) - set(test_descriptor_cols))
    missing_in_train = sorted(set(test_descriptor_cols) - set(train_descriptor_cols))
    if missing_in_test or missing_in_train:
        raise ValueError(
            "Train/test descriptor columns do not match -- they must come from the same "
            f"3. Data Split run. In train only (up to 10): {missing_in_test[:10]}. "
            f"In test only (up to 10): {missing_in_train[:10]}."
        )

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[train_descriptor_cols] = train_df[train_descriptor_cols].replace([np.inf, -np.inf], np.nan)
    test_df[test_descriptor_cols] = test_df[test_descriptor_cols].replace([np.inf, -np.inf], np.nan)

    train_nan_frac = train_df[train_descriptor_cols].isna().mean()
    retained = [c for c in train_descriptor_cols if train_nan_frac[c] <= descriptor_nan_threshold]
    dropped_by_threshold = [c for c in train_descriptor_cols if c not in retained]

    train_n_observed = train_df[retained].notna().sum() if retained else pd.Series(dtype=int)
    all_nan_in_train = [c for c in retained if train_n_observed.get(c, 0) == 0]
    if all_nan_in_train:
        retained = [c for c in retained if c not in all_nan_in_train]
    dropped = dropped_by_threshold + all_nan_in_train

    if not retained:
        raise ValueError(
            "No descriptor columns survived retention -- every training descriptor is either "
            f"above descriptor_nan_threshold={descriptor_nan_threshold} or has zero observed "
            "values in training. Check your threshold and input data."
        )

    train_df = train_df[train_id_cols + retained].copy()
    test_df = test_df[test_id_cols + retained].copy()

    train_df, train_excluded = _row_nan_fraction_report(train_df, "Name" in train_df.columns, retained, compound_nan_threshold, "train")
    test_df, test_excluded = _row_nan_fraction_report(test_df, "Name" in test_df.columns, retained, compound_nan_threshold, "test")
    excluded_rows = train_excluded + test_excluded

    imputer = SimpleImputer(strategy=method)
    train_imputed = imputer.fit_transform(train_df[retained])
    if train_imputed.shape[1] != len(retained) or len(imputer.statistics_) != len(retained):
        raise RuntimeError(
            f"Imputer produced {train_imputed.shape[1]} column(s) / "
            f"{len(imputer.statistics_)} statistic(s) but {len(retained)} retained "
            "descriptor(s) were expected -- likely an all-NaN column that slipped past the "
            "pre-filter. This is an internal invariant violation; please report it."
        )
    train_df[retained] = train_imputed
    test_df[retained] = imputer.transform(test_df[retained])

    for label, block in (("train", train_df[retained]), ("test", test_df[retained])):
        arr = block.to_numpy(dtype=float) if retained else np.empty((len(block), 0))
        if arr.size and not np.isfinite(arr).all():
            raise RuntimeError(
                f"{label} output still has non-finite descriptor value(s) after imputation -- "
                "this should not happen; please report it."
            )
    imputer_stats = dict(zip(retained, [float(v) for v in imputer.statistics_]))

    excluded_test_names = {e["Name"] for e in test_excluded}
    if excluded_test_names:
        if "Name" not in y_test_df.columns:
            raise ValueError(
                f"{len(excluded_test_names)} test compound(s) were excluded from X_test for "
                "missingness, but Y_test has no 'Name' column to filter the same compounds "
                "out -- cannot keep X_test/Y_test aligned. Provide a Y_test with a Name column."
            )
        y_names = y_test_df["Name"].astype(str).str.strip()
        y_test_filtered = y_test_df[~y_names.isin(excluded_test_names)].copy()
    else:
        y_test_filtered = y_test_df.copy()

    recipe = {
        "fit_scope": (
            "All statistics above (retained_descriptors, imputer_statistics) were fit on "
            "TRAINING data only and applied unchanged to test -- no test value was used to "
            "decide what gets kept or what a missing value gets filled with."
        ),
        "target_column": target_column,
        "compound_nan_threshold": compound_nan_threshold,
        "descriptor_nan_threshold": descriptor_nan_threshold,
        "imputation_method": method,
        "n_train_rows_used_for_fit": len(train_df),
        "n_retained_descriptors": len(retained),
        "n_dropped_descriptors": len(dropped),
        "retained_descriptors": retained,
        "dropped_descriptors": dropped,
        "dropped_descriptors_all_nan_in_train": all_nan_in_train,
        "imputer_statistics": imputer_stats,
    }
    return train_df, test_df, y_test_filtered, recipe, pd.DataFrame(excluded_rows)


class Remove_high_nan_compounds_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "descriptor_data_path": ("STRING", {"forceInput": True}),
            "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DESCRIPTORS",)
    FUNCTION = "run"
    CATEGORY = "QSAR/2. REGRESSION/OTHERS"
    OUTPUT_NODE = True

    def run(self, descriptor_data_path, threshold):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Regression", "04_Descriptor_Preprocessing")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(descriptor_data_path)
            initial = len(df)
            df, removed = remove_high_nan_rows_regression(df, threshold, REGRESSION_ID_COLS)
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(descriptor_data_path))[0]}_compounds_filtered.csv")
            df.to_csv(output_file, index=False)
            log_message = (
                "========================================\n"
                "🔹 High NaN Compound Removal Complete! 🔹\n"
                "========================================\n"
                f"✅ Initial Compounds: {initial}\n"
                f"✅ Removed Compounds: {removed}\n"
                f"✅ Final Compounds: {len(df)}\n"
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
                f"💾 Output: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}


class Remove_high_nan_descriptors_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "descriptor_data_path": ("STRING", {"forceInput": True}),
            "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DESCRIPTORS",)
    FUNCTION = "run"
    CATEGORY = "QSAR/2. REGRESSION/OTHERS"
    OUTPUT_NODE = True

    def run(self, descriptor_data_path, threshold):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Regression", "04_Descriptor_Preprocessing")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(descriptor_data_path)
            initial_cols = df.shape[1]
            df, removed = remove_high_nan_cols_regression(df, threshold, REGRESSION_ID_COLS)
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(descriptor_data_path))[0]}_descriptors_filtered.csv")
            df.to_csv(output_file, index=False)
            log_message = (
                "========================================\n"
                "🔹 High NaN Descriptor Removal Complete! 🔹\n"
                "========================================\n"
                f"✅ Initial Descriptors: {initial_cols}\n"
                f"✅ Removed Descriptors: {removed}\n"
                f"✅ Final Descriptors: {df.shape[1]}\n"
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
                f"💾 Output: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}


class Impute_missing_values_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "descriptor_data_path": ("STRING", {"forceInput": True}),
            "method": (["mean", "median", "most_frequent"],),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DESCRIPTORS",)
    FUNCTION = "run"
    CATEGORY = "QSAR/2. REGRESSION/OTHERS"
    OUTPUT_NODE = True

    def run(self, descriptor_data_path, method):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Regression", "04_Descriptor_Preprocessing")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(descriptor_data_path)
            df, count = impute_missing_values_regression(df, method)
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(descriptor_data_path))[0]}_imputed.csv")
            df.to_csv(output_file, index=False)
            log_message = (
                "========================================\n"
                "🔹 Imputation Complete! 🔹\n"
                "========================================\n"
                f"✅ Imputation Method: '{method}'\n"
                f"✅ Filled Missing Values: {count}\n"
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
                f"💾 Output: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(output_file),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}


class Descriptor_preprocessing_Regression:
    """Whole-dataset, single-file preprocessing (fits row/column thresholds
    and the imputer on everything given). For any pipeline that has a real
    hold-out set (3. Data Split), use Paired_Descriptor_Preprocessing
    instead: fitting this node's statistics on a file that mixes train and
    hold-out compounds leaks hold-out information into the training values.

    Even with no external hold-out, this is exploratory/fixed preprocessing,
    not a properly nested cross-validation step: fitting column retention
    and the imputer on the WHOLE dataset before running CV means every
    fold's "held-out" portion already influenced those statistics, which
    biases the CV estimate optimistically (Ambroise & McLachlan 2002, PNAS,
    doi:10.1073/pnas.102102699; Varma & Simon 2006, BMC Bioinformatics,
    doi:10.1186/1471-2105-7-91). Report results from this node as "fixed
    preprocessing, then CV" rather than as an unbiased internal-validation
    estimate. For a fully nested-clean estimate, imputation (and ideally
    feature selection) would need to be refit inside each CV fold, which
    this platform does not automate."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "descriptor_data_path": ("STRING", {
                "tooltip": "Whole-dataset preprocessing -- use only when there is NO "
                           "train/hold-out split (e.g. whole-dataset CV-only studies). If "
                           "your workflow uses '3. Data Split', use '4. Descriptor "
                           "Preprocessing' (Paired) instead, or this will fit imputation/"
                           "column-retention statistics across train+hold-out together and "
                           "leak hold-out information into the training values.",
            }),
            "compounds_nan_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "descriptors_nan_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "imputation_method": (["mean", "median", "most_frequent"],),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PREPROCESSED_DESCRIPTORS",)
    FUNCTION = "preprocess"
    CATEGORY = "QSAR/2. REGRESSION/OTHERS"
    OUTPUT_NODE = True

    def preprocess(self, descriptor_data_path, compounds_nan_threshold, descriptors_nan_threshold, imputation_method):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Regression", "04_Descriptor_Preprocessing")
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(descriptor_data_path)
            initial_shape = f"{df.shape[0]}x{df.shape[1]}"
            df, inf_summary, inf_report_paths = audit_missing_and_infinite_values(df, output_dir, REGRESSION_ID_COLS)
            df, cpd_removed = remove_high_nan_rows_regression(df, compounds_nan_threshold, REGRESSION_ID_COLS)
            df, desc_removed = remove_high_nan_cols_regression(df, descriptors_nan_threshold, REGRESSION_ID_COLS)
            df, imputed = impute_missing_values_regression(df, imputation_method)
            if df.empty:
                return {"ui": {"text": "⚠️ Warning: Preprocessing resulted in an empty dataset."}, "result": ("",)}
            final_shape = f"{df.shape[0]}x{df.shape[1]}"
            output_file = os.path.join(output_dir, "descriptors.csv")
            df.to_csv(output_file, index=False)
            summary = (
                "========================================\n"
                "🔹 Preprocessing Pipeline Complete! 🔹\n"
                "========================================\n"
                "⚠️ Whole-dataset mode -- fixed preprocessing (column retention + imputer fit on "
                "everything given), THEN cross-validation, not a nested-CV-clean estimate: every "
                "fold's held-out portion already influenced these statistics, which biases a "
                "subsequent CV score optimistically. Report as 'fixed preprocessing, then CV', "
                "not as unbiased internal validation.\n"
                f"📊 Initial Shape: {initial_shape}\n"
                f"{_format_missing_value_log(inf_summary, inf_report_paths)}"
                f"✅ Compounds Removed: {cpd_removed}\n"
                f"✅ Descriptors Removed: {desc_removed}\n"
                f"✅ Values Imputed: {imputed} ('{imputation_method}')\n"
                f"📊 Final Shape: {final_shape}\n"
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
                f"💾 Output: {os.path.basename(output_file)}\n"
                "========================================"
            )
            return {"ui": {"text": summary}, "result": (str(output_file),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}


class Paired_Descriptor_Preprocessing_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "training_data_path": ("STRING", {"tooltip": "3. Data Split's TRAINING_DATA output."}),
                "holdout_data_path": ("STRING", {"tooltip": "3. Data Split's X_TEST output (the descriptor file -- not Y_TEST)."}),
                "holdout_targets_path": ("STRING", {"tooltip": "3. Data Split's Y_TEST output. Test compounds excluded here for "
                                                        "missingness are also removed from this file (same Names) and "
                                                        "returned as FILTERED_HOLDOUT_TARGETS -- use that output in 8. Model "
                                                        "Validation, not the original Y_TEST, or the two will disagree "
                                                        "whenever a test compound gets dropped."}),
                "target_column": ("STRING", {"default": "value"}),
                "compound_nan_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                                       "tooltip": "Applied independently to train and test, AFTER column restriction (step below) -- not a fit, just a per-row threshold, so this is safe on both sides."}),
                "descriptor_nan_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                                         "tooltip": "Column retention is decided from TRAINING missingness only; the same retained columns are then applied to test."}),
                "imputation_method": (["mean", "median", "most_frequent"],),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("PREPROCESSED_TRAINING", "PREPROCESSED_HOLDOUT", "FILTERED_HOLDOUT_TARGETS", "PREPROCESSING_RECIPE")
    OUTPUT_TOOLTIPS = (
        None, None, None,
        "Training-fitted imputation values. Feed to the Screener.",
    )
    FUNCTION = "run"
    CATEGORY = "QSAR/2. REGRESSION"
    OUTPUT_NODE = True

    def run(self, training_data_path, holdout_data_path, holdout_targets_path, target_column,
            compound_nan_threshold, descriptor_nan_threshold, imputation_method):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Regression", "04_Descriptor_Preprocessing")
            os.makedirs(output_dir, exist_ok=True)
            train_df = pd.read_csv(training_data_path, dtype={"Name": str})
            test_df = pd.read_csv(holdout_data_path, dtype={"Name": str})
            y_test_df = pd.read_csv(holdout_targets_path, dtype={"Name": str})
            if target_column not in train_df.columns:
                raise ValueError(f"Target column '{target_column}' not found in training_data_path.")

            train_out, test_out, y_test_out, recipe, excluded_df = paired_preprocess_regression(
                train_df, test_df, y_test_df, target_column, compound_nan_threshold, descriptor_nan_threshold, imputation_method
            )

            train_path = os.path.join(output_dir, "train_preprocessed.csv")
            test_path = os.path.join(output_dir, "test_preprocessed.csv")
            y_test_filtered_path = os.path.join(output_dir, "y_test_filtered.csv")
            recipe_path = os.path.join(output_dir, "preprocessing_recipe.json")
            report_path = os.path.join(output_dir, "excluded_compounds_report.csv")

            train_out.to_csv(train_path, index=False)
            test_out.to_csv(test_path, index=False)
            y_test_out.to_csv(y_test_filtered_path, index=False)
            with open(recipe_path, "w") as f:
                json.dump(recipe, f, indent=2)
            if excluded_df.empty:
                excluded_df = pd.DataFrame(columns=["Name", "SMILES", "source", "row_nan_fraction", "threshold", "reason"])
            excluded_df.to_csv(report_path, index=False)

            n_excluded_train = int((excluded_df["source"] == "train").sum()) if len(excluded_df) else 0
            n_excluded_test = int((excluded_df["source"] == "test").sum()) if len(excluded_df) else 0
            log_message = (
                "========================================\n"
                "🔹 Paired Descriptor Preprocessing Complete! 🔹\n"
                "========================================\n"
                "ℹ️ Statistics fit on TRAINING only, applied unchanged to test (detail in recipe JSON).\n"
                f"📊 Training rows in: {len(train_df)}, out: {len(train_out)} (excluded for missingness: {n_excluded_train})\n"
                f"📊 Test rows in: {len(test_df)}, out: {len(test_out)} (excluded for missingness: {n_excluded_test})\n"
                f"📊 Y_test rows in: {len(y_test_df)}, out: {len(y_test_out)} (filtered to match PREPROCESSED_HOLDOUT)\n"
                f"✅ Descriptors retained: {recipe['n_retained_descriptors']} / "
                f"{recipe['n_retained_descriptors'] + recipe['n_dropped_descriptors']} "
                f"(dropped: {recipe['n_dropped_descriptors']}, threshold={descriptor_nan_threshold})\n"
                f"✅ Imputation method: '{imputation_method}'\n"
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
                f"💾 Train: {os.path.basename(train_path)}\n"
                f"💾 Test: {os.path.basename(test_path)}\n"
                f"💾 Y_test Filtered: {os.path.basename(y_test_filtered_path)}\n"
                f"💾 Recipe: {os.path.basename(recipe_path)}\n"
                f"💾 Excluded Compounds Report: {os.path.basename(report_path)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(train_path), str(test_path), str(y_test_filtered_path), str(recipe_path))}
        except Exception as e:
            error_msg = f"❌ Error in paired preprocessing: {e}\n\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": error_msg}, "result": ("", "", "", "")}


NODE_CLASS_MAPPINGS = {
    "Remove_high_nan_compounds_Regression": Remove_high_nan_compounds_Regression,
    "Remove_high_nan_descriptors_Regression": Remove_high_nan_descriptors_Regression,
    "Impute_missing_values_Regression": Impute_missing_values_Regression,
    "Descriptor_preprocessing_Regression": Descriptor_preprocessing_Regression,
    "Paired_Descriptor_Preprocessing_Regression": Paired_Descriptor_Preprocessing_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Remove_high_nan_compounds_Regression": "Remove High NaN Compounds",
    "Remove_high_nan_descriptors_Regression": "Remove High NaN Descriptors",
    "Impute_missing_values_Regression": "Impute Missing Values",
    "Descriptor_preprocessing_Regression": "Descriptor Preprocessing (Whole-Dataset / No Split)",
    "Paired_Descriptor_Preprocessing_Regression": "4. Descriptor Preprocessing",
}
