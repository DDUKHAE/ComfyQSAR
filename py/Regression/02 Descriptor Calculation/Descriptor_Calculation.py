import os
import sys
import json
import time
import glob
import subprocess
import numpy as np
import pandas as pd
import multiprocessing
import traceback
import tempfile
import folder_paths
from padelpy import padeldescriptor

# Runs padeldescriptor() in a brand-new interpreter process (plain
# `python -c <script>` subprocess, JSON args on argv -- nothing pickled)
# rather than multiprocessing.Process: this platform's custom nodes are
# loaded via importlib.util.spec_from_file_location with a filesystem-
# path-derived module name (see ComfyUI's load_custom_node), which is
# registered in *this* process's sys.modules but is not resolvable by a
# fresh interpreter via a normal `import <name>` -- so pickling a function
# defined in this module for multiprocessing's spawn start method fails
# the moment the child tries to un-pickle it (confirmed empirically). A
# plain subprocess has no such requirement: it only needs `padelpy`
# importable, which it is, since it runs with the same interpreter
# (sys.executable).
_PADEL_SUBPROCESS_SCRIPT = (
    "import sys, json\n"
    "from padelpy import padeldescriptor\n"
    "padeldescriptor(**json.loads(sys.argv[1]))\n"
)


def _cleanup_padel_outputs(d_file):
    """Removes the expected output file plus any split files PaDEL's own
    maxcpdperfile batching would leave behind (<stem>_1<ext>, <stem>_2<ext>,
    ...), so a previous attempt's output can never be mistaken for, or
    silently mixed with, the next attempt's."""
    if not d_file:
        return
    stem, ext = os.path.splitext(d_file)
    for path in [d_file] + glob.glob(f"{stem}_*{ext}"):
        try:
            os.remove(path)
        except OSError:
            pass


def _run_padel_with_retry(options, max_retries=3, retry_delay=2):
    """padeldescriptor() (PaDEL's own Java worker, via padelpy) can
    occasionally raise a CDK-side exception or silently produce no output
    for reasons unrelated to the input molecules. Each attempt runs in a
    fresh, isolated Python process (see _PADEL_SUBPROCESS_SCRIPT above)
    and any leftover output from the previous attempt is cleared first, so
    a reviewer running this platform themselves doesn't get stuck on a
    transient failure. Callers must not pass a nonzero maxcpdperfile here:
    that option makes PaDEL split its output into <stem>_1<ext>,
    <stem>_2<ext>, ... instead of writing d_file directly, which this
    function (and the row-recovery logic that reads d_file afterward)
    does not handle -- confirmed empirically to produce a 0-byte/missing
    base d_file even when the split files themselves contain valid
    descriptor rows."""
    last_error = None
    d_file = options.get("d_file")
    for attempt in range(1, max_retries + 1):
        _cleanup_padel_outputs(d_file)
        proc = subprocess.run(
            [sys.executable, "-c", _PADEL_SUBPROCESS_SCRIPT, json.dumps(options)],
            capture_output=True, text=True,
        )
        ok_output = d_file and os.path.exists(d_file) and os.path.getsize(d_file) > 0
        if proc.returncode == 0 and ok_output:
            if attempt > 1:
                print(f"PaDEL-Descriptor succeeded on attempt {attempt}/{max_retries}.")
            return
        if proc.returncode != 0:
            last_error = proc.stderr.strip()[-2000:] if proc.stderr else f"subprocess exited with code {proc.returncode}"
        else:
            last_error = (
                "padeldescriptor() returned without raising, but produced no output "
                "(0-byte or missing d_file)."
            )
        print(f"PaDEL-Descriptor attempt {attempt}/{max_retries} failed: {last_error}")
        if attempt < max_retries:
            time.sleep(retry_delay)
    _cleanup_padel_outputs(d_file)
    raise RuntimeError(f"PaDEL-Descriptor failed after {max_retries} attempt(s). Last error: {last_error}")


def build_padel_options(descriptor_type, detect_aromaticity, remove_salt, standardize_nitro,
                        retain_order, threads, waiting_jobs,
                        max_runtime, max_cpd_per_file, headless, log):
    # maxcpdperfile > 0 makes PaDEL split its output into <stem>_1<ext>,
    # <stem>_2<ext>, ... instead of writing to d_file directly -- neither
    # _run_padel_with_retry nor the row-recovery logic that reads d_file
    # afterward handles those split files, so a nonzero value here
    # reliably surfaces as a 0-byte/missing d_file (confirmed
    # empirically). Forced to 0 (unlimited, single output file) rather
    # than exposed as a user-facing option.
    max_cpd_per_file = 0
    return {
        "fingerprints": False,
        "d_2d": descriptor_type,
        "d_3d": not descriptor_type,
        "detectaromaticity": detect_aromaticity,
        "removesalt": remove_salt,
        "standardizenitro": standardize_nitro,
        # Always False: PaDEL's "use filename as molecule name" mode
        # overrides the per-molecule index tag this module relies on for
        # safe drop/value re-matching (confirmed empirically -- see the
        # Classification counterpart for the same fix). Not exposed to
        # the user since nothing here can recover from that override.
        "usefilenameasmolname": False,
        "retainorder": retain_order,
        "threads": threads,
        "waitingjobs": waiting_jobs,
        "maxruntime": max_runtime,
        "maxcpdperfile": max_cpd_per_file,
        "headless": headless,
        "log": log,
    }


def sanitize_invalid_numeric(df, id_cols):
    """Any non-id column that isn't already numeric dtype is a descriptor
    column carrying stray non-numeric garbage (e.g. a PaDEL error string in
    one cell forces the whole pandas column to object dtype) -- coerce it to
    numeric so unparseable cells become NaN and flow into the normal
    missing-value pipeline, instead of silently riding through as a string
    all the way to model training. Must run before any select_dtypes(
    include=np.number) filtering downstream, or these columns would be
    invisible to the missing-value audit entirely."""
    candidate_cols = [c for c in df.columns if c not in id_cols]
    df = df.copy()
    n_newly_nan = 0
    coerced_cols = []
    for c in candidate_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            before = df[c].notna().sum()
            df[c] = pd.to_numeric(df[c], errors="coerce")
            after = df[c].notna().sum()
            newly_nan = int(before - after)
            if newly_nan > 0:
                n_newly_nan += newly_nan
                coerced_cols.append(c)
    return df, n_newly_nan, coerced_cols


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


def _format_missing_value_log(n_coerced, coerced_cols, summary, report_paths):
    # "Missing values" (plain English) replaces "NaN"/"coerced"/"inf" jargon,
    # and a clean/expected cell (PaDEL simply couldn't compute a descriptor)
    # collapses to one line; only genuinely unusual findings (stray
    # non-numeric text, infinite values) get their own called-out sub-line,
    # and only when they actually occurred.
    total = summary["total_missing_or_inf"]
    if total == 0:
        return "✅ No missing or invalid values found\n"

    lines = [
        f"✅ Missing values: {total} cell(s) across "
        f"{summary['n_affected_descriptors']} descriptor(s), "
        f"{summary['n_affected_compounds']} compound(s)"
    ]
    if n_coerced:
        preview = ", ".join(coerced_cols[:5]) + ("..." if len(coerced_cols) > 5 else "")
        lines.append(
            f"   ⚠️  Non-numeric text found (treated as missing): {n_coerced} "
            f"cell(s) in {len(coerced_cols)} descriptor(s): {preview}"
        )
    if summary["total_inf"]:
        lines.append(f"   ⚠️  Infinite values found (treated as missing): {summary['total_inf']} cell(s)")
        lines.append(
            f"   ⚠️  Warning: Infinite descriptor values were detected. Review the affected "
            f"structures and descriptors in {os.path.basename(report_paths['detail'])} before modeling."
        )
    top = ", ".join(f"{r['descriptor']}({r['n_total_missing_or_inf']})" for r in summary["top_descriptors"])
    lines.append(f"📌 Most affected descriptors: {top}")
    lines.append(
        f"📋 Reports: {os.path.basename(report_paths['detail'])}, "
        f"{os.path.basename(report_paths['by_descriptor'])}, "
        f"{os.path.basename(report_paths['by_compound'])}"
    )
    return "\n".join(lines) + "\n"


def validate_metadata_and_target(df, target_column, id_cols):
    """Name blank/duplicate check (always, if Name is present) + target
    existence/numeric/finite/NaN check (only if target_column is actually
    present -- absent is expected and NOT an error for screening-only
    inputs that have no target yet, e.g. external candidates). Regression
    has no 0/1 constraint to check, but unlike Classification (where a
    non-numeric label already fails the {0,1} set-equality check for free)
    a stray non-numeric or infinite value here has nothing else to catch
    it, so it must be checked explicitly."""
    notes = []
    if "Name" in df.columns:
        names = df["Name"].astype(str).str.strip()
        names = names.where(names.str.lower() != "nan", "")
        n_blank = int((names == "").sum())
        if n_blank:
            raise ValueError(f"{n_blank} row(s) have a blank/missing Name value.")
        dup = names[names.duplicated()]
        if len(dup):
            dup_list = sorted(set(dup.tolist()))
            preview = dup_list[:10]
            more = f" (+{len(dup_list) - 10} more)" if len(dup_list) > 10 else ""
            raise ValueError(f"Duplicate Name value(s): {preview}{more}")

    if target_column in df.columns:
        y_raw = df[target_column]
        y_numeric = pd.to_numeric(y_raw, errors="coerce")
        bad_mask = y_numeric.isna() & y_raw.notna()
        if bad_mask.any():
            bad_vals = sorted(set(y_raw[bad_mask].astype(str).tolist()))
            preview = bad_vals[:10]
            more = f" (+{len(bad_vals) - 10} more)" if len(bad_vals) > 10 else ""
            raise ValueError(f"Target column '{target_column}' has non-numeric value(s): {preview}{more}")
        if y_numeric.isna().any():
            n_nan = int(y_numeric.isna().sum())
            raise ValueError(f"Target column '{target_column}' has {n_nan} missing (NaN) value(s).")
        if not np.isfinite(y_numeric.to_numpy()).all():
            n_inf = int((~np.isfinite(y_numeric.to_numpy())).sum())
            raise ValueError(f"Target column '{target_column}' has {n_inf} non-finite (inf/-inf) value(s).")
    else:
        notes.append(
            f"ℹ️ Target column '{target_column}' not found -- skipping label validation "
            "(expected for a screening-only input with no target yet).\n"
        )
    return notes


def sanitize_descriptors_file(descriptors_path, target_column, output_dir):
    """Standalone, dict-returning sanitization step (coerce invalid
    numerics, audit/replace inf, validate Name/target) -- same shape as the
    former standalone "2b" node's run(). Kept as a plain function, not
    folded directly into the class method below, so it can also be called
    directly with a synthetic descriptor CSV in tests that need to inject
    deliberately-bad values (all-NaN columns, infinite targets, etc.)
    without going through real PaDEL."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(descriptors_path, dtype={"Name": str})
        id_cols = [c for c in ("Name", "SMILES", target_column) if c in df.columns]
        df, n_coerced, coerced_cols = sanitize_invalid_numeric(df, id_cols)
        df, inf_summary, report_paths = audit_missing_and_infinite_values(df, output_dir, id_cols)
        notes = validate_metadata_and_target(df, target_column, id_cols)

        output_file = os.path.join(output_dir, "sanitized_descriptors.csv")
        df.to_csv(output_file, index=False)

        rel_dir = os.path.relpath(output_dir, folder_paths.get_output_directory())
        log_message = (
            f"{_format_missing_value_log(n_coerced, coerced_cols, inf_summary, report_paths)}"
            f"{''.join(notes)}"
            f"📁 Directory: {rel_dir}{os.sep}\n"
            f"💾 Output: {os.path.basename(output_file)}\n"
        )
        return {"ui": {"text": log_message}, "result": (str(output_file),)}
    except Exception as e:
        return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}


class Descriptor_Calculations_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "standardized_data_path": ("STRING", {"tooltip": "From 1's STANDARDIZED_DATA, not a raw file."}),
                "advanced": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "descriptor_type": ("BOOLEAN", {"default": True, "label_on": "2D", "label_off": "3D"}),
                "detect_aromaticity": ("BOOLEAN", {"default": True}),
                "remove_salt": ("BOOLEAN", {"default": True}),
                "standardize_nitro": ("BOOLEAN", {"default": True}),
                "log": ("BOOLEAN", {"default": False}),
                "retain_order": ("BOOLEAN", {"default": True}),
                "max_runtime": ("INT", {"default": 100000, "min": 1000, "max": 100000, "step": 1000}),
                "headless": ("BOOLEAN", {"default": True}),
                "threads": ("INT", {"default": -1, "min": -1, "max": multiprocessing.cpu_count(), "step": 1}),
                "waiting_jobs": ("INT", {"default": -1, "min": -1, "max": multiprocessing.cpu_count(), "step": 1}),
                "target_column": ("STRING", {
                    "default": "value",
                    "tooltip": "Only validated if present in the descriptor output -- absent is fine "
                               "for screening-only inputs with no target yet.",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DESCRIPTOR_MATRIX",)
    FUNCTION = "calculate_descriptors"
    CATEGORY = "QSAR/2. REGRESSION"
    OUTPUT_NODE = True

    def calculate_descriptors(self, standardized_data_path, advanced, descriptor_type=True,
                               detect_aromaticity=True, remove_salt=True, standardize_nitro=True,
                               log=False, retain_order=True,
                               max_runtime=10000, headless=True,
                               threads=-1, waiting_jobs=-1, target_column="value"):
        output_dir = os.path.join(folder_paths.get_output_directory(), "Regression", "02_Descriptor_Calculation")
        os.makedirs(output_dir, exist_ok=True)
        try:
            df = pd.read_csv(standardized_data_path)
            smiles_col = next((c for c in df.columns if 'smiles' in c.lower()), None)
            value_col = next((c for c in df.columns if c.lower() == 'value'), None)
            if smiles_col is None or value_col is None:
                raise ValueError("Input CSV must have 'SMILES' and 'value' columns.")
            # Drop NaN-SMILES rows from df itself (not just from a separate
            # smiles_list) before deriving smiles_list/values -- dropping
            # only smiles_list would shift it out of alignment with values
            # (still the original, unfiltered length) for every row after
            # the first NaN SMILES, silently pairing values with the wrong
            # compound. This node accepts an arbitrary user-supplied CSV, not
            # only Standardization's own (already-clean) output.
            df = df.dropna(subset=[smiles_col]).reset_index(drop=True)
            smiles_list = df[smiles_col].astype(str).tolist()
            values = df[value_col].values

            # Tag each molecule with its original index (SMILES<TAB>index) so
            # PaDEL's output can be matched back by an explicit id instead of
            # row position. PaDEL can silently drop molecules it can't
            # process, and empirically can drop *every subsequent* molecule
            # in the same batch too -- naive positional zipping (the previous
            # `values[:len(df_desc)]` fallback) would then silently pair
            # values with the wrong compounds.
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.smi', newline='') as tmp:
                tmp.write('\n'.join(f"{smi}\t{i}" for i, smi in enumerate(smiles_list)))
                smi_path = tmp.name

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_out:
                desc_path = tmp_out.name

            padel_options = build_padel_options(
                descriptor_type, detect_aromaticity, remove_salt, standardize_nitro,
                retain_order, threads, waiting_jobs,
                max_runtime, 0, headless, log
            )
            padel_options['mol_dir'] = smi_path
            padel_options['d_file'] = desc_path
            _run_padel_with_retry(padel_options)
            os.remove(smi_path)

            if not (os.path.exists(desc_path) and os.path.getsize(desc_path) > 0):
                raise ValueError("PaDEL descriptor calculation produced no output.")

            df_desc = pd.read_csv(desc_path)
            os.remove(desc_path)

            survived_idx = None
            try:
                survived_idx = df_desc['Name'].astype(int).tolist()
            except (ValueError, KeyError, TypeError):
                survived_idx = None

            n = len(smiles_list)
            valid = (
                survived_idx is not None
                and len(survived_idx) == len(df_desc)
                and len(set(survived_idx)) == len(survived_idx)
                and all(0 <= i < n for i in survived_idx)
            )
            if not valid:
                # No positional fallback even when the row count happens
                # to match len(values): a matching count does not prove
                # nothing was dropped/duplicated, and guessing here is
                # exactly the silent value/compound misalignment this
                # module exists to prevent.
                raise ValueError(
                    f"Could not recover a safe per-molecule index from PaDEL's "
                    f"'Name' output column for {len(df_desc)} row(s) (expected "
                    f"unique indices in [0, {n})). Refusing to align values "
                    "with descriptors by row position (this previously caused "
                    "silent compound/value misalignment when PaDEL dropped a "
                    "non-trailing molecule)."
                )
            dropped_indices = sorted(set(range(n)) - set(survived_idx))
            df_desc['value'] = [values[i] for i in survived_idx]
            df_desc['SMILES'] = [smiles_list[i] for i in survived_idx]

            merged_file = os.path.join(output_dir, "descriptors_with_values.csv")
            df_desc.to_csv(merged_file, index=False)

            dropped_set = set(dropped_indices)
            report_rows = [
                {"original_index": i, "SMILES": smi, "status": "dropped" if i in dropped_set else "kept"}
                for i, smi in enumerate(smiles_list)
            ]
            report_path = os.path.join(output_dir, "descriptor_calculation_report.csv")
            pd.DataFrame(report_rows).to_csv(report_path, index=False)

            dropped_note = ""
            # output_dir is reused across runs of this node, so a stale
            # padel_dropped_compounds.csv from a previous (dropped) run
            # must not linger and be mistaken for this run's result.
            dropped_path = os.path.join(output_dir, "padel_dropped_compounds.csv")
            if dropped_indices:
                pd.DataFrame({
                    "original_index": dropped_indices,
                    "SMILES": [smiles_list[i] for i in dropped_indices],
                }).to_csv(dropped_path, index=False)
                dropped_note = (
                    f"⚠️ PaDEL silently dropped {len(dropped_indices)} compound(s) -- see "
                    f"{os.path.basename(dropped_path)} / {os.path.basename(report_path)}\n"
                )
            elif os.path.exists(dropped_path):
                os.remove(dropped_path)

            non_descriptor_cols = sum(c in df_desc.columns for c in ("Name", "SMILES", "value"))

            # Sanitization (formerly node "2b"): coerce stray non-numeric
            # cells, audit/replace inf, validate Name/target -- merged into
            # this node so a workflow can never wire node 3 straight to the
            # raw PaDEL output and skip it by accident.
            sanitize_result = sanitize_descriptors_file(merged_file, target_column, output_dir)
            if not sanitize_result["result"][0]:
                return {
                    "ui": {"text": f"❌ Error during descriptor sanitization: {sanitize_result['ui']['text']}"},
                    "result": ("",),
                }
            sanitized_file = sanitize_result["result"][0]

            log_message = (
                "========================================\n"
                "🔹 Regression Descriptor Calculation & Sanitization Done! 🔹\n"
                "========================================\n"
                f"✅ Compounds: {len(df_desc)} (submitted {len(smiles_list)}, dropped: {len(dropped_indices)})\n"
                f"🔢 Descriptors: {df_desc.shape[1] - non_descriptor_cols}\n"
                f"{dropped_note}"
                f"{sanitize_result['ui']['text']}"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(sanitized_file),)}
        except RuntimeError:
            # PaDEL failed even after isolated-process retries. Deliberately
            # NOT caught into a UI-text/empty-result return here: ComfyUI
            # treats any normal return value as a completed, cacheable
            # execution, so swallowing this would let a genuinely failed run
            # be served back as a "successful" cached result on the next
            # queue -- exactly what made repeated failures look like the
            # same single attempt during the QDB116 investigation. Letting
            # it propagate marks this execution as failed instead.
            raise
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}

NODE_CLASS_MAPPINGS = {
    "Descriptor_Calculations_Regression": Descriptor_Calculations_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Descriptor_Calculations_Regression": "2. Descriptor Calculation",
}
