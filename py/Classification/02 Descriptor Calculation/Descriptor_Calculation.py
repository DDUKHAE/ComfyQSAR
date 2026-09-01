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
from rdkit import Chem
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
    function (and the row-recovery logic in calculate_descriptors_from_file
    that reads d_file afterward) does not handle -- confirmed empirically
    to produce a 0-byte/missing base d_file even when the split files
    themselves contain valid descriptor rows."""
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


def calculate_descriptors_from_file(input_path, options):
    """
    Returns (descriptor_records, original_smiles, dropped_indices).

    PaDEL can silently drop molecules it fails to process -- and, worse,
    can silently drop *every subsequent* molecule in the same batch once
    it hits one it can't handle (confirmed empirically: a single
    unparseable SMILES mid-batch caused all later, valid molecules to
    disappear from PaDEL's output with no error). Naively zipping PaDEL's
    output back to the input by row position is therefore unsafe. Instead,
    each molecule is tagged with its original index (SMILES<TAB>index) in
    the temp .smi file PaDEL consumes; PaDEL preserves this tag in its
    "Name" output column, so surviving rows can be matched back to their
    real input index (and hence their real SMILES) unambiguously, and
    genuinely-dropped compounds can be identified and reported instead of
    silently vanishing.
    """
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv') as tmp_desc_file:
        desc_output_path = tmp_desc_file.name

    options['d_file'] = desc_output_path

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    original_smiles = []

    if input_path.lower().endswith('.sdf'):
        # Mirrors the .smi branch's index-tagging: PaDEL can silently drop
        # SDF entries too, and its auto-generated Name ("AUTOGEN_<file>_<n>")
        # is a 1-indexed *output position*, not the original input index --
        # useless for recovering which compounds survived once a drop
        # happens. So each RDKit-parseable molecule's title line (_Name) is
        # set to its original index and the batch is rewritten to a fresh
        # temp SDF before handing it to PaDEL, exactly like the tagged .smi
        # file used for CSV/.smi input. Molecules RDKit itself can't parse
        # are never sent to PaDEL at all (recorded as dropped immediately) --
        # this brings SDF input in line with how Standardization already
        # gates SMI/CSV input through RDKit before descriptor calculation.
        suppl = Chem.SDMolSupplier(input_path, removeHs=False, strictParsing=False)
        tagged_mols = []
        for i, m in enumerate(suppl):
            if m is None:
                original_smiles.append(None)
                continue
            original_smiles.append(Chem.MolToSmiles(m))
            m.SetProp('_Name', str(i))
            tagged_mols.append(m)

        if not tagged_mols:
            raise ValueError(f"No valid molecules could be parsed by RDKit from {os.path.basename(input_path)}.")

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sdf') as tmp_sdf_file:
            sdf_input_path = tmp_sdf_file.name
        writer = Chem.SDWriter(sdf_input_path)
        for m in tagged_mols:
            writer.write(m)
        writer.close()

        options['mol_dir'] = sdf_input_path
        _run_padel_with_retry(options)
        os.remove(sdf_input_path)

    elif input_path.lower().endswith(('.csv', '.smi')):
        try:
            if input_path.lower().endswith('.smi'):
                # .smi files are headerless (one SMILES per line) -- reading
                # with the default header='infer' silently drops the first
                # compound by mistaking it for a column name (same bug
                # already fixed in Data_Load_and_Standardization.py).
                df = pd.read_csv(input_path, header=None, skip_blank_lines=True)
                df.columns = ['SMILES']
            else:
                df = pd.read_csv(input_path, skip_blank_lines=True)
            smiles_col = next((col for col in df.columns if 'smiles' in col.lower()), None)
            if smiles_col is None:
                if not df.empty:
                    smiles_col = df.columns[0]
                    df.rename(columns={df.columns[0]: 'SMILES'}, inplace=True)
                else:
                    raise ValueError(f"Input file {os.path.basename(input_path)} is empty or has no identifiable SMILES column.")
            else:
                df.rename(columns={smiles_col: 'SMILES'}, inplace=True)

            original_smiles = df['SMILES'].dropna().astype(str).tolist()
            if not original_smiles:
                raise ValueError(f"No valid SMILES strings found in {os.path.basename(input_path)} after filtering NaN values.")

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.smi', newline='') as tmp_smi_file:
                tmp_smi_file.write('\n'.join(f"{smi}\t{i}" for i, smi in enumerate(original_smiles)))
                smi_input_path = tmp_smi_file.name

        except pd.errors.EmptyDataError:
            raise ValueError(f"Input file {os.path.basename(input_path)} is empty.")
        except (ValueError, KeyError) as e:
            raise ValueError(f"Failed to read SMILES from {os.path.basename(input_path)}. Error: {e}")

        # Deliberately outside the try/except above: that block exists only
        # to give SMILES-parsing failures (bad CSV, no SMILES column) a
        # clear message, and must not catch _run_padel_with_retry's
        # RuntimeError -- doing so would relabel a genuine PaDEL failure as
        # "failed to read SMILES" and let it be caught by this function's
        # caller as a graceful, cacheable empty result instead of a real
        # node execution failure.
        options['mol_dir'] = smi_input_path
        _run_padel_with_retry(options)
        os.remove(smi_input_path)
    else:
        raise ValueError(f"Unsupported file format: {os.path.basename(input_path)}. Supported: .sdf, .csv, .smi")

    if os.path.exists(desc_output_path) and os.path.getsize(desc_output_path) > 0:
        df_desc = pd.read_csv(desc_output_path)
        os.remove(desc_output_path)

        dropped_indices = []
        if original_smiles:
            survived_idx = None
            try:
                survived_idx = df_desc['Name'].astype(int).tolist()
            except (ValueError, KeyError, TypeError):
                survived_idx = None

            n = len(original_smiles)
            valid = (
                survived_idx is not None
                and len(survived_idx) == len(df_desc)
                and len(set(survived_idx)) == len(survived_idx)
                and all(0 <= i < n for i in survived_idx)
            )
            if not valid:
                # Deliberately no positional fallback here, even when the
                # row count happens to match: a matching count does not
                # prove nothing was dropped (PaDEL could in principle drop
                # one and duplicate another), and this module's whole
                # purpose is to never assign SMILES to a compound by guessing
                # from row position. An unrecoverable index is a loud
                # failure, not a silent SMILES/descriptor mismatch.
                raise ValueError(
                    f"Could not recover a safe per-molecule index from PaDEL's "
                    f"'Name' output column for {len(df_desc)} row(s) (expected "
                    f"unique indices in [0, {n})). Refusing to align SMILES "
                    "with descriptors by row position."
                )
            df_desc['SMILES'] = [original_smiles[i] for i in survived_idx]
            dropped_indices = sorted(set(range(n)) - set(survived_idx))

        return df_desc.to_dict(orient='records'), original_smiles, dropped_indices
    return [], original_smiles, list(range(len(original_smiles)))

def build_padel_options(descriptor_type, detect_aromaticity, remove_salt, standardize_nitro,
                        retain_order, threads, waiting_jobs,
                        max_runtime, max_cpd_per_file, headless, log):
    # maxcpdperfile > 0 makes PaDEL split its output into <stem>_1<ext>,
    # <stem>_2<ext>, ... instead of writing to d_file directly -- neither
    # _run_padel_with_retry nor calculate_descriptors_from_file's row-
    # recovery logic (below) reads those split files, so a nonzero value
    # here reliably surfaces as a 0-byte/missing d_file (confirmed
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
        # Always False, regardless of any past caller setting: PaDEL's
        # "use filename as molecule name" mode overrides the per-molecule
        # _Name/index tag this module relies on for safe drop-tracking
        # (confirmed empirically -- it silently reverts every molecule in
        # a batch to "<filename>_<position>", defeating the tag). Since
        # nothing here can recover from that silently, this option is not
        # exposed to the user at all.
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
    downstream (excluding id_cols, e.g. Name/SMILES/Label -- metadata/target
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
    existence/NaN/binary-0-1 check (only if target_column is actually
    present -- absent is expected and NOT an error for screening-only
    inputs that have no label yet, e.g. external candidates)."""
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
        y = df[target_column]
        if y.isna().any():
            n_nan = int(y.isna().sum())
            raise ValueError(f"Target column '{target_column}' has {n_nan} missing (NaN) value(s).")
        unique_labels = set(pd.unique(y))
        if unique_labels != {0, 1}:
            raise ValueError(
                f"Classification target column '{target_column}' must be binary-encoded as 0/1 "
                f"(found: {sorted(unique_labels, key=str)}). Encode binary labels as 0/1 before "
                "running this node."
            )
    else:
        notes.append(
            f"ℹ️ Target column '{target_column}' not found -- skipping label validation "
            "(expected for a screening-only input with no label yet).\n"
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


class Descriptor_Calculations_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "standardized_positive_path": ("STRING", {"tooltip": "From 1's STANDARDIZED_POSITIVE, not a raw file."}),
                "standardized_negative_path": ("STRING", {"tooltip": "From 1's STANDARDIZED_NEGATIVE, not a raw file."}),
                "advanced": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "descriptor_type": ("BOOLEAN", {"default": True, "label_on": "2D", "label_off": "3D",
                                                "tooltip": "Choose descriptor type: 2D (faster) or 3D (more detailed)"}),
                "detect_aromaticity": ("BOOLEAN", {"default": True, "tooltip": "Detect and handle aromatic structures"}),
                "remove_salt": ("BOOLEAN", {"default": True, "tooltip": "Remove salt components from molecules"}),
                "standardize_nitro": ("BOOLEAN", {"default": True, "tooltip": "Standardize nitro groups"}),
                "log": ("BOOLEAN", {"default": False, "tooltip": "Enable PaDEL-Descriptor's internal logging to a file"}),
                "retain_order": ("BOOLEAN", {"default": True, "tooltip": "Keep original molecule order"}),
                "max_runtime": ("INT", {"default": 100000, "min": 1000, "max": 100000, "step": 1000,
                                        "tooltip": "Maximum calculation time per molecule (milliseconds)"}),
                "headless": ("BOOLEAN", {"default": True,
                                         "tooltip": "Run PaDEL-Descriptor without GUI (recommended for servers)"}),
                "threads": ("INT", {"default": -1, "min": -1, "max": multiprocessing.cpu_count(), "step": 1,
                                    "tooltip": f"Number of CPU threads for PaDEL-Descriptor (-1 for all available, 1-{multiprocessing.cpu_count()})"}),
                "waiting_jobs": ("INT", {"default": -1, "min": -1, "max": multiprocessing.cpu_count(), "step": 1,
                                         "tooltip": "Number of concurrent PaDEL-Descriptor jobs in queue (-1 for auto)"}),
                "target_column": ("STRING", {
                    "default": "Label",
                    "tooltip": "Only validated if present in the merged descriptors -- absent is fine "
                               "for screening-only inputs with no label yet.",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DESCRIPTOR_MATRIX",)
    FUNCTION = "calculate_and_merge_descriptors"
    CATEGORY = "QSAR/1. CLASSIFICATION"
    OUTPUT_NODE = True

    def calculate_and_merge_descriptors(self, standardized_positive_path, standardized_negative_path, advanced, descriptor_type,
                                        detect_aromaticity, remove_salt, standardize_nitro,
                                        retain_order, threads, waiting_jobs,
                                        max_runtime, headless, log,
                                        target_column="Label"):
        output_dir = os.path.join(folder_paths.get_output_directory(), "Classification", "02_Descriptor_Calculation")
        os.makedirs(output_dir, exist_ok=True)

        padel_options = build_padel_options(
            descriptor_type, detect_aromaticity, remove_salt, standardize_nitro,
            retain_order, threads, waiting_jobs,
            max_runtime, 0, headless, log
        )

        try:
            print(f"Calculating descriptors for positive compounds from: {os.path.basename(standardized_positive_path)}")
            pos_descriptors_list, pos_original_smiles, pos_dropped = calculate_descriptors_from_file(standardized_positive_path, padel_options.copy())
            df_positive = pd.DataFrame(pos_descriptors_list)
            if df_positive.empty:
                print(f"Warning: No descriptors calculated for positive compounds from {os.path.basename(standardized_positive_path)}.")
            df_positive['Label'] = 1

            print(f"Calculating descriptors for negative compounds from: {os.path.basename(standardized_negative_path)}")
            neg_descriptors_list, neg_original_smiles, neg_dropped = calculate_descriptors_from_file(standardized_negative_path, padel_options.copy())
            df_negative = pd.DataFrame(neg_descriptors_list)
            if df_negative.empty:
                print(f"Warning: No descriptors calculated for negative compounds from {os.path.basename(standardized_negative_path)}.")
            df_negative['Label'] = 0

            # PaDEL's "Name" column is each file's own local 0-based index
            # (see calculate_descriptors_from_file's docstring) -- positive
            # and negative are calculated independently, so both start over
            # at 0. Concatenating them as-is produces duplicate Names
            # (e.g. positive row 0 and negative row 0 both "0"), which is
            # silently wrong rather than merely cosmetic: every downstream
            # node that identifies a compound by Name (the sanitization
            # step's blank/duplicate check onward) would either collide
            # outright or, worse, could misattribute a positive compound's
            # row to a negative one after any Name-based join. Prefixing
            # with the source makes every Name globally unique before
            # anything downstream ever sees it.
            if 'Name' in df_positive.columns:
                df_positive['Name'] = 'positive:' + df_positive['Name'].astype(str)
            if 'Name' in df_negative.columns:
                df_negative['Name'] = 'negative:' + df_negative['Name'].astype(str)

            df_final = pd.concat([df_positive, df_negative], ignore_index=True)
            if df_final.empty:
                raise ValueError("No descriptors were calculated for either positive or negative compounds.")

            if 'Name' in df_final.columns:
                name_col = df_final.pop('Name')
                df_final.insert(0, 'Name', name_col)

            merged_file = os.path.join(output_dir, "final_merged_descriptors.csv")
            df_final.to_csv(merged_file, index=False)

            pos_dropped_set = set(pos_dropped)
            neg_dropped_set = set(neg_dropped)
            report_rows = (
                [{"Name": f"positive:{i}", "source": "positive", "original_index": i, "SMILES": smi,
                  "status": "dropped" if i in pos_dropped_set else "kept"}
                 for i, smi in enumerate(pos_original_smiles)] +
                [{"Name": f"negative:{i}", "source": "negative", "original_index": i, "SMILES": smi,
                  "status": "dropped" if i in neg_dropped_set else "kept"}
                 for i, smi in enumerate(neg_original_smiles)]
            )
            report_path = os.path.join(output_dir, "descriptor_calculation_report.csv")
            pd.DataFrame(report_rows).to_csv(report_path, index=False)

            n_dropped = len(pos_dropped) + len(neg_dropped)
            dropped_note = ""
            # output_dir is reused across runs of the same node, so a stale
            # padel_dropped_compounds.csv from a previous (dropped) run must
            # not be left behind to be mistaken for this run's result.
            dropped_path = os.path.join(output_dir, "padel_dropped_compounds.csv")
            if n_dropped:
                pd.DataFrame(
                    [r for r in report_rows if r["status"] == "dropped"]
                ).to_csv(dropped_path, index=False)
                dropped_note = (
                    f"⚠️ PaDEL silently dropped {n_dropped} compound(s) -- see "
                    f"{os.path.basename(dropped_path)} / {os.path.basename(report_path)}\n"
                )
            elif os.path.exists(dropped_path):
                os.remove(dropped_path)

            non_descriptor_cols = sum(c in df_final.columns for c in ("Name", "SMILES", "Label"))

            # Sanitization (formerly node "2b"): coerce stray non-numeric
            # cells, audit/replace inf, validate Name/target -- merged into
            # this node so a workflow can never wire node 3 straight to the
            # raw PaDEL merge and skip it by accident.
            sanitize_result = sanitize_descriptors_file(merged_file, target_column, output_dir)
            if not sanitize_result["result"][0]:
                return {
                    "ui": {"text": f"❌ Error during descriptor sanitization: {sanitize_result['ui']['text']}"},
                    "result": ("",),
                }
            sanitized_file = sanitize_result["result"][0]

            log_message = (
                "========================================\n"
                "🔹 Descriptor Calculation & Sanitization Done! 🔹\n"
                "========================================\n"
                f"✅ Positive Molecules: {len(df_positive)} (submitted {len(pos_original_smiles)}, dropped: {len(pos_dropped)})\n"
                f"✅ Negative Molecules: {len(df_negative)} (submitted {len(neg_original_smiles)}, dropped: {len(neg_dropped)})\n"
                f"📊 Total Molecules: {len(df_final)}\n"
                f"🔢 Total Descriptors: {df_final.shape[1] - non_descriptor_cols} (excluding Name, SMILES, Label)\n"
                f"{dropped_note}"
                f"{sanitize_result['ui']['text']}"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(sanitized_file),)}

        except ReferenceError as e:
            if "Java JRE 6+ not found" in str(e):
                error_message = (
                    "❌ PaDEL-Descriptor Error: Java JRE Not Found!\n\n"
                    "PaDEL-Descriptor requires Java to run. Please install a Java Runtime Environment (JRE) or JDK (version 11 is recommended) on your system and restart ComfyUI.\n\n"
                    f"Original Error: {e}"
                )
                return {"ui": {"text": [error_message]}, "result": ("",)}
            raise e
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
        except (FileNotFoundError, ValueError) as e:
            error_message = f"❌ Error during descriptor calculation: {e}\n\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": [error_message]}, "result": ("",)}
        except Exception as e:
            error_message = f"❌ An unexpected error occurred during descriptor calculation: {e}\n\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": [error_message]}, "result": ("",)}

NODE_CLASS_MAPPINGS = {
    "Descriptor_Calculations_Classification": Descriptor_Calculations_Classification,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Descriptor_Calculations_Classification": "2. Descriptor Calculation",
}
