import os
import numpy as np
import pandas as pd
import folder_paths
from sklearn.model_selection import train_test_split

class QSARDataSplit_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "descriptor_data_path": ("STRING", {
                    "default": "",
                    "tooltip": "From 2. Do not feed an already-preprocessed file -- preprocessing "
                               "(4) runs after this split.",
                }),
                "test_size": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.5, "step": 0.05}),
                "random_state": ("INT", {"default": 42, "min": 0, "max": 9999}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("TRAINING_DATA", "HOLDOUT_DATA", "HOLDOUT_TARGETS")
    FUNCTION = "execute"
    CATEGORY = "QSAR/2. REGRESSION"
    OUTPUT_NODE = True

    def execute(
        self,
        descriptor_data_path,
        test_size=0.2,
        random_state=42,
    ):
        # Fixed, not exposed as an INPUT_TYPES field -- Regression's
        # target column is always "value" throughout this codebase.
        target_column = "value"

        output_dir = os.path.join(folder_paths.get_output_directory(), "Regression", "03_Data_Split")
        os.makedirs(output_dir, exist_ok=True)
        # dtype={"Name": str} keeps an all-numeric-looking Name (e.g. "007")
        # as text -- otherwise pandas infers int64 and the leading zero is
        # gone before anything downstream can use it as an identifier.
        df = pd.read_csv(descriptor_data_path, dtype={"Name": str})

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        # Defensive re-check even though '2. Descriptor Calculation's
        # sanitization step should already have validated this -- this file
        # may not have come from node 2 at all.
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

        # Defensive re-check of the target too -- Regression has no {0,1}
        # constraint to catch a stray non-numeric value for free the way
        # Classification's label check does.
        y_numeric = pd.to_numeric(df[target_column], errors="coerce")
        bad_mask = y_numeric.isna() & df[target_column].notna()
        if bad_mask.any():
            bad_vals = sorted(set(df[target_column][bad_mask].astype(str).tolist()))
            raise ValueError(f"Target column '{target_column}' has non-numeric value(s): {bad_vals[:10]}")
        if y_numeric.isna().any():
            raise ValueError(f"Target column '{target_column}' has {int(y_numeric.isna().sum())} missing (NaN) value(s).")
        if not np.isfinite(y_numeric.to_numpy()).all():
            raise ValueError(f"Target column '{target_column}' has {int((~np.isfinite(y_numeric.to_numpy())).sum())} non-finite (inf/-inf) value(s).")

        # Name/SMILES are excluded from the split's own X (shape/ordering
        # unaffected) but not dropped from the dataset -- they ride along as
        # passenger columns in every output file so downstream traceability/
        # applicability-domain nodes can still identify each row's compound.
        # train_test_split preserves the original pandas index, so
        # meta.loc[X_train.index] realigns correctly.
        metadata_cols = [c for c in ("Name", "SMILES") if c in df.columns]
        feature_cols = [c for c in df.columns if c not in metadata_cols and c != target_column]
        X = df[feature_cols]
        y = df[target_column]
        meta = df[metadata_cols] if metadata_cols else None

        n_total = len(df)
        y_mean  = float(y.mean())
        y_std   = float(y.std())
        y_min   = float(y.min())
        y_max   = float(y.max())
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        def _with_meta(part_df, idx):
            if meta is None:
                return part_df
            meta_part = meta.loc[idx]
            out = part_df.copy()
            for col in reversed(metadata_cols):
                out.insert(0, col, meta_part[col].values)
            return out

        train_df = X_train.copy()
        train_df[target_column] = y_train.values
        train_df = _with_meta(train_df, X_train.index)

        test_df = X_test.copy()
        test_df[target_column] = y_test.values
        test_df = _with_meta(test_df, X_test.index)

        x_test_out = _with_meta(X_test, X_test.index)

        y_test_out = y_test.to_frame(target_column)
        y_test_out = _with_meta(y_test_out, X_test.index)

        output_train  = os.path.join(output_dir, "train_data.csv")
        output_test   = os.path.join(output_dir, "test_data.csv")
        output_x_test = os.path.join(output_dir, "X_test.csv")
        output_y_test = os.path.join(output_dir, "Y_test.csv")

        train_df.to_csv(output_train,  index=False)
        test_df.to_csv(output_test,   index=False)
        x_test_out.to_csv(output_x_test,  index=False)
        y_test_out.to_csv(output_y_test, index=False)

        actual_test_ratio = len(X_test) / n_total

        # Absolute paths are mostly install-specific noise (everything up to
        # ".../output/" varies by machine) -- only the part after the output
        # root is actually stable/meaningful, so show that relative path once
        # and list the co-located files together instead of repeating the
        # full path four times.
        rel_dir = os.path.relpath(output_dir, folder_paths.get_output_directory())
        log_message = (
            "========================================\n"
            "🔹 Data Split (Regression) Completed! 🔹\n"
            "========================================\n"
            f"📊 Total: {n_total} compounds -- target '{target_column}': "
            f"mean={y_mean:.3f}, std={y_std:.3f}, min={y_min:.3f}, max={y_max:.3f}\n"
            f"✂️ Split: {1-actual_test_ratio:.0%} train / {actual_test_ratio:.0%} test (seed {random_state})\n"
            f"   Train: {len(X_train)} (mean={float(y_train.mean()):.3f}, std={float(y_train.std()):.3f})\n"
            f"   Test:  {len(X_test)} (mean={float(y_test.mean()):.3f}, std={float(y_test.std()):.3f})\n"
            f"📁 Directory: {rel_dir}{os.sep}\n"
            f"💾 Outputs: train_data.csv, test_data.csv, X_test.csv, Y_test.csv\n"
            "========================================"
        )

        return {"ui": {"text": log_message}, "result": (output_train, output_x_test, output_y_test)}


NODE_CLASS_MAPPINGS = {
    "QSARDataSplit_Regression": QSARDataSplit_Regression
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QSARDataSplit_Regression": "3. Data Split"
}
