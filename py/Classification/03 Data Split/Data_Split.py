import os
import pandas as pd
import folder_paths
from sklearn.model_selection import train_test_split

class QSARDataSplit_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "descriptor_data_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "From 2. Do not feed an already-preprocessed file -- preprocessing "
                               "(4) runs after this split.",
                }),
                "test_size": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.05,
                    "max": 0.5,
                    "step": 0.05,
                    "display": "slider",
                }),
                "random_state": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("TRAINING_DATA", "HOLDOUT_DATA", "HOLDOUT_TARGETS")
    FUNCTION = "execute"
    CATEGORY = "QSAR/1. CLASSIFICATION"
    OUTPUT_NODE = True

    def execute(
        self,
        descriptor_data_path,
        test_size=0.2,
        random_state=42,
    ):
        # Fixed, not exposed as an INPUT_TYPES field -- Classification's
        # target column is always "Label" throughout this codebase.
        target_column = "Label"

        output_dir = os.path.join(folder_paths.get_output_directory(), "Classification", "03_Data_Split")
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

        # Name/SMILES are excluded from the split's own X (so stratification/
        # shape are unaffected) but are NOT dropped from the dataset -- they
        # ride along as passenger columns in every output file so that
        # downstream traceability/applicability-domain nodes can still
        # identify which compound each row is. train_test_split preserves
        # the original pandas index, so meta.loc[X_train.index] realigns
        # correctly without needing index=True on the CSVs (every
        # ML-consuming node downstream already excludes Name/SMILES from
        # its own feature matrix by name, not by position).
        metadata_cols = [c for c in ("Name", "SMILES") if c in df.columns]
        feature_cols = [c for c in df.columns if c not in metadata_cols and c != target_column]
        X = df[feature_cols]
        y = df[target_column]
        unique_labels = set(pd.unique(y))
        if unique_labels != {0, 1}:
            raise ValueError(
                f"Classification target column '{target_column}' must be binary-encoded as 0/1 "
                f"(found: {sorted(unique_labels, key=str)}). Encode binary labels as 0/1 before "
                "running this node."
            )
        meta = df[metadata_cols] if metadata_cols else None
        n_total   = len(df)
        n_class0  = int((y == 0).sum())
        n_class1  = int((y == 1).sum())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
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

        train_c0 = int((y_train == 0).sum())
        train_c1 = int((y_train == 1).sum())
        test_c0  = int((y_test  == 0).sum())
        test_c1  = int((y_test  == 1).sum())
        actual_test_ratio = len(X_test) / n_total

        # Absolute paths are mostly install-specific noise (everything up to
        # ".../output/" varies by machine) -- only the part after the output
        # root is actually stable/meaningful, so show that relative path once
        # and list the co-located files together instead of repeating the
        # full path four times.
        rel_dir = os.path.relpath(output_dir, folder_paths.get_output_directory())
        log_message = (
            "========================================\n"
            "🔹 Data Split Completed! 🔹\n"
            "========================================\n"
            f"📊 Total: {n_total} (Class 0: {n_class0}, Class 1: {n_class1})\n"
            f"✂️ Split: {1-actual_test_ratio:.0%} train / {actual_test_ratio:.0%} test (seed {random_state})\n"
            f"   Train: {len(X_train)} (Class 0: {train_c0}, Class 1: {train_c1})\n"
            f"   Test:  {len(X_test)} (Class 0: {test_c0}, Class 1: {test_c1})\n"
            f"📁 Directory: {rel_dir}{os.sep}\n"
            f"💾 Outputs: train_data.csv, test_data.csv, X_test.csv, Y_test.csv\n"
            "========================================"
        )

        return {"ui": {"text": log_message}, "result": (str(output_train), str(output_x_test), str(output_y_test))}

NODE_CLASS_MAPPINGS = {
    "QSARDataSplit_Classification": QSARDataSplit_Classification,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QSARDataSplit_Classification": "3. Data Split",
}
