"""8.3 Resampling Stability Assessment (formerly "8c Resampling Validation"):
leave_n default consistency between
INPUT_TYPES and the run() signature (사용자-1), fixed_descriptor_set scope
marker presence on every method branch, and leave-N-out pooling's
duplication-invariance (a compound drawn in multiple repeats must appear
as separate, non-deduplicated rows in the pooled raw output -- that's what
lets repeat frequency act as an implicit weight)."""
import os
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _helpers  # noqa: E402


class TestRegression8cLeaveNSignatureConsistency(unittest.TestCase):
    def setUp(self):
        _helpers.fresh_output_dir()
        self.mod = _helpers.load_node_module("Regression", "08 Model Evaluation/08-3 Resampling Stability Assessment", "Resampling_Validation.py")

    def test_run_default_matches_input_types_default(self):
        input_types_default = self.mod.ResamplingValidation_Regression.INPUT_TYPES()["optional"]["leave_n"][1]["default"]
        import inspect
        run_default = inspect.signature(self.mod.ResamplingValidation_Regression.run).parameters["leave_n"].default
        self.assertEqual(input_types_default, run_default,
                          "INPUT_TYPES leave_n default and run() signature leave_n default must match")
        self.assertEqual(run_default, 2)


class TestRegression8cFixedDescriptorSetMarker(unittest.TestCase):
    def setUp(self):
        _helpers.fresh_output_dir()
        self.mod = _helpers.load_node_module("Regression", "08 Model Evaluation/08-3 Resampling Stability Assessment", "Resampling_Validation.py",
                                              alias="r8c_marker")
        rng = np.random.default_rng(0)
        n = 20
        X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
        y = X["f1"] * 2 + rng.normal(scale=0.1, size=n)
        self.df = X.copy()
        self.df["value"] = y.values
        self.input_path = "/tmp/test_8c_reg_input.csv"
        self.df.to_csv(self.input_path, index=False)
        import joblib
        model = LinearRegression().fit(X, y)
        self.model_path = "/tmp/test_8c_reg_model.pkl"
        joblib.dump(model, self.model_path)

    def _summary_csv(self):
        import folder_paths
        return os.path.join(folder_paths.get_output_directory(), "Regression", "08_Model_Evaluation", "Resampling_Stability",
                             "Resampling_Validation_Results.csv")

    def test_loocv_has_marker(self):
        node = self.mod.ResamplingValidation_Regression()
        node.run(self.input_path, self.model_path, "value", "loocv")
        summary = pd.read_csv(self._summary_csv())
        self.assertEqual(summary["test_scope"].iloc[0], "fixed_descriptor_set")

    def test_repeated_kfold_has_marker(self):
        node = self.mod.ResamplingValidation_Regression()
        node.run(self.input_path, self.model_path, "value", "repeated_kfold", n_repeats=3, cv_splits=3)
        summary = pd.read_csv(self._summary_csv())
        self.assertEqual(summary["test_scope"].iloc[0], "fixed_descriptor_set")

    def test_repeated_leave_n_out_has_marker(self):
        node = self.mod.ResamplingValidation_Regression()
        node.run(self.input_path, self.model_path, "value", "repeated_leave_n_out", n_repeats=5, leave_n=2)
        summary = pd.read_csv(self._summary_csv())
        self.assertEqual(summary["test_scope"].iloc[0], "fixed_descriptor_set")


class TestRegression8cQ2LabelTerminology(unittest.TestCase):
    """2026-08-03 display-label clarification (progress/34): the pooled Q2
    log labels are clarified (LOO Q2 -> "LOO Q² (pooled out-of-fold;
    1-PRESS/SST_input)", etc.) but must NOT be described as "Q2F1" -- this
    node's Q2 is an internal, fixed-descriptor-set CV metric centered on
    the full training input's own mean, not Gramatica & Sangion's external
    Q2F1 (which needs a genuinely held-out set). Calculations/CSV columns
    (mean_r2, pooled_q2, ...) are unchanged."""

    def setUp(self):
        _helpers.fresh_output_dir()
        self.mod = _helpers.load_node_module("Regression", "08 Model Evaluation/08-3 Resampling Stability Assessment", "Resampling_Validation.py",
                                              alias="r8c_q2label")
        rng = np.random.default_rng(0)
        n = 20
        X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
        y = X["f1"] * 2 + rng.normal(scale=0.1, size=n)
        df = X.copy()
        df["value"] = y.values
        self.input_path = "/tmp/test_8c_q2label_input.csv"
        df.to_csv(self.input_path, index=False)
        import joblib
        model = LinearRegression().fit(X, y)
        self.model_path = "/tmp/test_8c_q2label_model.pkl"
        joblib.dump(model, self.model_path)

    def test_loocv_log_wording(self):
        node = self.mod.ResamplingValidation_Regression()
        result = node.run(self.input_path, self.model_path, "value", "loocv")
        text = result["ui"]["text"]
        self.assertIn("LOO Q² (pooled out-of-fold; 1-PRESS/SST_input)", text)
        self.assertNotIn("Q2F1", text)
        self.assertNotIn("Q²F1", text)

    def test_repeated_kfold_log_wording(self):
        node = self.mod.ResamplingValidation_Regression()
        result = node.run(self.input_path, self.model_path, "value", "repeated_kfold", n_repeats=3, cv_splits=3)
        text = result["ui"]["text"]
        self.assertIn("Repeated k-fold Q² (pooled out-of-fold per repeat)", text)
        self.assertNotIn("Q2F1", text)
        self.assertNotIn("Q²F1", text)

    def test_repeated_leave_n_out_log_wording(self):
        node = self.mod.ResamplingValidation_Regression()
        result = node.run(self.input_path, self.model_path, "value", "repeated_leave_n_out", n_repeats=5, leave_n=2)
        text = result["ui"]["text"]
        self.assertIn("Leave-N-out Q² (pooled held-out predictions; centered on full input-target mean)", text)
        self.assertNotIn("Q2F1", text)
        self.assertNotIn("Q²F1", text)

    def test_summary_csv_columns_unchanged(self):
        node = self.mod.ResamplingValidation_Regression()
        node.run(self.input_path, self.model_path, "value", "loocv")
        import folder_paths
        summary_path = os.path.join(folder_paths.get_output_directory(), "Regression", "08_Model_Evaluation",
                                     "Resampling_Stability", "Resampling_Validation_Results.csv")
        summary = pd.read_csv(summary_path)
        for col in ("mean_r2", "mean_rmse"):
            self.assertIn(col, summary.columns, f"CSV column {col!r} must be unchanged")


class TestRegression8cLeaveNOutDuplicationInvariance(unittest.TestCase):
    def setUp(self):
        _helpers.fresh_output_dir()
        self.mod = _helpers.load_node_module("Regression", "08 Model Evaluation/08-3 Resampling Stability Assessment", "Resampling_Validation.py",
                                              alias="r8c_dup")

    def test_pooled_raw_output_retains_duplicate_draws_not_deduplicated(self):
        # Small n_train + leave_n=2 + enough repeats guarantees (by
        # pigeonhole) that at least one compound is drawn in more than one
        # repeat.
        rng = np.random.default_rng(0)
        n_train = 10
        X = pd.DataFrame({"f1": rng.normal(size=n_train), "f2": rng.normal(size=n_train)})
        y = X["f1"] * 2 + rng.normal(scale=0.1, size=n_train)
        df = X.copy()
        df["value"] = y.values
        input_path = "/tmp/test_8c_dup_input.csv"
        df.to_csv(input_path, index=False)
        import joblib
        model = LinearRegression().fit(X, y)
        model_path = "/tmp/test_8c_dup_model.pkl"
        joblib.dump(model, model_path)

        n_repeats, leave_n = 20, 2
        node = self.mod.ResamplingValidation_Regression()
        node.run(input_path, model_path, "value", "repeated_leave_n_out", n_repeats=n_repeats, leave_n=leave_n)

        import folder_paths
        raw_path = os.path.join(folder_paths.get_output_directory(), "Regression", "08_Model_Evaluation", "Resampling_Stability",
                                 "Resampling_LeaveNOut_Raw_Predictions.csv")
        if not os.path.exists(raw_path):
            # fall back to whatever the raw predictions file is actually named
            outdir = os.path.dirname(raw_path)
            candidates = [f for f in os.listdir(outdir) if "raw" in f.lower() or "prediction" in f.lower()]
            self.assertTrue(candidates, f"no raw/prediction CSV found in {outdir}: {os.listdir(outdir)}")
            raw_path = os.path.join(outdir, candidates[0])
        raw_df = pd.read_csv(raw_path)

        self.assertEqual(len(raw_df), n_repeats * leave_n,
                          "pooled raw output must have exactly n_repeats*leave_n rows (no deduplication)")
        dup_counts = raw_df["original_row_index"].value_counts()
        self.assertTrue((dup_counts > 1).any(),
                         "with n_train=10, leave_n=2, n_repeats=20, at least one compound should be "
                         "drawn in more than one repeat -- if not, this test's premise needs a bigger n_repeats")


class TestClassification8cRepeatPooledDesign(unittest.TestCase):
    """Confirms the repeat-pooled redesign (loocv/repeated_kfold pool
    per-repeat OOF predictions into one score rather than averaging
    per-fold scores) is what's actually running, via the summary schema."""

    def setUp(self):
        _helpers.fresh_output_dir()
        self.mod = _helpers.load_node_module("Classification", "08 Model Evaluation/08-3 Resampling Stability Assessment", "Resampling_Validation.py",
                                              alias="c8c_pooled")
        rng = np.random.default_rng(0)
        n = 30
        X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
        y = (X["f1"] > 0).astype(int)
        df = X.copy()
        df["Label"] = y.values
        self.input_path = "/tmp/test_8c_cls_input.csv"
        df.to_csv(self.input_path, index=False)
        import joblib
        model = LogisticRegression().fit(X, y)
        self.model_path = "/tmp/test_8c_cls_model.pkl"
        joblib.dump(model, self.model_path)

    def test_repeated_stratified_kfold_reports_one_score_per_repeat(self):
        node = self.mod.ResamplingValidation_Classification()
        n_repeats = 4
        node.run(self.input_path, self.model_path, "Label", "repeated_stratified_kfold", n_repeats=n_repeats, cv_splits=3)
        import folder_paths
        raw_path = os.path.join(folder_paths.get_output_directory(), "Classification", "08_Model_Evaluation", "Resampling_Stability",
                                 "Resampling_Validation_Raw_Scores.csv")
        raw_df = pd.read_csv(raw_path)
        self.assertEqual(len(raw_df), n_repeats,
                          "repeat-pooled design: exactly one row (one pooled score) per repeat")


if __name__ == "__main__":
    unittest.main()
