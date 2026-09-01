"""02 (Descriptor Calculation -- its sanitize_descriptors_file() step,
formerly the standalone "02b Descriptor Value Sanitization" node before it
was merged into 02) -> 03 (Data Split) -> 04 (Paired Descriptor
Preprocessing) pipeline restructure. Covers the leak-fix itself
(train-fit-only statistics), the row-missingness-after-column-restriction
ordering, the High-severity fixes found in the follow-up review
(Y_TEST_FILTERED, all-NaN column / zero-retained-descriptor guards,
leading-zero Name preservation, Regression non-numeric/infinite target
rejection, dynamic id_cols in the audit function), and the 4-output
RETURN_TYPES (training/holdout/targets/recipe; the audit report is
disk-only, not a graph socket).
Classification and Regression are structurally identical, so most
scenarios are parameterized across both tracks; a few are Classification-
or Regression-only where the underlying code genuinely differs (e.g. the
0/1 label check only applies to Classification).
"""
import json
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _helpers  # noqa: E402

TRACKS = {
    "Classification": {"target_column": "Label"},
    "Regression": {"target_column": "value"},
}

_SKIP_REASON = (
    "RDKit/padelpy not importable in this interpreter -- node 2's module now imports "
    "both at module level (needed for real PaDEL calculation), so even these "
    "sanitization/split/preprocessing-only tests can't import it here, even though "
    "the sanitize_descriptors_file() logic they exercise never touches RDKit/PaDEL "
    "itself. Run with an RDKit+padelpy-capable interpreter to exercise these."
)


def _make_target(track, n, rng):
    if track == "Classification":
        return (rng.random(n) > 0.5).astype(int).tolist()
    return rng.normal(loc=2.0, scale=0.5, size=n).tolist()


class PipelineRestructureTestBase:
    """Mixin; concrete subclasses set self.track. Not collected directly
    (doesn't inherit TestCase) -- see the two subclasses below."""

    track = None

    def setUp(self):
        self.outdir = _helpers.fresh_output_dir()
        self.target_column = TRACKS[self.track]["target_column"]
        self.san = _helpers.load_node_module(self.track, "02 Descriptor Calculation",
                                              "Descriptor_Calculation.py", alias=f"desccalc_{self.track}")
        self.split_mod = _helpers.load_node_module(self.track, "03 Data Split",
                                                    "Data_Split.py", alias=f"split_{self.track}")
        self.prep = _helpers.load_node_module(self.track, "04 Descriptor Preprocessing",
                                               "Descriptor_Preprocessing.py", alias=f"prep_{self.track}")
        self.sanitize_fn = self.san.sanitize_descriptors_file
        self.san_output_dir = os.path.join(self.outdir, self.track, "02_Descriptor_Calculation")
        self.split_cls = getattr(self.split_mod, f"QSARDataSplit_{self.track}")
        self.prep_cls = getattr(self.prep, f"Paired_Descriptor_Preprocessing_{self.track}")

    def _sanitize(self, descriptors_path, target_column):
        """Matches the former standalone "02b" node's
        `self.san_cls().run(path, target_column=...)` call shape -- now a
        plain function call into node 2's merged sanitize_descriptors_file()."""
        return self.sanitize_fn(descriptors_path, target_column, self.san_output_dir)

    def _build_raw_dataset(self, n=40, n_bad_cols=5):
        rng = np.random.default_rng(0)
        names = [f"C{i:03d}" for i in range(n)]
        data = {"Name": names, "SMILES": ["CCO"] * n}
        for j in range(n_bad_cols):
            col = [np.nan] * n
            for keep_idx in (0, 10, 20, 30):
                if keep_idx < n:
                    col[keep_idx] = float(j + 1)
            data[f"descriptor_bad_{j}"] = col
        data["descriptor_ok"] = rng.normal(loc=5.0, scale=1.0, size=n).tolist()
        data[self.target_column] = _make_target(self.track, n, rng)
        return pd.DataFrame(data)

    def _run_through_04(self, raw_df, compound_thr=0.5, descriptor_thr=0.5, method="mean"):
        raw_path = os.path.join(self.outdir, "raw.csv")
        raw_df.to_csv(raw_path, index=False)
        out_san = self._sanitize(raw_path, self.target_column)
        self.assertNotIn("Error", out_san["ui"]["text"])
        sanitized_path = out_san["result"][0]

        out_split = self.split_cls().execute(sanitized_path, test_size=0.25, random_state=42)
        self.assertNotIn("❌", out_split["ui"]["text"])
        train_path, x_test_path, y_test_path = out_split["result"]

        out_pp = self.prep_cls().run(train_path, x_test_path, y_test_path, self.target_column,
                                      compound_thr, descriptor_thr, method)
        return out_pp, train_path, x_test_path, y_test_path

    def _recipe_path(self):
        return os.path.join(self.outdir, self.track, "04_Descriptor_Preprocessing", "preprocessing_recipe.json")

    def _load_recipe(self):
        with open(self._recipe_path()) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    def test_full_pipeline_runs_clean(self):
        raw_df = self._build_raw_dataset()
        out_pp, *_ = self._run_through_04(raw_df)
        self.assertNotIn("❌", out_pp["ui"]["text"])
        self.assertEqual(len(out_pp["result"]), 4, "RETURN_TYPES must be exactly 4 (training/holdout/targets/recipe; audit is disk-only)")

    def test_leak_fix_extreme_holdout_value_does_not_move_training_stats(self):
        raw_df = self._build_raw_dataset()
        out_pp_a, train_path, x_test_path, y_test_path = self._run_through_04(raw_df)
        recipe_a = self._load_recipe()
        mean_a = recipe_a["imputer_statistics"]["descriptor_ok"]

        x_test_df = pd.read_csv(x_test_path, dtype={"Name": str})
        x_test_df.loc[x_test_df.index[0], "descriptor_ok"] = 999999.0
        leaky_path = os.path.join(self.outdir, "leaky.csv")
        x_test_df.to_csv(leaky_path, index=False)

        out_pp_b = self.prep_cls().run(train_path, leaky_path, y_test_path, self.target_column, 0.5, 0.5, "mean")
        self.assertNotIn("❌", out_pp_b["ui"]["text"])
        recipe_b = self._load_recipe()
        mean_b = recipe_b["imputer_statistics"]["descriptor_ok"]
        self.assertAlmostEqual(mean_a, mean_b, places=9,
                                msg="training imputer statistic must be unaffected by an extreme hold-out value")

    def test_row_missingness_computed_after_column_restriction(self):
        # descriptor_bad_* are guaranteed dropped (>50% NaN in train by
        # construction); a test compound missing ONLY those columns should
        # NOT be excluded, since after column-restriction it has 0% missing.
        raw_df = self._build_raw_dataset()
        out_pp, train_path, x_test_path, y_test_path = self._run_through_04(raw_df)
        x_test_df = pd.read_csv(x_test_path, dtype={"Name": str})
        probe_idx = x_test_df.index[0]
        for j in range(5):
            x_test_df.loc[probe_idx, f"descriptor_bad_{j}"] = np.nan
        probe_name = x_test_df.loc[probe_idx, "Name"]
        probe_path = os.path.join(self.outdir, "probe.csv")
        x_test_df.to_csv(probe_path, index=False)

        out_probe = self.prep_cls().run(train_path, probe_path, y_test_path, self.target_column, 0.5, 0.5, "mean")
        test_out_path = out_probe["result"][1]
        test_out_df = pd.read_csv(test_out_path, dtype={"Name": str})
        self.assertIn(probe_name, test_out_df["Name"].values)

    def test_high1_test_row_exclusion_produces_matching_y_test_filtered(self):
        train_df = pd.DataFrame({
            "Name": [f"T{i}" for i in range(10)],
            "SMILES": ["CCO"] * 10,
            "f1": np.random.default_rng(1).normal(size=10),
        })
        train_df[self.target_column] = _make_target(self.track, 10, np.random.default_rng(1))
        test_df = pd.DataFrame({"Name": ["x", "y"], "SMILES": ["CCO", "CCO"], "f1": [np.nan, 1.23]})
        y_test_df = pd.DataFrame({"Name": ["x", "y"], self.target_column: _make_target(self.track, 2, np.random.default_rng(2))})
        train_p = os.path.join(self.outdir, "h1_train.csv")
        test_p = os.path.join(self.outdir, "h1_test.csv")
        y_p = os.path.join(self.outdir, "h1_ytest.csv")
        train_df.to_csv(train_p, index=False)
        test_df.to_csv(test_p, index=False)
        y_test_df.to_csv(y_p, index=False)

        out = self.prep_cls().run(train_p, test_p, y_p, self.target_column, 0.5, 0.5, "mean")
        self.assertNotIn("❌", out["ui"]["text"])
        test_out = pd.read_csv(out["result"][1], dtype={"Name": str})
        y_out = pd.read_csv(out["result"][2], dtype={"Name": str})
        self.assertEqual(test_out["Name"].tolist(), ["y"])
        self.assertEqual(y_out["Name"].tolist(), ["y"], "Y_TEST_FILTERED must drop the same compound as TEST_PREPROCESSED")

    def test_high2a_all_nan_in_train_column_force_dropped_not_crashed(self):
        train_df = pd.DataFrame({
            "Name": [f"T{i}" for i in range(10)],
            "f_all_nan": [np.nan] * 10,
            "f_ok": np.random.default_rng(2).normal(size=10),
        })
        train_df[self.target_column] = _make_target(self.track, 10, np.random.default_rng(2))
        test_df = pd.DataFrame({"Name": ["x"], "f_all_nan": [np.nan], "f_ok": [1.0]})
        y_test_df = pd.DataFrame({"Name": ["x"], self.target_column: _make_target(self.track, 1, np.random.default_rng(3))})
        train_p = os.path.join(self.outdir, "h2a_train.csv")
        test_p = os.path.join(self.outdir, "h2a_test.csv")
        y_p = os.path.join(self.outdir, "h2a_ytest.csv")
        train_df.to_csv(train_p, index=False)
        test_df.to_csv(test_p, index=False)
        y_test_df.to_csv(y_p, index=False)

        out = self.prep_cls().run(train_p, test_p, y_p, self.target_column, 0.5, 1.0, "mean")
        self.assertNotIn("❌", out["ui"]["text"], out["ui"]["text"])
        recipe = self._load_recipe()
        self.assertIn("f_all_nan", recipe["dropped_descriptors_all_nan_in_train"])
        self.assertIn("f_ok", recipe["retained_descriptors"])

    def test_high2b_zero_retained_descriptors_raises_explicit_error(self):
        train_df = pd.DataFrame({
            "Name": [f"T{i}" for i in range(10)],
            "f_all_nan": [np.nan] * 10,
        })
        train_df[self.target_column] = _make_target(self.track, 10, np.random.default_rng(4))
        test_df = pd.DataFrame({"Name": ["x"], "f_all_nan": [np.nan]})
        y_test_df = pd.DataFrame({"Name": ["x"], self.target_column: _make_target(self.track, 1, np.random.default_rng(5))})
        train_p = os.path.join(self.outdir, "h2b_train.csv")
        test_p = os.path.join(self.outdir, "h2b_test.csv")
        y_p = os.path.join(self.outdir, "h2b_ytest.csv")
        train_df.to_csv(train_p, index=False)
        test_df.to_csv(test_p, index=False)
        y_test_df.to_csv(y_p, index=False)

        out = self.prep_cls().run(train_p, test_p, y_p, self.target_column, 0.5, 1.0, "mean")
        self.assertIn("No descriptor columns survived retention", out["ui"]["text"])

    def test_high4_leading_zero_name_preserved_through_03(self):
        names = [f"{i:03d}" for i in range(7, 17)]
        df = pd.DataFrame({"Name": names, "f1": np.random.default_rng(3).normal(size=10)})
        df[self.target_column] = _make_target(self.track, 10, np.random.default_rng(3))
        p = os.path.join(self.outdir, "leadzero_in.csv")
        df.to_csv(p, index=False)
        out = self.split_cls().execute(p, test_size=0.3, random_state=42)
        train_out = pd.read_csv(out["result"][0], dtype={"Name": str})
        self.assertTrue(all(len(v) == 3 for v in train_out["Name"]),
                         f"leading zeros lost: {train_out['Name'].tolist()}")

    def test_major_audit_uses_dynamic_target_column_not_hardcoded(self):
        # A custom target column name should not be misattributed as a
        # descriptor by the sanitization audit.
        df = pd.DataFrame({
            "Name": ["a", "b", "c", "d"],
            "f1": [1.0, 2.0, 3.0, 4.0],
        })
        custom_target = "activity" if self.track == "Classification" else "custom_value"
        df[custom_target] = [0.0, np.inf, 0.0, 1.0] if self.track == "Classification" else [1.0, np.inf, 2.0, 3.0]
        p = os.path.join(self.outdir, "custom_target.csv")
        df.to_csv(p, index=False)
        out = self._sanitize(p, custom_target)
        # the inf value must surface as a target-level error (non-finite),
        # not get silently absorbed into descriptor NaN accounting
        self.assertIn("Error", out["ui"]["text"])
        self.assertTrue("non-finite" in out["ui"]["text"].lower() or "0/1" in out["ui"]["text"] or "nan" in out["ui"]["text"].lower())


@unittest.skipUnless(_helpers.rdkit_available() and _helpers.padelpy_available(), _SKIP_REASON)
class TestPipelineRestructureClassification(PipelineRestructureTestBase, unittest.TestCase):
    track = "Classification"

    def test_sanitization_rejects_non_01_label(self):
        df = pd.DataFrame({"Name": ["a", "b", "c"], "f1": [1.0, 2.0, 3.0], "Label": [0, 1, 2]})
        p = os.path.join(self.outdir, "bad_label.csv")
        df.to_csv(p, index=False)
        out = self._sanitize(p, "Label")
        self.assertIn("binary-encoded as 0/1", out["ui"]["text"])


@unittest.skipUnless(_helpers.rdkit_available() and _helpers.padelpy_available(), _SKIP_REASON)
class TestPipelineRestructureRegression(PipelineRestructureTestBase, unittest.TestCase):
    track = "Regression"

    def test_sanitization_rejects_non_numeric_target(self):
        df = pd.DataFrame({"Name": ["a", "b", "c"], "f1": [1.0, 2.0, 3.0], "value": ["1.0", "bad", "3.0"]})
        p = os.path.join(self.outdir, "bad_target.csv")
        df.to_csv(p, index=False)
        out = self._sanitize(p, "value")
        self.assertIn("non-numeric", out["ui"]["text"])

    def test_sanitization_rejects_infinite_target(self):
        df = pd.DataFrame({"Name": ["a", "b", "c"], "f1": [1.0, 2.0, 3.0], "value": [1.0, np.inf, 3.0]})
        p = os.path.join(self.outdir, "inf_target.csv")
        df.to_csv(p, index=False)
        out = self._sanitize(p, "value")
        self.assertIn("non-finite", out["ui"]["text"])


if __name__ == "__main__":
    unittest.main()
