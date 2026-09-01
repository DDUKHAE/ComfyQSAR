"""9. Applicability Domain Assessment (formerly "8d Applicability Domain Assessment").
Regression: h_star boundary (>=1.0 inclusive), NaN/inf pre-check on both
training and query descriptors, rank-deficient matrix detection. Always
runs (no RDKit dependency -- pure descriptor-space leverage).
Classification: invalid-training-SMILES report CSV + stale-cleanup, wording.
Skipped when RDKit isn't importable in the current interpreter (it isn't
in some sandboxed environments -- run with an RDKit-capable interpreter,
e.g. `/tmp/comfyqsar_zenodo_clean/env/bin/python3` in this project's
sandbox, to exercise it)."""
import os
import sys
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _helpers  # noqa: E402


class TestRegression8d(unittest.TestCase):
    def setUp(self):
        _helpers.fresh_output_dir()
        self.mod = _helpers.load_node_module("Regression", "09 Applicability Domain Assessment",
                                              "Applicability_Domain_Assessment.py", alias="reg8d")

    def _build(self, n, p, seed=0, extra_col=None):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
        if extra_col:
            X[extra_col] = X["f0"]
        y = X.filter(like="f").sum(axis=1)
        train_df = X.copy()
        train_df["Name"] = [f"T{i}" for i in range(n)]
        train_df["value"] = y.values
        model = LinearRegression().fit(X, y)
        model_path = f"/tmp/test_8dreg_model_{seed}.pkl"
        joblib.dump(model, model_path)
        return train_df, model_path, X, y

    def test_normal_case_no_warnings(self):
        train_df, model_path, X, y = self._build(31, 4)
        train_path = "/tmp/test_8dreg_train_normal.csv"
        train_df.to_csv(train_path, index=False)
        query_df = X.iloc[:5].copy()
        query_df["Name"] = [f"Q{i}" for i in range(5)]
        query_df["value"] = y.values[:5]
        query_path = "/tmp/test_8dreg_query_normal.csv"
        query_df.to_csv(query_path, index=False)

        node = self.mod.ApplicabilityDomain_Regression()
        out = node.run(train_path, query_path, model_path=model_path, target_column="value")
        self.assertNotIn("❌", out["ui"]["text"])
        self.assertIn("pseudoinverse used: no", out["ui"]["text"])
        self.assertNotIn("rank-deficient", out["ui"]["text"])

    def test_nan_in_training_descriptors_raises(self):
        train_df, model_path, X, y = self._build(31, 4, seed=1)
        train_df.loc[2, "f0"] = np.nan
        train_path = "/tmp/test_8dreg_train_nan.csv"
        train_df.to_csv(train_path, index=False)
        query_df = X.iloc[:5].copy()
        query_df["Name"] = [f"Q{i}" for i in range(5)]
        query_df["value"] = y.values[:5]
        query_path = "/tmp/test_8dreg_query_nan.csv"
        query_df.to_csv(query_path, index=False)

        node = self.mod.ApplicabilityDomain_Regression()
        out = node.run(train_path, query_path, model_path=model_path, target_column="value")
        self.assertIn("NaN/inf", out["ui"]["text"])

    def test_inf_in_query_descriptors_raises(self):
        train_df, model_path, X, y = self._build(31, 4, seed=2)
        train_path = "/tmp/test_8dreg_train_inf.csv"
        train_df.to_csv(train_path, index=False)
        query_df = X.iloc[:5].copy()
        query_df.loc[query_df.index[0], "f1"] = np.inf
        query_df["Name"] = [f"Q{i}" for i in range(5)]
        query_df["value"] = y.values[:5]
        query_path = "/tmp/test_8dreg_query_inf.csv"
        query_df.to_csv(query_path, index=False)

        node = self.mod.ApplicabilityDomain_Regression()
        out = node.run(train_path, query_path, model_path=model_path, target_column="value")
        self.assertIn("NaN/inf", out["ui"]["text"])

    def test_rank_deficient_matrix_detected(self):
        train_df, model_path, X, y = self._build(31, 4, seed=3, extra_col="f0_dup")
        train_path = "/tmp/test_8dreg_train_dup.csv"
        train_df.to_csv(train_path, index=False)
        query_df = X.iloc[:5].copy()
        query_df["Name"] = [f"Q{i}" for i in range(5)]
        query_df["value"] = y.values[:5]
        query_path = "/tmp/test_8dreg_query_dup.csv"
        query_df.to_csv(query_path, index=False)

        node = self.mod.ApplicabilityDomain_Regression()
        out = node.run(train_path, query_path, model_path=model_path, target_column="value")
        self.assertIn("rank-deficient", out["ui"]["text"])
        self.assertIn("rank 4 < 5", out["ui"]["text"])

    def test_h_star_boundary_inclusive_at_exactly_1(self):
        # h* = multiplier*(p+1)/n -- choose n, p, multiplier so h* lands
        # exactly on 1.0.
        n, p, multiplier = 8, 3, 2.0  # h* = 2*(3+1)/8 = 1.0
        train_df, model_path, X, y = self._build(n, p, seed=4)
        train_path = "/tmp/test_8dreg_train_hstar.csv"
        train_df.to_csv(train_path, index=False)
        query_df = X.iloc[:2].copy()
        query_df["Name"] = [f"Q{i}" for i in range(2)]
        query_df["value"] = y.values[:2]
        query_path = "/tmp/test_8dreg_query_hstar.csv"
        query_df.to_csv(query_path, index=False)

        node = self.mod.ApplicabilityDomain_Regression()
        out = node.run(train_path, query_path, model_path=model_path, target_column="value",
                        warning_leverage_multiplier=multiplier)
        self.assertIn("h* (1.0000)", out["ui"]["text"])
        self.assertIn("at or exceeds 1.0", out["ui"]["text"], "h*==1.0 must trigger the warning (inclusive boundary)")

    def test_nonlinear_advisory_present(self):
        train_df, model_path, X, y = self._build(31, 4, seed=5)
        train_path = "/tmp/test_8dreg_train_adv.csv"
        train_df.to_csv(train_path, index=False)
        query_df = X.iloc[:5].copy()
        query_df["Name"] = [f"Q{i}" for i in range(5)]
        query_df["value"] = y.values[:5]
        query_path = "/tmp/test_8dreg_query_adv.csv"
        query_df.to_csv(query_path, index=False)

        node = self.mod.ApplicabilityDomain_Regression()
        out = node.run(train_path, query_path, model_path=model_path, target_column="value")
        self.assertIn("does not by itself guarantee prediction", out["ui"]["text"])
        self.assertNotIn("RMSE-equivalent", out["ui"]["text"])


@unittest.skipUnless(_helpers.rdkit_available(),
                      "RDKit not importable in this interpreter -- run with an RDKit-capable "
                      "interpreter to exercise Classification 8d (e.g. "
                      "/tmp/comfyqsar_zenodo_clean/env/bin/python3 in this project's sandbox).")
class TestClassification8d(unittest.TestCase):
    def setUp(self):
        _helpers.fresh_output_dir()
        self.mod = _helpers.load_node_module("Classification", "09 Applicability Domain Assessment",
                                              "Applicability_Domain_Assessment.py", alias="cls8d")

    def _paths(self):
        import folder_paths
        return os.path.join(folder_paths.get_output_directory(), "Classification", "09_Applicability_Domain")

    def test_invalid_training_smiles_report_created(self):
        train_df = pd.DataFrame({
            "Name": [f"T{i}" for i in range(8)],
            "SMILES": ["CCO", "CCC", "CCN", "CCCl", "not_a_smiles!!", "CCF", "CCBr", "CCI"],
        })
        query_df = pd.DataFrame({"Name": ["Q1", "Q2"], "SMILES": ["CCO", "CCC"]})
        train_path = "/tmp/test_8dcls_train.csv"
        query_path = "/tmp/test_8dcls_query.csv"
        train_df.to_csv(train_path, index=False)
        query_df.to_csv(query_path, index=False)

        node = self.mod.ApplicabilityDomain_Classification()
        out = node.run(train_path, query_path, k_neighbors=3)
        self.assertIn("1 invalid", out["ui"]["text"])
        report_path = os.path.join(self._paths(), "Invalid_Training_SMILES_Report.csv")
        self.assertTrue(os.path.exists(report_path))
        report_df = pd.read_csv(report_path)
        self.assertEqual(report_df.iloc[0]["Name"], "T4")

    def test_stale_report_removed_on_clean_run(self):
        train_df_bad = pd.DataFrame({
            "Name": [f"T{i}" for i in range(8)],
            "SMILES": ["CCO", "CCC", "CCN", "CCCl", "not_a_smiles!!", "CCF", "CCBr", "CCI"],
        })
        query_df = pd.DataFrame({"Name": ["Q1", "Q2"], "SMILES": ["CCO", "CCC"]})
        train_path_bad = "/tmp/test_8dcls_train_bad.csv"
        query_path = "/tmp/test_8dcls_query2.csv"
        train_df_bad.to_csv(train_path_bad, index=False)
        query_df.to_csv(query_path, index=False)
        node = self.mod.ApplicabilityDomain_Classification()
        node.run(train_path_bad, query_path, k_neighbors=3)
        report_path = os.path.join(self._paths(), "Invalid_Training_SMILES_Report.csv")
        self.assertTrue(os.path.exists(report_path))

        train_df_clean = train_df_bad.copy()
        train_df_clean.loc[4, "SMILES"] = "CCS"
        train_path_clean = "/tmp/test_8dcls_train_clean.csv"
        train_df_clean.to_csv(train_path_clean, index=False)
        node.run(train_path_clean, query_path, k_neighbors=3)
        self.assertFalse(os.path.exists(report_path), "stale report file should be removed on a clean run")

    def test_wording_uses_kNN_mean_not_overclaimed_standard(self):
        train_df = pd.DataFrame({"Name": [f"T{i}" for i in range(6)], "SMILES": ["CCO", "CCC", "CCN", "CCCl", "CCF", "CCBr"]})
        query_df = pd.DataFrame({"Name": ["Q1"], "SMILES": ["CCO"]})
        train_path = "/tmp/test_8dcls_train_wording.csv"
        query_path = "/tmp/test_8dcls_query_wording.csv"
        train_df.to_csv(train_path, index=False)
        query_df.to_csv(query_path, index=False)
        node = self.mod.ApplicabilityDomain_Classification()
        out = node.run(train_path, query_path, k_neighbors=3)
        self.assertIn("kNN-mean Tanimoto AD with empirical percentile threshold", out["ui"]["text"])
        self.assertIn("Requested k:", out["ui"]["text"])
        self.assertIn("Effective k:", out["ui"]["text"])


if __name__ == "__main__":
    unittest.main()
