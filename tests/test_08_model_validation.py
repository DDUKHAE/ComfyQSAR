"""8.1 Hold-out & External Performance (formerly "08 Model Validation"):
Name-based X_test/Y_test alignment, leading-zero
dtype preservation, Classification's single-class-external domain-table
metric validity, and the Bootstrap CI additions (Balanced Accuracy + MCC
for Classification, CCC for Regression)."""
import os
import sys
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _helpers  # noqa: E402


class TestClassificationIDAlignment(unittest.TestCase):
    def setUp(self):
        _helpers.fresh_output_dir()
        self.mv = _helpers.load_node_module("Classification", "08 Model Evaluation/08-1 Hold-out & External Performance", "Model_Validation.py",
                                             alias="mv_cls_align")
        self.n = 6
        X = pd.DataFrame({"f1": np.random.default_rng(0).normal(size=self.n), "f2": np.random.default_rng(1).normal(size=self.n)})
        y = np.array([0, 1, 0, 1, 0, 1])
        self.model = LogisticRegression().fit(X, y)
        self.model_path = "/tmp/test_08cls_model.pkl"
        joblib.dump(self.model, self.model_path)
        self.X = X
        self.y = y
        self.feat_path = "/tmp/test_08cls_feat.txt"
        with open(self.feat_path, "w") as f:
            f.write("f1\nf2\n")

    def test_shuffled_y_reordered_by_name(self):
        names = [f"C{i}" for i in range(self.n)]
        x_df = self.X.copy()
        x_df.insert(0, "Name", names)
        y_df = pd.DataFrame({"Name": names, "Label": self.y})
        y_df = y_df.iloc[[3, 0, 5, 1, 4, 2]].reset_index(drop=True)
        x_path, y_path = "/tmp/test_08cls_x_shuf.csv", "/tmp/test_08cls_y_shuf.csv"
        x_df.to_csv(x_path, index=False)
        y_df.to_csv(y_path, index=False)
        _, _, y_test, names_out, _, alignment_mode = self.mv.load_classification_inputs(
            self.model_path, x_path, y_path, self.feat_path)
        self.assertEqual(alignment_mode, "id_matched")
        self.assertEqual(list(y_test), list(self.y))
        self.assertEqual(names_out, names)

    def test_duplicate_name_raises(self):
        names = [f"C{i}" for i in range(self.n)]
        x_df = self.X.copy()
        x_df.insert(0, "Name", names)
        x_df.loc[1, "Name"] = "C0"
        y_df = pd.DataFrame({"Name": names, "Label": self.y})
        x_path, y_path = "/tmp/test_08cls_x_dup.csv", "/tmp/test_08cls_y_dup.csv"
        x_df.to_csv(x_path, index=False)
        y_df.to_csv(y_path, index=False)
        with self.assertRaises(ValueError) as ctx:
            self.mv.load_classification_inputs(self.model_path, x_path, y_path, self.feat_path)
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_mismatched_name_set_raises(self):
        names = [f"C{i}" for i in range(self.n)]
        x_df = self.X.copy()
        x_df.insert(0, "Name", names)
        y_df = pd.DataFrame({"Name": names[:-1] + ["ZZZ"], "Label": self.y})
        x_path, y_path = "/tmp/test_08cls_x_mm.csv", "/tmp/test_08cls_y_mm.csv"
        x_df.to_csv(x_path, index=False)
        y_df.to_csv(y_path, index=False)
        with self.assertRaises(ValueError) as ctx:
            self.mv.load_classification_inputs(self.model_path, x_path, y_path, self.feat_path)
        self.assertIn("ZZZ", str(ctx.exception))

    def test_nan_target_raises(self):
        names = [f"C{i}" for i in range(self.n)]
        x_df = self.X.copy()
        x_df.insert(0, "Name", names)
        y_vals = self.y.astype(float)
        y_vals[2] = np.nan
        y_df = pd.DataFrame({"Name": names, "Label": y_vals})
        x_path, y_path = "/tmp/test_08cls_x_nan.csv", "/tmp/test_08cls_y_nan.csv"
        x_df.to_csv(x_path, index=False)
        y_df.to_csv(y_path, index=False)
        with self.assertRaises(ValueError) as ctx:
            self.mv.load_classification_inputs(self.model_path, x_path, y_path, self.feat_path)
        self.assertIn("C2", str(ctx.exception))

    def test_legacy_no_name_fallback_matching_length_ok(self):
        x_path, y_path = "/tmp/test_08cls_x_legacy.csv", "/tmp/test_08cls_y_legacy.csv"
        self.X.to_csv(x_path, index=False)
        pd.DataFrame({"Label": self.y}).to_csv(y_path, index=False)
        _, _, y_test, _, _, alignment_mode = self.mv.load_classification_inputs(
            self.model_path, x_path, y_path, self.feat_path)
        self.assertEqual(alignment_mode, "row_order_fallback")
        self.assertEqual(list(y_test), list(self.y))

    def test_legacy_no_name_mismatched_length_raises(self):
        x_path, y_path = "/tmp/test_08cls_x_legacy2.csv", "/tmp/test_08cls_y_legacy2.csv"
        self.X.to_csv(x_path, index=False)
        pd.DataFrame({"Label": self.y[:-1]}).to_csv(y_path, index=False)
        with self.assertRaises(ValueError) as ctx:
            self.mv.load_classification_inputs(self.model_path, x_path, y_path, self.feat_path)
        self.assertIn("different lengths", str(ctx.exception))

    def test_leading_zero_name_preserved(self):
        names = [f"{i:03d}" for i in range(10)]
        X10 = pd.DataFrame({"f1": np.random.default_rng(0).normal(size=10), "f2": np.random.default_rng(1).normal(size=10)})
        model10 = LogisticRegression().fit(X10, [0, 1] * 5)
        model_path10 = "/tmp/test_08cls_model10.pkl"
        joblib.dump(model10, model_path10)
        x_df10 = X10.copy()
        x_df10.insert(0, "Name", names)
        y_df10 = pd.DataFrame({"Name": names, "Label": [0, 1] * 5})
        x_path, y_path = "/tmp/test_08cls_x_leadzero.csv", "/tmp/test_08cls_y_leadzero.csv"
        x_df10.to_csv(x_path, index=False)
        y_df10.to_csv(y_path, index=False)
        _, _, _, names_out, _, _ = self.mv.load_classification_inputs(model_path10, x_path, y_path, self.feat_path)
        self.assertEqual(names_out, names, f"leading zeros lost: {names_out}")


class TestClassificationDomainTableMetrics(unittest.TestCase):
    def setUp(self):
        _helpers.fresh_output_dir()
        self.mv = _helpers.load_node_module("Classification", "08 Model Evaluation/08-1 Hold-out & External Performance", "Model_Validation.py",
                                             alias="mv_cls_domain")
        X = pd.DataFrame({"f1": np.random.default_rng(0).normal(size=20), "f2": np.random.default_rng(1).normal(size=20)})
        y_train = np.array([0, 1] * 10)
        self.model = LogisticRegression().fit(X, y_train)
        self.X_test = X.iloc[:10]

    def test_two_class_all_metrics_defined(self):
        y = np.array([0, 1] * 5)
        metrics, _, _ = self.mv.calculate_classification_metrics(self.model, self.X_test, y)
        self.assertTrue(all(v is not None for v in metrics.values()))

    def test_all_negative_only_accuracy_and_specificity_defined(self):
        y = np.zeros(10, dtype=int)
        metrics, _, _ = self.mv.calculate_classification_metrics(self.model, self.X_test, y)
        self.assertIsNotNone(metrics["accuracy"])
        self.assertIsNotNone(metrics["specificity"])
        for k in ("recall", "precision", "f1_score", "balanced_accuracy", "mcc", "roc_auc"):
            self.assertIsNone(metrics[k], f"{k} should be None for all-negative external set")

    def test_all_positive_only_accuracy_and_recall_always_defined(self):
        y = np.ones(10, dtype=int)
        metrics, _, _ = self.mv.calculate_classification_metrics(self.model, self.X_test, y)
        self.assertIsNotNone(metrics["accuracy"])
        self.assertIsNotNone(metrics["recall"])
        for k in ("specificity", "balanced_accuracy", "mcc", "roc_auc"):
            self.assertIsNone(metrics[k], f"{k} should be None for all-positive external set")

    def test_single_class_does_not_crash(self):
        y = np.zeros(10, dtype=int)
        try:
            self.mv.calculate_classification_metrics(self.model, self.X_test, y)
        except Exception as e:
            self.fail(f"single-class external set raised unexpectedly: {e}")

    def test_multiclass_model_blocked(self):
        class StubMulticlass:
            classes_ = np.array([0, 1, 2])

            def predict(self, X):
                return np.zeros(len(X), dtype=int)

            def predict_proba(self, X):
                return np.zeros((len(X), 3))
        with self.assertRaises(ValueError) as ctx:
            self.mv.calculate_classification_metrics(StubMulticlass(), self.X_test, np.array([0, 1] * 5))
        self.assertIn("binary", str(ctx.exception).lower())

    def test_bootstrap_ci_includes_balanced_accuracy_and_mcc(self):
        y = np.array([0, 1] * 5)
        metrics, y_pred, y_proba = self.mv.calculate_classification_metrics(self.model, self.X_test, y)
        ci_df = self.mv.bootstrap_classification_ci(y, y_pred, y_proba, metrics, n_bootstrap=200, seed=42)
        self.assertIn("Balanced Accuracy", ci_df["Metric"].values)
        self.assertIn("MCC", ci_df["Metric"].values)

    def test_stale_bootstrap_ci_file_removed_when_disabled(self):
        node = self.mv.Model_Validation_Classification()
        names = [f"C{i}" for i in range(10)]
        x_df = self.X_test.copy()
        x_df.insert(0, "Name", names)
        y_df = pd.DataFrame({"Name": names, "Label": [0, 1] * 5})
        model_path = "/tmp/test_08cls_stale_model.pkl"
        joblib.dump(self.model, model_path)
        x_path, y_path = "/tmp/test_08cls_stale_x.csv", "/tmp/test_08cls_stale_y.csv"
        x_df.to_csv(x_path, index=False)
        y_df.to_csv(y_path, index=False)
        feat_path = "/tmp/test_08cls_stale_feat.txt"
        with open(feat_path, "w") as f:
            f.write("f1\nf2\n")

        node.validate_model(model_path, feat_path, x_path, y_path, compute_bootstrap_ci=True, n_bootstrap=100)
        import folder_paths
        ci_path = os.path.join(folder_paths.get_output_directory(), "Classification", "08_Model_Evaluation",
                                "Holdout_External_Performance", "Bootstrap_CI_Results.csv")
        self.assertTrue(os.path.exists(ci_path))
        node.validate_model(model_path, feat_path, x_path, y_path, compute_bootstrap_ci=False)
        self.assertFalse(os.path.exists(ci_path), "stale Bootstrap_CI_Results.csv should be removed when compute_bootstrap_ci=False")


class TestRegressionBootstrapCI(unittest.TestCase):
    def setUp(self):
        _helpers.fresh_output_dir()
        self.mv = _helpers.load_node_module("Regression", "08 Model Evaluation/08-1 Hold-out & External Performance", "Model_Validation.py",
                                             alias="mv_reg_ci")

    def test_ccc_present_in_bootstrap_ci(self):
        rng = np.random.default_rng(0)
        y_test = rng.normal(loc=5, scale=2, size=20)
        y_pred = y_test + rng.normal(scale=0.5, size=20)

        class StubModel:
            def predict(self, X):
                return y_pred
        metrics, _ = self.mv.calculate_regression_metrics(StubModel(), pd.DataFrame(np.zeros((20, 1))), y_test)
        ci_df = self.mv.bootstrap_regression_ci(y_test, y_pred, metrics, n_bootstrap=200, seed=42)
        self.assertIn("CCC", ci_df["Metric"].values)


class TestRegressionMetricLabelTerminology(unittest.TestCase):
    """2026-08-03 display-label clarification (progress/34): only the log
    text/comments changed ("Predictive R2" -> "Test-set R2 (Q2F2)",
    "CV Predictive R2" -> "Mean CV R2" elsewhere) -- calculations and CSV
    schema (dict keys, CSV "Metric" values) are untouched. These tests lock
    that down: the numbers must be bit-identical to before, the CSV keys
    must be the legacy names, and the new/old wording must appear/disappear
    from the log exactly as intended."""

    def setUp(self):
        _helpers.fresh_output_dir()
        self.mv = _helpers.load_node_module("Regression", "08 Model Evaluation/08-1 Hold-out & External Performance", "Model_Validation.py",
                                             alias="mv_reg_labels")
        rng = np.random.default_rng(0)
        n = 30
        X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
        y = X["f1"] * 2 - X["f2"] + rng.normal(scale=0.3, size=n)
        self.model = LinearRegression().fit(X, y)
        self.X, self.y = X, y.to_numpy()

        self.model_path = "/tmp/test_08reg_labels_model.pkl"
        joblib.dump(self.model, self.model_path)
        self.feat_path = "/tmp/test_08reg_labels_feat.txt"
        with open(self.feat_path, "w") as f:
            f.write("f1\nf2\n")

        names = [f"C{i}" for i in range(n)]
        x_df = X.copy()
        x_df.insert(0, "SMILES", "")
        x_df.insert(0, "Name", names)
        y_df = pd.DataFrame({"Name": names, "value": self.y})
        self.x_path = "/tmp/test_08reg_labels_x.csv"
        self.y_path = "/tmp/test_08reg_labels_y.csv"
        x_df.to_csv(self.x_path, index=False)
        y_df.to_csv(self.y_path, index=False)

        train_names = [f"T{i}" for i in range(40)]
        Xt = pd.DataFrame({"f1": rng.normal(size=40), "f2": rng.normal(size=40)})
        yt = Xt["f1"] * 2 - Xt["f2"] + rng.normal(scale=0.3, size=40)
        train_df = Xt.copy()
        train_df.insert(0, "Name", train_names)
        train_df["value"] = yt.values
        self.train_path = "/tmp/test_08reg_labels_train.csv"
        train_df.to_csv(self.train_path, index=False)

    def test_test_set_r2_equals_q2f2_and_matches_sklearn_r2_score(self):
        from sklearn.metrics import r2_score
        metrics, y_pred = self.mv.calculate_regression_metrics(self.model, self.X, self.y)
        self.assertAlmostEqual(metrics["predictive_r2"], metrics["q2f2"], places=12,
                                msg="Test-set R2 (dict key 'predictive_r2') must be numerically identical to Q2F2")
        self.assertAlmostEqual(metrics["predictive_r2"], r2_score(self.y, y_pred), places=12,
                                msg="Test-set R2 must still be exactly sklearn's r2_score (unchanged calculation)")

    def test_csv_metric_keys_are_unchanged_legacy_names(self):
        node = self.mv.Model_Validation_Regression()
        node.validate_model(self.model_path, self.feat_path, self.x_path, self.y_path,
                             training_data_path=self.train_path, compute_bootstrap_ci=False)
        import folder_paths
        eval_path = os.path.join(folder_paths.get_output_directory(), "Regression", "08_Model_Evaluation",
                                  "Holdout_External_Performance", "Evaluation_Results_ExternalTestSet.csv")
        eval_df = pd.read_csv(eval_path)
        for key in ("Predictive_R2", "Pearson_r2", "Q2F1", "Q2F2", "Q2F3"):
            self.assertIn(key, eval_df["Metric"].values, f"CSV must keep legacy Metric key {key!r}")

    def test_log_uses_test_set_r2_wording_not_old_predictive_r2_wording(self):
        node = self.mv.Model_Validation_Regression()
        result = node.validate_model(self.model_path, self.feat_path, self.x_path, self.y_path,
                                      training_data_path=self.train_path, compute_bootstrap_ci=False)
        text = result["ui"]["text"]
        self.assertIn("Test-set R²", text)
        self.assertIn("Q2F2 (test-mean reference; identical to Test-set R²)", text)
        self.assertIn("Pearson r² (corr(y_true, y_pred)²; squared correlation)", text)
        self.assertNotIn("Predictive R²", text, "old 'Predictive R²' wording must not appear in the 8.1 log anymore")

    def test_q2f1_q2f3_are_na_without_training_data_path_others_still_computed(self):
        node = self.mv.Model_Validation_Regression()
        result = node.validate_model(self.model_path, self.feat_path, self.x_path, self.y_path,
                                      training_data_path="", compute_bootstrap_ci=False)
        text = result["ui"]["text"]
        self.assertIn("Q2F1: N/A", text)
        self.assertIn("Q2F3: N/A", text)
        self.assertNotIn("Q2F2: N/A", text, "Q2F2/Test-set R2/Pearson r2 do not need training data and must still compute")


if __name__ == "__main__":
    unittest.main()
