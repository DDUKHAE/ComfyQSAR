"""Classification target must be strictly {0,1} in training-context nodes
(04, 07, 08c, 8b -- strict equality) and a subset of {0,1} in 08's
external/query context (issubset -- single-class external sets allowed)."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _helpers  # noqa: E402


class TestDataSplitLabelContract(unittest.TestCase):
    def setUp(self):
        _helpers.fresh_output_dir()
        self.mod = _helpers.load_node_module("Classification", "03 Data Split", "Data_Split.py")

    def _make_csv(self, path, labels):
        n = len(labels)
        df = pd.DataFrame({
            "Name": [f"C{i}" for i in range(n)],
            "f1": np.random.default_rng(0).normal(size=n),
            "Label": labels,
        })
        df.to_csv(path, index=False)
        return path

    def test_rejects_non_01_labels(self):
        p = self._make_csv("/tmp/test_split_bad_labels.csv", [1, 2] * 5)
        node = self.mod.QSARDataSplit_Classification()
        with self.assertRaises(ValueError) as ctx:
            node.execute(p, test_size=0.2, random_state=42)
        self.assertIn("0/1", str(ctx.exception))

    def test_accepts_01_labels(self):
        p = self._make_csv("/tmp/test_split_good_labels.csv", [0, 1] * 5)
        node = self.mod.QSARDataSplit_Classification()
        out = node.execute(p, test_size=0.2, random_state=42)
        self.assertNotIn("❌", out["ui"]["text"])


class TestHyperparameterTuningLabelContract(unittest.TestCase):
    def setUp(self):
        _helpers.fresh_output_dir()
        self.mod = _helpers.load_node_module(
            "Classification", "07 Hyperparameter Tuning & Model Training",
            "Hyperparameter_Tuning_and_Model_Training.py",
        )

    def test_rejects_non_01_labels(self):
        n = 10
        df = pd.DataFrame({
            "f1": np.random.default_rng(0).normal(size=n),
            "Label": [-1, 1] * 5,
        })
        p = "/tmp/test_07_bad_labels.csv"
        df.to_csv(p, index=False)
        node = self.mod.Hyperparameter_Grid_Search_Classification()
        out = node.perform_grid_search(p, "logistic", "Label")
        self.assertIn("0/1", out["ui"]["text"])


class TestModelValidation08LabelContract(unittest.TestCase):
    """08 is the external/query context -- a single-class {0} or {1} set is
    allowed (issubset), but anything outside {0,1} must still fail."""

    def setUp(self):
        _helpers.fresh_output_dir()
        self.mod = _helpers.load_node_module("Classification", "08 Model Evaluation/08-1 Hold-out & External Performance",
                                              "Model_Validation.py", alias="mv_labelcontract")

    def test_single_class_external_allowed(self):
        n = 6
        x_df = pd.DataFrame({"Name": [f"C{i}" for i in range(n)], "f1": np.random.default_rng(0).normal(size=n)})
        y_df = pd.DataFrame({"Name": [f"C{i}" for i in range(n)], "Label": [0] * n})
        x_path, y_path = "/tmp/test_08_x_single.csv", "/tmp/test_08_y_single.csv"
        x_df.to_csv(x_path, index=False)
        y_df.to_csv(y_path, index=False)
        feat_path = "/tmp/test_08_feat_single.txt"
        with open(feat_path, "w") as f:
            f.write("f1\n")
        import joblib
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression().fit(pd.DataFrame({"f1": [0, 1, 0, 1]}), [0, 1, 0, 1])
        model_path = "/tmp/test_08_model_single.pkl"
        joblib.dump(model, model_path)
        # should not raise
        self.mod.load_classification_inputs(model_path, x_path, y_path, feat_path)

    def test_non_01_external_rejected(self):
        n = 4
        x_df = pd.DataFrame({"Name": [f"C{i}" for i in range(n)], "f1": np.random.default_rng(0).normal(size=n)})
        y_df = pd.DataFrame({"Name": [f"C{i}" for i in range(n)], "Label": [2, 3, 2, 3]})
        x_path, y_path = "/tmp/test_08_x_bad.csv", "/tmp/test_08_y_bad.csv"
        x_df.to_csv(x_path, index=False)
        y_df.to_csv(y_path, index=False)
        feat_path = "/tmp/test_08_feat_bad.txt"
        with open(feat_path, "w") as f:
            f.write("f1\n")
        import joblib
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression().fit(pd.DataFrame({"f1": [0, 1, 0, 1]}), [0, 1, 0, 1])
        model_path = "/tmp/test_08_model_bad.pkl"
        joblib.dump(model, model_path)
        with self.assertRaises(ValueError):
            self.mod.load_classification_inputs(model_path, x_path, y_path, feat_path)


class TestResamplingValidation8cLabelContract(unittest.TestCase):
    def setUp(self):
        _helpers.fresh_output_dir()
        self.mod = _helpers.load_node_module("Classification", "08 Model Evaluation/08-3 Resampling Stability Assessment",
                                              "Resampling_Validation.py")

    def test_rejects_non_01_labels(self):
        n = 10
        df = pd.DataFrame({
            "f1": np.random.default_rng(0).normal(size=n),
            "Label": ["a", "b"] * 5,
        })
        p = "/tmp/test_8c_bad_labels.csv"
        df.to_csv(p, index=False)
        node = self.mod.ResamplingValidation_Classification()
        out = node.run(p, "nonexistent.pkl", "Label", "loocv")
        self.assertIn("0/1", out["ui"]["text"])


class TestYScrambling8bLabelContract(unittest.TestCase):
    def setUp(self):
        _helpers.fresh_output_dir()
        self.mod = _helpers.load_node_module("Classification", "08 Model Evaluation/08-2 Chance-Correlation Test",
                                              "Y_Scrambling_Validation.py")

    def test_rejects_non_01_labels(self):
        n = 10
        df = pd.DataFrame({
            "f1": np.random.default_rng(0).normal(size=n),
            "Label": [1, 2] * 5,
        })
        p = "/tmp/test_8b_bad_labels.csv"
        df.to_csv(p, index=False)
        node = self.mod.YScramblingValidation_Classification()
        out = node.run(p, "nonexistent.pkl", "Label")
        self.assertIn("0/1", out["ui"]["text"])

    def test_accepts_01_labels_and_runs(self):
        import joblib
        from sklearn.linear_model import LogisticRegression
        n = 20
        X = pd.DataFrame({"f1": np.random.default_rng(0).normal(size=n)})
        y = np.array([0, 1] * (n // 2))
        model = LogisticRegression().fit(X, y)
        model_path = "/tmp/test_8b_model.pkl"
        joblib.dump(model, model_path)
        df = X.copy()
        df["Label"] = y
        p = "/tmp/test_8b_good_labels.csv"
        df.to_csv(p, index=False)
        node = self.mod.YScramblingValidation_Classification()
        out = node.run(p, model_path, "Label", n_permutations=10, cv_splits=3)
        self.assertNotIn("❌", out["ui"]["text"])


if __name__ == "__main__":
    unittest.main()
