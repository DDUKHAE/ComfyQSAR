"""Node 6 (Descriptor Combination) regression tests.

Covers two things found during the num_cores/loky performance investigation
(2026-07-28): (1) node 6 must actually complete under REAL multiprocessing
(n_jobs=2) when loaded the way the real ComfyUI loader loads it (no
sys.modules registration) -- a test harness that DOES register the module
in sys.modules masks a real cloudpickle-serialization difference and can
report a false "ModuleNotFoundError" that never happens in production, or
vice versa hide a real one; (2) num_cores now defaults to 1 and small
searches force serial execution regardless of the requested value.
"""
import os
import unittest

import numpy as np
import pandas as pd

import _helpers


def _make_dataset(n_rows=40, n_features=6, target_col="Label", classification=True, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n_rows, n_features)), columns=[f"f{i}" for i in range(n_features)])
    if classification:
        df[target_col] = rng.integers(0, 2, size=n_rows)
    else:
        df[target_col] = rng.normal(size=n_rows)
    return df


class TestDescriptorCombinationClassification(unittest.TestCase):
    def setUp(self):
        _helpers.fresh_output_dir()
        self.mod = _helpers.load_node_module(
            "Classification", "06 Descriptor Combination", "Descriptor_Combination_MemoryE.py",
            alias="test06_cls",
        )

    def _write_csv(self, df):
        path = os.path.join(_helpers._OUTPUT_ROOT, "input.csv")
        df.to_csv(path, index=False)
        return path

    def test_default_num_cores_is_1(self):
        input_types = self.mod.Feature_Combination_Search.INPUT_TYPES()
        self.assertEqual(input_types["required"]["num_cores"][1]["default"], 1)

    def test_small_combination_count_forces_serial_regardless_of_request(self):
        # 4 features, max_features=2 -> C(4,2) = 6 combinations, well under
        # the force-serial threshold (16) even though num_cores=-1 is
        # requested (i.e. "use every core").
        df = _make_dataset(n_rows=30, n_features=4)
        path = self._write_csv(df)
        node = self.mod.Feature_Combination_Search()
        res = node.descriptor_combination_classification_MemoryE(path, 2, -1, 3, 2000)
        text = res["ui"]["text"]
        self.assertIn("Effective cores: 1", text)
        self.assertIn("Total combinations evaluated: 6", text)
        self.assertNotIn("per-worker startup overhead", text)  # warning suppressed when forced serial
        self.assertNotEqual(res["result"][0], "")

    def test_real_multiprocessing_with_n_jobs_2_completes(self):
        # 6 features, max_features=3 -> C(6,2)+C(6,3) = 15+20 = 35
        # combinations -- above the force-serial threshold, so this
        # genuinely exercises the loky backend with n_jobs=2. Loaded via
        # _helpers.load_node_module, which (as of this fix) does NOT
        # register the module in sys.modules, matching the real ComfyUI
        # loader -- this is the actual regression test for the
        # false-positive ModuleNotFoundError found during manual testing.
        df = _make_dataset(n_rows=30, n_features=6)
        path = self._write_csv(df)
        node = self.mod.Feature_Combination_Search()
        res = node.descriptor_combination_classification_MemoryE(path, 3, 2, 3, 2000)
        text = res["ui"]["text"]
        self.assertIn("Effective cores: 2", text)
        self.assertIn("Total combinations evaluated: 35", text)
        self.assertNotEqual(res["result"][0], "")
        self.assertTrue(os.path.isfile(res["result"][0]))


class TestDescriptorCombinationRegression(unittest.TestCase):
    def setUp(self):
        _helpers.fresh_output_dir()
        self.mod = _helpers.load_node_module(
            "Regression", "06 Descriptor Combination", "Descriptor_Combination_MemoryE.py",
            alias="test06_reg",
        )

    def _write_csv(self, df):
        path = os.path.join(_helpers._OUTPUT_ROOT, "input.csv")
        df.to_csv(path, index=False)
        return path

    def test_default_num_cores_is_1(self):
        input_types = self.mod.Regression_Feature_Combination_Search.INPUT_TYPES()
        self.assertEqual(input_types["required"]["num_cores"][1]["default"], 1)

    def test_small_combination_count_forces_serial_regardless_of_request(self):
        df = _make_dataset(n_rows=30, n_features=4, target_col="value", classification=False)
        path = self._write_csv(df)
        node = self.mod.Regression_Feature_Combination_Search()
        res = node.find_best_combinations(path, 2, -1, 3, 2000, "value")
        text = res["ui"]["text"]
        self.assertIn("Effective cores: 1", text)
        self.assertIn("Total combinations evaluated: 6", text)
        self.assertNotIn("per-worker startup overhead", text)
        self.assertNotEqual(res["result"][0], "")

    def test_real_multiprocessing_with_n_jobs_2_completes(self):
        df = _make_dataset(n_rows=30, n_features=6, target_col="value", classification=False)
        path = self._write_csv(df)
        node = self.mod.Regression_Feature_Combination_Search()
        res = node.find_best_combinations(path, 3, 2, 3, 2000, "value")
        text = res["ui"]["text"]
        self.assertIn("Effective cores: 2", text)
        self.assertIn("Total combinations evaluated: 35", text)
        self.assertNotEqual(res["result"][0], "")
        self.assertTrue(os.path.isfile(res["result"][0]))


if __name__ == "__main__":
    unittest.main()
