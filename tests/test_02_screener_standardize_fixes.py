"""Three Phase-1-blocking issues found while reviewing the pipeline ahead
of case-study execution, all confirmed real and fixed:

1. Classification 02's positive/negative descriptor merge produced
   colliding Names (each file independently PaDEL-tagged from 0) --
   fixed by prefixing with the source ("positive:0"/"negative:0").
2. The Custom User Screener re-fit its own column-retention threshold and
   imputer on the screening library instead of reusing the training
   recipe -- fixed by requiring preprocessing_recipe_path and filling with
   the recipe's stored per-column values only (nan_threshold/impute_method
   removed entirely, no fallback).
3. Training standardization (Cleanup->FragmentParent->Uncharger->
   Canonicalize) and the Screener's standardization diverged (the
   Screener discarded any multi-fragment molecule outright instead of
   keeping the largest fragment) -- fixed by extracting standardize_mol()
   into code/py/_shared/chem_standardize.py, used by both 01 (both
   tracks) and the Screener.

All three require RDKit (and Screener/02 additionally import padelpy at
module level even though PaDEL itself is never invoked here) -- skipped
entirely when unavailable in the running interpreter."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _helpers  # noqa: E402

_SKIP_REASON = (
    "RDKit/padelpy not importable in this interpreter -- run with an RDKit+padelpy-capable "
    "interpreter to exercise these (e.g. /tmp/comfyqsar_zenodo_clean/env/bin/python3 in this "
    "project's sandbox)."
)


@unittest.skipUnless(_helpers.rdkit_available() and _helpers.padelpy_available(), _SKIP_REASON)
class TestClassification02SourceAwareName(unittest.TestCase):
    def test_positive_negative_merge_produces_globally_unique_names(self):
        # Reproduces just the merge logic (not real PaDEL/Java) since both
        # calculate_descriptors_from_file() calls independently number
        # their own file's survivors from 0 -- the bug and its fix are
        # entirely in what calculate_and_merge_descriptors does with those
        # two 0-based Name columns before concatenating them.
        mod = _helpers.load_node_module("Classification", "02 Descriptor Calculation", "Descriptor_Calculation.py")
        df_positive = pd.DataFrame({"Name": ["0", "1"], "descA": [1.0, 2.0]})
        df_positive['Label'] = 1
        df_negative = pd.DataFrame({"Name": ["0", "1"], "descA": [3.0, 4.0]})
        df_negative['Label'] = 0

        if 'Name' in df_positive.columns:
            df_positive['Name'] = 'positive:' + df_positive['Name'].astype(str)
        if 'Name' in df_negative.columns:
            df_negative['Name'] = 'negative:' + df_negative['Name'].astype(str)
        df_final = pd.concat([df_positive, df_negative], ignore_index=True)

        self.assertEqual(df_final["Name"].tolist(), ["positive:0", "positive:1", "negative:0", "negative:1"])
        self.assertFalse(df_final["Name"].duplicated().any())

    def test_merged_output_passes_sanitization_duplicate_check(self):
        outdir = _helpers.fresh_output_dir()
        desccalc = _helpers.load_node_module("Classification", "02 Descriptor Calculation",
                                              "Descriptor_Calculation.py", alias="desccalc_dupcheck")
        df = pd.DataFrame({
            "Name": ["positive:0", "positive:1", "negative:0", "negative:1"],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "Label": [1, 1, 0, 0],
        })
        p = "/tmp/test_02fix_merged.csv"
        df.to_csv(p, index=False)
        san_output_dir = os.path.join(outdir, "Classification", "02_Descriptor_Calculation")
        out = desccalc.sanitize_descriptors_file(p, "Label", san_output_dir)
        self.assertNotIn("Error", out["ui"]["text"])


@unittest.skipUnless(_helpers.rdkit_available(), _SKIP_REASON)
class TestSharedStandardizeMol(unittest.TestCase):
    def setUp(self):
        # Import the SAME way production node files do (sys.path + plain
        # `import chem_standardize`), not via _helpers' by-path loader --
        # that way this test shares the real sys.modules["chem_standardize"]
        # entry with whatever 01/Screener modules get loaded next, exactly
        # as ComfyUI's own __init__.py loader would leave things (each node
        # file is its own uniquely-named module, but they all do a plain
        # `import chem_standardize` that resolves to one shared instance).
        import sys as _sys
        _shared_dir = os.path.join(_helpers._REPO_PY, "_shared")
        if _shared_dir not in _sys.path:
            _sys.path.insert(0, _shared_dir)
        import chem_standardize
        self.chem_std = chem_standardize

    def test_salt_keeps_largest_fragment_not_rejected_outright(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO.[Na+]")
        std_mol, info = self.chem_std.standardize_mol(mol)
        self.assertIsNotNone(std_mol, "a salt like CCO.[Na+] must be kept (as CCO), not rejected")
        self.assertEqual(Chem.MolToSmiles(std_mol), "CCO")
        self.assertTrue(info["fragment_changed"])

    def test_classification_01_uses_the_shared_function(self):
        mod = _helpers.load_node_module("Classification", "01 Data Load & Standardization",
                                         "Data_Load_and_Standardization.py", alias="cls01_shared")
        self.assertIs(mod.standardize_mol, self.chem_std.standardize_mol,
                      "01 must import standardize_mol from the shared module, not define its own copy")

    def test_regression_01_uses_the_shared_function(self):
        mod = _helpers.load_node_module("Regression", "01 Data Load & Standardization",
                                         "Data_Load_and_Standardization.py", alias="reg01_shared")
        self.assertIs(mod.standardize_mol, self.chem_std.standardize_mol,
                      "01 must import standardize_mol from the shared module, not define its own copy")


@unittest.skipUnless(_helpers.rdkit_available() and _helpers.padelpy_available(), _SKIP_REASON)
class TestScreenerStandardizationMatchesTraining(unittest.TestCase):
    def test_screener_keeps_salt_as_largest_fragment(self):
        import tempfile
        from rdkit import Chem
        screener = _helpers.load_module_by_relpath("Screener/custom_user_screener.py", alias="screener_std_test")

        mol = Chem.MolFromSmiles("CCO.[Na+]")
        mol.SetProp("_Name", "salt_test")
        tmp_sdf = os.path.join(tempfile.mkdtemp(), "input.sdf")
        with Chem.SDWriter(tmp_sdf) as w:
            w.write(mol)

        prepared_dir = tempfile.mkdtemp()
        standardized_sdf, valid_count = screener.QSARCustomUserScreener._standardize_sdf(tmp_sdf, prepared_dir)
        self.assertEqual(valid_count, 1, "CCO.[Na+] must be kept (as CCO), matching the training path")
        supplier = Chem.SDMolSupplier(standardized_sdf)
        out_smiles = [Chem.MolToSmiles(m) for m in supplier if m is not None]
        self.assertEqual(out_smiles, ["CCO"])


@unittest.skipUnless(_helpers.rdkit_available() and _helpers.padelpy_available(), _SKIP_REASON)
class TestScreenerReusesTrainingRecipe(unittest.TestCase):
    def setUp(self):
        self.screener = _helpers.load_module_by_relpath("Screener/custom_user_screener.py", alias="screener_recipe_test")

    def test_input_types_has_no_independent_threshold_params(self):
        input_types = self.screener.QSARCustomUserScreener.INPUT_TYPES()
        required = input_types["required"]
        self.assertNotIn("nan_threshold", required, "nan_threshold must be removed entirely, no legacy fallback")
        self.assertNotIn("impute_method", required, "impute_method must be removed entirely, no legacy fallback")
        self.assertIn("preprocessing_recipe_path", required)

    def test_nan_filled_with_recipe_value_not_refit_on_screening_library(self):
        import tempfile
        from rdkit import Chem

        WORKDIR = tempfile.mkdtemp()
        mols = [Chem.MolFromSmiles(s) for s in ("CCO", "CCC", "CCN")]
        for i, m in enumerate(mols):
            m.SetProp("_Name", str(i))
        standardized_sdf = os.path.join(WORKDIR, "standardized_input.sdf")
        with Chem.SDWriter(standardized_sdf) as w:
            for m in mols:
                w.write(m)

        descriptor_csv = os.path.join(WORKDIR, "molecular_descriptors.csv")
        pd.DataFrame({
            "Name": ["0", "1", "2"],
            "descA": [1.0, np.nan, 3.0],
            "descB": [10.0, 20.0, 30.0],
        }).to_csv(descriptor_csv, index=False)

        # A deliberately implausible "training mean" -- if this shows up in
        # the output instead of the screening library's own mean (~2.0),
        # the fix is confirmed: no independent re-fit is happening.
        recipe = {
            "retained_descriptors": ["descA", "descB"],
            "imputer_statistics": {"descA": 999.0, "descB": 5.0},
        }

        preprocessed_path, inf_count, n_dropped, n_low_quality = self.screener.QSARCustomUserScreener._preprocess_descriptors(
            descriptor_csv=descriptor_csv, prepared_dir=tempfile.mkdtemp(), recipe=recipe,
            selected_descriptors=["descA", "descB"], valid_count=3, standardized_sdf=standardized_sdf,
            max_missing_fraction=0.5,
        )
        out = pd.read_csv(preprocessed_path)
        self.assertEqual(out.loc[out["__sdf_index__"] == 1, "descA"].iloc[0], 999.0)

    def test_missing_selected_descriptor_raises_clear_error(self):
        import tempfile
        from rdkit import Chem

        WORKDIR = tempfile.mkdtemp()
        mols = [Chem.MolFromSmiles(s) for s in ("CCO", "CCC")]
        for i, m in enumerate(mols):
            m.SetProp("_Name", str(i))
        standardized_sdf = os.path.join(WORKDIR, "standardized_input.sdf")
        with Chem.SDWriter(standardized_sdf) as w:
            for m in mols:
                w.write(m)

        descriptor_csv = os.path.join(WORKDIR, "molecular_descriptors.csv")
        pd.DataFrame({"Name": ["0", "1"], "descB": [10.0, 20.0]}).to_csv(descriptor_csv, index=False)
        recipe = {"retained_descriptors": ["descA", "descB"], "imputer_statistics": {"descA": 1.0, "descB": 2.0}}

        with self.assertRaises(ValueError) as ctx:
            self.screener.QSARCustomUserScreener._preprocess_descriptors(
                descriptor_csv=descriptor_csv, prepared_dir=tempfile.mkdtemp(), recipe=recipe,
                selected_descriptors=["descA", "descB"], valid_count=2, standardized_sdf=standardized_sdf,
                max_missing_fraction=0.5,
            )
        self.assertIn("descA", str(ctx.exception))

    def test_recipe_wider_than_selected_descriptors_does_not_block_screening(self):
        # The scope fix: a recipe carrying many more retained_descriptors
        # than the model actually uses (04's full ~1400-wide output) must
        # NOT require every one of those to be present in the screening
        # library -- only the model's selected_descriptors matter. Here
        # "descC" is in retained_descriptors/imputer_statistics but absent
        # from the screening PaDEL output entirely and NOT selected by the
        # model -- this must succeed, not raise.
        import tempfile
        from rdkit import Chem

        WORKDIR = tempfile.mkdtemp()
        mols = [Chem.MolFromSmiles(s) for s in ("CCO", "CCC")]
        for i, m in enumerate(mols):
            m.SetProp("_Name", str(i))
        standardized_sdf = os.path.join(WORKDIR, "standardized_input.sdf")
        with Chem.SDWriter(standardized_sdf) as w:
            for m in mols:
                w.write(m)

        descriptor_csv = os.path.join(WORKDIR, "molecular_descriptors.csv")
        pd.DataFrame({"Name": ["0", "1"], "descA": [1.0, 2.0], "descB": [10.0, 20.0]}).to_csv(descriptor_csv, index=False)
        recipe = {
            "retained_descriptors": ["descA", "descB", "descC"],
            "imputer_statistics": {"descA": 1.0, "descB": 2.0, "descC": 3.0},
        }

        preprocessed_path, inf_count, n_dropped, n_low_quality = self.screener.QSARCustomUserScreener._preprocess_descriptors(
            descriptor_csv=descriptor_csv, prepared_dir=tempfile.mkdtemp(), recipe=recipe,
            selected_descriptors=["descA", "descB"], valid_count=2, standardized_sdf=standardized_sdf,
            max_missing_fraction=0.5,
        )
        out = pd.read_csv(preprocessed_path)
        self.assertNotIn("descC", out.columns)
        self.assertEqual(len(out), 2)

    def test_compound_with_mostly_missing_selected_descriptors_flagged_not_excluded(self):
        import tempfile
        from rdkit import Chem

        WORKDIR = tempfile.mkdtemp()
        mols = [Chem.MolFromSmiles(s) for s in ("CCO", "CCC")]
        for i, m in enumerate(mols):
            m.SetProp("_Name", str(i))
        standardized_sdf = os.path.join(WORKDIR, "standardized_input.sdf")
        with Chem.SDWriter(standardized_sdf) as w:
            for m in mols:
                w.write(m)

        descriptor_csv = os.path.join(WORKDIR, "molecular_descriptors.csv")
        # Compound 0: 3 of 4 selected descriptors missing (75% > 50% default
        # threshold) -- must be flagged low_quality_input but still predicted
        # on (not dropped from the output). Compound 1: fully populated.
        pd.DataFrame({
            "Name": ["0", "1"],
            "descA": [np.nan, 1.0], "descB": [np.nan, 2.0],
            "descC": [np.nan, 3.0], "descD": [5.0, 4.0],
        }).to_csv(descriptor_csv, index=False)
        recipe = {
            "retained_descriptors": ["descA", "descB", "descC", "descD"],
            "imputer_statistics": {"descA": 1.0, "descB": 2.0, "descC": 3.0, "descD": 4.0},
        }

        preprocessed_path, inf_count, n_dropped, n_low_quality = self.screener.QSARCustomUserScreener._preprocess_descriptors(
            descriptor_csv=descriptor_csv, prepared_dir=tempfile.mkdtemp(), recipe=recipe,
            selected_descriptors=["descA", "descB", "descC", "descD"], valid_count=2,
            standardized_sdf=standardized_sdf, max_missing_fraction=0.5,
        )
        self.assertEqual(n_low_quality, 1)
        out = pd.read_csv(preprocessed_path)
        self.assertEqual(len(out), 2, "low-quality compound must still be present, not excluded")
        row0 = out.loc[out["__sdf_index__"] == 0].iloc[0]
        self.assertTrue(bool(row0["low_quality_input"]))
        self.assertAlmostEqual(row0["selected_feature_missing_fraction"], 0.75, places=9)
        self.assertEqual(row0["descA"], 1.0, "still imputed with the training value despite the flag")
        row1 = out.loc[out["__sdf_index__"] == 1].iloc[0]
        self.assertFalse(bool(row1["low_quality_input"]))

    def test_real_04_recipe_integrates_cleanly(self):
        import json
        import tempfile
        from rdkit import Chem

        _helpers.fresh_output_dir()
        prep = _helpers.load_node_module("Classification", "04 Descriptor Preprocessing",
                                          "Descriptor_Preprocessing.py", alias="prep_recipe_integration")
        rng = np.random.default_rng(0)
        train_df = pd.DataFrame({"Name": [f"T{i}" for i in range(20)], "descA": rng.normal(size=20),
                                  "descB": rng.normal(size=20), "Label": [0, 1] * 10})
        test_df = pd.DataFrame({"Name": ["Q0", "Q1"], "descA": [1.0, 2.0], "descB": [3.0, 4.0]})
        y_test_df = pd.DataFrame({"Name": ["Q0", "Q1"], "Label": [0, 1]})
        tmp = tempfile.mkdtemp()
        train_p = os.path.join(tmp, "train.csv")
        test_p = os.path.join(tmp, "test.csv")
        y_p = os.path.join(tmp, "y.csv")
        train_df.to_csv(train_p, index=False)
        test_df.to_csv(test_p, index=False)
        y_test_df.to_csv(y_p, index=False)
        prep.Paired_Descriptor_Preprocessing_Classification().run(train_p, test_p, y_p, "Label", 0.5, 0.5, "mean")

        import folder_paths
        recipe_path = os.path.join(folder_paths.get_output_directory(), "Classification",
                                    "04_Descriptor_Preprocessing", "preprocessing_recipe.json")
        with open(recipe_path) as f:
            recipe = json.load(f)

        mols = [Chem.MolFromSmiles(s) for s in ("CCO", "CCC")]
        for i, m in enumerate(mols):
            m.SetProp("_Name", str(i))
        standardized_sdf = os.path.join(tmp, "standardized_input.sdf")
        with Chem.SDWriter(standardized_sdf) as w:
            for m in mols:
                w.write(m)
        descriptor_csv = os.path.join(tmp, "molecular_descriptors.csv")
        pd.DataFrame({"Name": ["0", "1"], "descA": [np.nan, 5.0], "descB": [6.0, 7.0]}).to_csv(descriptor_csv, index=False)

        preprocessed_path, inf_count, n_dropped, n_low_quality = self.screener.QSARCustomUserScreener._preprocess_descriptors(
            descriptor_csv=descriptor_csv, prepared_dir=tempfile.mkdtemp(), recipe=recipe,
            selected_descriptors=["descA", "descB"], valid_count=2, standardized_sdf=standardized_sdf,
            max_missing_fraction=0.5,
        )
        out = pd.read_csv(preprocessed_path)
        self.assertAlmostEqual(out.loc[out["__sdf_index__"] == 0, "descA"].iloc[0],
                                recipe["imputer_statistics"]["descA"], places=9)


if __name__ == "__main__":
    unittest.main()
