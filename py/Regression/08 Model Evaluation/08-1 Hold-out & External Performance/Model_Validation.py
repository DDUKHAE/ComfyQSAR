import os
import pandas as pd
import numpy as np
import joblib
import traceback
import folder_paths
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def _clean_name_col(df):
    """Stripped string Name series if the column exists, else None. Blank
    strings and the literal text 'nan' (from a stray NaN read as text
    upstream) are normalized to '' so blank-detection catches both."""
    if "Name" not in df.columns:
        return None
    s = df["Name"].astype(str).str.strip()
    return s.where(s.str.lower() != "nan", "")

def load_regression_inputs(trained_model_path, x_path, y_path, features_path):
    model = joblib.load(trained_model_path)
    # dtype={"Name": str} forces the Name column to stay text even when every
    # value looks numeric (e.g. "007") -- otherwise pandas' own dtype
    # inference reads it as int64 (007 -> 7) before .astype(str) ever runs,
    # so the leading zero is already gone by the time we try to preserve it.
    # A dtype dict entry for a column that isn't present is a no-op, so this
    # is safe even when X/Y have no Name column at all.
    x_test_df = pd.read_csv(x_path, dtype={"Name": str})
    y_test_df = pd.read_csv(y_path, dtype={"Name": str})

    if "value" in y_test_df.columns:
        target_col = "value"
    elif y_test_df.shape[1] == 1:
        target_col = y_test_df.columns[0]
    else:
        raise ValueError(f"Could not determine target column in {os.path.basename(y_path)}.")

    x_name = _clean_name_col(x_test_df)
    y_name = _clean_name_col(y_test_df)

    # ID-based alignment when BOTH files carry a Name column -- never assume
    # row order matches just because two files happen to have the same
    # length (independent upstream filtering/sorting could silently
    # desynchronize them, pairing a prediction with the wrong compound's
    # ground truth).
    if x_name is not None and y_name is not None:
        for label, path, names in (("X_test", x_path, x_name), ("Y_test", y_path, y_name)):
            n_blank = int((names == "").sum())
            if n_blank:
                raise ValueError(f"{label} ({os.path.basename(path)}) has {n_blank} blank/missing Name value(s).")
            dup = names[names.duplicated()]
            if len(dup):
                dup_list = sorted(set(dup.tolist()))
                preview = dup_list[:10]
                more = f" (+{len(dup_list) - 10} more)" if len(dup_list) > 10 else ""
                raise ValueError(f"{label} ({os.path.basename(path)}) has duplicate Name value(s): {preview}{more}")

        x_set, y_set = set(x_name.tolist()), set(y_name.tolist())
        if x_set != y_set:
            only_x = sorted(x_set - y_set)[:10]
            only_y = sorted(y_set - x_set)[:10]
            raise ValueError(
                f"X_test ({os.path.basename(x_path)}) and Y_test ({os.path.basename(y_path)}) "
                f"Name sets do not match. In X_test only (up to 10): {only_x}. "
                f"In Y_test only (up to 10): {only_y}."
            )

        y_lookup = y_test_df.copy()
        y_lookup["Name"] = y_name
        y_reordered = y_lookup.set_index("Name").loc[x_name.tolist()].reset_index()
        y_test = y_reordered[target_col].to_numpy()
        alignment_mode = "id_matched"
        names = x_name.tolist()
    else:
        if len(x_test_df) != len(y_test_df):
            raise ValueError(
                f"X_test ({os.path.basename(x_path)}, {len(x_test_df)} rows) and Y_test "
                f"({os.path.basename(y_path)}, {len(y_test_df)} rows) have different lengths, and "
                "at least one file has no 'Name' column to align by -- cannot safely combine them."
            )
        y_test = y_test_df[target_col].to_numpy()
        alignment_mode = "row_order_fallback"
        names = x_name.tolist() if x_name is not None else [str(i) for i in range(len(x_test_df))]

    nan_mask = pd.isna(y_test)
    if nan_mask.any():
        n_nan = int(nan_mask.sum())
        if alignment_mode == "id_matched":
            bad_names = x_name.tolist()
            bad_names = [n for n, is_nan in zip(bad_names, nan_mask) if is_nan][:10]
            raise ValueError(f"Target has {n_nan} missing (NaN) value(s), e.g. Name={bad_names}.")
        raise ValueError(f"Target has {n_nan} missing (NaN) value(s) (row_order_fallback -- no Name available to identify them).")

    smiles = x_test_df["SMILES"].astype(str).tolist() if "SMILES" in x_test_df.columns else [""] * len(x_test_df)

    with open(features_path, "r") as f:
        selected_features = [line.strip() for line in f if line.strip()]
    missing = [ft for ft in selected_features if ft not in x_test_df.columns]
    if missing:
        raise ValueError(f"Missing features in X_test: {', '.join(missing)}")
    return model, x_test_df[selected_features], y_test, names, smiles, alignment_mode


def _regression_through_origin_r2(y_dep, y_indep):
    """R^2 of the intercept-free regression y_dep = k * y_indep (least squares,
    no constant term) -- the r0^2 used by Roy's r2m metric."""
    y_dep = np.asarray(y_dep, dtype=float)
    y_indep = np.asarray(y_indep, dtype=float)
    denom = float(np.sum(y_indep ** 2))
    if denom <= 0:
        return float("nan")
    k = float(np.sum(y_indep * y_dep) / denom)
    y_hat = k * y_indep
    ss_res = float(np.sum((y_dep - y_hat) ** 2))
    ss_tot = float(np.sum((y_dep - y_dep.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _ccc(y_true, y_pred):
    """Lin's Concordance Correlation Coefficient (Chirico & Gramatica 2011)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    n = len(y_true)
    numerator = 2 * np.sum((y_true - mean_true) * (y_pred - mean_pred))
    denominator = (
        np.sum((y_true - mean_true) ** 2)
        + np.sum((y_pred - mean_pred) ** 2)
        + n * (mean_true - mean_pred) ** 2
    )
    return float(numerator / denominator) if denominator > 0 else float("nan")


def calculate_regression_metrics(model, x_test, y_test, y_train=None):
    """
    y_train (optional): the training-set target values, needed only for
    Q2F1/Q2F3 (Gramatica & Sangion 2016, doi:10.1021/acs.jcim.6b00088), which
    are defined against the training-set mean/variance rather than the test
    set's own. If not supplied, those two are reported as None (skipped, not
    silently guessed) -- CCC, Q2F2, and r2m never need training data at all.
    """
    y_pred = model.predict(x_test)
    y_test_arr = np.asarray(y_test, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, y_pred)

    ccc = _ccc(y_test_arr, y_pred_arr)

    # r2m (Roy, Kar & Das 2015, doi:10.1007/978-3-319-17281-1_2): r2 is the
    # ordinary (intercept) correlation; r0^2/r0'^2 are the two
    # regression-through-origin directions. abs() inside the sqrt guards the
    # edge case r0^2 > r2 (can happen with small test sets) instead of
    # producing NaN -- the raw pre-abs difference is kept in the output for
    # transparency (r2_minus_r0_2_*).
    if len(y_test_arr) > 1 and np.std(y_test_arr) > 0 and np.std(y_pred_arr) > 0:
        r2_ordinary = float(np.corrcoef(y_test_arr, y_pred_arr)[0, 1] ** 2)
    else:
        r2_ordinary = float("nan")
    r0_2_forward = _regression_through_origin_r2(y_test_arr, y_pred_arr)
    r0_2_reverse = _regression_through_origin_r2(y_pred_arr, y_test_arr)
    r2_minus_r0_2_forward = r2_ordinary - r0_2_forward
    r2_minus_r0_2_reverse = r2_ordinary - r0_2_reverse
    if not np.isnan(r2_ordinary):
        r2m_forward = r2_ordinary * (1 - np.sqrt(abs(r2_minus_r0_2_forward)))
        r2m_reverse = r2_ordinary * (1 - np.sqrt(abs(r2_minus_r0_2_reverse)))
        r2m_overall = (r2m_forward + r2m_reverse) / 2
        delta_r2m = abs(r2m_forward - r2m_reverse)
    else:
        r2m_forward = r2m_reverse = r2m_overall = delta_r2m = float("nan")

    # Q2F1/Q2F2/Q2F3 (Gramatica & Sangion 2016): same numerator (test-set
    # SSE), different denominators. The paper's conclusion is to report all
    # three together rather than pick one as definitive.
    sse = float(np.sum((y_test_arr - y_pred_arr) ** 2))
    q2f2_denom = float(np.sum((y_test_arr - y_test_arr.mean()) ** 2))
    q2f2 = 1 - sse / q2f2_denom if q2f2_denom > 0 else float("nan")

    q2f1 = None
    q2f3 = None
    if y_train is not None and len(y_train) > 0:
        y_train_arr = np.asarray(y_train, dtype=float)
        q2f1_denom = float(np.sum((y_test_arr - y_train_arr.mean()) ** 2))
        q2f1 = 1 - sse / q2f1_denom if q2f1_denom > 0 else float("nan")

        n_test = len(y_test_arr)
        n_train = len(y_train_arr)
        train_var_per_n = (
            float(np.sum((y_train_arr - y_train_arr.mean()) ** 2)) / n_train if n_train > 0 else float("nan")
        )
        sse_per_n = sse / n_test if n_test > 0 else float("nan")
        q2f3 = 1 - sse_per_n / train_var_per_n if train_var_per_n and train_var_per_n > 0 else float("nan")

    metrics = {
        # Dict key "predictive_r2" is a LEGACY name kept for CSV/schema
        # compatibility with already-finalized case-study results (QDB258/
        # 260/261) -- the quantity itself is the test-set coefficient of
        # determination (1-SSE/SST_test, sklearn's r2_score), i.e. Q2F2 under
        # the Gramatica & Sangion 2016 naming used elsewhere in this file.
        # "pearson_r2" (squared correlation) is a different quantity that
        # happens to share the casual name "R2" in QSAR usage -- keeping
        # them under distinct keys here is what let this same confusion
        # (comparing a literature Pearson r2 against this platform's
        # test-set R2 as if they were the same number) get caught during
        # QDB261 analysis. See progress/34 for the 2026-08-03 display-label
        # clarification (no change to these keys or their values).
        "predictive_r2": r2, "pearson_r2": r2_ordinary,
        "mse": mse, "rmse": rmse, "mae": mae,
        "ccc": ccc,
        "q2f1": q2f1, "q2f2": q2f2, "q2f3": q2f3,
        "r2m_forward": r2m_forward, "r2m_reverse": r2m_reverse,
        "r2m_overall": r2m_overall, "delta_r2m": delta_r2m,
        "r2_minus_r0_2_forward": r2_minus_r0_2_forward,
        "r2_minus_r0_2_reverse": r2_minus_r0_2_reverse,
    }
    return metrics, y_pred


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.4f}"


def bootstrap_regression_ci(y_test, y_pred, metrics, n_bootstrap=2000, seed=42):
    """Bootstrap 95% CI for Predictive_R2/Pearson_r2/RMSE/MAE/CCC. Resamples
    compound indices once per replicate (same mechanics as
    thesis_metric_verification/calculate_manuscript_uncertainty.py, so
    results are bit-for-bit reproducible against that prior analysis). A
    replicate's Predictive_R2 is skipped (not NaN-appended) when the
    resampled y_test has zero variance -- SS_tot would be 0, making it
    undefined for that draw. Pearson_r2 is skipped the same way when either
    the resampled actual or predicted values are constant (correlation
    undefined). CCC is skipped when its own denominator collapses to 0
    (degenerate all-constant draw). Q2F1-3/r2m are explicitly out of
    bootstrap-CI scope -- their denominators reference the training set,
    not just this resampled draw, so a per-replicate CI isn't a
    like-for-like comparison the way it is for these four."""
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_test)
    rng = np.random.default_rng(seed)
    predictive_r2_vals, pearson_r2_vals, rmse_vals, mae_vals, ccc_vals = [], [], [], [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_test[idx], y_pred[idx]
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        if ss_tot > 0:
            ss_res = float(np.sum((yt - yp) ** 2))
            predictive_r2_vals.append(1 - ss_res / ss_tot)
        if np.std(yt) > 0 and np.std(yp) > 0:
            pearson_r2_vals.append(float(np.corrcoef(yt, yp)[0, 1] ** 2))
        rmse_vals.append(float(np.sqrt(mean_squared_error(yt, yp))))
        mae_vals.append(float(mean_absolute_error(yt, yp)))
        ccc_i = _ccc(yt, yp)
        if not np.isnan(ccc_i):
            ccc_vals.append(ccc_i)

    rows = []
    for key, label, vals in (
        ("predictive_r2", "Predictive_R2", predictive_r2_vals),
        ("pearson_r2", "Pearson_r2", pearson_r2_vals),
        ("rmse", "RMSE", rmse_vals),
        ("mae", "MAE", mae_vals),
        ("ccc", "CCC", ccc_vals),
    ):
        v = np.asarray(vals, dtype=float)
        n_valid = len(v)
        if n_valid > 0:
            ci_lo, ci_hi = float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))
        else:
            ci_lo, ci_hi = float("nan"), float("nan")
        rows.append({
            "Metric": label, "Point_Estimate": metrics[key],
            "CI_Lower": ci_lo, "CI_Upper": ci_hi,
            "N_Bootstrap": n_bootstrap, "Bootstrap_Seed": seed,
            "N_Valid_Replicates": n_valid, "N_Invalid_Replicates": n_bootstrap - n_valid,
        })
    return pd.DataFrame(rows)


def save_regression_results(output_dir, y_test, y_pred, metrics, names, smiles, alignment_mode,
                             compute_bootstrap_ci=True, n_bootstrap=2000, bootstrap_seed=42):
    pred_df = pd.DataFrame({
        "Name": names, "SMILES": smiles, "Actual": y_test, "Predicted": y_pred,
        "Alignment_Mode": alignment_mode,
    })
    pred_path = os.path.join(output_dir, "Actual_vs_Predicted.csv")
    pred_df.to_csv(pred_path, index=False)

    # Meets_Recommended_Threshold is advisory only (reported alongside the
    # value, never used to gate/fail the run) -- literature-recommended
    # thresholds: CCC>0.8, r2m(overall)>0.5, delta_r2m<0.2.
    rows = [
        # "Predictive_R2" is a legacy CSV key (kept for compatibility with
        # already-finalized case-study results) -- the value is the
        # test-set coefficient of determination, 1-SSE/SST_test, i.e. the
        # same quantity as "Q2F2" below. See progress/34.
        ("Predictive_R2", metrics["predictive_r2"], None),
        # Pearson_r2 (squared correlation) is a different quantity from
        # Predictive_R2 (1-SSE/SST_test) -- they're easy to conflate under the
        # single casual name "R2" (this distinction is what QDB261's
        # reference-model comparison turned on: the published external
        # "R2test" was Pearson_r2, not this platform's Predictive_R2).
        ("Pearson_r2", metrics["pearson_r2"], None),
        ("MSE", metrics["mse"], None),
        ("RMSE", metrics["rmse"], None),
        ("MAE", metrics["mae"], None),
        ("CCC", metrics["ccc"], metrics["ccc"] > 0.8 if metrics["ccc"] is not None and not np.isnan(metrics["ccc"]) else None),
        ("Q2F1", metrics["q2f1"], None),
        # Q2F2 is mathematically identical to Predictive_R2 (same SSE
        # numerator, same test-set-SST denominator) -- both are kept since
        # Q2F1/Q2F2/Q2F3 are conventionally reported together as a triplet
        # (Gramatica & Sangion 2016), but they are not independent numbers.
        ("Q2F2", metrics["q2f2"], None),
        ("Q2F3", metrics["q2f3"], None),
        ("r2m_forward", metrics["r2m_forward"], None),
        ("r2m_reverse", metrics["r2m_reverse"], None),
        ("r2m_overall", metrics["r2m_overall"],
         metrics["r2m_overall"] > 0.5 if metrics["r2m_overall"] is not None and not np.isnan(metrics["r2m_overall"]) else None),
        ("delta_r2m", metrics["delta_r2m"],
         metrics["delta_r2m"] < 0.2 if metrics["delta_r2m"] is not None and not np.isnan(metrics["delta_r2m"]) else None),
        ("r2_minus_r0_2_forward", metrics["r2_minus_r0_2_forward"], None),
        ("r2_minus_r0_2_reverse", metrics["r2_minus_r0_2_reverse"], None),
        ("Alignment_Mode", alignment_mode, None),
    ]
    eval_df = pd.DataFrame(rows, columns=["Metric", "Value", "Meets_Recommended_Threshold"])
    eval_path = os.path.join(output_dir, "Evaluation_Results_ExternalTestSet.csv")
    eval_df.to_csv(eval_path, index=False)

    ci_path = None
    ci_df = None
    stale_ci_path = os.path.join(output_dir, "Bootstrap_CI_Results.csv")
    if compute_bootstrap_ci:
        ci_df = bootstrap_regression_ci(y_test, y_pred, metrics, n_bootstrap, bootstrap_seed)
        ci_path = stale_ci_path
        ci_df.to_csv(ci_path, index=False)
    elif os.path.exists(stale_ci_path):
        # A prior run with compute_bootstrap_ci=True left this file behind --
        # remove it so it can't be mistaken for this run's (nonexistent) CI.
        os.remove(stale_ci_path)

    return eval_path, pred_path, ci_path, ci_df

class Model_Validation_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trained_model_path": ("STRING", {"tooltip": "From 7."}),
                "descriptor_list_path": ("STRING", {"tooltip": "From 7. The .txt name list, not a CSV."}),
                "holdout_data_path": ("STRING", {"tooltip": "Use 4's PREPROCESSED_HOLDOUT -- not 3's HOLDOUT_DATA (no train-fitted imputation yet)."}),
                "holdout_targets_path": ("STRING", {"tooltip": "Use 4's FILTERED_HOLDOUT_TARGETS -- not 3's HOLDOUT_TARGETS (4 may drop compounds, so the row sets must match)."}),
            },
            "optional": {
                "training_data_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional. Only needed for Q2F1/Q2F3; skipped if empty.",
                }),
                "compute_bootstrap_ci": ("BOOLEAN", {"default": True}),
                "n_bootstrap": ("INT", {"default": 2000, "min": 100, "max": 100000, "step": 100}),
                "bootstrap_seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("TRAINED_MODEL", "SELECTED_DESCRIPTOR_LIST")
    OUTPUT_TOOLTIPS = (
        "Same value as this node's own trained_model_path input -- routed through here so "
        "downstream nodes (e.g. External Data Screening) only run after this validation.",
        "Same value as this node's own descriptor_list_path input -- routed through for the same reason.",
    )
    FUNCTION = "validate_model"
    CATEGORY = "QSAR/2. REGRESSION/8. Model Evaluation"
    OUTPUT_NODE = True

    def validate_model(self, trained_model_path, descriptor_list_path, holdout_data_path, holdout_targets_path, training_data_path="",
                        compute_bootstrap_ci=True, n_bootstrap=2000, bootstrap_seed=42):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Regression", "08_Model_Evaluation", "Holdout_External_Performance")
            os.makedirs(output_dir, exist_ok=True)
            model, x_test_filtered, y_test, names, smiles, alignment_mode = load_regression_inputs(
                trained_model_path, holdout_data_path, holdout_targets_path, descriptor_list_path
            )
            y_train = None
            q2f_note = " (training_data_path not provided -- skipped)"
            if training_data_path:
                train_df = pd.read_csv(training_data_path)
                if "value" in train_df.columns:
                    y_train = train_df["value"].values
                    q2f_note = ""
                else:
                    q2f_note = " (training_data_path has no 'value' column -- skipped)"
            metrics, y_pred = calculate_regression_metrics(model, x_test_filtered, y_test, y_train=y_train)
            eval_path, pred_path, ci_path, ci_df = save_regression_results(
                output_dir, y_test, y_pred, metrics, names, smiles, alignment_mode,
                compute_bootstrap_ci, n_bootstrap, bootstrap_seed,
            )

            ccc_flag = " (meets recommended >0.8)" if metrics["ccc"] is not None and not np.isnan(metrics["ccc"]) and metrics["ccc"] > 0.8 else ""
            r2m_flag = " (meets recommended >0.5)" if metrics["r2m_overall"] is not None and not np.isnan(metrics["r2m_overall"]) and metrics["r2m_overall"] > 0.5 else ""
            delta_flag = " (meets recommended <0.2)" if metrics["delta_r2m"] is not None and not np.isnan(metrics["delta_r2m"]) and metrics["delta_r2m"] < 0.2 else ""

            log_message = (
                "========================================\n"
                "🔹 8.1 Hold-out & External Performance (Regression) Done! 🔹\n"
                "========================================\n"
                f"📌 Model: {os.path.basename(trained_model_path)}\n"
                f"📊 Test-set R² (1-SSE/SST_test; identical to Q²F2): {metrics['predictive_r2']:.4f}\n"
                f"📊 Pearson r² (corr(y_true, y_pred)²; squared correlation): {_fmt(metrics['pearson_r2'])}\n"
                f"📊 MSE: {metrics['mse']:.4f}\n"
                f"📊 RMSE: {metrics['rmse']:.4f}\n"
                f"📊 MAE: {metrics['mae']:.4f}\n"
                "----------------------------------------\n"
                f"📊 CCC: {_fmt(metrics['ccc'])}{ccc_flag}\n"
                f"📊 Q2F1: {_fmt(metrics['q2f1'])}{q2f_note if metrics['q2f1'] is None else ''}\n"
                f"📊 Q2F2 (test-mean reference; identical to Test-set R²): {_fmt(metrics['q2f2'])}\n"
                f"📊 Q2F3: {_fmt(metrics['q2f3'])}{q2f_note if metrics['q2f3'] is None else ''}\n"
                f"📊 r2m(overall): {_fmt(metrics['r2m_overall'])}{r2m_flag}\n"
                f"📊 delta_r2m: {_fmt(metrics['delta_r2m'])}{delta_flag}\n"
                "----------------------------------------\n"
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
                f"💾 Evaluation Results: {os.path.basename(eval_path)}\n"
                f"💾 Predictions: {os.path.basename(pred_path)}\n"
                f"📌 Alignment mode: {alignment_mode}\n"
                + (
                    "⚠️ X_test/Y_test could not be aligned by 'Name' (missing on at least one side) -- "
                    "fell back to row order. Verify both files came from the same, unreordered pipeline run.\n"
                    if alignment_mode == "row_order_fallback" else ""
                )
                + (
                    (
                        f"----------------------------------------\n"
                        f"📈 Bootstrap 95% CI (n_bootstrap={n_bootstrap}, seed={bootstrap_seed}):\n"
                        + "\n".join(
                            f"    {row['Metric']}: {row['Point_Estimate']:.4f} [{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}]"
                            for _, row in ci_df.iterrows()
                        )
                        + f"\n💾 Bootstrap CI: {os.path.basename(ci_path)}\n"
                    ) if compute_bootstrap_ci and ci_df is not None else ""
                )
                + "========================================"
            )
            return {
                "ui": {"text": log_message},
                "result": (
                    str(trained_model_path),
                    str(descriptor_list_path),
                )
            }
        except Exception as e:
            return {
                "ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"},
                "result": ("", "")
            }


NODE_CLASS_MAPPINGS = {
    "Model_Validation_Regression": Model_Validation_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Model_Validation_Regression": "8.1 Hold-out & External Performance",
}
