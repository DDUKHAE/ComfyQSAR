import os
import traceback
import numpy as np
import pandas as pd
import joblib
import folder_paths


def _leverage_status(lev, r, h_star):
    high_leverage = lev > h_star
    if r is None:
        return "high_leverage" if high_leverage else "in_domain"
    outlier = abs(r) > 3
    if high_leverage and outlier:
        return "high_leverage_outlier"
    if high_leverage:
        return "high_leverage"
    if outlier:
        return "outlier"
    return "in_domain"


def _leverage_ad_core(train_df, query_df, target_column, warning_leverage_multiplier, model_path=""):
    """Shared Williams/leverage AD core, used by both the Evaluation-Set AD
    node (hold-out/external query, which also computes standardized
    residuals when model_path + target_column are available) and the
    Screening-Candidate AD node (screening query, which has no ground-truth
    endpoint and so only ever gets leverage). Returns per-compound arrays
    aligned 1:1 with train_df/query_df row order, plus a stats dict."""
    has_query = query_df is not None

    metadata_cols_train = [c for c in ("Name", "SMILES", target_column) if c in train_df.columns]
    X_train = train_df.drop(columns=metadata_cols_train).select_dtypes(include=[np.number])
    descriptor_cols = X_train.columns.tolist()
    if not descriptor_cols:
        raise ValueError("No numeric descriptor columns found in training_data_path.")

    if has_query:
        missing_cols = [c for c in descriptor_cols if c not in query_df.columns]
        if missing_cols:
            preview = missing_cols[:5]
            more = "..." if len(missing_cols) > 5 else ""
            raise ValueError(f"holdout_data_path is missing descriptor columns used in training: {preview}{more}")
        X_query = query_df[descriptor_cols]
    else:
        X_query = None

    train_finite = np.isfinite(X_train.to_numpy(dtype=float))
    if not train_finite.all():
        bad_names = (
            train_df["Name"].iloc[np.where(~train_finite.all(axis=1))[0]].tolist()
            if "Name" in train_df.columns else np.where(~train_finite.all(axis=1))[0].tolist()
        )
        raise ValueError(
            f"training_data_path has NaN/inf in {len(bad_names)} descriptor row(s) "
            f"(e.g. {bad_names[:10]}) -- leverage matrix math requires a fully imputed, "
            "finite matrix. Use '4. Descriptor Preprocessing's TRAIN_PREPROCESSED output, "
            "not a raw/unimputed file."
        )
    if has_query:
        query_finite = np.isfinite(X_query.to_numpy(dtype=float))
        if not query_finite.all():
            bad_names = (
                query_df["Name"].iloc[np.where(~query_finite.all(axis=1))[0]].tolist()
                if "Name" in query_df.columns else np.where(~query_finite.all(axis=1))[0].tolist()
            )
            raise ValueError(
                f"holdout_data_path has NaN/inf in {len(bad_names)} descriptor row(s) "
                f"(e.g. {bad_names[:10]}) -- use '4. Descriptor Preprocessing's "
                "TEST_PREPROCESSED output (or the equivalent preprocessed screening CSV), "
                "not a raw/unimputed file."
            )

    n_train, p = X_train.shape
    if n_train <= p:
        raise ValueError(
            f"Leverage/hat-matrix AD requires more training compounds ({n_train}) than "
            f"descriptors ({p}) -- X^T X is not invertible otherwise."
        )

    # Standardize using training-set mean/std; query compounds are
    # projected into the SAME space (never re-standardized on their
    # own), matching how leverage/AD is defined relative to the
    # training descriptor space.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0).replace(0, 1.0)
    X_train_std = ((X_train - mean) / std).to_numpy()
    X_query_std = ((X_query - mean) / std).to_numpy() if has_query else None

    XtX = X_train_std.T @ X_train_std
    # matrix_rank is checked independently of whether inv() raises --
    # a near-singular (but not exactly singular) matrix can produce
    # a numerically "successful" inv() that is nonetheless wildly
    # inaccurate, so rank deficiency needs its own explicit check
    # rather than being inferred only from a caught LinAlgError.
    matrix_rank = int(np.linalg.matrix_rank(XtX))
    rank_deficient = matrix_rank < p
    try:
        XtX_inv = np.linalg.inv(XtX)
        used_pinv = False
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
        used_pinv = True

    # A full-rank XtX can still be numerically near-singular enough
    # that inv() "succeeds" while being unreliable -- matrix_rank's
    # SVD-based tolerance and inv()'s LU-based failure point are
    # different criteria, so a matrix can pass rank_deficient=False
    # yet still be poorly conditioned. Checked only when NOT already
    # rank_deficient (that warning already covers the same
    # underlying "unreliable inversion" concern for the more severe
    # case). 1e10 reflects double precision's ~16 significant
    # digits: a condition number at that scale means roughly 10 of
    # those digits are lost, well before inv() itself would raise.
    ILL_CONDITIONED_THRESHOLD = 1e10
    condition_number = float(np.linalg.cond(XtX))
    ill_conditioned = (not rank_deficient) and (condition_number >= ILL_CONDITIONED_THRESHOLD)

    # h_i = 1/n + x_i^T (X^T X)^-1 x_i, where X is mean-centered.
    # The "+1/n" term is required for consistency with h* =
    # 3(p+1)/n: that threshold assumes the *intercept-included*
    # hat matrix, whose trace is (p+1) (p descriptor slopes + 1
    # intercept), so its diagonal averages to (p+1)/n over the
    # training set. Without the 1/n term, this is instead the
    # through-origin hat matrix (trace p, average p/n) -- leverage
    # would be systematically under-reported by exactly 1/n per
    # compound (most visibly: the training-set mean point, whose
    # centered coordinates are all zero, would get leverage 0
    # instead of the correct 1/n), silently making the AD boundary
    # more permissive than h* was calibrated for. Standardizing
    # (dividing by std) on top of centering doesn't change any
    # leverage value -- hat-matrix leverage is invariant to
    # per-column rescaling -- so this correction only needed the
    # missing intercept term, not a different scaling.
    # Same formula applied to the training compounds themselves --
    # every Williams plot needs BOTH the training cloud and the
    # query points on the same (leverage, residual) axes; the
    # training-set leverage is what tells a reader whether a query
    # point's leverage is actually unusual relative to the model's
    # own training distribution, not just an absolute number.
    train_leverage = 1.0 / n_train + np.einsum('ij,jk,ik->i', X_train_std, XtX_inv, X_train_std)
    query_leverage = (
        1.0 / n_train + np.einsum('ij,jk,ik->i', X_query_std, XtX_inv, X_query_std)
        if has_query else np.array([])
    )
    h_star = warning_leverage_multiplier * (p + 1) / n_train

    # h* assumes (p+1)/n is a small fraction (so that 3x the average
    # leverage still meaningfully separates outliers from the bulk).
    # When descriptors are a large share of the training size, h*
    # itself can exceed 1 -- since training-point leverage is
    # mathematically bounded in [1/n, 1], a threshold above 1 can
    # never flag ANY training point as high-leverage regardless of
    # how extreme it is, silently making the check uninformative
    # rather than strict. This is a distinct condition from the
    # n_train<=p hard error above (which blocks non-invertibility);
    # h*>1 can happen even when n_train>p if p is still a large
    # fraction of n_train.
    h_star_warning = ""
    if h_star >= 1.0:
        h_star_warning = (
            f"\n⚠️ h* ({h_star:.4f}) is at or exceeds 1.0 -- with {p} descriptors and only "
            f"{n_train} training compounds, this AD threshold is likely uninformative "
            "(no training point's leverage can mathematically exceed 1). Consider "
            "further feature reduction (05-2/06) before relying on this AD result."
        )
    rank_warning = ""
    if rank_deficient:
        rank_warning = (
            f"\n⚠️ Descriptor matrix is rank-deficient (rank {matrix_rank} < {p} descriptors) -- "
            "likely collinear/duplicate descriptors survived selection. Leverage values from a "
            "rank-deficient (X^T X)^-1 can be unreliable even where the computation itself "
            "completes without error; consider further feature reduction (05-2/06)."
        )
    cond_warning = ""
    if ill_conditioned:
        cond_warning = (
            f"\n⚠️ Descriptor matrix is ill-conditioned (condition number {condition_number:.2e}) "
            f"despite full rank ({matrix_rank}/{p}) -- likely near-collinear descriptors. Leverage "
            "values from (X^T X)^-1 can still be numerically unreliable; consider further feature "
            "reduction (05-2/06)."
        )

    model = joblib.load(model_path) if model_path else None
    train_residuals = None
    query_residuals = None
    train_resid_std = None
    if model is not None and target_column in train_df.columns:
        # Standardized against the TRAINING set's own residual
        # spread -- i.e. each residual is reported as "how many
        # training-residual standard deviations away", not an
        # RMSE-equivalent quantity. The query set can be a handful
        # of compounds (or just one), too few to give a stable SD of
        # its own, whereas the training residuals reflect the
        # model's genuine noise level and let any number of query
        # points be standardized consistently against it (standard
        # Williams-plot practice). ddof=0 (population std, divide by
        # n) is used deliberately -- matches this codebase's other
        # residual-scale statistics (RMSE) rather than the ddof=1
        # sample-variance convention.
        y_train_true = train_df[target_column].values
        y_train_pred = model.predict(X_train)
        resid_train_raw = y_train_true - y_train_pred
        train_resid_std = resid_train_raw.std(ddof=0)
        train_residuals = resid_train_raw / train_resid_std if train_resid_std > 0 else resid_train_raw
        if has_query and target_column in query_df.columns:
            y_true = query_df[target_column].values
            y_pred = model.predict(X_query)
            resid = y_true - y_pred
            query_residuals = resid / train_resid_std if train_resid_std > 0 else resid

    return {
        "descriptor_cols": descriptor_cols, "n_train": n_train, "p": p,
        "matrix_rank": matrix_rank, "rank_deficient": rank_deficient,
        "condition_number": condition_number, "ill_conditioned": ill_conditioned,
        "used_pinv": used_pinv, "h_star": h_star,
        "h_star_warning": h_star_warning, "rank_warning": rank_warning, "cond_warning": cond_warning,
        "train_leverage": train_leverage, "query_leverage": query_leverage,
        "train_residuals": train_residuals, "query_residuals": query_residuals,
        "has_query": has_query,
    }


class ApplicabilityDomain_Regression:
    """Evaluation-Set Applicability Domain: interprets the applicability
    scope of hold-out/external performance (Model Validation), with a full
    Williams-plot leverage + standardized-residual characterization. For
    the applicability domain of external screening candidates, use
    ScreeningCandidateAD_Regression instead -- same leverage method and
    training set, but keyed to preserve screening prediction metadata, and
    leverage-only (no residual axis, since screening candidates have no
    observed endpoint)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "training_data_path": ("STRING", {
                    "tooltip": "Exact training dataset used by Node 7, after preprocessing and any "
                               "optional descriptor selection or combination.",
                }),
            },
            "optional": {
                "holdout_data_path": ("STRING", {
                    "default": "",
                    "tooltip": "4's PREPROCESSED_HOLDOUT. Must be imputed -- NaN breaks the leverage "
                               "matrix. Leave empty for a training-only characterization. For "
                               "screening candidates, use 'Screening-Candidate AD (Regression)' instead.",
                }),
                "mode": (["auto", "manual", "disabled"], {
                    "default": "auto",
                    "tooltip": "auto/manual both run the same computation (the descriptor matrix "
                               "is always available, no SMILES needed); disabled does nothing.",
                }),
                "model_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: trained model .pkl, used only to compute standardized "
                               "residuals (the Williams plot's second axis) when holdout_data_path "
                               "also has the target column. Leverage itself never needs a model.",
                }),
                "target_column": ("STRING", {"default": "value"}),
                "warning_leverage_multiplier": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("EVALUATION_AD_REPORT",)
    FUNCTION = "run"
    CATEGORY = "QSAR/2. REGRESSION"
    OUTPUT_NODE = True

    def run(self, training_data_path, holdout_data_path="", mode="auto", model_path="",
            target_column="value", warning_leverage_multiplier=3.0):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Regression", "09_Applicability_Domain")
            os.makedirs(output_dir, exist_ok=True)

            if mode == "disabled":
                return {"ui": {"text": "⏭️ Applicability Domain assessment disabled (mode=disabled)."}, "result": ("",)}

            has_query = bool(holdout_data_path)

            train_df = pd.read_csv(training_data_path)
            query_df = pd.read_csv(holdout_data_path) if has_query else None

            core = _leverage_ad_core(train_df, query_df, target_column, warning_leverage_multiplier, model_path)
            h_star = core["h_star"]

            rows = []
            for i in range(core["n_train"]):
                lev = float(core["train_leverage"][i])
                r = float(core["train_residuals"][i]) if core["train_residuals"] is not None else None
                rows.append({
                    "dataset_role": "train",
                    "compound_id": train_df["Name"].iloc[i] if "Name" in train_df.columns else None,
                    "SMILES": train_df["SMILES"].iloc[i] if "SMILES" in train_df.columns else None,
                    "leverage": lev,
                    "standardized_residual": r,
                    "h_star_threshold": h_star,
                    "ad_status": _leverage_status(lev, r, h_star),
                })
            if has_query:
                for i in range(len(query_df)):
                    lev = float(core["query_leverage"][i])
                    r = float(core["query_residuals"][i]) if core["query_residuals"] is not None else None
                    rows.append({
                        "dataset_role": "query",
                        "compound_id": query_df["Name"].iloc[i] if "Name" in query_df.columns else None,
                        "SMILES": query_df["SMILES"].iloc[i] if "SMILES" in query_df.columns else None,
                        "leverage": lev,
                        "standardized_residual": r,
                        "h_star_threshold": h_star,
                        "ad_status": _leverage_status(lev, r, h_star),
                    })

            report_df = pd.DataFrame(rows)
            report_path = os.path.join(output_dir, "AD_Assessment_Report.csv")
            report_df.to_csv(report_path, index=False)
            residuals = core["query_residuals"]

            query_mask = report_df["dataset_role"] == "query"
            train_mask = report_df["dataset_role"] == "train"
            n_high_lev = int(report_df.loc[query_mask, "ad_status"].isin(["high_leverage", "high_leverage_outlier"]).sum())
            n_high_lev_train = int(report_df.loc[train_mask, "ad_status"].isin(["high_leverage", "high_leverage_outlier"]).sum())
            residual_note = "" if residuals is not None else " (query residuals skipped: need model_path + target_column present in both training and query data)"
            if has_query:
                query_line = f"📊 Query compounds: {len(query_df)}, high-leverage: {n_high_lev}{residual_note}\n"
                output_note = "(both train and query rows, dataset_role column)"
            else:
                query_line = (
                    "📊 Query compounds: N/A -- no holdout_data_path provided (training-only "
                    "Applicability Domain characterization)\n"
                )
                output_note = "(train rows only, dataset_role column -- no query set provided)"

            log_message = (
                "========================================\n"
                "🔹 9. Descriptor-Space Applicability Domain Complete! 🔹\n"
                "========================================\n"
                "📌 Method: Williams/leverage AD (descriptor-space leverage --\n"
                "   suited to regression QSAR workflows)\n"
                "ℹ️ Leverage is a descriptor-space diagnostic (how unusual a compound's descriptors\n"
                "   are relative to training) -- it does not by itself guarantee prediction\n"
                "   reliability for nonlinear models, whose local behavior within the training\n"
                "   descriptor space isn't fully captured by a linear hat-matrix leverage value.\n"
                f"📌 Training compounds: {core['n_train']}, descriptors: {core['p']}\n"
                f"📌 Descriptor matrix rank: {core['matrix_rank']}/{core['p']}, pseudoinverse used: {'yes' if core['used_pinv'] else 'no'}{core['rank_warning']}{core['cond_warning']}\n"
                f"📌 Warning leverage h*: {h_star:.4f} ({warning_leverage_multiplier}*(p+1)/n){core['h_star_warning']}\n"
                f"📊 Training high-leverage: {n_high_lev_train}/{core['n_train']}\n"
                f"{query_line}"
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
                f"💾 Output: {os.path.basename(report_path)} {output_note}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(report_path),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}


class ScreeningCandidateAD_Regression:
    """Screening-Candidate Applicability Domain: same Williams/leverage AD
    method and training set as ApplicabilityDomain_Regression, applied to
    external screening candidates instead of a hold-out/external test set.
    Leverage-only (no standardized-residual axis) -- screening candidates
    have no observed endpoint, so a residual cannot be computed. Preserves
    every column already present in screening_results_path (prediction,
    library, rank, etc.) and appends leverage/ad_status to it. in_domain
    does not confirm a prediction is correct, only that the candidate's
    descriptors lie within the training descriptor space; out_of_domain
    means the prediction is an extrapolation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "training_data_path": ("STRING", {
                    "tooltip": "Exact training dataset used by Node 7, after preprocessing and any "
                               "optional descriptor selection or combination -- the same file given to "
                               "step 9's evaluation-set AD node, or the domain boundary won't match.",
                }),
                "screening_results_path": ("STRING", {
                    "tooltip": "The Screener's SCREENING_RESULTS output.",
                }),
            },
            "optional": {
                "mode": (["auto", "manual", "disabled"], {
                    "default": "auto",
                    "tooltip": "auto/manual both run the same computation; disabled does nothing.",
                }),
                "target_column": ("STRING", {
                    "default": "value",
                    "tooltip": "Excludes the training endpoint column from the descriptor matrix -- "
                               "no residual is computed, so this need not exist in screening_results_path.",
                }),
                "warning_leverage_multiplier": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0,
                    "tooltip": "Keep identical to the evaluation AD node."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("SCREENING_AD_REPORT",)
    FUNCTION = "run"
    CATEGORY = "QSAR/3. SCREENER"
    OUTPUT_NODE = True

    def run(self, training_data_path, screening_results_path, mode="auto", target_column="value", warning_leverage_multiplier=3.0):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Screening", "Applicability_Domain", "Regression")
            os.makedirs(output_dir, exist_ok=True)

            if mode == "disabled":
                return {"ui": {"text": "⏭️ Screening-Candidate Applicability Domain disabled (mode=disabled)."}, "result": ("",)}

            train_df = pd.read_csv(training_data_path)
            screening_df = pd.read_csv(screening_results_path)

            core = _leverage_ad_core(train_df, screening_df, target_column=target_column, warning_leverage_multiplier=warning_leverage_multiplier, model_path="")
            h_star = core["h_star"]

            ad_rows = []
            for i in range(len(screening_df)):
                lev = float(core["query_leverage"][i])
                ad_rows.append({
                    "leverage": lev,
                    "h_star_threshold": h_star,
                    "ad_status": _leverage_status(lev, None, h_star),
                })
            ad_df = pd.DataFrame(ad_rows)
            report_df = pd.concat([screening_df.reset_index(drop=True), ad_df.reset_index(drop=True)], axis=1)
            report_path = os.path.join(output_dir, "Screening_Candidate_AD_Report.csv")
            report_df.to_csv(report_path, index=False)

            n_high_lev = int((ad_df["ad_status"] == "high_leverage").sum())

            log_message = (
                "========================================\n"
                "🔹 Screening-Candidate Applicability Domain Complete! 🔹\n"
                "========================================\n"
                "📌 Method: Williams/leverage AD, same training set and parameters as the\n"
                "   Evaluation-Set AD node -- only the query changes (screening candidates,\n"
                "   not hold-out/external compounds). Leverage only -- no standardized-residual\n"
                "   axis, since screening candidates have no observed endpoint.\n"
                "ℹ️ in_domain means the candidate's descriptors are within the training descriptor\n"
                "   space, NOT that its prediction is correct. high_leverage means the prediction\n"
                "   is an extrapolation and should be treated with reduced confidence, not dropped.\n"
                f"📌 Training compounds: {core['n_train']}, descriptors: {core['p']}\n"
                f"📌 Descriptor matrix rank: {core['matrix_rank']}/{core['p']}, pseudoinverse used: {'yes' if core['used_pinv'] else 'no'}{core['rank_warning']}{core['cond_warning']}\n"
                f"📌 Warning leverage h*: {h_star:.4f} ({warning_leverage_multiplier}*(p+1)/n){core['h_star_warning']}\n"
                f"📊 Screening candidates: {len(screening_df)}, high-leverage: {n_high_lev}\n"
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
                f"💾 Output: {os.path.basename(report_path)} (all screening_results_path columns preserved, AD columns appended)\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(report_path),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}


NODE_CLASS_MAPPINGS = {
    "ApplicabilityDomain_Regression": ApplicabilityDomain_Regression,
    "ScreeningCandidateAD_Regression": ScreeningCandidateAD_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplicabilityDomain_Regression": "9. Descriptor-Space Applicability Domain",
    "ScreeningCandidateAD_Regression": "Screening-Candidate AD (Regression)",
}
