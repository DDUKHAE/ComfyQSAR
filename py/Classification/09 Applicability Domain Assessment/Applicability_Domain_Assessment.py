import os
import traceback
import numpy as np
import pandas as pd
import joblib
import folder_paths
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def _compute_morgan_fp(smiles, radius=2, n_bits=2048):
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def _tanimoto_knn_ad_core(train_df, query_df, fingerprint_radius, fingerprint_bits, k_neighbors, ad_percentile_threshold):
    """Shared kNN-mean Tanimoto AD core, used by both the Evaluation-Set AD
    node (hold-out/external query, keyed on compound_id/SMILES only) and the
    Screening-Candidate AD node (screening query, whose caller concatenates
    these columns onto the full screening-results row so prediction/library
    metadata survives). Returns an ad_df aligned 1:1 with query_df's row
    order (not its index), plus a stats dict for logging."""
    train_fps = []
    train_ids = []
    invalid_train_rows = []
    train_names = train_df["Name"] if "Name" in train_df.columns else None
    for idx, smi in enumerate(train_df["SMILES"]):
        fp = _compute_morgan_fp(smi, fingerprint_radius, fingerprint_bits)
        name_val = str(train_names.iloc[idx]) if train_names is not None else str(idx)
        if fp is not None:
            train_fps.append(fp)
            train_ids.append(name_val)
        else:
            invalid_train_rows.append({"row_index": idx, "Name": name_val, "SMILES": smi})
    if not train_fps:
        raise ValueError("No valid training-set SMILES could be parsed for fingerprinting.")
    n_train_valid = len(train_fps)
    n_train_invalid = len(invalid_train_rows)
    k_eff = min(k_neighbors, len(train_fps) - 1)
    if k_eff < 1:
        raise ValueError("Need at least 2 valid training compounds for kNN-based AD.")

    # Decision criterion: mean Tanimoto similarity to the k nearest
    # OTHER training compounds -- an empirical kNN-mean Tanimoto AD
    # with a percentile-derived threshold. Averaging over k neighbors
    # rather than a single nearest neighbor is a common choice in the
    # AD literature (e.g. Sheridan et al. 2004; Sahigara et al. 2012,
    # Molecules), but those papers compare several AD approaches
    # rather than establishing k-averaging as the one standard --
    # treat this as one reasonable, literature-informed choice, not
    # "the" standard method. Threshold derivation mirrors the same
    # logic: for each training compound, its own leave-self-out
    # kNN-mean similarity; the chosen low percentile of that
    # distribution becomes the domain boundary.
    self_knn_means = []
    for i in range(len(train_fps)):
        others = train_fps[:i] + train_fps[i + 1:]
        if not others:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(train_fps[i], others)
        self_knn_means.append(float(np.mean(sorted(sims, reverse=True)[:k_eff])))
    threshold = float(np.percentile(self_knn_means, ad_percentile_threshold)) if self_knn_means else 0.0

    ad_rows = []
    for _, row in query_df.iterrows():
        smi = row.get("SMILES")
        fp = _compute_morgan_fp(smi, fingerprint_radius, fingerprint_bits)
        if fp is None:
            ad_rows.append({
                "knn_mean_similarity": None, "nearest_neighbor_similarity": None,
                "nearest_training_compound_id": None,
                "threshold_used": threshold, "ad_status": "unparseable_smiles",
            })
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        nearest_idx = int(np.argmax(sims))
        nearest_sim = float(sims[nearest_idx])
        knn_mean = float(np.mean(sorted(sims, reverse=True)[:k_eff]))
        # AD decision is based on knn_mean (standard kNN-AD, see note
        # above) -- nearest_neighbor_similarity/nearest_training_
        # compound_id are reported for interpretability only (e.g.
        # "which training compound is this screening hit closest
        # to") and never drive ad_status.
        status = "in_domain" if knn_mean >= threshold else "out_of_domain"
        ad_rows.append({
            "knn_mean_similarity": knn_mean,
            "nearest_neighbor_similarity": nearest_sim,
            "nearest_training_compound_id": train_ids[nearest_idx],
            "threshold_used": threshold, "ad_status": status,
        })

    ad_df = pd.DataFrame(ad_rows)
    n_in = int((ad_df["ad_status"] == "in_domain").sum())
    n_out = int((ad_df["ad_status"] == "out_of_domain").sum())
    n_bad = int((ad_df["ad_status"] == "unparseable_smiles").sum())
    stats = {
        "threshold": threshold, "k_eff": k_eff, "k_requested": k_neighbors,
        "n_train_valid": n_train_valid, "n_train_invalid": n_train_invalid,
        "invalid_train_rows": invalid_train_rows,
        "n_in": n_in, "n_out": n_out, "n_bad": n_bad,
    }
    return ad_df, stats


class ApplicabilityDomain_Classification:
    """Evaluation-Set Applicability Domain: interprets the applicability
    scope of hold-out/external performance (Model Validation). For the
    applicability domain of external screening candidates, use
    ScreeningCandidateAD_Classification instead -- same underlying AD
    method and training set, but keyed to preserve screening
    prediction/library metadata rather than a bare compound_id/SMILES
    report."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "training_data_path": ("STRING", {
                    "tooltip": "Exact training dataset used by Node 7, after preprocessing and any "
                               "optional descriptor selection or combination.",
                }),
                "holdout_data_path": ("STRING", {
                    "tooltip": "From 4's PREPROCESSED_HOLDOUT -- not 3's raw HOLDOUT_DATA (no "
                               "train-fitted imputation yet). Only the descriptor columns present in "
                               "training_data_path are used; any extra columns are ignored, so the "
                               "wider 4-stage file is fine as-is. For screening candidates, use the "
                               "separate 'Screening-Candidate AD (Classification)' node instead.",
                }),
            },
            "optional": {
                "mode": (["auto", "manual", "disabled"], {
                    "default": "auto",
                    "tooltip": "auto: run automatically if both files have a SMILES column, skip "
                               "silently otherwise. manual: same computation, but error instead of "
                               "skipping if SMILES is missing. disabled: do nothing. (Ignored for "
                               "ad_method='rf_proximity', which needs descriptor columns, not SMILES.)",
                }),
                "ad_method": (["tanimoto_knn", "rf_proximity"], {
                    "default": "tanimoto_knn",
                    "tooltip": "tanimoto_knn (default): structure-similarity AD from Morgan "
                               "fingerprints, unchanged from before. rf_proximity: tree-ensemble "
                               "AD using a trained model's own leaf co-occurrence across trees "
                               "(Breiman's RF proximity) -- only valid for a tree-ensemble "
                               "classifier (e.g. RandomForestClassifier) trained by this platform's "
                               "own '7. Hyperparameter Tuning & Model Training'; it does not "
                               "reconstruct a different RF implementation's (e.g. R randomForest/"
                               "PMML) own proximity matrix.",
                }),
                "fingerprint_radius": ("INT", {"default": 2, "min": 1, "max": 4}),
                "fingerprint_bits": ("INT", {"default": 2048, "min": 256, "max": 4096}),
                "k_neighbors": ("INT", {"default": 5, "min": 1, "max": 50}),
                "ad_percentile_threshold": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 50.0,
                    "tooltip": "Percentile of the training set's own leave-self-out kNN-mean "
                               "similarity distribution used as the in/out-of-domain cutoff.",
                }),
                "model_path": ("STRING", {
                    "default": "",
                    "tooltip": "Required only for ad_method='rf_proximity': a tree-ensemble "
                               "classifier .pkl (e.g. from '7. Hyperparameter Tuning & Model "
                               "Training', algorithm=random_forest) exposing scikit-learn's "
                               ".apply(X). Ignored for ad_method='tanimoto_knn'.",
                }),
                "target_column": ("STRING", {"default": "Label"}),
                "proximity_cutoff": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0,
                    "tooltip": "rf_proximity only: minimum fraction of trees in which a training "
                               "compound must share a query compound's terminal leaf to count as "
                               "'similar' (Breiman's RF proximity, paper's own default 0.7).",
                }),
                "min_similar_training": ("INT", {
                    "default": 4, "min": 1, "max": 1000,
                    "tooltip": "rf_proximity only: minimum number of training compounds at or "
                               "above proximity_cutoff for a query compound to be in_domain.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("EVALUATION_AD_REPORT",)
    FUNCTION = "run"
    CATEGORY = "QSAR/1. CLASSIFICATION"
    OUTPUT_NODE = True

    def run(self, training_data_path, holdout_data_path, mode="auto", ad_method="tanimoto_knn",
            fingerprint_radius=2, fingerprint_bits=2048, k_neighbors=5, ad_percentile_threshold=5.0,
            model_path="", target_column="Label", proximity_cutoff=0.7, min_similar_training=4):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Classification", "09_Applicability_Domain")
            os.makedirs(output_dir, exist_ok=True)

            if mode == "disabled":
                return {"ui": {"text": "⏭️ Applicability Domain assessment disabled (mode=disabled)."}, "result": ("",)}

            if ad_method == "rf_proximity":
                return self._run_rf_proximity(
                    training_data_path, holdout_data_path, model_path, target_column,
                    proximity_cutoff, min_similar_training, output_dir,
                )

            train_df = pd.read_csv(training_data_path)
            query_df = pd.read_csv(holdout_data_path)

            if "SMILES" not in train_df.columns or "SMILES" not in query_df.columns:
                if mode == "auto":
                    return {
                        "ui": {"text": "⏭️ Applicability Domain skipped (mode=auto): no SMILES column "
                                       "in training_data_path or holdout_data_path."},
                        "result": ("",),
                    }
                raise ValueError("training_data_path and holdout_data_path must both contain a 'SMILES' column.")

            ad_df, stats = _tanimoto_knn_ad_core(
                train_df, query_df, fingerprint_radius, fingerprint_bits, k_neighbors, ad_percentile_threshold
            )

            query_names = query_df["Name"] if "Name" in query_df.columns else None
            id_cols = pd.DataFrame({
                "compound_id": query_names.astype(str).values if query_names is not None else [None] * len(query_df),
                "SMILES": query_df["SMILES"].values,
            })
            report_df = pd.concat([id_cols.reset_index(drop=True), ad_df.reset_index(drop=True)], axis=1)
            report_path = os.path.join(output_dir, "AD_Assessment_Report.csv")
            report_df.to_csv(report_path, index=False)

            invalid_smiles_report_path = os.path.join(output_dir, "Invalid_Training_SMILES_Report.csv")
            if stats["n_train_invalid"] > 0:
                pd.DataFrame(stats["invalid_train_rows"]).to_csv(invalid_smiles_report_path, index=False)
            elif os.path.exists(invalid_smiles_report_path):
                # Stale from a prior run where some training SMILES failed to
                # parse -- this run has none, so remove it rather than leave
                # a leftover file implying there's still a problem.
                os.remove(invalid_smiles_report_path)

            log_message = (
                "========================================\n"
                "🔹 9. Structure-Similarity Applicability Domain Complete! 🔹\n"
                "========================================\n"
                "📌 Method: kNN-mean Tanimoto AD with empirical percentile threshold\n"
                "   (structure-similarity domain -- suited to classification /\n"
                "   external-screening workflows)\n"
                "ℹ️ ad_status is decided by mean similarity to the k nearest training compounds.\n"
                "   nearest_neighbor_similarity/nearest_training_compound_id are reported\n"
                "   separately for interpretability only and do not affect the decision.\n"
                f"📌 Requested k: {stats['k_requested']}, Effective k: {stats['k_eff']}"
                + (" (reduced -- fewer valid training compounds than requested k+1)\n" if stats['k_eff'] != stats['k_requested'] else "\n")
                + f"📌 Training SMILES: {stats['n_train_valid']} valid, {stats['n_train_invalid']} invalid (unparseable)\n"
                + (f"💾 Invalid Training SMILES Report: {os.path.basename(invalid_smiles_report_path)}\n" if stats['n_train_invalid'] > 0 else "")
                + f"📌 Threshold ({ad_percentile_threshold:.0f}th pct of training self kNN-mean similarity): {stats['threshold']:.4f}\n"
                f"📊 Query compounds: in_domain={stats['n_in']}, out_of_domain={stats['n_out']}, unparseable={stats['n_bad']}\n"
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
                f"💾 Output: {os.path.basename(report_path)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(report_path),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}

    def _run_rf_proximity(self, training_data_path, holdout_data_path, model_path, target_column,
                           proximity_cutoff, min_similar_training, output_dir):
        try:
            if not model_path:
                raise ValueError("model_path is required when ad_method='rf_proximity'.")
            model = joblib.load(model_path)
            if not hasattr(model, "apply"):
                raise ValueError(
                    f"The model loaded from model_path does not support .apply() (terminal-leaf "
                    f"lookup) -- rf_proximity only works with a tree-ensemble classifier (e.g. "
                    f"scikit-learn RandomForestClassifier), got {type(model).__name__}."
                )

            train_df = pd.read_csv(training_data_path)
            query_df = pd.read_csv(holdout_data_path)
            metadata_cols_train = [c for c in ("Name", "SMILES", target_column) if c in train_df.columns]
            metadata_cols_query = [c for c in ("Name", "SMILES", target_column) if c in query_df.columns]
            X_train = train_df.drop(columns=metadata_cols_train).select_dtypes(include=[np.number])
            X_query = query_df[X_train.columns] if all(c in query_df.columns for c in X_train.columns) else None
            if X_query is None:
                missing = [c for c in X_train.columns if c not in query_df.columns]
                raise ValueError(
                    f"holdout_data_path is missing {len(missing)} descriptor column(s) present in "
                    f"training_data_path (e.g. {missing[:5]}) -- both files must share the exact "
                    "descriptor set the model was trained on."
                )
            n_features_expected = getattr(model, "n_features_in_", None)
            if n_features_expected is not None and X_train.shape[1] != n_features_expected:
                raise ValueError(
                    f"training_data_path has {X_train.shape[1]} descriptor column(s) but the model "
                    f"was trained on {n_features_expected} -- pass the same feature-selected CSV "
                    "used to train this model (e.g. '7. Hyperparameter Tuning & Model Training''s "
                    "own input_file), not a different descriptor set."
                )

            train_ids = (train_df["Name"] if "Name" in train_df.columns else pd.Series(range(len(train_df)))).astype(str).tolist()

            # Breiman's RF proximity: fraction of trees in which two compounds
            # land in the same terminal leaf. .apply(X) returns, for every
            # compound, which leaf index it landed in per tree -- comparing
            # leaf indices tree-by-tree (not the leaf index *value* across
            # different compounds' absolute value, which is meaningless
            # across trees except as an equality test) and averaging over
            # trees gives the proximity fraction directly, without ever
            # materializing an (n_train, n_query, n_trees) array.
            leaves_train = model.apply(X_train.to_numpy())
            leaves_query = model.apply(X_query.to_numpy())
            n_trees = leaves_train.shape[1]
            proximity = np.zeros((len(X_query), len(X_train)), dtype=np.float64)
            for t in range(n_trees):
                proximity += (leaves_query[:, t][:, None] == leaves_train[:, t][None, :])
            proximity /= n_trees

            rows = []
            for i in range(len(X_query)):
                compound_id = query_df["Name"].iloc[i] if "Name" in query_df.columns else str(i)
                smiles = query_df["SMILES"].iloc[i] if "SMILES" in query_df.columns else None
                prox_row = proximity[i]
                above_cutoff_idx = np.where(prox_row >= proximity_cutoff)[0]
                n_above = int(len(above_cutoff_idx))
                # Nearest IDs reported by descending proximity (paper's own
                # worked examples list neighbors ranked by proximity, not by
                # Compound Id order).
                ranked = above_cutoff_idx[np.argsort(-prox_row[above_cutoff_idx])]
                nearest_ids = ";".join(train_ids[j] for j in ranked)
                rows.append({
                    "compound_id": compound_id, "SMILES": smiles,
                    "rf_proximity_max": float(prox_row.max()) if len(prox_row) else 0.0,
                    "n_training_above_cutoff": n_above,
                    "nearest_training_compound_ids": nearest_ids,
                    "proximity_cutoff": proximity_cutoff,
                    "min_similar_training": min_similar_training,
                    "ad_status": "in_domain" if n_above >= min_similar_training else "out_of_domain",
                })

            report_df = pd.DataFrame(rows)
            report_path = os.path.join(output_dir, "AD_Assessment_Report.csv")
            report_df.to_csv(report_path, index=False)

            n_in = int((report_df["ad_status"] == "in_domain").sum())
            n_out = int((report_df["ad_status"] == "out_of_domain").sum())
            log_message = (
                "========================================\n"
                "🔹 9. RF-Proximity Applicability Domain Complete! 🔹\n"
                "========================================\n"
                "📌 Method: Random Forest proximity (Breiman) -- fraction of trees in which a "
                "query and training compound share a terminal leaf\n"
                "ℹ️ Uses this platform's own trained model (model_path) -- does not reconstruct "
                "any other RF implementation's (e.g. R randomForest/PMML) proximity matrix.\n"
                f"📌 Training compounds: {len(X_train)}, trees: {n_trees}, descriptors: {X_train.shape[1]}\n"
                f"📌 proximity_cutoff: {proximity_cutoff}, min_similar_training: {min_similar_training}\n"
                f"📊 Query compounds: {len(X_query)}, in_domain={n_in}, out_of_domain={n_out}\n"
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
                f"💾 Output: {os.path.basename(report_path)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(report_path),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}


class ScreeningCandidateAD_Classification:
    """Screening-Candidate Applicability Domain: same kNN-mean Tanimoto AD
    method and training set as ApplicabilityDomain_Classification, applied
    to external screening candidates instead of a hold-out/external test
    set. Unlike the evaluation-set report (compound_id/SMILES only), this
    node preserves every column already present in screening_results_path
    (prediction_value, library, rank, etc.) and appends AD columns to it --
    in_domain does not confirm a prediction is correct, only that the
    candidate lies within the training chemical space; out_of_domain means
    the prediction is an extrapolation."""

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
                    "tooltip": "auto: run automatically if both files have a SMILES column, skip "
                               "silently otherwise. manual: same computation, but error instead of "
                               "skipping if SMILES is missing. disabled: do nothing.",
                }),
                "fingerprint_radius": ("INT", {"default": 2, "min": 1, "max": 4,
                    "tooltip": "Keep identical to the evaluation AD node."}),
                "fingerprint_bits": ("INT", {"default": 2048, "min": 256, "max": 4096,
                    "tooltip": "Keep identical to the evaluation AD node."}),
                "k_neighbors": ("INT", {"default": 5, "min": 1, "max": 50,
                    "tooltip": "Keep identical to the evaluation AD node."}),
                "ad_percentile_threshold": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 50.0,
                    "tooltip": "Keep identical to the evaluation AD node.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("SCREENING_AD_REPORT",)
    FUNCTION = "run"
    CATEGORY = "QSAR/3. SCREENER"
    OUTPUT_NODE = True

    def run(self, training_data_path, screening_results_path, mode="auto",
            fingerprint_radius=2, fingerprint_bits=2048, k_neighbors=5, ad_percentile_threshold=5.0):
        try:
            output_dir = os.path.join(folder_paths.get_output_directory(), "Screening", "Applicability_Domain", "Classification")
            os.makedirs(output_dir, exist_ok=True)

            if mode == "disabled":
                return {"ui": {"text": "⏭️ Screening-Candidate Applicability Domain disabled (mode=disabled)."}, "result": ("",)}

            train_df = pd.read_csv(training_data_path)
            screening_df = pd.read_csv(screening_results_path)

            if "SMILES" not in train_df.columns or "SMILES" not in screening_df.columns:
                if mode == "auto":
                    return {
                        "ui": {"text": "⏭️ Screening-Candidate Applicability Domain skipped (mode=auto): "
                                       "no SMILES column in training_data_path or screening_results_path."},
                        "result": ("",),
                    }
                raise ValueError("training_data_path and screening_results_path must both contain a 'SMILES' column.")

            ad_df, stats = _tanimoto_knn_ad_core(
                train_df, screening_df, fingerprint_radius, fingerprint_bits, k_neighbors, ad_percentile_threshold
            )

            report_df = pd.concat([screening_df.reset_index(drop=True), ad_df.reset_index(drop=True)], axis=1)
            report_path = os.path.join(output_dir, "Screening_Candidate_AD_Report.csv")
            report_df.to_csv(report_path, index=False)

            log_message = (
                "========================================\n"
                "🔹 Screening-Candidate Applicability Domain Complete! 🔹\n"
                "========================================\n"
                "📌 Method: kNN-mean Tanimoto AD, same training set and parameters as the\n"
                "   Evaluation-Set AD node -- only the query changes (screening candidates,\n"
                "   not hold-out/external compounds).\n"
                "ℹ️ in_domain means the candidate is within the training chemical space, NOT\n"
                "   that its prediction is correct. out_of_domain means the prediction is an\n"
                "   extrapolation and should be treated with reduced confidence, not dropped.\n"
                f"📌 Requested k: {stats['k_requested']}, Effective k: {stats['k_eff']}"
                + (" (reduced -- fewer valid training compounds than requested k+1)\n" if stats['k_eff'] != stats['k_requested'] else "\n")
                + f"📌 Training SMILES: {stats['n_train_valid']} valid, {stats['n_train_invalid']} invalid (unparseable)\n"
                + f"📌 Threshold ({ad_percentile_threshold:.0f}th pct of training self kNN-mean similarity): {stats['threshold']:.4f}\n"
                f"📊 Screening candidates: in_domain={stats['n_in']}, out_of_domain={stats['n_out']}, unparseable={stats['n_bad']}\n"
                f"📁 Directory: {os.path.relpath(output_dir, folder_paths.get_output_directory())}{os.sep}\n"
                f"💾 Output: {os.path.basename(report_path)} (all screening_results_path columns preserved, AD columns appended)\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(report_path),)}
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}


NODE_CLASS_MAPPINGS = {
    "ApplicabilityDomain_Classification": ApplicabilityDomain_Classification,
    "ScreeningCandidateAD_Classification": ScreeningCandidateAD_Classification,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplicabilityDomain_Classification": "9. Structure-Similarity Applicability Domain",
    "ScreeningCandidateAD_Classification": "Screening-Candidate AD (Classification)",
}
