from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from antinetwork.killtest import make_regression_pipeline

OFFICIAL_ASSAYS = ["HIC", "Tm2", "Titer", "PR_CHO", "AC-SINS_pH7.4"]
OFFICIAL_FOLD_COL = "hierarchical_cluster_IgG_isotype_stratified_fold"
OFFICIAL_HIGHER_IS_BETTER = {
    "HIC": False,
    "Tm2": True,
    "Titer": True,
    "PR_CHO": False,
    "AC-SINS_pH7.4": False,
    "SEC_%monomer": True,
}

OFFICIAL_PROPERTY_TARGETS = {
    "Hydrophobicity": {
        "assay": "HIC",
        "competition_top_spearman": 0.708,
        "first_bar_spearman": 0.65,
    },
    "Thermostability": {
        "assay": "Tm2",
        "competition_top_spearman": 0.392,
        "first_bar_spearman": 0.39,
    },
    "Polyreactivity": {
        "assay": "PR_CHO",
        "competition_top_spearman": 0.356,
        "first_bar_spearman": 0.36,
    },
    "Self-association": {
        "assay": "AC-SINS_pH7.4",
        "competition_top_spearman": 0.337,
        "first_bar_spearman": 0.34,
    },
    "Titer": {
        "assay": "Titer",
        "competition_top_spearman": 0.310,
        "first_bar_spearman": 0.31,
    },
}

GINKGO_EXTERNAL_TARGETS = {
    "Hydrophobicity": {
        "assay": "HIC",
        "ginkgo_train_column": "hic_rt_avg",
        "ginkgo_test_column": "hic_rt_avg",
        "competition_top_spearman": 0.708,
    },
    "Thermostability": {
        "assay": "Tm2",
        "ginkgo_train_column": "tm2_nanodsf_avg",
        "ginkgo_test_column": "tm2_nanodsf_avg",
        "competition_top_spearman": 0.392,
    },
    "Polyreactivity": {
        "assay": "PR_CHO",
        "ginkgo_train_column": "polyreactivity_prscore_cho_avg",
        "ginkgo_test_column": "polyreactivity_prscore_cho_avg",
        "competition_top_spearman": 0.356,
    },
    "Self-association": {
        "assay": "AC-SINS_pH7.4",
        "ginkgo_train_column": "acsins_dLmax_ph7.4_avg",
        "ginkgo_test_column": "acsins_dLmax_ph7.4_avg",
        "competition_top_spearman": 0.337,
    },
    "Titer": {
        "assay": "Titer",
        "ginkgo_train_column": "titer_productionbatch1_avg",
        "ginkgo_test_column": "titer_avg",
        "competition_top_spearman": 0.310,
    },
    "Aggregation": {
        "assay": "SEC_%monomer",
        "ginkgo_train_column": "sec_%monomer_avg",
        "ginkgo_test_column": "sec_%monomer_avg",
        "competition_top_spearman": np.nan,
    },
}

ID_COLUMNS = {
    "antibody_id",
    "antibody_name",
    "mseq",
    "vh_protein_sequence",
    "vl_protein_sequence",
    "hc_protein_sequence",
    "lc_protein_sequence",
    "heavy_aligned_aho",
    "light_aligned_aho",
}


def normalize_ginkgo_sequences(sequences: pd.DataFrame) -> pd.DataFrame:
    """Return Ginkgo workbook sequences with the columns used by local features."""
    out = sequences.copy()
    if "antibody_name" not in out.columns:
        out["antibody_name"] = out["antibody_id"]
    if "vh_protein_sequence" not in out.columns and "protein_sequence" in out.columns:
        out["vh_protein_sequence"] = out["protein_sequence"]
    if (
        "vl_protein_sequence" not in out.columns
        and "lc_protein_sequence" not in out.columns
        and "protein_sequence" in out.columns
    ):
        out["vl_protein_sequence"] = ""
    if "vl_protein_sequence" not in out.columns and "lc_protein_sequence" in out.columns:
        out["vl_protein_sequence"] = out["lc_protein_sequence"]
    if "heavy_aligned_aho" not in out.columns:
        out["heavy_aligned_aho"] = out["vh_protein_sequence"]
    if "light_aligned_aho" not in out.columns:
        out["light_aligned_aho"] = out["vl_protein_sequence"]
    return out


def load_ginkgo_workbook(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load the relevant sheets from a GDPa Excel workbook."""
    path = Path(path)
    return {
        "sequences": normalize_ginkgo_sequences(pd.read_excel(path, sheet_name="Sequences")),
        "assay_average": pd.read_excel(path, sheet_name="Assay Data - average"),
        "assay_tidy": pd.read_excel(path, sheet_name="Assay Data - tidy format"),
    }


def ginkgo_column_mapping_table(
    train_average: pd.DataFrame,
    test_average: pd.DataFrame,
    targets: dict[str, dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Describe which workbook columns can be externally benchmarked."""
    targets = targets or GINKGO_EXTERNAL_TARGETS
    rows = []
    for property_name, spec in targets.items():
        train_col = str(spec["ginkgo_train_column"])
        test_col = str(spec["ginkgo_test_column"])
        rows.append(
            {
                "property": property_name,
                "assay": spec["assay"],
                "ginkgo_train_column": train_col,
                "ginkgo_test_column": test_col,
                "train_present": train_col in train_average.columns,
                "test_present": test_col in test_average.columns,
                "published_top_spearman": spec["competition_top_spearman"],
            }
        )
    return pd.DataFrame(rows)


def make_ginkgo_external_dataset(
    workbook: dict[str, pd.DataFrame],
    *,
    targets: dict[str, dict[str, object]] | None = None,
    split: str = "train",
) -> pd.DataFrame:
    """Merge workbook sequences and average assay labels into local benchmark columns."""
    targets = targets or GINKGO_EXTERNAL_TARGETS
    sequences = workbook["sequences"].copy()
    average = workbook["assay_average"].copy()
    out = sequences.merge(average, on="antibody_id", how="left", suffixes=("", "_assay"))
    if "antibody_name" not in out.columns:
        out["antibody_name"] = out["antibody_id"]

    key = "ginkgo_train_column" if split == "train" else "ginkgo_test_column"
    for spec in targets.values():
        column = str(spec[key])
        assay = str(spec["assay"])
        out[assay] = out[column] if column in out.columns else np.nan
    return out


def evaluate_external_predictions(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    assays: list[str] | None = None,
    model: str = "model",
    dataset: str = "GDPa3_external",
) -> pd.DataFrame:
    """Evaluate frozen GDPa1-trained predictions on an external labelled set."""
    assays = assays or [assay for assay in OFFICIAL_ASSAYS if assay in predictions.columns]
    merged = truth[["antibody_name", *assays]].merge(
        predictions[["antibody_name", *assays]],
        on="antibody_name",
        how="left",
        suffixes=("_true", "_pred"),
    )
    rows = []
    for assay in assays:
        y_true = merged[f"{assay}_true"].to_numpy(dtype=float)
        y_pred = merged[f"{assay}_pred"].to_numpy(dtype=float)
        rows.append(
            official_metric_row(
                y_true,
                y_pred,
                assay,
                "external",
                model,
                dataset=dataset,
            )
        )
    return pd.DataFrame(rows)


def map_sequence_named_predictions_to_external_ids(
    predictions: pd.DataFrame,
    external_sequences: pd.DataFrame,
    *,
    external_id_col: str = "antibody_id",
) -> pd.DataFrame:
    """Map old heldout prediction names to external workbook IDs by exact VH/VL sequence."""
    external = normalize_ginkgo_sequences(external_sequences)
    required = {"vh_protein_sequence", "vl_protein_sequence"}
    missing_predictions = required - set(predictions.columns)
    missing_external = required - set(external.columns)
    if missing_predictions:
        raise ValueError(f"Prediction table is missing sequence columns: {sorted(missing_predictions)}")
    if missing_external:
        raise ValueError(f"External sequence table is missing sequence columns: {sorted(missing_external)}")

    pred = predictions.copy()
    pred["_sequence_key"] = (
        pred["vh_protein_sequence"].astype(str) + "|" + pred["vl_protein_sequence"].astype(str)
    )
    external = external.copy()
    external["_sequence_key"] = (
        external["vh_protein_sequence"].astype(str) + "|" + external["vl_protein_sequence"].astype(str)
    )
    key_to_id = external[["_sequence_key", external_id_col]].drop_duplicates("_sequence_key")
    mapped = pred.merge(key_to_id, on="_sequence_key", how="left", validate="many_to_one")
    if mapped[external_id_col].isna().any():
        missing = int(mapped[external_id_col].isna().sum())
        raise ValueError(f"{missing} prediction rows could not be mapped to external sequence IDs")
    mapped["legacy_antibody_name"] = mapped["antibody_name"]
    mapped["antibody_name"] = mapped[external_id_col]
    return mapped.drop(columns=["_sequence_key", external_id_col])


def compare_external_to_competition_targets(external_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare GDPa3 external metrics to published private-test reference scores."""
    targets = pd.DataFrame(
        [
            {"property": property_name, **spec}
            for property_name, spec in GINKGO_EXTERNAL_TARGETS.items()
        ]
    ).rename(columns={"competition_top_spearman": "published_top_spearman"})
    comparison = targets.merge(
        external_metrics[["assay", "model", "spearman", "n"]],
        on="assay",
        how="left",
    )
    comparison["delta_vs_published_top"] = comparison["spearman"] - comparison["published_top_spearman"]
    comparison["status"] = np.select(
        [
            comparison["published_top_spearman"].isna(),
            comparison["spearman"] >= comparison["published_top_spearman"],
            comparison["spearman"] >= 0.8 * comparison["published_top_spearman"],
        ],
        ["exploratory_no_prize_target", "meets_or_beats_top", "approaches_top"],
        default="below_top",
    )
    return comparison[
        [
            "property",
            "assay",
            "n",
            "spearman",
            "published_top_spearman",
            "delta_vs_published_top",
            "status",
            "model",
        ]
    ]


def recall_at_top_fraction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    assay: str,
    fraction: float = 0.10,
) -> float:
    """Mirror the official AbDev top-fraction recall metric."""
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = np.asarray(y_true, dtype=float)[valid]
    y_pred = np.asarray(y_pred, dtype=float)[valid]
    if len(y_true) == 0:
        return np.nan

    if not OFFICIAL_HIGHER_IS_BETTER[assay]:
        y_true = -y_true
        y_pred = -y_pred

    k = int(len(y_true) * fraction)
    if k == 0:
        return np.nan

    true_top = set(np.argsort(y_true)[-k:].tolist())
    pred_top = set(np.argsort(y_pred)[-k:].tolist())
    return len(true_top.intersection(pred_top)) / k


def official_metric_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    assay: str,
    fold: str,
    model: str,
    split: str = "test",
    dataset: str = "GDPa1_cross_validation",
) -> dict[str, object]:
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    row: dict[str, object] = {
        "spearman": np.nan,
        "top_10_recall": np.nan,
        "fold": fold,
        "dataset": dataset,
        "assay": assay,
        "model": model,
        "split": split,
        "n": int(valid.sum()),
    }
    if valid.sum() >= 3:
        row["spearman"] = spearmanr(y_pred[valid], y_true[valid], nan_policy="omit").correlation
        row["top_10_recall"] = recall_at_top_fraction(y_true[valid], y_pred[valid], assay)
    return row


def evaluate_official_cv_predictions(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    assays: list[str] | None = None,
    model: str = "model",
    fold_col: str = OFFICIAL_FOLD_COL,
) -> pd.DataFrame:
    """Evaluate one out-of-fold prediction table using official fold semantics."""
    assays = assays or [assay for assay in OFFICIAL_ASSAYS if assay in predictions.columns]
    merged = truth[["antibody_name", fold_col, *assays]].merge(
        predictions[["antibody_name", *assays]],
        on="antibody_name",
        how="left",
        suffixes=("_true", "_pred"),
    )

    rows = []
    for assay in assays:
        per_fold = []
        for fold in sorted(merged[fold_col].dropna().unique()):
            fold_mask = merged[fold_col] == fold
            y_true = merged.loc[fold_mask, f"{assay}_true"].to_numpy(dtype=float)
            y_pred = merged.loc[fold_mask, f"{assay}_pred"].to_numpy(dtype=float)
            row = official_metric_row(y_true, y_pred, assay, str(fold), model)
            rows.append(row)
            per_fold.append(row)

        y_true = merged[f"{assay}_true"].to_numpy(dtype=float)
        y_pred = merged[f"{assay}_pred"].to_numpy(dtype=float)
        rows.append(official_metric_row(y_true, y_pred, assay, "aggregated", model))
        spearman_values = [row["spearman"] for row in per_fold]
        recall_values = [row["top_10_recall"] for row in per_fold]
        rows.append(
            {
                "spearman": np.nanmean(spearman_values),
                "top_10_recall": np.nan if np.isnan(recall_values).all() else np.nanmean(recall_values),
                "fold": "average",
                "dataset": "GDPa1_cross_validation",
                "assay": assay,
                "model": model,
                "split": "test",
                "n": int(sum(row["n"] for row in per_fold)),
            }
        )
    return pd.DataFrame(rows)


def load_official_cv_results(evaluation_dir: str | Path) -> pd.DataFrame:
    """Load official AbDev benchmark CV metric files."""
    evaluation_dir = Path(evaluation_dir)
    frames = []
    for path in sorted(evaluation_dir.glob("*_cv.csv")):
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No *_cv.csv files found in {evaluation_dir}")
    return pd.concat(frames, ignore_index=True)


def summarize_average_fold(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return the official average-fold rows as a compact model/assay table."""
    out = metrics[(metrics["fold"].astype(str) == "average") & (metrics["split"] == "test")].copy()
    return out.sort_values(["assay", "spearman"], ascending=[True, False]).reset_index(drop=True)


def official_property_target_table() -> pd.DataFrame:
    """Return published AbDev competition top Spearman targets."""
    rows = []
    for property_name, values in OFFICIAL_PROPERTY_TARGETS.items():
        rows.append({"property": property_name, **values})
    return pd.DataFrame(rows)


def compare_to_competition_targets(local_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare local official-fold model metrics with published competition targets."""
    targets = official_property_target_table()
    average = local_metrics[local_metrics["fold"].astype(str) == "average"].copy()
    best_local = (
        average.sort_values(["assay", "spearman"], ascending=[True, False])
        .groupby("assay", as_index=False)
        .first()
        .rename(columns={"model": "best_local_model", "spearman": "best_local_spearman"})
    )
    comparison = targets.merge(best_local, on="assay", how="left")
    comparison["delta_vs_competition_top"] = (
        comparison["best_local_spearman"] - comparison["competition_top_spearman"]
    )
    comparison["delta_vs_first_bar"] = (
        comparison["best_local_spearman"] - comparison["first_bar_spearman"]
    )
    comparison["status"] = np.select(
        [
            comparison["best_local_spearman"] >= comparison["competition_top_spearman"],
            comparison["best_local_spearman"] >= comparison["first_bar_spearman"],
        ],
        ["meets_or_beats_top", "near_competitive"],
        default="below_bar",
    )
    return comparison


def load_official_prediction_tables(prediction_root: str | Path) -> dict[str, pd.DataFrame]:
    """Load official out-of-fold prediction tables keyed by model directory name."""
    prediction_root = Path(prediction_root)
    tables = {}
    for path in sorted(prediction_root.glob("*/predictions.csv")):
        tables[path.parent.name] = pd.read_csv(path)
    if not tables:
        raise FileNotFoundError(f"No prediction tables found in {prediction_root}")
    return tables


def load_processed_feature_tables(feature_dir: str | Path) -> pd.DataFrame:
    """Concatenate available official processed tabular features for GDPa1."""
    feature_dir = Path(feature_dir)
    frames = []
    index = None
    for path in sorted(feature_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        if "antibody_name" not in frame:
            continue
        frame = frame.set_index("antibody_name")
        if index is None:
            index = frame.index
        frame = frame.reindex(index)
        numeric_cols = [
            col for col in frame.select_dtypes(include="number").columns if col not in ID_COLUMNS
        ]
        if numeric_cols:
            prefix = path.stem.lower().replace(" ", "_")
            frames.append(frame[numeric_cols].add_prefix(f"{prefix}__"))
    if not frames:
        raise FileNotFoundError(f"No numeric processed features found in {feature_dir}")
    return pd.concat(frames, axis=1)


def official_fold_model_predictions(
    df: pd.DataFrame,
    features: pd.DataFrame,
    assays: list[str] | None = None,
    model_label: str = "ridge",
    fold_col: str = OFFICIAL_FOLD_COL,
    random_state: int = 7,
) -> pd.DataFrame:
    """Train ridge models on official folds and return out-of-fold predictions."""
    assays = assays or OFFICIAL_ASSAYS
    predictions = pd.DataFrame(
        {
            "antibody_name": df["antibody_name"].to_numpy(),
            fold_col: df[fold_col].to_numpy(),
        },
        index=df.index,
    )
    for assay in assays:
        predictions[assay] = np.nan
        valid = df[assay].notna() & df[fold_col].notna()
        for fold in sorted(df.loc[valid, fold_col].unique()):
            train = valid & (df[fold_col] != fold)
            test = valid & (df[fold_col] == fold)
            estimator = make_regression_pipeline(
                features.loc[train],
                model_name="ridge",
                random_state=random_state,
            )
            estimator.fit(features.loc[train], df.loc[train, assay].to_numpy(dtype=float))
            predictions.loc[test, assay] = estimator.predict(features.loc[test])
    predictions.attrs["model_label"] = model_label
    return predictions


def train_full_model_predictions(
    train_df: pd.DataFrame,
    train_features: pd.DataFrame,
    predict_df: pd.DataFrame,
    predict_features: pd.DataFrame,
    assays: list[str] | None = None,
    model_label: str = "ridge",
    random_state: int = 7,
) -> pd.DataFrame:
    """Train one model per assay on all available labels and predict new sequences."""
    assays = assays or OFFICIAL_ASSAYS
    predictions = pd.DataFrame({"antibody_name": predict_df["antibody_name"].to_numpy()})
    for assay in assays:
        predictions[assay] = np.nan
        if assay not in train_df.columns:
            continue
        valid = train_df[assay].notna()
        if valid.sum() < 10:
            continue
        estimator = make_regression_pipeline(
            train_features.loc[valid],
            model_name="ridge",
            random_state=random_state,
        )
        estimator.fit(train_features.loc[valid], train_df.loc[valid, assay].to_numpy(dtype=float))
        predictions[assay] = estimator.predict(predict_features)
    predictions.attrs["model_label"] = model_label
    return predictions


def residual_charge_after_official_predictions(
    df: pd.DataFrame,
    charge_features: pd.DataFrame,
    official_predictions: dict[str, pd.DataFrame],
    assays: list[str] | None = None,
    fold_col: str = OFFICIAL_FOLD_COL,
    random_state: int = 7,
) -> pd.DataFrame:
    """Test whether charge predicts residual variation left by official OOF predictions."""
    assays = assays or OFFICIAL_ASSAYS
    rows = []
    for official_model, preds in official_predictions.items():
        available_assays = [assay for assay in assays if assay in preds.columns]
        if not available_assays:
            continue
        merged = df[["antibody_name", fold_col, *available_assays]].merge(
            preds[["antibody_name", *available_assays]],
            on="antibody_name",
            how="left",
            suffixes=("_true", "_official"),
        )
        merged.index = df.index

        for assay in available_assays:
            official_oof = merged[f"{assay}_official"].to_numpy(dtype=float)
            final_pred = np.full(len(merged), np.nan)
            residual_true = np.full(len(merged), np.nan)
            residual_pred = np.full(len(merged), np.nan)
            calibrated_official = np.full(len(merged), np.nan)
            valid = merged[f"{assay}_true"].notna() & merged[f"{assay}_official"].notna()

            for fold in sorted(merged.loc[valid, fold_col].unique()):
                train = valid & (merged[fold_col] != fold)
                test = valid & (merged[fold_col] == fold)
                if train.sum() < 10 or test.sum() < 3:
                    continue

                y_train = merged.loc[train, f"{assay}_true"].to_numpy(dtype=float)
                y_test = merged.loc[test, f"{assay}_true"].to_numpy(dtype=float)
                off_train = merged.loc[train, f"{assay}_official"].to_numpy(dtype=float).reshape(-1, 1)
                off_test = merged.loc[test, f"{assay}_official"].to_numpy(dtype=float).reshape(-1, 1)

                calibration = LinearRegression().fit(off_train, y_train)
                base_train = calibration.predict(off_train)
                base_test = calibration.predict(off_test)
                residual_train = y_train - base_train

                estimator = make_regression_pipeline(
                    charge_features.loc[train],
                    model_name="ridge",
                    random_state=random_state,
                )
                estimator.fit(charge_features.loc[train], residual_train)
                res_pred = estimator.predict(charge_features.loc[test])

                calibrated_official[test] = base_test
                residual_true[test] = y_test - base_test
                residual_pred[test] = res_pred
                final_pred[test] = base_test + res_pred

            y_true = merged[f"{assay}_true"].to_numpy(dtype=float)
            official_row = official_metric_row(y_true, official_oof, assay, "aggregated", official_model)
            calibrated_row = official_metric_row(
                y_true,
                calibrated_official,
                assay,
                "aggregated",
                f"{official_model}_calibrated",
            )
            final_row = official_metric_row(
                y_true,
                final_pred,
                assay,
                "aggregated",
                f"{official_model}_plus_charge_residual",
            )
            residual_valid = ~np.isnan(residual_true) & ~np.isnan(residual_pred)
            residual_spearman = np.nan
            if residual_valid.sum() >= 3:
                residual_spearman = spearmanr(
                    residual_pred[residual_valid],
                    residual_true[residual_valid],
                    nan_policy="omit",
                ).correlation

            rows.append(
                {
                    "official_model": official_model,
                    "assay": assay,
                    "n": int((~np.isnan(final_pred) & ~np.isnan(y_true)).sum()),
                    "official_spearman": official_row["spearman"],
                    "calibrated_official_spearman": calibrated_row["spearman"],
                    "plus_charge_spearman": final_row["spearman"],
                    "delta_vs_official": final_row["spearman"] - official_row["spearman"],
                    "delta_vs_calibrated": final_row["spearman"] - calibrated_row["spearman"],
                    "residual_spearman": residual_spearman,
                }
            )
    return pd.DataFrame(rows)
