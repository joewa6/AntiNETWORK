from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from antinetwork.killtest import make_regression_pipeline, metric_row, model_cv_results


def feature_families(feature_names: list[str] | pd.Index) -> pd.DataFrame:
    """Assign interpretable families to crude physical sequence features."""
    rows = []
    for feature in feature_names:
        name = str(feature)
        chain_region = "global"
        if name.startswith("vh_"):
            chain_region = "vh"
        elif name.startswith("vl_"):
            chain_region = "vl"
        elif name.startswith("heavy_cdr_"):
            chain_region = "heavy_cdr"
        elif name.startswith("light_cdr_"):
            chain_region = "light_cdr"
        elif name.startswith("fv_cdr_") or name.startswith("cdr_"):
            chain_region = "cdr"

        if (
            "hydrophobic" in name
            or "aromatic" in name
            or "gravy" in name
            or "aliphatic" in name
        ):
            chemistry = "hydrophobic_aromatic"
        elif (
            "proline" in name
            or "glycine" in name
            or "flexible" in name
            or "composition_entropy" in name
        ):
            chemistry = "conformational_entropy"
        elif "cysteine" in name or "disulfide" in name:
            chemistry = "disulfide"
        elif (
            "charge" in name
            or "positive" in name
            or "negative" in name
            or "lys_arg" in name
            or "acid_base" in name
        ):
            chemistry = "charge"
        elif "length" in name:
            chemistry = "length"
        elif "imbalance" in name or "asymmetry" in name:
            chemistry = "imbalance"
        else:
            chemistry = "composition_other"

        rows.append(
            {
                "feature": name,
                "chain_region": chain_region,
                "chemistry": chemistry,
                "is_cdr": "cdr" in chain_region or name.startswith("cdr_"),
                "is_framework_or_full_variable": chain_region in {"vh", "vl"},
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def cross_validated_predictions(
    df: pd.DataFrame,
    features: pd.DataFrame,
    assay: str,
    cv_kind: str = "cluster",
    groups: pd.Series | None = None,
    model_name: str = "ridge",
    random_state: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return y and out-of-fold predictions for one assay."""
    valid = df[assay].notna()
    x = features.loc[valid]
    y = df.loc[valid, assay].to_numpy(dtype=float)

    if cv_kind == "random":
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        split_groups = None
    elif cv_kind == "cluster":
        if groups is None:
            raise ValueError("groups must be provided for cluster CV")
        split_groups = groups.loc[valid].to_numpy()
        cv = GroupKFold(n_splits=min(5, len(np.unique(split_groups))))
    else:
        raise ValueError(f"Unknown cv_kind: {cv_kind}")

    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in cv.split(x, y, groups=split_groups):
        estimator = make_regression_pipeline(x, model_name=model_name, random_state=random_state)
        estimator.fit(x.iloc[train_idx], y[train_idx])
        preds[test_idx] = estimator.predict(x.iloc[test_idx])

    return y, preds


def leave_one_family_out(
    df: pd.DataFrame,
    features: pd.DataFrame,
    assays: list[str],
    family_table: pd.DataFrame,
    family_column: str,
    cv_kind: str = "cluster",
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    """Measure performance drop after removing one feature family."""
    baseline = model_cv_results(
        df,
        features,
        assays,
        model_name="ridge",
        cv_kind=cv_kind,
        groups=groups,
        feature_set=f"all_features_{cv_kind}",
    )

    rows = []
    for family in sorted(family_table[family_column].dropna().unique()):
        drop_cols = family_table.index[family_table[family_column] == family].tolist()
        kept = features.drop(columns=drop_cols)
        if kept.shape[1] == 0:
            continue
        result = model_cv_results(
            df,
            kept,
            assays,
            model_name="ridge",
            cv_kind=cv_kind,
            groups=groups,
            feature_set=f"without_{family}",
        )
        merged = baseline.merge(result, on="assay", suffixes=("_all", "_without"))
        for _, row in merged.iterrows():
            rows.append(
                {
                    "assay": row["assay"],
                    "family_column": family_column,
                    "family_removed": family,
                    "baseline_spearman": row["spearman_all"],
                    "without_spearman": row["spearman_without"],
                    "spearman_drop": row["spearman_all"] - row["spearman_without"],
                    "baseline_rmse": row["rmse_all"],
                    "without_rmse": row["rmse_without"],
                    "rmse_increase": row["rmse_without"] - row["rmse_all"],
                }
            )
    return pd.DataFrame(rows)


def bootstrap_ridge_coefficients(
    df: pd.DataFrame,
    features: pd.DataFrame,
    assay: str,
    n_boot: int = 300,
    random_state: int = 7,
) -> pd.DataFrame:
    """Bootstrap standardized ridge coefficients for one assay."""
    rng = np.random.default_rng(random_state)
    valid = df[assay].notna()
    x = features.loc[valid]
    y = df.loc[valid, assay].to_numpy(dtype=float)

    coefficients = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        estimator = make_regression_pipeline(x, model_name="ridge", random_state=random_state)
        estimator.fit(x.iloc[idx], y[idx])
        model = estimator.named_steps["model"]
        coefficients.append(model.coef_)

    coef = np.vstack(coefficients)
    return pd.DataFrame(
        {
            "feature": x.columns,
            "coef_mean": coef.mean(axis=0),
            "coef_sd": coef.std(axis=0),
            "coef_q025": np.quantile(coef, 0.025, axis=0),
            "coef_q975": np.quantile(coef, 0.975, axis=0),
            "abs_coef_mean": np.abs(coef).mean(axis=0),
        }
    ).sort_values("abs_coef_mean", ascending=False)


def permutation_importance_by_assay(
    df: pd.DataFrame,
    features: pd.DataFrame,
    assays: list[str],
    n_repeats: int = 50,
    random_state: int = 7,
) -> pd.DataFrame:
    """Fit ridge models and compute training-set permutation importance.

    This is interpretability, not performance estimation. Use with the CV
    results nearby.
    """

    def spearman_scorer(estimator, x, y):
        pred = estimator.predict(x)
        return spearmanr(y, pred).statistic

    rows = []
    for assay in assays:
        valid = df[assay].notna()
        x = features.loc[valid]
        y = df.loc[valid, assay].to_numpy(dtype=float)
        estimator = make_regression_pipeline(x, model_name="ridge", random_state=random_state)
        estimator.fit(x, y)
        result = permutation_importance(
            estimator,
            x,
            y,
            scoring=spearman_scorer,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )
        for feature, mean, sd in zip(x.columns, result.importances_mean, result.importances_std):
            rows.append(
                {
                    "assay": assay,
                    "feature": feature,
                    "importance_mean": mean,
                    "importance_sd": sd,
                }
            )
    return pd.DataFrame(rows).sort_values(["assay", "importance_mean"], ascending=[True, False])


def grouped_importance(
    importance: pd.DataFrame,
    families: pd.DataFrame,
    family_column: str,
) -> pd.DataFrame:
    """Aggregate feature-level importance by family."""
    return (
        importance.join(families[[family_column]], on="feature")
        .groupby(["assay", family_column], dropna=False)["importance_mean"]
        .sum()
        .reset_index()
        .sort_values(["assay", "importance_mean"], ascending=[True, False])
    )


def family_expectation_table(
    chemistry_drop: pd.DataFrame,
    interaction_assays: list[str],
) -> pd.DataFrame:
    """Compact mechanistic falsification table."""
    subset = chemistry_drop[chemistry_drop["assay"].isin(interaction_assays)]
    rows = []
    expectations = {
        "SMAC": "hydrophobic_aromatic",
        "HIC": "hydrophobic_aromatic",
        "AC-SINS_pH6.0": "charge",
        "AC-SINS_pH7.4": "charge",
        "HAC": "charge",
        "PR_CHO": "hydrophobic_aromatic",
        "PR_Ova": "hydrophobic_aromatic",
    }
    for assay, expected_family in expectations.items():
        assay_rows = subset[subset["assay"] == assay].sort_values("spearman_drop", ascending=False)
        if assay_rows.empty:
            continue
        top = assay_rows.iloc[0]
        expected = assay_rows[assay_rows["family_removed"] == expected_family]
        expected_drop = np.nan if expected.empty else expected["spearman_drop"].iloc[0]
        rows.append(
            {
                "assay": assay,
                "expected_family": expected_family,
                "top_removed_family": top["family_removed"],
                "top_drop": top["spearman_drop"],
                "expected_family_drop": expected_drop,
                "mechanistic_alignment": top["family_removed"] == expected_family
                or (pd.notna(expected_drop) and expected_drop > 0.03),
            }
        )
    return pd.DataFrame(rows)


def one_row_metric(
    df: pd.DataFrame,
    features: pd.DataFrame,
    assay: str,
    cv_kind: str,
    groups: pd.Series | None = None,
) -> dict[str, object]:
    y, pred = cross_validated_predictions(df, features, assay, cv_kind=cv_kind, groups=groups)
    return metric_row(assay, y, pred, f"ridge_{cv_kind}")


def build_bias_control_features(
    df: pd.DataFrame,
    sequence_cluster: pd.Series,
    metadata_cols: list[str],
) -> pd.DataFrame:
    """Build metadata + sequence-cluster covariates for bias control."""
    controls = df[metadata_cols].copy()
    controls["sequence_cluster"] = sequence_cluster.astype(str).to_numpy()
    for column in controls.columns:
        controls[column] = controls[column].astype(str)
    return controls


def residualize_assays(
    df: pd.DataFrame,
    assays: list[str],
    controls: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regress each assay on controls and return residuals plus control fit metrics."""
    residuals = pd.DataFrame(index=df.index)
    rows = []

    for assay in assays:
        valid = df[assay].notna()
        x = controls.loc[valid]
        y = df.loc[valid, assay].to_numpy(dtype=float)
        model = make_control_model(x)
        model.fit(x, y)
        pred = model.predict(x)
        residuals.loc[valid, assay] = y - pred

        total = np.sum((y - y.mean()) ** 2)
        unexplained = np.sum((y - pred) ** 2)
        rows.append(
            {
                "assay": assay,
                "n": int(valid.sum()),
                "control_r2": np.nan if total == 0 else 1 - unexplained / total,
                "residual_sd": float(np.std(y - pred)),
            }
        )

    return residuals, pd.DataFrame(rows)


def make_control_model(controls: pd.DataFrame) -> Pipeline:
    numeric_cols = controls.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in controls.columns if col not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers=transformers)),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 13))),
        ]
    )


def residual_signal_experiment(
    residuals: pd.DataFrame,
    features: pd.DataFrame,
    assays: list[str],
    groups: pd.Series,
    n_shuffles: int = 100,
    feature_set_label: str = "charge",
    random_state: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare residual prediction against shuffled-feature and random-feature controls."""
    rng = np.random.default_rng(random_state)
    observed_label = (
        "charge_residual" if feature_set_label == "charge" else f"{feature_set_label}_residual"
    )
    shuffled_label = (
        "shuffled_charge" if feature_set_label == "charge" else f"shuffled_{feature_set_label}"
    )
    random_label = (
        "random_gaussian"
        if feature_set_label == "charge"
        else f"random_gaussian_{feature_set_label}"
    )
    observed = model_cv_results(
        residuals,
        features,
        assays,
        model_name="ridge",
        cv_kind="cluster",
        groups=groups,
        feature_set=observed_label,
        random_state=random_state,
    )

    control_rows = []
    for repeat in range(n_shuffles):
        shuffled = features.iloc[rng.permutation(len(features))].reset_index(drop=True)
        shuffled.index = features.index
        shuffled_result = model_cv_results(
            residuals,
            shuffled,
            assays,
            model_name="ridge",
            cv_kind="cluster",
            groups=groups,
            feature_set=shuffled_label,
            random_state=random_state + repeat + 1,
        )
        shuffled_result["repeat"] = repeat
        control_rows.append(shuffled_result)

        random_values = pd.DataFrame(
            rng.normal(size=features.shape),
            index=features.index,
            columns=features.columns,
        )
        random_result = model_cv_results(
            residuals,
            random_values,
            assays,
            model_name="ridge",
            cv_kind="cluster",
            groups=groups,
            feature_set=random_label,
            random_state=random_state + repeat + 1,
        )
        random_result["repeat"] = repeat
        control_rows.append(random_result)

    controls = pd.concat(control_rows, ignore_index=True)
    return observed, controls


def residual_control_summary(
    observed: pd.DataFrame,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize observed residual prediction against null controls."""
    rows = []
    for _, obs in observed.iterrows():
        assay_controls = controls[controls["assay"] == obs["assay"]]
        for model_name, subset in assay_controls.groupby("model"):
            null = subset["spearman"].dropna()
            if null.empty:
                continue
            rows.append(
                {
                    "assay": obs["assay"],
                    "control": model_name,
                    "observed_spearman": obs["spearman"],
                    "null_mean": null.mean(),
                    "null_q95": null.quantile(0.95),
                    "empirical_p_greater": (1 + (null >= obs["spearman"]).sum()) / (len(null) + 1),
                    "observed_minus_null_mean": obs["spearman"] - null.mean(),
                }
            )
    return pd.DataFrame(rows)
