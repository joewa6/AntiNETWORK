from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ASSAY_DIRECTIONS = {
    "Titer": "low",
    "Purity": "low",
    "SEC %Monomer": "low",
    "SMAC": "high",
    "HIC": "high",
    "HAC": "high",
    "PR_CHO": "high",
    "PR_Ova": "high",
    "AC-SINS_pH6.0": "high",
    "AC-SINS_pH7.4": "high",
    "Tonset": "low",
    "Tm1": "low",
    "Tm2": "low",
}

INTERACTION_ASSAYS = [
    "SMAC",
    "HIC",
    "HAC",
    "PR_CHO",
    "PR_Ova",
    "AC-SINS_pH6.0",
    "AC-SINS_pH7.4",
]


def define_failure_matrix(
    df: pd.DataFrame,
    assay_directions: dict[str, str] | None = None,
    quantile: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Binarize assays into crude worst-quantile failure labels."""
    directions = assay_directions or ASSAY_DIRECTIONS
    labels = pd.DataFrame(index=df.index)
    thresholds = []

    for assay, direction in directions.items():
        values = df[assay]
        if direction == "high":
            threshold = values.quantile(1 - quantile)
            labels[assay] = (values >= threshold).where(values.notna(), pd.NA)
            threshold_side = ">="
        elif direction == "low":
            threshold = values.quantile(quantile)
            labels[assay] = (values <= threshold).where(values.notna(), pd.NA)
            threshold_side = "<="
        else:
            raise ValueError(f"Unknown assay direction for {assay}: {direction}")

        labels[assay] = labels[assay].astype("boolean")
        thresholds.append(
            {
                "assay": assay,
                "risk_direction": direction,
                "threshold_side": threshold_side,
                "threshold": threshold,
                "measured": values.notna().sum(),
                "failures": labels[assay].sum(skipna=True),
            }
        )

    return labels, pd.DataFrame(thresholds).set_index("assay")


def jaccard_similarity(labels: pd.DataFrame) -> pd.DataFrame:
    """Jaccard similarity between assay failure labels, ignoring missing pairs."""
    out = pd.DataFrame(np.eye(labels.shape[1]), index=labels.columns, columns=labels.columns)
    for i, source in enumerate(labels.columns):
        for target in labels.columns[i + 1 :]:
            valid = labels[source].notna() & labels[target].notna()
            source_fail = labels.loc[valid, source].astype(bool)
            target_fail = labels.loc[valid, target].astype(bool)
            union = (source_fail | target_fail).sum()
            value = np.nan if union == 0 else (source_fail & target_fail).sum() / union
            out.loc[source, target] = value
            out.loc[target, source] = value
    return out


def sequence_knn_baseline(
    df: pd.DataFrame,
    identity: pd.DataFrame,
    assays: list[str],
    k: int = 8,
    exclude_groups: pd.Series | None = None,
    model_label: str = "sequence_knn",
) -> pd.DataFrame:
    """Leave-one-out kNN assay prediction using sequence identity only."""
    rows = []
    sim = identity.to_numpy(copy=True)
    np.fill_diagonal(sim, -np.inf)
    group_values = None if exclude_groups is None else exclude_groups.to_numpy()

    for assay in assays:
        y = df[assay].to_numpy(dtype=float)
        preds = np.full(len(y), np.nan)
        mean_neighbor_identity = np.full(len(y), np.nan)

        for i in range(len(y)):
            valid = ~np.isnan(y)
            valid[i] = False
            if group_values is not None:
                valid &= group_values != group_values[i]
            candidates = np.where(valid)[0]
            if len(candidates) == 0 or np.isnan(y[i]):
                continue
            ordered = candidates[np.argsort(sim[i, candidates])[::-1]]
            neighbors = ordered[: min(k, len(ordered))]
            preds[i] = np.nanmean(y[neighbors])
            mean_neighbor_identity[i] = np.nanmean(sim[i, neighbors])

        rows.append(metric_row(assay, y, preds, model_label, extra={"k": k}))
        rows[-1]["mean_neighbor_identity"] = np.nanmean(mean_neighbor_identity)

    return pd.DataFrame(rows)


def sequence_clusters(identity: pd.DataFrame, threshold: float = 0.85) -> pd.Series:
    """Cluster antibodies by sequence identity using average-linkage distance."""
    distance = 1 - identity.to_numpy()
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    z = linkage(condensed, method="average")
    clusters = fcluster(z, t=1 - threshold, criterion="distance")
    return pd.Series(clusters, index=identity.index, name=f"seq_cluster_{threshold:g}")


def metric_row(
    assay: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    row: dict[str, object] = {
        "assay": assay,
        "model": model,
        "n": int(valid.sum()),
        "pearson": np.nan,
        "spearman": np.nan,
        "rmse": np.nan,
    }
    if valid.sum() >= 3:
        row["pearson"] = pearsonr(y_true[valid], y_pred[valid]).statistic
        row["spearman"] = spearmanr(y_true[valid], y_pred[valid]).statistic
        row["rmse"] = float(np.sqrt(mean_squared_error(y_true[valid], y_pred[valid])))
    if extra:
        row.update(extra)
    return row


def model_cv_results(
    df: pd.DataFrame,
    features: pd.DataFrame,
    assays: list[str],
    model_name: str = "ridge",
    cv_kind: str = "random",
    groups: pd.Series | None = None,
    feature_set: str | None = None,
    random_state: int = 7,
) -> pd.DataFrame:
    """Evaluate descriptor models with random CV or sequence-cluster grouped CV."""
    rows = []

    for assay in assays:
        valid = df[assay].notna()
        x = features.loc[valid]
        y = df.loc[valid, assay].to_numpy(dtype=float)
        if len(y) < 20:
            continue

        if cv_kind == "random":
            cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
            split_groups = None
        elif cv_kind == "cluster":
            if groups is None:
                raise ValueError("groups must be provided for cluster CV")
            split_groups = groups.loc[valid].to_numpy()
            n_groups = len(np.unique(split_groups))
            cv = GroupKFold(n_splits=min(5, n_groups))
        else:
            raise ValueError(f"Unknown cv_kind: {cv_kind}")

        estimator = make_regression_pipeline(x, model_name=model_name, random_state=random_state)
        pred = cross_val_predict(estimator, x, y, cv=cv, groups=split_groups)
        label = f"{model_name}_{cv_kind}_cv" if feature_set is None else f"{feature_set}_{cv_kind}_cv"
        rows.append(metric_row(assay, y, pred, label))

    return pd.DataFrame(rows)


def make_regression_pipeline(
    features: pd.DataFrame,
    model_name: str = "ridge",
    random_state: int = 7,
) -> Pipeline:
    numeric_cols = features.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in features.columns if col not in numeric_cols]

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

    if model_name == "ridge":
        model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    elif model_name == "elasticnet":
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            alphas=np.logspace(-3, 1, 9),
            max_iter=20_000,
            cv=3,
            random_state=random_state,
        )
    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=8,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers=transformers)),
            ("model", model),
        ]
    )


def verdict_table(
    sequence_metrics: pd.DataFrame,
    physical_random: pd.DataFrame,
    physical_cluster: pd.DataFrame,
    interaction_assays: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize the kill-test decision rules in a compact table."""
    interaction = interaction_assays or INTERACTION_ASSAYS

    seq = sequence_metrics.set_index("assay")
    phys_random = physical_random.set_index("assay")
    phys_cluster = physical_cluster.set_index("assay")
    common = [assay for assay in interaction if assay in seq.index and assay in phys_random.index]

    seq_mean = seq.loc[common, "spearman"].mean()
    phys_random_mean = phys_random.loc[common, "spearman"].mean()
    phys_cluster_mean = phys_cluster.loc[
        [assay for assay in common if assay in phys_cluster.index], "spearman"
    ].mean()

    rows = [
        {
            "question": "Interaction assays form coherent module",
            "criterion": "Co-failure/correlation modules visible",
            "result": "Inspect networks and clustered heatmaps",
            "decision": "manual",
        },
        {
            "question": "Sequence similarity explains everything",
            "criterion": "Sequence kNN mean Spearman on interaction assays",
            "result": round(seq_mean, 3),
            "decision": "caution" if seq_mean >= 0.4 else "pass",
        },
        {
            "question": "Physical descriptors beat sequence baseline",
            "criterion": "Physical random-CV mean Spearman > sequence kNN",
            "result": round(phys_random_mean - seq_mean, 3),
            "decision": "pass" if phys_random_mean > seq_mean else "fail",
        },
        {
            "question": "Physical descriptors survive cluster CV",
            "criterion": "Cluster-CV physical descriptors retain signal",
            "result": round(phys_cluster_mean, 3),
            "decision": "pass" if phys_cluster_mean >= 0.2 else "caution",
        },
        {
            "question": "HAC evidence strength",
            "criterion": "HAC has sparse coverage",
            "result": "supporting only",
            "decision": "do not overclaim",
        },
    ]
    return pd.DataFrame(rows)
