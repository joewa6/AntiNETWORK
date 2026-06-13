from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import LinearSVR, SVR

from antinetwork.killtest import ASSAY_DIRECTIONS


DEFAULT_COMPETITIVE_MODELS = ["ridge", "random_forest", "hist_gradient_boosting"]
TM_STRUCTURE_MODELS = [
    "ridge",
    "elasticnet",
    "linear_svm",
    "rbf_svm",
    "gam_spline",
    "random_forest",
    "hist_gradient_boosting",
]


@dataclass(frozen=True)
class RankerEvaluationResult:
    """Competitive ranker metrics plus out-of-fold predictions."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame


def make_numeric_regression_pipeline(
    model_name: str,
    random_state: int = 7,
    pca_components: int | None = None,
) -> Pipeline:
    """Create a numeric-only pipeline for already-assembled feature matrices."""
    if model_name == "ridge":
        model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    elif model_name == "elasticnet":
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            alphas=np.logspace(-2, 2, 9),
            max_iter=100_000,
            tol=1e-3,
            cv=3,
            random_state=random_state,
            selection="random",
        )
    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=8,
            random_state=random_state,
            n_jobs=1,
        )
    elif model_name == "linear_svm":
        model = LinearSVR(
            C=1.0,
            epsilon=0.1,
            loss="squared_epsilon_insensitive",
            max_iter=20_000,
            random_state=random_state,
        )
    elif model_name == "rbf_svm":
        model = SVR(C=1.0, epsilon=0.1, gamma="scale")
    elif model_name == "gam_spline":
        model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    elif model_name == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=150,
            l2_regularization=0.1,
            min_samples_leaf=12,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if model_name == "gam_spline":
        steps.append(("spline", AdaptiveSplineTransformer(n_knots=4, degree=3)))
    if pca_components is not None:
        steps.append(("pca", AdaptivePCA(n_components=pca_components, random_state=random_state)))
    steps.append(("model", model))
    return Pipeline(steps)


class AdaptivePCA(BaseEstimator, TransformerMixin):
    """PCA that caps n_components to the fitted matrix to avoid fold-size leakage/errors."""

    def __init__(self, n_components: int = 64, random_state: int = 7):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "AdaptivePCA":
        max_components = min(x.shape[0] - 1, x.shape[1], self.n_components)
        self.n_components_ = max(int(max_components), 0)
        self.pca_: PCA | None
        if self.n_components_ >= 1:
            self.pca_ = PCA(n_components=self.n_components_, random_state=self.random_state)
            self.pca_.fit(x)
        else:
            self.pca_ = None
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.pca_ is None:
            return x
        return self.pca_.transform(x)


class AdaptiveSplineTransformer(BaseEstimator, TransformerMixin):
    """Spline basis that caps knots for tiny folds/features."""

    def __init__(self, n_knots: int = 4, degree: int = 3):
        self.n_knots = n_knots
        self.degree = degree

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "AdaptiveSplineTransformer":
        n_unique = min(len(np.unique(x[:, idx])) for idx in range(x.shape[1])) if x.shape[1] else 0
        n_knots = max(2, min(self.n_knots, n_unique))
        self.transformer_: SplineTransformer | None
        if n_unique > 1:
            self.transformer_ = SplineTransformer(
                n_knots=n_knots,
                degree=min(self.degree, n_knots - 1),
                include_bias=False,
            )
            self.transformer_.fit(x)
        else:
            self.transformer_ = None
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.transformer_ is None:
            return x
        return self.transformer_.transform(x)


def competitive_metric_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    assay: str,
    model: str,
    feature_set: str,
    dataset: str = "GDPa1_cluster_cv",
    bad_fraction: float = 0.20,
) -> dict[str, object]:
    """Rank, triage, and calibration metrics for developability screening."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    row: dict[str, object] = {
        "dataset": dataset,
        "assay": assay,
        "feature_set": feature_set,
        "model": model,
        "n": int(valid.sum()),
        "spearman": np.nan,
        "pearson": np.nan,
        "r2": np.nan,
        "rmse": np.nan,
        "label_sd": np.nan,
        "rmse_over_sd": np.nan,
        "worst_20_recall": np.nan,
        "worst_20_auroc": np.nan,
    }
    if valid.sum() < 3:
        return row

    yt = y_true[valid]
    yp = y_pred[valid]
    row["spearman"] = spearmanr(yt, yp).statistic
    row["pearson"] = pearsonr(yt, yp).statistic
    row["r2"] = r2_score(yt, yp)
    row["rmse"] = float(np.sqrt(mean_squared_error(yt, yp)))
    row["label_sd"] = float(np.std(yt))
    row["rmse_over_sd"] = row["rmse"] / row["label_sd"] if row["label_sd"] > 0 else np.nan
    row["worst_20_recall"] = worst_fraction_recall(yt, yp, assay, fraction=bad_fraction)
    row["worst_20_auroc"] = worst_fraction_auroc(yt, yp, assay, fraction=bad_fraction)
    return row


def worst_fraction_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    assay: str,
    fraction: float = 0.20,
) -> float:
    """Recall for the worst assay tail, using developability-risk direction."""
    yt, yp = _risk_oriented_arrays(y_true, y_pred, assay)
    if len(yt) == 0:
        return np.nan
    k = int(np.ceil(len(yt) * fraction))
    if k <= 0:
        return np.nan
    true_worst = set(np.argsort(yt)[-k:].tolist())
    pred_worst = set(np.argsort(yp)[-k:].tolist())
    return len(true_worst & pred_worst) / k


def worst_fraction_auroc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    assay: str,
    fraction: float = 0.20,
) -> float:
    """AUROC for classifying the worst assay tail."""
    yt, yp = _risk_oriented_arrays(y_true, y_pred, assay)
    if len(yt) < 3:
        return np.nan
    k = int(np.ceil(len(yt) * fraction))
    if k <= 0 or k >= len(yt):
        return np.nan
    labels = np.zeros(len(yt), dtype=int)
    labels[np.argsort(yt)[-k:]] = 1
    if labels.min() == labels.max():
        return np.nan
    return float(roc_auc_score(labels, yp))


def evaluate_competitive_rankers(
    df: pd.DataFrame,
    feature_sets: dict[str, pd.DataFrame],
    assays: list[str],
    groups: pd.Series | None,
    model_names: list[str] | None = None,
    cv_kind: str = "cluster",
    random_state: int = 7,
    dataset: str = "GDPa1_cluster_cv",
    pca_components: int | None = None,
) -> RankerEvaluationResult:
    """Evaluate feature/model stacks with identical grouped CV semantics."""
    model_names = model_names or DEFAULT_COMPETITIVE_MODELS
    common_index = _common_index(feature_sets)
    analysis_df = df.loc[common_index]
    analysis_groups = groups.loc[common_index] if groups is not None else None

    metrics = []
    prediction_frames = []
    for feature_set, features in feature_sets.items():
        x_all = features.loc[common_index].select_dtypes(include="number")
        for model_name in model_names:
            predictions = pd.DataFrame(index=common_index)
            predictions["feature_set"] = feature_set
            predictions["model"] = model_name
            for assay in assays:
                if assay not in analysis_df.columns:
                    continue
                y, pred = cross_validated_ranker_predictions(
                    analysis_df,
                    x_all,
                    assay,
                    groups=analysis_groups,
                    model_name=model_name,
                    cv_kind=cv_kind,
                    random_state=random_state,
                    pca_components=pca_components,
                )
                predictions[assay] = np.nan
                valid_index = analysis_df.index[analysis_df[assay].notna()]
                predictions.loc[valid_index, assay] = pred
                metrics.append(
                    competitive_metric_row(
                        y,
                        pred,
                        assay=assay,
                        model=model_name,
                        feature_set=feature_set,
                        dataset=dataset,
                    )
                )
            prediction_frames.append(predictions.reset_index(names="row_index"))

    return RankerEvaluationResult(
        metrics=pd.DataFrame(metrics),
        predictions=pd.concat(prediction_frames, ignore_index=True),
    )


def train_full_ranker_predictions(
    train_df: pd.DataFrame,
    train_features: pd.DataFrame,
    predict_df: pd.DataFrame,
    predict_features: pd.DataFrame,
    assays: list[str],
    model_by_assay: dict[str, str] | None = None,
    default_model: str = "ridge",
    random_state: int = 7,
    pca_components: int | None = None,
) -> pd.DataFrame:
    """Train frozen assay-specific rankers on all training labels and predict new rows."""
    model_by_assay = model_by_assay or {}
    predictions = pd.DataFrame({"antibody_name": predict_df["antibody_name"].to_numpy()})
    for assay in assays:
        predictions[assay] = np.nan
        if assay not in train_df.columns:
            continue
        valid = train_df[assay].notna()
        if valid.sum() < 10:
            continue
        model_name = model_by_assay.get(assay, default_model)
        estimator = make_numeric_regression_pipeline(
            model_name,
            random_state=random_state,
            pca_components=pca_components,
        )
        estimator.fit(train_features.loc[valid], train_df.loc[valid, assay].to_numpy(dtype=float))
        predictions[assay] = estimator.predict(predict_features)
    return predictions


def evaluate_external_ranker_predictions(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    assays: list[str],
    model_by_assay: dict[str, str] | None = None,
    feature_set: str = "external_features",
    dataset: str = "GDPa3_external",
) -> pd.DataFrame:
    """Evaluate frozen external predictions with the competitive triage metrics."""
    model_by_assay = model_by_assay or {}
    merged = truth[["antibody_name", *assays]].merge(
        predictions[["antibody_name", *assays]],
        on="antibody_name",
        how="left",
        suffixes=("_true", "_pred"),
    )
    rows = []
    for assay in assays:
        rows.append(
            competitive_metric_row(
                merged[f"{assay}_true"].to_numpy(dtype=float),
                merged[f"{assay}_pred"].to_numpy(dtype=float),
                assay=assay,
                model=model_by_assay.get(assay, "model"),
                feature_set=feature_set,
                dataset=dataset,
            )
        )
    return pd.DataFrame(rows)


def cross_validated_ranker_predictions(
    df: pd.DataFrame,
    features: pd.DataFrame,
    assay: str,
    groups: pd.Series | None,
    model_name: str,
    cv_kind: str = "cluster",
    random_state: int = 7,
    pca_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return y and out-of-fold predictions for one assay/model/feature set."""
    valid = df[assay].notna()
    x = features.loc[valid]
    y = df.loc[valid, assay].to_numpy(dtype=float)
    pred = np.full(len(y), np.nan)
    if len(y) < 20:
        return y, pred

    if cv_kind == "random":
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        split_groups = None
    elif cv_kind == "cluster":
        if groups is None:
            raise ValueError("groups must be provided for cluster CV")
        split_groups = groups.loc[valid].to_numpy()
        n_groups = len(np.unique(split_groups))
        if n_groups < 2:
            return y, pred
        cv = GroupKFold(n_splits=min(5, n_groups))
    else:
        raise ValueError(f"Unknown cv_kind: {cv_kind}")

    for train_idx, test_idx in cv.split(x, y, groups=split_groups):
        estimator = make_numeric_regression_pipeline(
            model_name,
            random_state=random_state,
            pca_components=pca_components,
        )
        estimator.fit(x.iloc[train_idx], y[train_idx])
        pred[test_idx] = estimator.predict(x.iloc[test_idx])
    return y, pred


def select_best_rankers(
    metrics: pd.DataFrame,
    primary_metric: str = "spearman",
) -> pd.DataFrame:
    """Choose the best model/feature stack per assay by a predeclared metric."""
    return (
        metrics.sort_values(["assay", primary_metric], ascending=[True, False])
        .groupby("assay", as_index=False)
        .first()
        .sort_values("assay")
        .reset_index(drop=True)
    )


def _risk_oriented_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    assay: str,
) -> tuple[np.ndarray, np.ndarray]:
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = np.asarray(y_true, dtype=float)[valid]
    yp = np.asarray(y_pred, dtype=float)[valid]
    direction = ASSAY_DIRECTIONS.get(assay, "high")
    if direction == "low":
        yt = -yt
        yp = -yp
    elif direction != "high":
        raise ValueError(f"Unknown risk direction for {assay}: {direction}")
    return yt, yp


def _common_index(feature_sets: dict[str, pd.DataFrame]) -> pd.Index:
    indexes = [frame.index for frame in feature_sets.values()]
    if not indexes:
        raise ValueError("feature_sets must not be empty.")
    common = indexes[0]
    for index in indexes[1:]:
        common = common.intersection(index)
    return common
