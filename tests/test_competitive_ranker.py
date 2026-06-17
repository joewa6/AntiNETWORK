import numpy as np
import pandas as pd

from antinetwork.competitive_ranker import (
    competitive_metric_row,
    evaluate_external_ranker_predictions,
    evaluate_competitive_rankers,
    make_numeric_regression_pipeline,
    select_best_rankers,
    train_full_ranker_predictions,
    worst_fraction_auroc,
    worst_fraction_recall,
)


def test_worst_fraction_metrics_follow_assay_risk_direction():
    y_true = np.arange(10, dtype=float)
    good_high_bad = np.arange(10, dtype=float)
    bad_high_bad = good_high_bad[::-1]

    assert worst_fraction_recall(y_true, good_high_bad, "HIC", fraction=0.2) == 1.0
    assert worst_fraction_recall(y_true, bad_high_bad, "HIC", fraction=0.2) == 0.0
    assert worst_fraction_auroc(y_true, good_high_bad, "HIC", fraction=0.2) == 1.0

    # Titer is low-risk-direction in killtest terms, so the smallest labels are worst.
    good_low_bad = np.arange(10, dtype=float)
    assert worst_fraction_recall(y_true, good_low_bad, "Titer", fraction=0.2) == 1.0


def test_competitive_metric_row_reports_rank_triage_and_calibration():
    y_true = np.linspace(0, 1, 30)
    y_pred = y_true + 0.01

    row = competitive_metric_row(
        y_true,
        y_pred,
        assay="HIC",
        model="ridge",
        feature_set="toy",
    )

    assert row["spearman"] == 1.0
    assert row["r2"] > 0.99
    assert row["worst_20_recall"] == 1.0
    assert row["worst_20_auroc"] == 1.0


def test_evaluate_competitive_rankers_and_select_best_runs_grouped_cv():
    n = 50
    feature = np.linspace(-1, 1, n)
    df = pd.DataFrame(
        {
            "antibody_name": [f"ab{i}" for i in range(n)],
            "HIC": 2.0 + feature,
        }
    )
    feature_sets = {
        "signal": pd.DataFrame({"feature": feature}),
        "weak": pd.DataFrame({"feature": np.sin(np.arange(n))}),
    }
    groups = pd.Series(np.repeat(np.arange(10), 5))

    result = evaluate_competitive_rankers(
        df,
        feature_sets,
        assays=["HIC"],
        groups=groups,
        model_names=["ridge", "random_forest"],
    )
    best = select_best_rankers(result.metrics)

    assert set(result.metrics["feature_set"]) == {"signal", "weak"}
    assert set(result.metrics["model"]) == {"ridge", "random_forest"}
    assert best.loc[0, "assay"] == "HIC"
    assert best.loc[0, "spearman"] > 0.9


def test_tm_structure_model_registry_builds_non_neural_models():
    x = pd.DataFrame({"feature": np.linspace(-1, 1, 20)})
    y = np.linspace(0, 1, 20)

    for model_name in ["elasticnet", "linear_svm", "rbf_svm", "gam_spline"]:
        estimator = make_numeric_regression_pipeline(model_name, random_state=7)
        estimator.fit(x, y)
        pred = estimator.predict(x)

        assert pred.shape == y.shape


def test_train_full_ranker_predictions_and_external_metrics():
    train = pd.DataFrame(
        {
            "antibody_name": [f"train{i}" for i in range(30)],
            "HIC": np.linspace(0.0, 1.0, 30),
        }
    )
    test = pd.DataFrame(
        {
            "antibody_name": [f"test{i}" for i in range(10)],
            "HIC": np.linspace(0.0, 1.0, 10),
        }
    )
    train_features = pd.DataFrame({"feature": train["HIC"]})
    test_features = pd.DataFrame({"feature": test["HIC"]})

    predictions = train_full_ranker_predictions(
        train,
        train_features,
        test,
        test_features,
        assays=["HIC"],
        model_by_assay={"HIC": "ridge"},
    )
    metrics = evaluate_external_ranker_predictions(
        test,
        predictions,
        assays=["HIC"],
        model_by_assay={"HIC": "ridge"},
    )

    assert predictions["antibody_name"].tolist() == test["antibody_name"].tolist()
    assert predictions["HIC"].notna().all()
    assert metrics.loc[0, "dataset"] == "GDPa3_external"
    assert metrics.loc[0, "worst_20_auroc"] == 1.0
