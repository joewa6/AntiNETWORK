import pandas as pd

from antinetwork.gdpa import fold_summary, missingness_report, network_from_correlations


def test_missingness_report_orders_missing_columns_first():
    df = pd.DataFrame({"complete": [1, 2, 3], "some_missing": [1, None, None]})

    report = missingness_report(df)

    assert report.index[0] == "some_missing"
    assert report.loc["some_missing", "missing_count"] == 2


def test_fold_summary_uses_available_columns():
    df = pd.DataFrame({"random_fold": [0, 0, 1], "value": [1.0, 2.0, 3.0]})

    summary = fold_summary(df)

    assert set(summary) == {"random_fold"}
    assert summary["random_fold"].loc[0] == 2


def test_network_from_correlations_adds_strong_edges():
    df = pd.DataFrame(
        {
            "assay_a": [1, 2, 3, 4, 5],
            "assay_b": [2, 4, 6, 8, 10],
            "assay_c": [5, 1, 4, 2, 3],
        }
    )

    graph = network_from_correlations(df, min_abs_corr=0.95)

    assert graph.has_edge("assay_a", "assay_b")
    assert graph["assay_a"]["assay_b"]["correlation"] > 0

