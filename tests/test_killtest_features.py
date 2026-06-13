import numpy as np
import pandas as pd

from antinetwork.features import build_physical_features, pairwise_aligned_identity
from antinetwork.killtest import define_failure_matrix, jaccard_similarity, sequence_knn_baseline


def test_define_failure_matrix_respects_assay_direction():
    df = pd.DataFrame({"high_bad": [1, 2, 3, 4, 5], "low_bad": [1, 2, 3, 4, 5]})

    labels, thresholds = define_failure_matrix(
        df,
        assay_directions={"high_bad": "high", "low_bad": "low"},
        quantile=0.2,
    )

    assert labels["high_bad"].tolist() == [False, False, False, False, True]
    assert labels["low_bad"].tolist() == [True, False, False, False, False]
    assert thresholds.loc["high_bad", "threshold_side"] == ">="
    assert thresholds.loc["low_bad", "threshold_side"] == "<="


def test_jaccard_similarity_ignores_missing_pairs():
    labels = pd.DataFrame(
        {
            "a": pd.Series([True, True, False, pd.NA], dtype="boolean"),
            "b": pd.Series([True, False, False, True], dtype="boolean"),
        }
    )

    jaccard = jaccard_similarity(labels)

    assert np.isclose(jaccard.loc["a", "b"], 0.5)


def test_physical_features_and_identity_have_expected_columns():
    df = pd.DataFrame(
        {
            "vh_protein_sequence": ["AKDEFWY", "AKDEFWF"],
            "vl_protein_sequence": ["KKDDVV", "KKDDVI"],
            "heavy_aligned_aho": ["AKDEFWY" + "-" * 142, "AKDEFWF" + "-" * 142],
            "light_aligned_aho": ["KKDDVV" + "-" * 143, "KKDDVI" + "-" * 143],
        }
    )

    features = build_physical_features(df)
    identity = pairwise_aligned_identity(df)

    assert "fv_net_charge" in features.columns
    assert "vh_vl_charge_imbalance" in features.columns
    assert identity.shape == (2, 2)
    assert identity.iloc[0, 0] == 1
    assert 0 < identity.iloc[0, 1] < 1


def test_sequence_knn_baseline_returns_metrics():
    df = pd.DataFrame({"assay": [1.0, 2.0, 10.0, 11.0]})
    identity = pd.DataFrame(
        [
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.8],
            [0.1, 0.1, 0.8, 1.0],
        ]
    )

    metrics = sequence_knn_baseline(df, identity, ["assay"], k=1)

    assert metrics.loc[0, "assay"] == "assay"
    assert metrics.loc[0, "n"] == 4
    assert metrics.loc[0, "rmse"] > 0

