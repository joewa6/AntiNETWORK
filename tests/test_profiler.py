import numpy as np
import pandas as pd

from antinetwork.profiler import build_axis_report, percentile_score, risk_label


def _reference_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fv_cdr_hydrophobic_fraction": [0.2, 0.3, 0.4, 0.5],
            "fv_cdr_aromatic_fraction": [0.05, 0.1, 0.15, 0.2],
            "vh_hydrophobic_fraction": [0.35, 0.4, 0.45, 0.5],
            "vl_hydrophobic_fraction": [0.30, 0.35, 0.40, 0.45],
            "cdr_h3_length": [8.0, 10.0, 12.0, 18.0],
            "fv_cdr_positive_count": [2.0, 4.0, 6.0, 8.0],
            "fv_cdr_net_charge": [-1.0, 1.0, 3.0, 5.0],
            "cdr_charge_fraction_of_fv": [0.2, 0.4, 0.6, 0.8],
            "vh_vl_charge_imbalance": [1.0, 2.0, 4.0, 8.0],
            "heavy_cdr_positive_count": [1.0, 2.0, 5.0, 7.0],
            "vh_vl_hydrophobicity_imbalance": [0.01, 0.04, 0.08, 0.12],
            "fv_net_charge": [-3.0, 0.0, 3.0, 6.0],
            "vh_length": [110.0, 112.0, 114.0, 116.0],
            "vl_length": [105.0, 107.0, 109.0, 111.0],
        }
    )


def test_percentile_score_and_risk_label():
    assert percentile_score(3.0, pd.Series([1.0, 2.0, 3.0, 4.0])) == 0.75
    assert risk_label(0.8) == "high"
    assert risk_label(0.6) == "medium"
    assert risk_label(0.2) == "low"
    assert risk_label(np.nan) == "unknown"


def test_build_axis_report_returns_expected_axes():
    reference = _reference_features()
    row = reference.iloc[-1].copy()
    network_projection = pd.Series(
        {
            "network_risk_score": 0.8,
            "mapped_regime": "finite_reversible_clusters",
            "mapped_largest_cluster_fraction": 0.4,
            "mapped_percolation_probability": 0.2,
            "mapped_mean_degree": 1.3,
        }
    )

    report = build_axis_report(
        row,
        reference,
        antibody_name="toy_mAb",
        network_projection_row=network_projection,
    )

    assert report["axis"].tolist() == [
        "Hydrophobicity",
        "Polyreactivity",
        "Local self-association",
        "Bulk network / viscosity",
        "Thermostability",
        "Titer / expression",
    ]
    assert report.loc[report["axis"] == "Bulk network / viscosity", "risk_label"].iloc[0] == "high"
    assert (
        report.loc[report["axis"] == "Bulk network / viscosity", "confidence"].iloc[0]
        == "low-medium"
    )
    assert report.loc[report["axis"] == "Thermostability", "risk_label"].iloc[0] == "not assigned"
    assert report.loc[report["axis"] == "Titer / expression", "risk_label"].iloc[0] == "not assigned"
    assert {"evidence_strength", "benchmark_support", "mutation_direction"}.issubset(report.columns)
    assert report["main_features"].str.len().min() > 0
