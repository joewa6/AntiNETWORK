import pandas as pd

from antinetwork.physics_attribution import (
    build_bias_control_features,
    feature_families,
    family_expectation_table,
    residual_control_summary,
    residualize_assays,
)


def test_feature_families_tags_expected_chemistry_and_regions():
    families = feature_families(
        [
            "vh_net_charge",
            "vl_hydrophobic_fraction",
            "vh_aliphatic_index",
            "fv_cdr_proline_fraction",
            "fv_predicted_disulfide_pairs",
            "fv_cdr_aromatic_count",
            "cdr_h3_length",
            "vh_vl_charge_imbalance",
        ]
    )

    assert families.loc["vh_net_charge", "chemistry"] == "charge"
    assert families.loc["vl_hydrophobic_fraction", "chemistry"] == "hydrophobic_aromatic"
    assert families.loc["vh_aliphatic_index", "chemistry"] == "hydrophobic_aromatic"
    assert families.loc["fv_cdr_proline_fraction", "chemistry"] == "conformational_entropy"
    assert families.loc["fv_predicted_disulfide_pairs", "chemistry"] == "disulfide"
    assert families.loc["fv_cdr_aromatic_count", "is_cdr"]
    assert families.loc["cdr_h3_length", "chemistry"] == "length"
    assert families.loc["vh_vl_charge_imbalance", "chemistry"] == "charge"


def test_family_expectation_table_marks_expected_drop_as_alignment():
    drops = pd.DataFrame(
        {
            "assay": ["SMAC", "SMAC", "AC-SINS_pH7.4", "AC-SINS_pH7.4"],
            "family_removed": [
                "hydrophobic_aromatic",
                "charge",
                "hydrophobic_aromatic",
                "charge",
            ],
            "spearman_drop": [0.05, 0.01, 0.0, 0.04],
        }
    )

    table = family_expectation_table(drops, ["SMAC", "AC-SINS_pH7.4"])

    assert table.loc[table["assay"] == "SMAC", "mechanistic_alignment"].iloc[0]
    assert table.loc[table["assay"] == "AC-SINS_pH7.4", "mechanistic_alignment"].iloc[0]


def test_residualize_assays_removes_simple_metadata_signal():
    df = pd.DataFrame(
        {
            "assay": [1.0, 1.2, 4.0, 4.2],
            "hc_subtype": ["A", "A", "B", "B"],
            "status": ["x", "x", "y", "y"],
        }
    )
    clusters = pd.Series([0, 0, 1, 1])
    controls = build_bias_control_features(df, clusters, ["hc_subtype", "status"])

    residuals, summary = residualize_assays(df, ["assay"], controls)

    assert summary.loc[0, "control_r2"] > 0.9
    assert abs(residuals["assay"].mean()) < 1e-9


def test_residual_control_summary_computes_empirical_p():
    observed = pd.DataFrame({"assay": ["a"], "model": ["charge"], "spearman": [0.5]})
    controls = pd.DataFrame(
        {
            "assay": ["a", "a", "a"],
            "model": ["shuffle", "shuffle", "shuffle"],
            "spearman": [0.1, 0.2, 0.6],
        }
    )

    summary = residual_control_summary(observed, controls)

    assert summary.loc[0, "observed_minus_null_mean"] > 0
    assert summary.loc[0, "empirical_p_greater"] == 0.5
