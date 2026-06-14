import numpy as np
import pandas as pd

from antinetwork.topology_falsification import (
    positive_control_topology_signal,
    scramble_surface_chemistry,
    scramble_surface_chemistry_with_diagnostics,
    summarize_gate_metrics,
    summarize_positive_controls,
    summarize_scramble_diagnostics,
    surface_composition_features,
    topology_features,
)


def _toy_surface() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "resname": ["LYS", "ASP", "PHE", "TYR", "ARG", "GLU"],
            "aa": ["K", "D", "F", "Y", "R", "E"],
            "x": [0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "is_cdr": [True, True, True, False, False, False],
            "region_bin": ["cdr", "cdr", "cdr", "framework", "framework", "framework"],
        }
    )


def test_region_preserving_scramble_keeps_surface_composition_by_region():
    residues = _toy_surface()
    rng = np.random.default_rng(1)

    scrambled = scramble_surface_chemistry(residues, rng)

    original = surface_composition_features(residues)
    null = surface_composition_features(scrambled)
    assert original == null
    assert scrambled.loc[scrambled["region_bin"] == "cdr", "aa"].isin(["K", "D", "F"]).all()
    assert scrambled.loc[scrambled["region_bin"] == "framework", "aa"].isin(["Y", "R", "E"]).all()
    assert (scrambled.loc[scrambled["region_bin"] == "cdr", "aa"].to_numpy() != ["K", "D", "F"]).all()
    assert (
        scrambled.loc[scrambled["region_bin"] == "framework", "aa"].to_numpy() != ["Y", "R", "E"]
    ).all()


def test_scramble_diagnostics_report_identity_movement_and_tiny_bins():
    residues = _toy_surface()
    residues.loc[0, "region_bin"] = "tiny"
    rng = np.random.default_rng(1)

    _, diagnostics = scramble_surface_chemistry_with_diagnostics(residues, rng)
    summary = summarize_scramble_diagnostics(
        diagnostics.assign(
            antibody_index=0,
            real_scrambled_feature_correlation_mean=0.25,
        )
    )

    assert diagnostics.loc[diagnostics["region_bin"] == "tiny", "non_derangeable_bin"].iloc[0]
    assert diagnostics["identity_moved_fraction"].dropna().between(0, 1).all()
    assert "mean fraction of residues moved" in summary["diagnostic"].tolist()
    assert "mean real-vs-scrambled feature correlation" in summary["diagnostic"].tolist()


def test_topology_features_change_when_chemistry_moves_on_fixed_graph():
    residues = _toy_surface()
    scrambled = residues.copy()
    scrambled["resname"] = ["LYS", "PHE", "ASP", "TYR", "ARG", "GLU"]
    scrambled["aa"] = ["K", "F", "D", "Y", "R", "E"]

    real = topology_features(residues, distance_cutoff=1.5)
    moved = topology_features(scrambled, distance_cutoff=1.5)

    assert real["hydrophobic_positive_adjacent_edge_count"] != moved[
        "hydrophobic_positive_adjacent_edge_count"
    ]


def test_gate_verdict_requires_real_to_beat_scrambled_and_surface():
    metrics = pd.DataFrame(
        [
            {"assay": "HIC", "feature_block": "global_composition", "n": 120, "spearman": 0.2},
            {"assay": "HIC", "feature_block": "surface_composition", "n": 120, "spearman": 0.24},
            {"assay": "HIC", "feature_block": "real_topology", "n": 120, "spearman": 0.34},
            {
                "assay": "HIC",
                "feature_block": "scrambled_topology_000",
                "n": 120,
                "spearman": 0.25,
            },
            {
                "assay": "HIC",
                "feature_block": "scrambled_topology_001",
                "n": 120,
                "spearman": 0.26,
            },
            {"assay": "HIC", "feature_block": "label_shuffle_000", "n": 120, "spearman": 0.01},
            {"assay": "HIC", "feature_block": "label_shuffle_001", "n": 120, "spearman": -0.02},
            {"assay": "SMAC", "feature_block": "global_composition", "n": 120, "spearman": 0.2},
            {"assay": "SMAC", "feature_block": "surface_composition", "n": 120, "spearman": 0.3},
            {"assay": "SMAC", "feature_block": "real_topology", "n": 120, "spearman": 0.31},
            {
                "assay": "SMAC",
                "feature_block": "scrambled_topology_000",
                "n": 120,
                "spearman": 0.3,
            },
            {
                "assay": "SMAC",
                "feature_block": "scrambled_topology_001",
                "n": 120,
                "spearman": 0.31,
            },
            {"assay": "SMAC", "feature_block": "label_shuffle_000", "n": 120, "spearman": 0.0},
        ]
    )

    verdicts = summarize_gate_metrics(metrics)

    assert verdicts.loc[verdicts["assay"] == "HIC", "verdict"].iloc[0] == "PASS"
    assert verdicts.loc[verdicts["assay"] == "SMAC", "verdict"].iloc[0] == "FAIL"


def test_positive_control_summary_predeclares_expected_behaviors():
    metrics = pd.DataFrame(
        [
            {
                "assay": "positive_control_composition",
                "feature_block": "surface_composition",
                "spearman": 0.95,
            },
            {
                "assay": "positive_control_composition",
                "feature_block": "real_topology",
                "spearman": 0.93,
            },
            {
                "assay": "positive_control_composition",
                "feature_block": "scrambled_topology_000",
                "spearman": 0.92,
            },
            {
                "assay": "positive_control_topology",
                "feature_block": "surface_composition",
                "spearman": 0.05,
            },
            {
                "assay": "positive_control_topology",
                "feature_block": "real_topology",
                "spearman": 0.9,
            },
            {
                "assay": "positive_control_topology",
                "feature_block": "scrambled_topology_000",
                "spearman": 0.2,
            },
            {
                "assay": "positive_control_shuffled",
                "feature_block": "surface_composition",
                "spearman": 0.03,
            },
            {
                "assay": "positive_control_shuffled",
                "feature_block": "real_topology",
                "spearman": -0.04,
            },
            {
                "assay": "positive_control_shuffled",
                "feature_block": "scrambled_topology_000",
                "spearman": 0.02,
            },
        ]
    )

    summary = summarize_positive_controls(metrics)

    assert summary["verdict"].tolist() == ["PASS", "PASS", "PASS"]


def test_positive_control_topology_signal_detects_synthetic_topology():
    n = 180
    rng = np.random.default_rng(3)
    index = pd.RangeIndex(n)
    composition = rng.normal(size=n)
    topology = rng.normal(size=n)
    noise = rng.normal(scale=0.05, size=n)
    groups = pd.Series(np.repeat(np.arange(30), 6), index=index)
    feature_blocks = {
        "surface_composition": pd.DataFrame(
            {
                "surface_cdr_hydrophobic_fraction": composition,
                "surface_all_hydrophobic_fraction": composition + noise,
            },
            index=index,
        ),
        "real_topology": pd.DataFrame(
            {
                "surface_cdr_hydrophobic_fraction": composition,
                "hydrophobic_largest_patch_size": topology,
                "hydrophobic_positive_adjacent_edge_count": topology + noise,
            },
            index=index,
        ),
        "scrambled_topology_000": pd.DataFrame(
            {
                "hydrophobic_largest_patch_size": rng.normal(size=n),
                "hydrophobic_positive_adjacent_edge_count": rng.normal(size=n),
            },
            index=index,
        ),
        "scrambled_topology_001": pd.DataFrame(
            {
                "hydrophobic_largest_patch_size": rng.normal(size=n),
                "hydrophobic_positive_adjacent_edge_count": rng.normal(size=n),
            },
            index=index,
        ),
    }

    result = positive_control_topology_signal(
        feature_blocks,
        groups=groups,
        random_state=3,
    )
    summary = result.summary.set_index("test")

    assert summary.loc["positive_control_composition", "verdict"] == "PASS"
    assert summary.loc["positive_control_topology", "verdict"] == "PASS"
    assert summary.loc["positive_control_shuffled", "verdict"] == "PASS"
