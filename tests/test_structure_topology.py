from pathlib import Path

import pandas as pd

from antinetwork.gdpa import MAIN_CSV
from antinetwork.structure_topology import (
    build_charge_topology_features,
    build_structure_stability_features,
    charge_topology_features_for_structure,
    structure_stability_features_for_structure,
    topology_feature_groups,
)


def test_charge_topology_features_for_real_pdb_if_available():
    pdb_path = Path("data/raw/hf_snapshot/structures/adalimumab.pdb")
    if not pdb_path.exists():
        return

    df = pd.read_csv(Path("data/raw") / MAIN_CSV)
    row = df[df["antibody_name"] == "adalimumab"].iloc[0]
    features = charge_topology_features_for_structure(
        pdb_path,
        row["heavy_aligned_aho"],
        row["light_aligned_aho"],
    )

    assert features["structure_available"] == 1.0
    assert features["structure_residue_count"] > 100
    assert "pH7p4_largest_positive_patch_size" in features
    assert "pH6p0_charge_dipole_proxy" in features


def test_build_charge_topology_features_marks_missing_structure():
    df = pd.DataFrame(
        {
            "antibody_name": ["definitely_missing"],
            "heavy_aligned_aho": ["A" * 149],
            "light_aligned_aho": ["A" * 149],
        }
    )

    features = build_charge_topology_features(df, structure_dir="data/raw/hf_snapshot/structures")

    assert features.loc[0, "structure_available"] == 0.0


def test_structure_stability_features_for_real_pdb_if_available():
    pdb_path = Path("data/raw/hf_snapshot/structures/adalimumab.pdb")
    if not pdb_path.exists():
        return

    df = pd.read_csv(Path("data/raw") / MAIN_CSV)
    row = df[df["antibody_name"] == "adalimumab"].iloc[0]
    features = structure_stability_features_for_structure(
        pdb_path,
        row["heavy_aligned_aho"],
        row["light_aligned_aho"],
    )

    assert features["structure_available"] == 1.0
    assert features["interface_contact_count"] > 0
    assert features["interface_residue_count"] > 0
    assert "hydrophobic_largest_patch_size" in features
    assert "surface_charge_dipole_proxy" in features
    assert "cdr_h3_loop_entropy_proxy" in features
    assert "packing_worst10_mean_degree" in features
    assert "interface_graph_mean_degree" in features
    assert "interface_graph_diameter" in features
    assert "defect_buried_charged_residue_count" in features
    assert "defect_unsatisfied_polar_atom_proxy_count" in features
    assert "defect_h3_contact_density" in features
    assert "defect_interface_worst5_packing_mean_degree" in features


def test_build_structure_stability_features_marks_missing_structure():
    df = pd.DataFrame(
        {
            "antibody_name": ["definitely_missing"],
            "heavy_aligned_aho": ["A" * 149],
            "light_aligned_aho": ["A" * 149],
        }
    )

    features = build_structure_stability_features(
        df,
        structure_dir="data/raw/hf_snapshot/structures",
    )

    assert features.loc[0, "structure_available"] == 0.0


def test_topology_feature_groups_tags_patch_features():
    groups = topology_feature_groups(
        ["pH7p4_largest_positive_patch_size", "pH6p0_charge_dipole_proxy"]
    )

    assert groups.loc["pH7p4_largest_positive_patch_size", "topology_family"] == "patch"
    assert groups.loc["pH6p0_charge_dipole_proxy", "topology_family"] == "topology_asymmetry"
