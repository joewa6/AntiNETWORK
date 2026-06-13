import pandas as pd

from scripts.run_tm_structure_tier_audit import build_tm_tier_feature_sets
from scripts.run_tm_structure_tier_audit import _add_subtype_residual_assay


def test_build_tm_tier_feature_sets_splits_mechanism_blocks():
    index = pd.Index([0, 1])
    physics = pd.DataFrame({"vh_net_charge": [1.0, 2.0]}, index=index)
    structure = pd.DataFrame(
        {
            "interface_contact_count": [10.0, 11.0],
            "interface_graph_mean_degree": [1.0, 2.0],
            "hydrophobic_largest_patch_size": [4.0, 5.0],
            "aromatic_largest_patch_size": [2.0, 3.0],
            "positive_largest_patch_size": [3.0, 4.0],
            "surface_charge_dipole_proxy": [0.2, 0.3],
            "cdr_h3_length": [12.0, 13.0],
            "cdr_h3_loop_entropy_proxy": [0.1, 0.2],
            "packing_worst10_mean_degree": [5.0, 6.0],
            "defect_buried_charged_residue_count": [1.0, 2.0],
            "defect_h3_contact_density": [7.0, 8.0],
        },
        index=index,
    )
    metadata = pd.DataFrame({"hc_subtype": ["IgG1", "IgG4"], "lc_subtype": ["Kappa", "Lambda"]}, index=index)

    feature_sets = build_tm_tier_feature_sets(physics, structure, metadata)

    assert "tier1_interface" in feature_sets
    assert "tier1_hydrophobicity" in feature_sets
    assert "tier1_electrostatics" in feature_sets
    assert "tier2_frustration" in feature_sets
    assert "tier2_defects" in feature_sets
    assert "tier3_interface_graph" in feature_sets
    assert "sequence_plus_structure_all" in feature_sets
    assert "interface_contact_count" in feature_sets["tier1_interface"]
    assert "interface_graph_mean_degree" in feature_sets["tier3_interface_graph"]
    assert "defect_buried_charged_residue_count" in feature_sets["tier2_defects"]


def test_add_subtype_residual_assay_uses_train_subtype_means():
    train = pd.DataFrame(
        {
            "Tm2": [10.0, 12.0, 20.0, 22.0],
            "hc_subtype": ["A", "A", "B", "B"],
            "lc_subtype": ["K", "K", "L", "L"],
        }
    )
    test = pd.DataFrame(
        {
            "Tm2": [13.0, 23.0, 99.0],
            "hc_subtype": ["A", "B", "C"],
            "lc_subtype": ["K", "L", "K"],
        }
    )

    _add_subtype_residual_assay(train, test, assay="Tm2")

    assert train["Tm2_subtype_residual"].round(6).tolist() == [-1.0, 1.0, -1.0, 1.0]
    assert test["Tm2_subtype_residual"].round(6).tolist() == [2.0, 2.0, 83.0]
