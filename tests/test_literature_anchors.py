import pandas as pd

from antinetwork.literature_anchors import (
    LiteratureMappingCoefficients,
    build_literature_condition_features,
    calibration_rows,
    calibrate_literature_mapping,
    leave_one_paper_out_literature_fit,
    numeric_first,
    numeric_max,
    project_literature_conditions,
)
from antinetwork.patchy_network import network_regime_sweep


def _toy_anchors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "anchor_id": ["A1", "A2", "A3", "A4", "A5"],
            "paper_key": ["p1", "p1", "p2", "p2", "theory"],
            "system_id": ["s1", "s1", "s2", "s2", "s3"],
            "antibody_label": ["m1", "m1", "m2", "m2", "model"],
            "condition_id": ["low", "high", "mitigated", "clustered", "context"],
            "pH": ["6.0", "7.0", "6.0", "6.0", "not_used"],
            "buffer": ["histidine", "histidine", "citrate", "citrate", "not_used"],
            "salt_type": ["NaCl", "NaCl", "NaCl", "NaCl", "not_used"],
            "salt_mM": ["150 mM", "30 mM", "150 mM", "10 mM", "not_used"],
            "additive_type": ["arginine", "none", "arginine", "none", "none"],
            "additive_concentration": ["100 mM", "none", "150 mM", "none", "none"],
            "protein_concentration_mg_ml": ["60 for viscosity", "200 mg/mL", "200", "200", ""],
            "temperature_C": ["25", "25", "25", "25", ""],
            "reported_mechanism": [
                "charge-mediated Fab-Fc RSA",
                "charge-mediated Fab-Fc RSA",
                "hydrophobic short-range attraction",
                "Fab-Fab reversible self-association",
                "theory anchor",
            ],
            "self_association_measure": ["low", "high", "low", "clustered", ""],
            "network_regime_target": [
                "dispersed_fluid",
                "high_viscosity_charge_mediated_RSA",
                "dispersed_fluid",
                "finite_reversible_clusters",
                "theory_anchor_not_used_for_numeric_fit",
            ],
            "target_score_0_1_2": [0.0, 2.0, 0.0, 1.0, None],
            "confidence": ["high", "high", "high", "high", "high"],
            "use_for": ["fit", "fit", "fit", "fit", "context"],
        }
    )


def _toy_regime_map() -> pd.DataFrame:
    return network_regime_sweep(
        attraction_values=[0.0, 1.4, 2.8],
        repulsion_values=[0.0, 0.8],
        patch_valences=[1, 3, 5],
        n_particles=12,
        n_steps=4,
        density=0.12,
        seed=9,
    )


def test_numeric_extractors_handle_curated_text_fields():
    assert numeric_first("~14.5 at 60 mg/mL") == 14.5
    assert numeric_max("60 for viscosity; 10 for DLS") == 60.0


def test_build_literature_condition_features_filters_theory_rows():
    rows = calibration_rows(_toy_anchors())
    features = build_literature_condition_features(rows, include_mechanism=True)

    assert len(rows) == 4
    assert "charge_mechanism_signal" in features.columns
    assert features["arginine_or_excipient"].sum() == 2.0


def test_project_literature_conditions_returns_network_risk():
    rows = calibration_rows(_toy_anchors())
    features = build_literature_condition_features(rows)
    coefficients = LiteratureMappingCoefficients(
        concentration_to_attraction=1.0,
        salt_screening_to_attraction=1.0,
        additive_to_mitigation=0.5,
        salt_to_repulsion=0.5,
    )

    projected = project_literature_conditions(features, _toy_regime_map(), coefficients)

    assert len(projected) == len(rows)
    assert "network_risk_score" in projected.columns
    assert projected["mapped_patch_valence"].isin([1, 3, 5]).all()


def test_calibrate_and_leave_one_paper_out_runs_on_toy_anchors():
    rows = calibration_rows(_toy_anchors())
    features = build_literature_condition_features(rows, include_mechanism=True)
    grid = pd.DataFrame(
        {
            "concentration_to_attraction": [0.0, 1.0],
            "salt_screening_to_attraction": [0.5, 1.0],
            "additive_to_mitigation": [0.0, 0.5],
            "salt_to_repulsion": [0.0, 0.5],
            "mechanism_to_attraction": [0.0, 1.0],
            "hydrophobic_to_attraction": [0.0, 0.5],
            "mechanism_to_valence": [0.0, 1.0],
        }
    )

    calibration = calibrate_literature_mapping(
        features,
        rows["target_score_0_1_2"],
        _toy_regime_map(),
        grid,
    )
    lopo = leave_one_paper_out_literature_fit(rows, features, _toy_regime_map(), grid)

    assert len(calibration) == len(grid)
    assert "training_spearman" in calibration.columns
    assert "heldout_spearman" in lopo.columns
