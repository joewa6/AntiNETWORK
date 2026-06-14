import numpy as np
import pandas as pd

from antinetwork.patchy_network import (
    NetworkMappingCoefficients,
    aggregate_simulation_observables,
    apply_solution_conditions,
    archetype_representative_features,
    calibrate_global_mapping,
    charge_archetype_table,
    ionic_strength_screening,
    network_regime_sweep,
    parameters_from_feature_row,
    ph_charge_factor,
    project_archetypes_to_regime_map,
    regime_parameters,
    simulate_patchy_network,
    simulate_representative_archetypes,
)


def _toy_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fv_net_charge": [0.0, 8.0, -7.0, 5.0],
            "fv_cdr_net_charge": [0.0, 4.0, -4.0, 1.0],
            "vh_vl_charge_imbalance": [1.0, 7.0, 6.0, 2.0],
            "fv_cdr_hydrophobic_fraction": [0.30, 0.50, 0.35, 0.48],
        }
    )


def test_charge_archetype_table_labels_expected_states():
    archetypes = charge_archetype_table(_toy_features())

    assert archetypes.loc[0, "net_charge_state"] == "balanced_charge"
    assert archetypes.loc[1, "cdr_charge_state"] == "cdr_positive"
    assert archetypes.loc[2, "cdr_charge_state"] == "cdr_negative"
    assert "high_charge_asymmetry" in archetypes.loc[1, "charge_archetype"]


def test_simulate_patchy_network_is_seed_deterministic():
    params = parameters_from_feature_row(_toy_features().loc[1])

    first = simulate_patchy_network(params, n_particles=18, n_steps=8, seed=123)
    second = simulate_patchy_network(params, n_particles=18, n_steps=8, seed=123)
    first_summary = aggregate_simulation_observables(first)
    second_summary = aggregate_simulation_observables(second)

    assert first.frame_metrics.shape[0] == 8
    assert first.final_graph.number_of_nodes() == 18
    assert np.allclose(first_summary, second_summary)
    assert "largest_cluster_fraction_mean" in first_summary.index
    assert params.patch_valence >= 2


def test_regime_parameters_support_patch_valence():
    params = regime_parameters(
        attraction_strength=1.2,
        repulsion_strength=0.4,
        patch_valence=4,
    )
    result = simulate_patchy_network(params, n_particles=14, n_steps=4, density=0.03, seed=8)

    assert params.patch_valence == 4
    assert result.final_graph.number_of_nodes() == 14


def test_solution_condition_perturbations_are_directional():
    params = regime_parameters(
        attraction_strength=1.2,
        repulsion_strength=0.8,
        patch_valence=3,
    )
    low_salt = apply_solution_conditions(params, ph=7.4, ionic_strength_mM=10)
    high_salt = apply_solution_conditions(params, ph=7.4, ionic_strength_mM=200)
    low_ph = apply_solution_conditions(params, ph=6.0, ionic_strength_mM=50)

    assert ionic_strength_screening(10) > ionic_strength_screening(200)
    assert ph_charge_factor(6.0) > ph_charge_factor(7.4)
    assert low_salt.fab_charge_strength > high_salt.fab_charge_strength
    assert low_ph.fab_charge_strength > params.fab_charge_strength * ionic_strength_screening(50)


def test_network_regime_sweep_returns_grid_and_regime_labels():
    sweep = network_regime_sweep(
        attraction_values=[0.0, 1.4],
        repulsion_values=[0.0, 1.0],
        patch_valences=[1, 3],
        n_particles=12,
        n_steps=4,
        density=0.03,
        seed=9,
    )

    assert len(sweep) == 8
    assert {"attraction_strength", "repulsion_strength", "patch_valence", "regime"}.issubset(
        sweep.columns
    )
    assert sweep["regime"].notna().all()


def test_project_archetypes_to_regime_map_returns_fixed_network_risk():
    features = _toy_features()
    representatives = archetype_representative_features(features)
    regime_map = network_regime_sweep(
        attraction_values=[0.0, 1.4],
        repulsion_values=[0.0, 1.0],
        patch_valences=[1, 3, 5],
        n_particles=12,
        n_steps=4,
        density=0.03,
        seed=9,
    )
    coefficients = NetworkMappingCoefficients(
        cdr_charge_to_attraction=1.0,
        hydrophobicity_to_attraction=0.5,
        net_charge_to_repulsion=1.0,
    )

    projected = project_archetypes_to_regime_map(representatives, regime_map, coefficients)

    assert len(projected) == len(representatives)
    assert "network_risk_score" in projected.columns
    assert projected["mapped_patch_valence"].isin([1, 3, 5]).all()


def test_calibrate_global_mapping_scores_candidate_coefficients():
    features = _toy_features()
    representatives = archetype_representative_features(features)
    regime_map = network_regime_sweep(
        attraction_values=[0.0, 1.4],
        repulsion_values=[0.0, 1.0],
        patch_valences=[1, 3, 5],
        n_particles=12,
        n_steps=4,
        density=0.03,
        seed=9,
    )
    grid = pd.DataFrame(
        {
            "cdr_charge_to_attraction": [0.5, 1.0],
            "hydrophobicity_to_attraction": [0.0, 0.5],
            "net_charge_to_repulsion": [0.5, 1.0],
        }
    )
    target = pd.Series(np.arange(len(representatives)), index=representatives.index)

    calibration = calibrate_global_mapping(representatives, regime_map, target, grid)

    assert len(calibration) == len(grid)
    assert "training_spearman" in calibration.columns


def test_simulate_representative_archetypes_returns_network_observables():
    observables = simulate_representative_archetypes(
        _toy_features(),
        n_particles=12,
        n_steps=5,
        seed=7,
    )

    assert "mean_degree_mean" in observables.columns
    assert "n_antibodies_in_archetype" in observables.columns
    assert observables["n_antibodies_in_archetype"].sum() == 4.0
