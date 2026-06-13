from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PatchyParticleParameters:
    """Minimal, falsifiable particle parameters derived from charge descriptors."""

    net_charge: float
    cdr_charge: float
    vh_vl_charge_imbalance: float
    cdr_hydrophobic_fraction: float
    fab_charge_strength: float
    core_charge_strength: float
    same_charge_repulsion: float
    hydrophobic_attraction: float
    contact_cutoff: float = 0.7
    patch_valence: int = 2


@dataclass(frozen=True)
class NetworkSimulationResult:
    frame_metrics: pd.DataFrame
    contact_lifetimes: pd.Series
    final_graph: nx.Graph


@dataclass(frozen=True)
class NetworkMappingCoefficients:
    """Global charge-to-regime coefficients fitted on one endpoint, then frozen."""

    cdr_charge_to_attraction: float
    hydrophobicity_to_attraction: float
    net_charge_to_repulsion: float


def charge_archetype_table(features: pd.DataFrame) -> pd.DataFrame:
    """Classify antibodies into crude interaction archetypes from sequence features."""
    required = {
        "fv_net_charge",
        "fv_cdr_net_charge",
        "vh_vl_charge_imbalance",
        "fv_cdr_hydrophobic_fraction",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Missing required charge feature columns: {missing}")

    rows = []
    imbalance_threshold = max(4.0, features["vh_vl_charge_imbalance"].quantile(0.75))
    hydrophobic_threshold = max(0.42, features["fv_cdr_hydrophobic_fraction"].quantile(0.75))

    for idx, row in features.iterrows():
        net = float(row["fv_net_charge"])
        cdr = float(row["fv_cdr_net_charge"])
        imbalance = float(row["vh_vl_charge_imbalance"])
        hydro = float(row["fv_cdr_hydrophobic_fraction"])

        if net >= 3:
            net_state = "globally_positive"
        elif net <= -3:
            net_state = "globally_negative"
        else:
            net_state = "balanced_charge"

        if cdr >= 2:
            cdr_state = "cdr_positive"
        elif cdr <= -2:
            cdr_state = "cdr_negative"
        else:
            cdr_state = "cdr_balanced"

        imbalance_state = (
            "high_charge_asymmetry" if imbalance >= imbalance_threshold else "low_charge_asymmetry"
        )
        hydrophobic_state = "hydrophobic_cdr" if hydro >= hydrophobic_threshold else "ordinary_cdr"
        archetype = "|".join([net_state, cdr_state, imbalance_state, hydrophobic_state])

        rows.append(
            {
                "net_charge_state": net_state,
                "cdr_charge_state": cdr_state,
                "vh_vl_imbalance_state": imbalance_state,
                "hydrophobic_state": hydrophobic_state,
                "charge_archetype": archetype,
            }
        )

    return pd.DataFrame(rows, index=features.index)


def parameters_from_feature_row(row: pd.Series) -> PatchyParticleParameters:
    """Map one antibody's charge descriptors onto a deliberately simple particle."""
    net_charge = float(row["fv_net_charge"])
    cdr_charge = float(row["fv_cdr_net_charge"])
    imbalance = float(row["vh_vl_charge_imbalance"])
    hydro = float(row["fv_cdr_hydrophobic_fraction"])

    charge_directionality = min((abs(cdr_charge) + 0.5 * imbalance) / 14.0, 1.0)
    net_repulsion = min(abs(net_charge) / 24.0, 1.0)
    hydro_strength = min(max((hydro - 0.28) / 0.28, 0.0), 1.0)
    patch_valence = int(np.clip(np.ceil(1 + charge_directionality + hydro_strength), 1, 4))

    return PatchyParticleParameters(
        net_charge=net_charge,
        cdr_charge=cdr_charge,
        vh_vl_charge_imbalance=imbalance,
        cdr_hydrophobic_fraction=hydro,
        fab_charge_strength=0.2 + 1.4 * charge_directionality,
        core_charge_strength=0.15 + 0.8 * min(abs(net_charge) / 18.0, 1.0),
        same_charge_repulsion=0.1 + 1.2 * net_repulsion,
        hydrophobic_attraction=0.05 + 1.1 * hydro_strength,
        patch_valence=patch_valence,
    )


def regime_parameters(
    attraction_strength: float,
    repulsion_strength: float,
    patch_valence: int,
    hydrophobic_attraction: float = 0.0,
    contact_cutoff: float = 0.7,
) -> PatchyParticleParameters:
    """Create interpretable simulator parameters for a soft-matter regime sweep."""
    if attraction_strength < 0:
        raise ValueError("attraction_strength must be non-negative.")
    if repulsion_strength < 0:
        raise ValueError("repulsion_strength must be non-negative.")
    if patch_valence < 1:
        raise ValueError("patch_valence must be at least 1.")

    return PatchyParticleParameters(
        net_charge=repulsion_strength,
        cdr_charge=1.0,
        vh_vl_charge_imbalance=float(patch_valence),
        cdr_hydrophobic_fraction=0.0,
        fab_charge_strength=float(attraction_strength),
        core_charge_strength=float(attraction_strength),
        same_charge_repulsion=float(repulsion_strength),
        hydrophobic_attraction=float(hydrophobic_attraction),
        contact_cutoff=float(contact_cutoff),
        patch_valence=int(patch_valence),
    )


def ionic_strength_screening(ionic_strength_mM: float, reference_mM: float = 50.0) -> float:
    """Return a simple Debye-like screening factor for ionic-strength perturbations."""
    if ionic_strength_mM < 0:
        raise ValueError("ionic_strength_mM must be non-negative.")
    if reference_mM <= 0:
        raise ValueError("reference_mM must be positive.")
    return float(np.sqrt(reference_mM / (reference_mM + ionic_strength_mM)))


def ph_charge_factor(ph: float, reference_ph: float = 7.4, slope: float = 0.12) -> float:
    """Crude protonation proxy: lower pH increases effective charge-patch strength."""
    return float(np.clip(1.0 + slope * (reference_ph - ph), 0.4, 1.8))


def apply_solution_conditions(
    params: PatchyParticleParameters,
    ph: float = 7.4,
    ionic_strength_mM: float = 50.0,
) -> PatchyParticleParameters:
    """Perturb patch parameters by pH and ionic strength.

    This is a v0 formulation axis, not a quantitative electrostatics model.
    Higher salt screens charge-mediated attraction and repulsion; lower pH
    increases charge-patch strength through a simple protonation proxy.
    """
    screening = ionic_strength_screening(ionic_strength_mM)
    ph_factor = ph_charge_factor(ph)
    attraction_factor = screening * ph_factor
    repulsion_factor = screening / ph_factor

    return PatchyParticleParameters(
        net_charge=params.net_charge,
        cdr_charge=params.cdr_charge,
        vh_vl_charge_imbalance=params.vh_vl_charge_imbalance,
        cdr_hydrophobic_fraction=params.cdr_hydrophobic_fraction,
        fab_charge_strength=params.fab_charge_strength * attraction_factor,
        core_charge_strength=params.core_charge_strength * attraction_factor,
        same_charge_repulsion=params.same_charge_repulsion * repulsion_factor,
        hydrophobic_attraction=params.hydrophobic_attraction,
        contact_cutoff=params.contact_cutoff,
        patch_valence=params.patch_valence,
    )


def _rotation_matrices(angles: np.ndarray) -> np.ndarray:
    cos = np.cos(angles)
    sin = np.sin(angles)
    matrices = np.empty((len(angles), 2, 2), dtype=float)
    matrices[:, 0, 0] = cos
    matrices[:, 0, 1] = -sin
    matrices[:, 1, 0] = sin
    matrices[:, 1, 1] = cos
    return matrices


def _sticky_patch_offsets(patch_valence: int) -> dict[str, np.ndarray]:
    if patch_valence == 1:
        return {"sticky_0": np.array([0.72, 0.0])}
    angles = np.linspace(-0.85, 0.85, patch_valence)
    return {
        f"sticky_{idx}": np.array([0.72 * np.cos(angle), 0.72 * np.sin(angle)])
        for idx, angle in enumerate(angles)
    }


def _patch_positions(
    positions: np.ndarray,
    angles: np.ndarray,
    params: PatchyParticleParameters,
) -> dict[str, np.ndarray]:
    base_offsets = {
        **_sticky_patch_offsets(params.patch_valence),
        "core": np.array([-0.38, 0.0]),
        "hydrophobic": np.array([0.12, 0.52]),
    }
    rotations = _rotation_matrices(angles)
    return {
        name: positions + np.einsum("nij,j->ni", rotations, offset)
        for name, offset in base_offsets.items()
    }


def _charge_signs(params: PatchyParticleParameters) -> dict[str, int]:
    sticky_sign = int(np.sign(params.cdr_charge))
    if sticky_sign == 0:
        sticky_sign = 1 if params.vh_vl_charge_imbalance >= 4 else 0
    core_sign = int(-np.sign(params.net_charge or sticky_sign))
    signs = {f"sticky_{idx}": sticky_sign for idx in range(params.patch_valence)}
    signs["core"] = core_sign
    return signs


def _contact_graph(
    positions: np.ndarray,
    angles: np.ndarray,
    params: PatchyParticleParameters,
    box_size: float,
) -> nx.Graph:
    patches = _patch_positions(positions, angles, params)
    charge_signs = _charge_signs(params)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(positions)))
    neighbor_cutoff = 2.0
    attraction_cutoff = params.contact_cutoff * (0.7 + 0.45 * min(params.fab_charge_strength, 2.0))
    repulsion_cutoff = params.contact_cutoff * (0.7 + 0.35 * min(params.same_charge_repulsion, 2.0))
    hydrophobic_cutoff = params.contact_cutoff * (
        0.7 + 0.45 * min(params.hydrophobic_attraction, 2.0)
    )

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            delta = positions[i] - positions[j]
            delta -= box_size * np.round(delta / box_size)
            if np.linalg.norm(delta) > neighbor_cutoff:
                continue

            complement_score = 0.0
            repulsion_score = 0.0
            for patch_i, sign_i in charge_signs.items():
                if sign_i == 0:
                    continue
                for patch_j, sign_j in charge_signs.items():
                    if sign_j == 0:
                        continue
                    distance = np.linalg.norm(patches[patch_i][i] - patches[patch_j][j])
                    if sign_i * sign_j < 0:
                        if distance > attraction_cutoff:
                            continue
                        closeness = 1.0 - distance / attraction_cutoff
                        complement_score = max(complement_score, closeness)
                    else:
                        if distance > repulsion_cutoff:
                            continue
                        closeness = 1.0 - distance / repulsion_cutoff
                        repulsion_score = max(repulsion_score, closeness)

            hydro_distance = np.linalg.norm(patches["hydrophobic"][i] - patches["hydrophobic"][j])
            hydro_score = (
                1.0 - hydro_distance / hydrophobic_cutoff
                if hydro_distance <= hydrophobic_cutoff
                else 0.0
            )

            attractive_weight = (
                params.fab_charge_strength * complement_score
                + params.hydrophobic_attraction * hydro_score
            )
            repulsive_weight = params.same_charge_repulsion * repulsion_score
            if attractive_weight <= 0 and repulsive_weight <= 0:
                continue

            if attractive_weight > 0 and repulsive_weight > 0:
                edge_type = "mixed"
            elif hydro_score > 0 and complement_score == 0:
                edge_type = "hydrophobic"
            elif complement_score > 0:
                edge_type = "charge_complementary"
            else:
                edge_type = "same_charge_repulsive"

            graph.add_edge(
                i,
                j,
                edge_type=edge_type,
                weight=float(attractive_weight - repulsive_weight),
                attractive_weight=float(attractive_weight),
                repulsive_weight=float(repulsive_weight),
            )
    return graph


def _positive_subgraph(graph: nx.Graph) -> nx.Graph:
    positive_edges = [
        (u, v)
        for u, v, attrs in graph.edges(data=True)
        if attrs.get("attractive_weight", 0.0) > attrs.get("repulsive_weight", 0.0)
    ]
    subgraph = nx.Graph()
    subgraph.add_nodes_from(graph.nodes)
    subgraph.add_edges_from(positive_edges)
    return subgraph


def graph_observables(graph: nx.Graph) -> dict[str, float]:
    """Summarize contact-network regime from one graph frame."""
    n_nodes = graph.number_of_nodes()
    positive = _positive_subgraph(graph)
    degrees = np.array([degree for _, degree in positive.degree()], dtype=float)
    components = list(nx.connected_components(positive))
    largest = max((len(component) for component in components), default=0)
    edge_types = nx.get_edge_attributes(graph, "edge_type")
    repulsive_edges = sum(edge_type == "same_charge_repulsive" for edge_type in edge_types.values())

    return {
        "n_edges": float(graph.number_of_edges()),
        "attractive_edges": float(positive.number_of_edges()),
        "mean_degree": float(degrees.mean()) if len(degrees) else 0.0,
        "largest_cluster_fraction": float(largest / max(n_nodes, 1)),
        "degree_cv": float(degrees.std() / degrees.mean()) if degrees.mean() > 0 else 0.0,
        "repulsive_edge_fraction": float(repulsive_edges / max(graph.number_of_edges(), 1)),
        "percolated": float(largest / max(n_nodes, 1) >= 0.5),
    }


def simulate_patchy_network(
    params: PatchyParticleParameters,
    n_particles: int = 150,
    n_steps: int = 100,
    box_size: float | None = None,
    density: float = 0.08,
    brownian_step: float = 0.18,
    rotational_step: float = 0.25,
    seed: int = 7,
) -> NetworkSimulationResult:
    """Simulate transient contacts among identical coarse-grained antibodies.

    This is a minimal physical hypothesis test, not an antibody solution
    predictor. It asks whether charge-derived patch rules can generate distinct
    network observables.
    """
    if n_particles <= 1:
        raise ValueError("n_particles must be greater than 1.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if density <= 0:
        raise ValueError("density must be positive.")

    rng = np.random.default_rng(seed)
    box_size = float(box_size or np.sqrt(n_particles / density))
    positions = rng.uniform(0, box_size, size=(n_particles, 2))
    angles = rng.uniform(-np.pi, np.pi, size=n_particles)

    previous_positive_edges: set[tuple[int, int]] = set()
    active_lifetimes: dict[tuple[int, int], int] = {}
    completed_lifetimes: list[int] = []
    metrics = []
    final_graph = nx.Graph()

    for step in range(n_steps):
        positions = (positions + rng.normal(0.0, brownian_step, size=positions.shape)) % box_size
        angles = angles + rng.normal(0.0, rotational_step, size=n_particles)
        graph = _contact_graph(positions, angles, params, box_size)
        positive = _positive_subgraph(graph)
        positive_edges = {tuple(sorted(edge)) for edge in positive.edges()}

        ended = previous_positive_edges - positive_edges
        for edge in ended:
            completed_lifetimes.append(active_lifetimes.pop(edge, 1))
        for edge in positive_edges:
            active_lifetimes[edge] = active_lifetimes.get(edge, 0) + 1

        union = previous_positive_edges | positive_edges
        turnover = (
            1.0 - len(previous_positive_edges & positive_edges) / len(union)
            if union
            else 0.0
        )

        row = graph_observables(graph)
        row["step"] = float(step)
        row["edge_turnover"] = float(turnover)
        metrics.append(row)

        previous_positive_edges = positive_edges
        final_graph = graph

    completed_lifetimes.extend(active_lifetimes.values())
    return NetworkSimulationResult(
        frame_metrics=pd.DataFrame(metrics),
        contact_lifetimes=pd.Series(completed_lifetimes, dtype=float, name="contact_lifetime"),
        final_graph=final_graph,
    )


def aggregate_simulation_observables(result: NetworkSimulationResult) -> pd.Series:
    """Aggregate frame-level metrics into one descriptor vector."""
    metrics = result.frame_metrics.drop(columns=["step"], errors="ignore")
    summary = {}
    for column in metrics.columns:
        summary[f"{column}_mean"] = float(metrics[column].mean())
        summary[f"{column}_max"] = float(metrics[column].max())
    summary["cluster_lifetime_mean"] = float(result.contact_lifetimes.mean())
    summary["cluster_lifetime_p90"] = float(result.contact_lifetimes.quantile(0.9))
    return pd.Series(summary)


def classify_regime(row: pd.Series) -> str:
    """Assign a coarse physical regime from aggregate network observables."""
    largest = float(row.get("largest_cluster_fraction_mean", 0.0))
    percolated = float(row.get("percolated_mean", 0.0))
    degree = float(row.get("mean_degree_mean", 0.0))
    lifetime = float(row.get("cluster_lifetime_mean", 0.0))

    if percolated >= 0.35 or largest >= 0.5:
        return "percolated_network"
    if largest >= 0.18 or degree >= 1.0 or lifetime >= 2.5:
        return "finite_reversible_clusters"
    return "dispersed_fluid"


def network_regime_sweep(
    attraction_values: list[float] | np.ndarray,
    repulsion_values: list[float] | np.ndarray,
    patch_valences: list[int] | np.ndarray,
    n_particles: int = 80,
    n_steps: int = 40,
    density: float = 0.04,
    seed: int = 7,
) -> pd.DataFrame:
    """Sweep attraction, repulsion, and valence to test simulator regime diversity."""
    rows = []
    run_idx = 0
    for patch_valence in patch_valences:
        for attraction in attraction_values:
            for repulsion in repulsion_values:
                params = regime_parameters(
                    attraction_strength=float(attraction),
                    repulsion_strength=float(repulsion),
                    patch_valence=int(patch_valence),
                )
                result = simulate_patchy_network(
                    params,
                    n_particles=n_particles,
                    n_steps=n_steps,
                    density=density,
                    seed=seed + run_idx,
                )
                observable = aggregate_simulation_observables(result)
                observable["attraction_strength"] = float(attraction)
                observable["repulsion_strength"] = float(repulsion)
                observable["patch_valence"] = int(patch_valence)
                observable["regime"] = classify_regime(observable)
                rows.append(observable)
                run_idx += 1
    return pd.DataFrame(rows)


def archetype_representative_features(features: pd.DataFrame) -> pd.DataFrame:
    """Return one medoid-like feature row per charge archetype."""
    archetypes = charge_archetype_table(features)
    rows = []
    coordinate_cols = [
        "fv_net_charge",
        "fv_cdr_net_charge",
        "vh_vl_charge_imbalance",
        "fv_cdr_hydrophobic_fraction",
    ]
    for archetype, members in archetypes.groupby("charge_archetype"):
        member_features = features.loc[members.index]
        center = member_features[coordinate_cols].median()
        distances = member_features[coordinate_cols].sub(center).abs().sum(axis=1)
        representative_idx = distances.idxmin()
        row = features.loc[representative_idx, coordinate_cols].copy()
        row["charge_archetype"] = archetype
        row["representative_index"] = representative_idx
        row["n_antibodies_in_archetype"] = float(len(member_features))
        rows.append(row)
    return pd.DataFrame(rows).set_index("charge_archetype").sort_index()


def calibrated_mapping_coordinates(
    archetype_features: pd.DataFrame,
    coefficients: NetworkMappingCoefficients,
) -> pd.DataFrame:
    """Map archetype features to attraction, repulsion, and fixed valence bins."""
    required = {
        "fv_net_charge",
        "fv_cdr_net_charge",
        "vh_vl_charge_imbalance",
        "fv_cdr_hydrophobic_fraction",
    }
    missing = sorted(required - set(archetype_features.columns))
    if missing:
        raise ValueError(f"Missing required archetype feature columns: {missing}")

    out = archetype_features.copy()
    cdr_scale = max(float(out["fv_cdr_net_charge"].abs().max()), 1.0)
    net_scale = max(float(out["fv_net_charge"].abs().max()), 1.0)
    hydro_signal = ((out["fv_cdr_hydrophobic_fraction"] - 0.28) / 0.28).clip(lower=0.0)

    out["attraction_strength"] = (
        coefficients.cdr_charge_to_attraction * out["fv_cdr_net_charge"].abs() / cdr_scale
        + coefficients.hydrophobicity_to_attraction * hydro_signal
    )
    out["repulsion_strength"] = (
        coefficients.net_charge_to_repulsion * out["fv_net_charge"].abs() / net_scale
    )

    imbalance = out["vh_vl_charge_imbalance"]
    low, high = imbalance.quantile([0.33, 0.66])
    out["patch_valence"] = np.select(
        [imbalance <= low, imbalance <= high],
        [1, 3],
        default=5,
    ).astype(int)
    return out


def project_archetypes_to_regime_map(
    archetype_features: pd.DataFrame,
    regime_map: pd.DataFrame,
    coefficients: NetworkMappingCoefficients,
) -> pd.DataFrame:
    """Project calibrated archetype coordinates onto the nearest simulated regime point."""
    coordinates = calibrated_mapping_coordinates(archetype_features, coefficients)
    regime_required = {
        "attraction_strength",
        "repulsion_strength",
        "patch_valence",
        "largest_cluster_fraction_mean",
        "percolated_mean",
        "mean_degree_mean",
        "edge_turnover_mean",
    }
    missing = sorted(regime_required - set(regime_map.columns))
    if missing:
        raise ValueError(f"Missing required regime map columns: {missing}")

    rows = []
    for archetype, row in coordinates.iterrows():
        nearest = regime_map.assign(
            distance=(regime_map["attraction_strength"] - row["attraction_strength"]).abs()
            + (regime_map["repulsion_strength"] - row["repulsion_strength"]).abs()
            + 0.5 * (regime_map["patch_valence"] - row["patch_valence"]).abs()
        ).sort_values("distance").iloc[0]
        projected = row.to_dict()
        projected.update(
            {
                "mapped_attraction_strength": float(nearest["attraction_strength"]),
                "mapped_repulsion_strength": float(nearest["repulsion_strength"]),
                "mapped_patch_valence": int(nearest["patch_valence"]),
                "mapped_regime": nearest.get("regime", "unclassified"),
                "mapped_largest_cluster_fraction": float(nearest["largest_cluster_fraction_mean"]),
                "mapped_percolation_probability": float(nearest["percolated_mean"]),
                "mapped_mean_degree": float(nearest["mean_degree_mean"]),
                "mapped_edge_turnover": float(nearest["edge_turnover_mean"]),
            }
        )
        rows.append(pd.Series(projected, name=archetype))

    projected = pd.DataFrame(rows)
    projected["network_risk_score"] = (
        projected["mapped_largest_cluster_fraction"]
        + projected["mapped_percolation_probability"]
        + 0.25 * projected["mapped_mean_degree"]
        - 0.25 * projected["mapped_edge_turnover"]
    )
    return projected


def _spearman(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 4 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(valid.iloc[:, 0].rank().corr(valid.iloc[:, 1].rank()))


def calibrate_global_mapping(
    archetype_features: pd.DataFrame,
    regime_map: pd.DataFrame,
    target: pd.Series,
    coefficient_grid: pd.DataFrame,
) -> pd.DataFrame:
    """Fit global mapping coefficients by target rank alignment on one endpoint."""
    rows = []
    for _, candidate in coefficient_grid.iterrows():
        coefficients = NetworkMappingCoefficients(
            cdr_charge_to_attraction=float(candidate["cdr_charge_to_attraction"]),
            hydrophobicity_to_attraction=float(candidate["hydrophobicity_to_attraction"]),
            net_charge_to_repulsion=float(candidate["net_charge_to_repulsion"]),
        )
        projected = project_archetypes_to_regime_map(archetype_features, regime_map, coefficients)
        rho = _spearman(projected["network_risk_score"], target)
        rows.append(
            {
                **candidate.to_dict(),
                "training_spearman": rho,
                "training_abs_spearman": abs(rho) if pd.notna(rho) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("training_abs_spearman", ascending=False)


def simulate_representative_archetypes(
    features: pd.DataFrame,
    n_particles: int = 150,
    n_steps: int = 100,
    density: float = 0.08,
    seed: int = 7,
) -> pd.DataFrame:
    """Simulate the medoid-like member of each charge archetype."""
    archetypes = charge_archetype_table(features)
    rows = []
    for archetype, members in archetypes.groupby("charge_archetype"):
        member_features = features.loc[members.index]
        center = member_features[
            [
                "fv_net_charge",
                "fv_cdr_net_charge",
                "vh_vl_charge_imbalance",
                "fv_cdr_hydrophobic_fraction",
            ]
        ].median()
        distances = (
            member_features[
                [
                    "fv_net_charge",
                    "fv_cdr_net_charge",
                    "vh_vl_charge_imbalance",
                    "fv_cdr_hydrophobic_fraction",
                ]
            ]
            .sub(center)
            .abs()
            .sum(axis=1)
        )
        representative_idx = distances.idxmin()
        params = parameters_from_feature_row(features.loc[representative_idx])
        result = simulate_patchy_network(
            params,
            n_particles=n_particles,
            n_steps=n_steps,
            density=density,
            seed=seed,
        )
        observable = aggregate_simulation_observables(result)
        observable["charge_archetype"] = archetype
        observable["representative_index"] = representative_idx
        observable["n_antibodies_in_archetype"] = float(len(member_features))
        rows.append(observable)
    return pd.DataFrame(rows).set_index("charge_archetype").sort_index()
