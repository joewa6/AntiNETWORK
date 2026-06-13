from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from antinetwork.features import AROMATIC, HYDROPHOBIC, build_physical_features
from antinetwork.killtest import make_regression_pipeline, metric_row
from antinetwork.structure_topology import (
    NEGATIVE_RESIDUES,
    POSITIVE_RESIDUES,
    parse_structure_residues,
    structure_path_for_antibody,
    surface_mask,
)

DEFAULT_GATE_ASSAYS = [
    "HIC",
    "SMAC",
    "AC-SINS_pH6.0",
    "AC-SINS_pH7.4",
    "PR_CHO",
    "PR_Ova",
    "HAC",
]

TOPOLOGY_PROPERTIES = {
    "positive": lambda frame: frame["resname"].isin(POSITIVE_RESIDUES),
    "negative": lambda frame: frame["resname"].isin(NEGATIVE_RESIDUES),
    "hydrophobic": lambda frame: frame["aa"].isin(HYDROPHOBIC),
    "aromatic": lambda frame: frame["aa"].isin(AROMATIC),
}


@dataclass(frozen=True)
class TopologyGateResult:
    """Container for a full topology falsification run."""

    metrics: pd.DataFrame
    verdicts: pd.DataFrame
    feature_blocks: dict[str, pd.DataFrame]
    structure_subset: pd.Index
    scramble_diagnostics: pd.DataFrame


@dataclass(frozen=True)
class TopologyPositiveControlResult:
    """Synthetic-label calibration for the topology falsification detector."""

    metrics: pd.DataFrame
    summary: pd.DataFrame
    synthetic_labels: pd.DataFrame


@dataclass(frozen=True)
class TopologyFeatureBlockResult:
    """Feature blocks plus diagnostics for the composition-matched scramble null."""

    feature_blocks: dict[str, pd.DataFrame]
    scramble_diagnostics: pd.DataFrame


def prepare_surface_residues(
    pdb_path: str | Path,
    heavy_aligned_aho: str,
    light_aligned_aho: str,
    surface_radius: float = 10.0,
    surface_quantile: float = 0.45,
) -> pd.DataFrame:
    """Parse one Fv structure and retain the approximate exposed residue nodes."""
    residues = parse_structure_residues(Path(pdb_path), heavy_aligned_aho, light_aligned_aho)
    if residues.empty:
        return residues

    coords = residues[["x", "y", "z"]].to_numpy(dtype=float)
    residues = residues.copy()
    residues["is_surface_proxy"] = surface_mask(
        coords,
        radius=surface_radius,
        quantile=surface_quantile,
    )
    surface = residues[residues["is_surface_proxy"]].copy()
    if surface.empty:
        surface = residues.copy()
    surface["region_bin"] = np.where(surface["is_cdr"], "cdr", "framework")
    return surface.reset_index(drop=True)


def surface_graph(residues: pd.DataFrame, distance_cutoff: float = 10.0) -> nx.Graph:
    """Build a residue contact graph over exposed residue proxy coordinates."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(residues)))
    if residues.empty:
        return graph

    coords = residues[["x", "y", "z"]].to_numpy(dtype=float)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            distance = float(np.linalg.norm(coords[i] - coords[j]))
            if distance <= distance_cutoff:
                graph.add_edge(i, j, distance=distance)
    return graph


def surface_composition_features(residues: pd.DataFrame) -> dict[str, float]:
    """Composition of exposed residues without using their spatial arrangement."""
    rows: dict[str, float] = {
        "surface_residue_count": float(len(residues)),
        "surface_cdr_fraction": float(residues["is_cdr"].mean()) if len(residues) else np.nan,
    }
    for region_name, region in _region_frames(residues).items():
        prefix = f"surface_{region_name}"
        n = len(region)
        rows[f"{prefix}_residue_count"] = float(n)
        for name, selector in TOPOLOGY_PROPERTIES.items():
            count = float(selector(region).sum()) if n else 0.0
            rows[f"{prefix}_{name}_count"] = count
            rows[f"{prefix}_{name}_fraction"] = count / n if n else np.nan
    return rows


def topology_features(
    residues: pd.DataFrame,
    distance_cutoff: float = 10.0,
) -> dict[str, float]:
    """Patch and adjacency descriptors that depend on residue positions."""
    graph = surface_graph(residues, distance_cutoff=distance_cutoff)
    rows = {
        "surface_graph_nodes": float(graph.number_of_nodes()),
        "surface_graph_edges": float(graph.number_of_edges()),
        "surface_graph_mean_degree": _mean_degree(graph),
    }

    selected_nodes: dict[str, set[int]] = {}
    for name, selector in TOPOLOGY_PROPERTIES.items():
        nodes = set(np.flatnonzero(selector(residues).to_numpy()))
        selected_nodes[name] = nodes
        rows.update(_patch_summary(graph, residues, nodes, name))

    for left, right in [
        ("hydrophobic", "positive"),
        ("hydrophobic", "negative"),
        ("aromatic", "positive"),
        ("aromatic", "negative"),
        ("positive", "negative"),
    ]:
        rows.update(_adjacency_summary(graph, residues, selected_nodes[left], selected_nodes[right], left, right))

    return rows


def scramble_surface_chemistry(
    residues: pd.DataFrame,
    rng: np.random.Generator,
    preserve_region_bins: bool = True,
) -> pd.DataFrame:
    """Randomize residue identities on the same surface graph.

    The default null preserves exposed residue identities/counts separately
    within CDR and framework bins for each antibody.
    """
    scrambled = residues.copy()
    bins = ["all"]
    if preserve_region_bins:
        bins = list(scrambled["region_bin"].dropna().unique())

    for bin_name in bins:
        if preserve_region_bins:
            mask = scrambled["region_bin"] == bin_name
        else:
            mask = pd.Series(True, index=scrambled.index)
        idx = scrambled.index[mask].to_numpy()
        if len(idx) <= 1:
            continue
        shuffled = _deranged_indices(idx, rng)
        scrambled.loc[idx, ["resname", "aa"]] = residues.loc[shuffled, ["resname", "aa"]].to_numpy()
    return scrambled


def scramble_surface_chemistry_with_diagnostics(
    residues: pd.DataFrame,
    rng: np.random.Generator,
    preserve_region_bins: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomize residue identities and report null-quality diagnostics."""
    scrambled = residues.copy()
    bins = ["all"]
    if preserve_region_bins:
        bins = list(scrambled["region_bin"].dropna().unique())

    rows = []
    for bin_name in bins:
        if preserve_region_bins:
            mask = scrambled["region_bin"] == bin_name
        else:
            mask = pd.Series(True, index=scrambled.index)
        idx = scrambled.index[mask].to_numpy()
        n_nodes = len(idx)
        if n_nodes <= 1:
            rows.append(_scramble_bin_diagnostic(bin_name, residues, scrambled, idx, idx))
            continue
        shuffled = _deranged_indices(idx, rng)
        scrambled.loc[idx, ["resname", "aa"]] = residues.loc[shuffled, ["resname", "aa"]].to_numpy()
        rows.append(_scramble_bin_diagnostic(bin_name, residues, scrambled, idx, shuffled))
    return scrambled, pd.DataFrame(rows)


def build_feature_blocks(
    df: pd.DataFrame,
    structure_dir: str | Path = "data/raw/hf_snapshot/structures",
    n_scrambles: int = 30,
    random_state: int = 7,
    surface_radius: float = 10.0,
    surface_quantile: float = 0.45,
    topology_distance_cutoff: float = 10.0,
) -> dict[str, pd.DataFrame]:
    """Build global, surface, real-topology, and composition-matched scramble blocks."""
    return build_feature_blocks_with_diagnostics(
        df=df,
        structure_dir=structure_dir,
        n_scrambles=n_scrambles,
        random_state=random_state,
        surface_radius=surface_radius,
        surface_quantile=surface_quantile,
        topology_distance_cutoff=topology_distance_cutoff,
    ).feature_blocks


def build_feature_blocks_with_diagnostics(
    df: pd.DataFrame,
    structure_dir: str | Path = "data/raw/hf_snapshot/structures",
    n_scrambles: int = 30,
    random_state: int = 7,
    surface_radius: float = 10.0,
    surface_quantile: float = 0.45,
    topology_distance_cutoff: float = 10.0,
) -> TopologyFeatureBlockResult:
    """Build feature blocks and scramble diagnostics on the same structure subset."""
    if n_scrambles < 1:
        raise ValueError("n_scrambles must be at least 1.")

    structure_dir = Path(structure_dir)
    rng = np.random.default_rng(random_state)
    global_features = build_physical_features(df)

    surface_rows = []
    real_rows = []
    scramble_rows: list[list[dict[str, float]]] = [[] for _ in range(n_scrambles)]
    diagnostic_rows = []
    subset = []

    for idx, row in df.iterrows():
        pdb_path = structure_path_for_antibody(structure_dir, row["antibody_name"])
        if not pdb_path.exists():
            continue
        residues = prepare_surface_residues(
            pdb_path,
            row["heavy_aligned_aho"],
            row["light_aligned_aho"],
            surface_radius=surface_radius,
            surface_quantile=surface_quantile,
        )
        if residues.empty:
            continue

        subset.append(idx)
        surface_rows.append(pd.Series(surface_composition_features(residues), name=idx))
        real_rows.append(
            pd.Series(
                topology_features(residues, distance_cutoff=topology_distance_cutoff),
                name=idx,
            )
        )
        for scramble_idx in range(n_scrambles):
            scrambled, diagnostics = scramble_surface_chemistry_with_diagnostics(
                residues,
                rng=rng,
                preserve_region_bins=True,
            )
            diagnostics["antibody_index"] = idx
            diagnostics["antibody_name"] = row["antibody_name"]
            diagnostics["scramble_index"] = scramble_idx
            diagnostic_rows.extend(diagnostics.to_dict("records"))
            scramble_rows[scramble_idx].append(
                pd.Series(
                    topology_features(scrambled, distance_cutoff=topology_distance_cutoff),
                    name=idx,
                )
            )

    subset_index = pd.Index(subset)
    blocks: dict[str, pd.DataFrame] = {
        "global_composition": global_features.loc[subset_index].copy(),
        "surface_composition": pd.DataFrame(surface_rows).loc[subset_index],
        "real_topology": pd.DataFrame(real_rows).loc[subset_index],
    }
    for scramble_idx, rows in enumerate(scramble_rows):
        blocks[f"scrambled_topology_{scramble_idx:03d}"] = pd.DataFrame(rows).loc[subset_index]

    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics = add_real_scrambled_feature_correlations(blocks, diagnostics)
    return TopologyFeatureBlockResult(feature_blocks=blocks, scramble_diagnostics=diagnostics)


def evaluate_feature_blocks(
    df: pd.DataFrame,
    feature_blocks: dict[str, pd.DataFrame],
    assays: list[str] | None = None,
    groups: pd.Series | None = None,
    model_name: str = "ridge",
    cv_kind: str = "cluster",
    n_label_shuffles: int = 30,
    random_state: int = 7,
) -> pd.DataFrame:
    """Evaluate each predeclared feature block on the same antibody subset."""
    assays = assays or DEFAULT_GATE_ASSAYS
    subset = _common_feature_index(feature_blocks)
    analysis_df = df.loc[subset]
    analysis_groups = groups.loc[subset] if groups is not None else None

    rows = []
    for block_name, features in feature_blocks.items():
        for assay in assays:
            metric = _cross_validated_metric(
                analysis_df,
                features.loc[subset],
                assay,
                model_name=model_name,
                cv_kind=cv_kind,
                groups=analysis_groups,
                random_state=random_state,
            )
            metric["feature_block"] = block_name
            rows.append(metric)

    rng = np.random.default_rng(random_state)
    real_features = feature_blocks["real_topology"].loc[subset]
    for shuffle_idx in range(n_label_shuffles):
        for assay in assays:
            metric = _cross_validated_metric(
                analysis_df,
                real_features,
                assay,
                model_name=model_name,
                cv_kind=cv_kind,
                groups=analysis_groups,
                random_state=random_state,
                shuffled_labels=rng,
            )
            metric["feature_block"] = f"label_shuffle_{shuffle_idx:03d}"
            rows.append(metric)

    return pd.DataFrame(rows)


def summarize_gate_metrics(
    metrics: pd.DataFrame,
    low_n_cutoff: int = 100,
    min_delta: float = 0.05,
    label_shuffle_abs_cutoff: float = 0.15,
) -> pd.DataFrame:
    """Create the minimum topology falsification table with predeclared verdicts."""
    rows = []
    for assay, assay_metrics in metrics.groupby("assay", sort=False):
        lookup = assay_metrics.set_index("feature_block")
        scrambled = assay_metrics[assay_metrics["feature_block"].str.startswith("scrambled_topology_")]
        label = assay_metrics[assay_metrics["feature_block"].str.startswith("label_shuffle_")]

        global_score = _score_from_lookup(lookup, "global_composition")
        surface_score = _score_from_lookup(lookup, "surface_composition")
        real_score = _score_from_lookup(lookup, "real_topology")
        scrambled_mean = float(scrambled["spearman"].mean()) if not scrambled.empty else np.nan
        scrambled_sd = float(scrambled["spearman"].std(ddof=0)) if len(scrambled) else np.nan
        label_mean = float(label["spearman"].mean()) if not label.empty else np.nan
        label_sd = float(label["spearman"].std(ddof=0)) if len(label) else np.nan
        n = int(lookup.loc["real_topology", "n"]) if "real_topology" in lookup.index else 0

        delta_surface = real_score - surface_score
        delta_scrambled = real_score - scrambled_mean
        negative_control_fail = pd.notna(label_mean) and abs(label_mean) >= label_shuffle_abs_cutoff
        beats_scrambled = (
            pd.notna(delta_scrambled)
            and delta_scrambled > min_delta
            and real_score > scrambled_mean + scrambled_sd
        )
        beats_surface = pd.notna(delta_surface) and delta_surface > min_delta

        if negative_control_fail:
            verdict = "NEGATIVE CONTROL FAIL"
        elif n and n < low_n_cutoff:
            if beats_scrambled and beats_surface:
                verdict = "PASS; LOW-N CAUTION"
            elif beats_scrambled:
                verdict = "WEAK PASS; LOW-N CAUTION"
            else:
                verdict = "FAIL; LOW-N CAUTION"
        elif beats_scrambled and beats_surface:
            verdict = "PASS"
        elif beats_scrambled:
            verdict = "WEAK PASS"
        else:
            verdict = "FAIL"

        rows.append(
            {
                "assay": assay,
                "n": n,
                "global_composition_spearman": global_score,
                "surface_composition_spearman": surface_score,
                "real_topology_spearman": real_score,
                "scrambled_topology_spearman_mean": scrambled_mean,
                "scrambled_topology_spearman_sd": scrambled_sd,
                "label_shuffle_spearman_mean": label_mean,
                "label_shuffle_spearman_sd": label_sd,
                "delta_real_minus_surface": delta_surface,
                "delta_real_minus_scrambled": delta_scrambled,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def run_topology_falsification_gate(
    df: pd.DataFrame,
    groups: pd.Series,
    assays: list[str] | None = None,
    structure_dir: str | Path = "data/raw/hf_snapshot/structures",
    n_scrambles: int = 30,
    n_label_shuffles: int = 30,
    model_name: str = "ridge",
    random_state: int = 7,
) -> TopologyGateResult:
    """Run the full real-topology versus composition-matched-null gate."""
    block_result = build_feature_blocks_with_diagnostics(
        df,
        structure_dir=structure_dir,
        n_scrambles=n_scrambles,
        random_state=random_state,
    )
    blocks = block_result.feature_blocks
    metrics = evaluate_feature_blocks(
        df,
        blocks,
        assays=assays,
        groups=groups,
        model_name=model_name,
        cv_kind="cluster",
        n_label_shuffles=n_label_shuffles,
        random_state=random_state,
    )
    verdicts = summarize_gate_metrics(metrics)
    return TopologyGateResult(
        metrics=metrics,
        verdicts=verdicts,
        feature_blocks=blocks,
        structure_subset=_common_feature_index(blocks),
        scramble_diagnostics=block_result.scramble_diagnostics,
    )


def positive_control_topology_signal(
    feature_blocks: dict[str, pd.DataFrame],
    groups: pd.Series,
    model_name: str = "ridge",
    random_state: int = 7,
) -> TopologyPositiveControlResult:
    """Check that the gate can detect known composition and topology signals.

    This is not biology. It is a detector calibration:
    - composition labels should be learnable from surface composition;
    - topology labels should be learnable from real topology more than scrambles;
    - shuffled synthetic labels should not be learnable.
    """
    subset = _common_feature_index(feature_blocks)
    labels = synthetic_positive_control_labels(
        feature_blocks,
        random_state=random_state,
    ).loc[subset]
    groups = groups.loc[subset]

    rows = []
    for block_name, features in feature_blocks.items():
        for assay in labels.columns:
            metric = _cross_validated_metric(
                labels,
                features.loc[subset],
                assay,
                model_name=model_name,
                cv_kind="cluster",
                groups=groups,
                random_state=random_state,
            )
            metric["feature_block"] = block_name
            rows.append(metric)

    metrics = pd.DataFrame(rows)
    summary = summarize_positive_controls(metrics)
    return TopologyPositiveControlResult(
        metrics=metrics,
        summary=summary,
        synthetic_labels=labels,
    )


def synthetic_positive_control_labels(
    feature_blocks: dict[str, pd.DataFrame],
    random_state: int = 7,
) -> pd.DataFrame:
    """Create fake labels from known composition/topology sources."""
    subset = _common_feature_index(feature_blocks)
    surface = feature_blocks["surface_composition"].loc[subset]
    real = feature_blocks["real_topology"].loc[subset]
    rng = np.random.default_rng(random_state)

    composition_sources = _available_columns(
        surface,
        [
            "surface_cdr_hydrophobic_fraction",
            "surface_all_hydrophobic_fraction",
            "surface_cdr_positive_fraction",
            "surface_all_aromatic_fraction",
        ],
    )
    topology_sources = _available_columns(
        real,
        [
            "hydrophobic_largest_patch_size",
            "hydrophobic_positive_adjacent_edge_count",
            "hydrophobic_largest_patch_cdr_fraction",
            "aromatic_largest_patch_size",
            "positive_largest_patch_size",
        ],
    )
    if not composition_sources:
        raise ValueError("No usable surface-composition columns for positive control.")
    if not topology_sources:
        raise ValueError("No usable real-topology columns for positive control.")

    composition_signal = _zscore_frame(surface[composition_sources]).sum(axis=1)
    topology_signal = _topology_excess_signal(feature_blocks, topology_sources, subset)
    shuffled_topology = pd.Series(
        rng.permutation(topology_signal.to_numpy()),
        index=topology_signal.index,
    )

    return pd.DataFrame(
        {
            "positive_control_composition": composition_signal,
            "positive_control_topology": topology_signal,
            "positive_control_shuffled": shuffled_topology,
        },
        index=subset,
    )


def summarize_positive_controls(
    metrics: pd.DataFrame,
    topology_margin: float = 0.15,
    composition_tolerance: float = 0.10,
    shuffled_abs_cutoff: float = 0.15,
) -> pd.DataFrame:
    """Summarize whether synthetic positive controls behaved as expected."""
    rows = []
    for assay, assay_metrics in metrics.groupby("assay", sort=False):
        lookup = assay_metrics.set_index("feature_block")
        scrambled = assay_metrics[assay_metrics["feature_block"].str.startswith("scrambled_topology_")]
        surface_score = _score_from_lookup(lookup, "surface_composition")
        real_score = _score_from_lookup(lookup, "real_topology")
        scrambled_mean = float(scrambled["spearman"].mean()) if not scrambled.empty else np.nan
        scrambled_sd = float(scrambled["spearman"].std(ddof=0)) if len(scrambled) else np.nan

        if assay == "positive_control_composition":
            delta = real_score - surface_score
            passed = pd.notna(delta) and abs(delta) <= composition_tolerance
            expectation = "surface composition ~= real topology"
        elif assay == "positive_control_topology":
            delta = real_score - scrambled_mean
            passed = (
                pd.notna(delta)
                and delta > topology_margin
                and real_score > scrambled_mean + scrambled_sd
            )
            expectation = "real topology > scrambled topology"
        elif assay == "positive_control_shuffled":
            delta = real_score
            passed = pd.notna(real_score) and abs(real_score) <= shuffled_abs_cutoff
            expectation = "all feature blocks near zero"
        else:
            delta = np.nan
            passed = False
            expectation = "unknown"

        rows.append(
            {
                "test": assay,
                "expectation": expectation,
                "surface_composition_spearman": surface_score,
                "real_topology_spearman": real_score,
                "scrambled_topology_spearman_mean": scrambled_mean,
                "scrambled_topology_spearman_sd": scrambled_sd,
                "diagnostic_delta": delta,
                "verdict": "PASS" if passed else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def add_real_scrambled_feature_correlations(
    feature_blocks: dict[str, pd.DataFrame],
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Attach real-vs-scrambled topology feature correlations by scramble."""
    if diagnostics.empty:
        return diagnostics

    real = feature_blocks["real_topology"]
    rows = []
    common_numeric = real.select_dtypes(include="number").columns
    for block_name, block in feature_blocks.items():
        if not block_name.startswith("scrambled_topology_"):
            continue
        scramble_index = int(block_name.rsplit("_", 1)[-1])
        common = common_numeric.intersection(block.select_dtypes(include="number").columns)
        feature_correlations = []
        for column in common:
            corr = _safe_spearman(real[column], block[column])
            if pd.notna(corr):
                feature_correlations.append(corr)
        rows.append(
            {
                "scramble_index": scramble_index,
                "real_scrambled_feature_correlation_mean": (
                    float(np.mean(feature_correlations)) if feature_correlations else np.nan
                ),
                "real_scrambled_feature_correlation_median": (
                    float(np.median(feature_correlations)) if feature_correlations else np.nan
                ),
            }
        )

    correlation_summary = pd.DataFrame(rows)
    return diagnostics.merge(correlation_summary, on="scramble_index", how="left")


def summarize_scramble_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Compact diagnostics for whether the scramble null actually moved chemistry."""
    if diagnostics.empty:
        return pd.DataFrame()

    rows = [
        {
            "diagnostic": "mean fraction of residues moved",
            "value": float(diagnostics["identity_moved_fraction"].mean()),
            "why_it_matters": "checks scramble is not identity-ish",
        },
        {
            "diagnostic": "mean CDR moved fraction",
            "value": _mean_for_region(diagnostics, "cdr", "identity_moved_fraction"),
            "why_it_matters": "checks CDR null quality",
        },
        {
            "diagnostic": "mean framework moved fraction",
            "value": _mean_for_region(diagnostics, "framework", "identity_moved_fraction"),
            "why_it_matters": "checks framework null quality",
        },
        {
            "diagnostic": "antibodies with non-derangeable bins",
            "value": float(
                diagnostics.loc[
                    diagnostics["non_derangeable_bin"], "antibody_index"
                ].nunique()
            ),
            "why_it_matters": "identifies weak null cases from tiny bins",
        },
        {
            "diagnostic": "mean real-vs-scrambled feature correlation",
            "value": float(diagnostics["real_scrambled_feature_correlation_mean"].mean()),
            "why_it_matters": "detects residual topology leakage",
        },
    ]
    return pd.DataFrame(rows)


def _cross_validated_metric(
    df: pd.DataFrame,
    features: pd.DataFrame,
    assay: str,
    model_name: str,
    cv_kind: str,
    groups: pd.Series | None,
    random_state: int,
    shuffled_labels: np.random.Generator | None = None,
) -> dict[str, object]:
    if assay not in df.columns:
        return metric_row(assay, np.array([]), np.array([]), model_name)

    valid = df[assay].notna()
    x = features.loc[valid]
    y = df.loc[valid, assay].to_numpy(dtype=float)
    if shuffled_labels is not None and len(y):
        y = shuffled_labels.permutation(y)

    if len(y) < 20:
        return metric_row(assay, y, np.full(len(y), np.nan), model_name)

    if cv_kind == "random":
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        split_groups = None
    elif cv_kind == "cluster":
        if groups is None:
            raise ValueError("groups must be provided for cluster CV")
        split_groups = groups.loc[valid].to_numpy()
        n_groups = len(np.unique(split_groups))
        if n_groups < 2:
            return metric_row(assay, y, np.full(len(y), np.nan), model_name)
        cv = GroupKFold(n_splits=min(5, n_groups))
    else:
        raise ValueError(f"Unknown cv_kind: {cv_kind}")

    pred = np.full(len(y), np.nan)
    for train_idx, test_idx in cv.split(x, y, groups=split_groups):
        estimator = make_regression_pipeline(x, model_name=model_name, random_state=random_state)
        estimator.fit(x.iloc[train_idx], y[train_idx])
        pred[test_idx] = estimator.predict(x.iloc[test_idx])

    return metric_row(assay, y, pred, f"{model_name}_{cv_kind}_cv")


def _region_frames(residues: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "all": residues,
        "cdr": residues[residues["is_cdr"]],
        "framework": residues[~residues["is_cdr"]],
    }


def _patch_summary(
    graph: nx.Graph,
    residues: pd.DataFrame,
    nodes: set[int],
    prefix: str,
) -> dict[str, float]:
    subgraph = graph.subgraph(nodes)
    components = [set(component) for component in nx.connected_components(subgraph)]
    sizes = np.array([len(component) for component in components], dtype=float)
    largest = components[int(np.argmax(sizes))] if len(sizes) else set()

    rows = {
        f"{prefix}_patch_count": float(len(components)),
        f"{prefix}_largest_patch_size": float(sizes.max()) if len(sizes) else 0.0,
        f"{prefix}_mean_patch_size": float(sizes.mean()) if len(sizes) else 0.0,
        f"{prefix}_patch_fragmentation": float(len(components) / max(len(nodes), 1)),
        f"{prefix}_largest_patch_cdr_fraction": (
            float(residues.loc[list(largest), "is_cdr"].mean()) if largest else np.nan
        ),
    }
    for region_name, region in _region_frames(residues).items():
        region_nodes = nodes & set(region.index)
        rows[f"{prefix}_{region_name}_selected_count"] = float(len(region_nodes))
    return rows


def _adjacency_summary(
    graph: nx.Graph,
    residues: pd.DataFrame,
    left_nodes: set[int],
    right_nodes: set[int],
    left: str,
    right: str,
) -> dict[str, float]:
    prefix = f"{left}_{right}"
    edges = [
        (u, v)
        for u, v in graph.edges()
        if (u in left_nodes and v in right_nodes) or (u in right_nodes and v in left_nodes)
    ]
    centroid_distance = _centroid_distance(residues, left_nodes, right_nodes)
    return {
        f"{prefix}_adjacent_edge_count": float(len(edges)),
        f"{prefix}_adjacent_edge_fraction": float(len(edges) / max(graph.number_of_edges(), 1)),
        f"{prefix}_centroid_distance": centroid_distance,
    }


def _centroid_distance(residues: pd.DataFrame, left_nodes: set[int], right_nodes: set[int]) -> float:
    if not left_nodes or not right_nodes:
        return np.nan
    left_coords = residues.loc[list(left_nodes), ["x", "y", "z"]].to_numpy(dtype=float)
    right_coords = residues.loc[list(right_nodes), ["x", "y", "z"]].to_numpy(dtype=float)
    return float(np.linalg.norm(left_coords.mean(axis=0) - right_coords.mean(axis=0)))


def _mean_degree(graph: nx.Graph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    return float(np.mean([degree for _, degree in graph.degree()]))


def _common_feature_index(feature_blocks: dict[str, pd.DataFrame]) -> pd.Index:
    indexes = [block.index for block in feature_blocks.values()]
    common = indexes[0]
    for index in indexes[1:]:
        common = common.intersection(index)
    return common


def _score_from_lookup(lookup: pd.DataFrame, feature_block: str) -> float:
    if feature_block not in lookup.index:
        return np.nan
    return float(lookup.loc[feature_block, "spearman"])


def _scramble_bin_diagnostic(
    bin_name: str,
    original: pd.DataFrame,
    scrambled: pd.DataFrame,
    idx: np.ndarray,
    shuffled: np.ndarray,
) -> dict[str, float | str | bool]:
    n_nodes = len(idx)
    if n_nodes == 0:
        return {
            "region_bin": bin_name,
            "n_nodes": 0,
            "index_moved_fraction": np.nan,
            "identity_moved_fraction": np.nan,
            "non_derangeable_bin": True,
        }

    original_identity = original.loc[idx, "aa"].to_numpy()
    scrambled_identity = scrambled.loc[idx, "aa"].to_numpy()
    index_moved = shuffled != idx
    identity_moved = scrambled_identity != original_identity
    return {
        "region_bin": bin_name,
        "n_nodes": int(n_nodes),
        "index_moved_fraction": float(index_moved.mean()),
        "identity_moved_fraction": float(identity_moved.mean()),
        "non_derangeable_bin": bool(n_nodes <= 1),
    }


def _deranged_indices(idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle indexes while avoiding fixed points when possible."""
    if len(idx) <= 1:
        return idx.copy()
    if len(idx) == 2:
        return idx[::-1].copy()

    shuffled = idx.copy()
    for _ in range(100):
        rng.shuffle(shuffled)
        if np.all(shuffled != idx):
            return shuffled.copy()

    # Deterministic fallback: a one-step rotation is always a derangement for len > 1.
    return np.roll(idx, 1)


def _topology_excess_signal(
    feature_blocks: dict[str, pd.DataFrame],
    topology_sources: list[str],
    subset: pd.Index,
) -> pd.Series:
    """Synthetic target for topology beyond composition-matched scramble expectation."""
    real = feature_blocks["real_topology"].loc[subset, topology_sources].astype(float)
    mean = real.mean(axis=0)
    std = real.std(axis=0).replace(0, np.nan)
    real_signal = real.sub(mean, axis=1).div(std, axis=1).fillna(0.0).sum(axis=1)

    scramble_signals = []
    for block_name, block in feature_blocks.items():
        if not block_name.startswith("scrambled_topology_"):
            continue
        available = [column for column in topology_sources if column in block.columns]
        if available != topology_sources:
            continue
        scrambled = block.loc[subset, topology_sources].astype(float)
        scramble_signals.append(
            scrambled.sub(mean, axis=1).div(std, axis=1).fillna(0.0).sum(axis=1)
        )

    if not scramble_signals:
        return real_signal

    scramble_expectation = pd.concat(scramble_signals, axis=1).mean(axis=1)
    return real_signal - scramble_expectation


def _available_columns(frame: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [column for column in candidates if column in frame.columns]


def _zscore_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.astype(float)
    std = numeric.std(axis=0).replace(0, np.nan)
    return numeric.sub(numeric.mean(axis=0), axis=1).div(std, axis=1).fillna(0.0)


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    valid = pd.concat([left, right], axis=1).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(valid.iloc[:, 0].rank().corr(valid.iloc[:, 1].rank()))


def _mean_for_region(diagnostics: pd.DataFrame, region: str, column: str) -> float:
    subset = diagnostics[diagnostics["region_bin"] == region]
    return float(subset[column].mean()) if not subset.empty else np.nan
