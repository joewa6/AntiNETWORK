from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AxisDefinition:
    axis: str
    feature_weights: dict[str, float]
    suggested_assay: str
    confidence: str
    evidence_strength: str
    benchmark_support: str
    mutation_direction: str
    mechanistic_note: str
    assign_risk: bool = True


AXIS_DEFINITIONS = [
    AxisDefinition(
        axis="Hydrophobicity",
        feature_weights={
            "fv_cdr_hydrophobic_fraction": 0.35,
            "fv_cdr_aromatic_fraction": 0.30,
            "vh_hydrophobic_fraction": 0.15,
            "vl_hydrophobic_fraction": 0.15,
            "cdr_h3_length": 0.05,
        },
        suggested_assay="HIC / SMAC",
        confidence="medium-high",
        evidence_strength="medium-high",
        benchmark_support="HIC/SMAC-like endpoints align with hydrophobic/aromatic local features.",
        mutation_direction=(
            "Inspect exposed CDR/framework aromatic or bulky hydrophobic residues; consider reducing "
            "solvent-exposed aromatic clustering while preserving binding."
        ),
        mechanistic_note="Exposed aromatic or hydrophobic CDR/framework burden.",
    ),
    AxisDefinition(
        axis="Polyreactivity",
        feature_weights={
            "fv_cdr_positive_count": 0.25,
            "fv_cdr_net_charge": 0.20,
            "cdr_charge_fraction_of_fv": 0.20,
            "vh_vl_charge_imbalance": 0.20,
            "fv_cdr_hydrophobic_fraction": 0.15,
        },
        suggested_assay="PR_CHO / PR_Ova",
        confidence="medium",
        evidence_strength="medium",
        benchmark_support="PR endpoints retain strong charge-baseline behavior with local-feature contribution.",
        mutation_direction=(
            "Reduce excessive positive CDR charge or charge imbalance; check hydrophobic CDR surfaces "
            "that may drive nonspecific biological matrix binding."
        ),
        mechanistic_note="Positive charge, CDR charge density, and nonspecific binding surfaces.",
    ),
    AxisDefinition(
        axis="Local self-association",
        feature_weights={
            "heavy_cdr_positive_count": 0.25,
            "cdr_h3_length": 0.25,
            "fv_cdr_hydrophobic_fraction": 0.25,
            "fv_cdr_aromatic_fraction": 0.15,
            "fv_cdr_positive_count": 0.10,
        },
        suggested_assay="AC-SINS",
        confidence="medium",
        evidence_strength="medium",
        benchmark_support="AC-SINS residuals align better with local CDR patch features than network risk.",
        mutation_direction=(
            "Inspect H3 length, heavy-chain CDR positive charge, and aromatic/hydrophobic CDR burden; "
            "prioritize conservative edits outside binding-critical contacts."
        ),
        mechanistic_note="Local CDR patch stickiness and immobilized-assay geometry.",
    ),
    AxisDefinition(
        axis="Thermostability",
        feature_weights={
            "cdr_h3_length": 0.30,
            "vh_vl_hydrophobicity_imbalance": 0.25,
            "vh_vl_charge_imbalance": 0.20,
            "fv_cdr_hydrophobic_fraction": 0.15,
            "fv_net_charge": 0.10,
        },
        suggested_assay="DSF / DSC",
        confidence="low",
        evidence_strength="low",
        benchmark_support="Current sequence-only descriptors are not validated for fold stability.",
        mutation_direction="Do not mutate from this module alone; run DSF/DSC or use a dedicated stability model.",
        mechanistic_note="Crude sequence-only stability proxy; use structure or stability models when available.",
        assign_risk=False,
    ),
    AxisDefinition(
        axis="Titer / expression",
        feature_weights={
            "vh_length": 0.20,
            "vl_length": 0.20,
            "fv_cdr_hydrophobic_fraction": 0.20,
            "fv_net_charge": 0.20,
            "vh_vl_charge_imbalance": 0.20,
        },
        suggested_assay="Expression screen",
        confidence="low",
        evidence_strength="low",
        benchmark_support="Titer prediction is process- and host-cell-dependent; current proxies are weak.",
        mutation_direction=(
            "Do not interpret molecular percentile proxies as expression liability; validate in the "
            "intended expression system."
        ),
        mechanistic_note="Weak-confidence process flag; expression is not purely molecular.",
        assign_risk=False,
    ),
]


def percentile_score(value: float, reference: pd.Series) -> float:
    """Return a 0-1 high-risk percentile for one value against a reference series."""
    reference = reference.dropna().astype(float)
    if pd.isna(value) or reference.empty:
        return np.nan
    return float((reference <= float(value)).mean())


def risk_label(score: float) -> str:
    if pd.isna(score):
        return "unknown"
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def display_score(score: float, assign_risk: bool = True) -> float:
    if not assign_risk:
        return np.nan
    return score


def _feature_percentiles(
    feature_row: pd.Series,
    reference_features: pd.DataFrame,
    feature_weights: dict[str, float],
) -> dict[str, float]:
    scores = {}
    for feature in feature_weights:
        if feature not in feature_row.index or feature not in reference_features.columns:
            continue
        scores[feature] = percentile_score(feature_row[feature], reference_features[feature])
    return scores


def weighted_axis_score(
    feature_row: pd.Series,
    reference_features: pd.DataFrame,
    feature_weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    percentiles = _feature_percentiles(feature_row, reference_features, feature_weights)
    if not percentiles:
        return np.nan, {}
    weighted_sum = 0.0
    weight_total = 0.0
    for feature, percentile in percentiles.items():
        if pd.isna(percentile):
            continue
        weight = feature_weights[feature]
        weighted_sum += weight * percentile
        weight_total += weight
    if weight_total == 0:
        return np.nan, percentiles
    return float(weighted_sum / weight_total), percentiles


def top_feature_drivers(
    feature_percentiles: dict[str, float],
    feature_row: pd.Series,
    n: int = 3,
) -> list[str]:
    ordered = sorted(
        feature_percentiles.items(),
        key=lambda item: item[1] if pd.notna(item[1]) else -1,
        reverse=True,
    )
    drivers = []
    for feature, percentile in ordered[:n]:
        value = feature_row.get(feature, np.nan)
        if pd.isna(value):
            drivers.append(f"{feature} high vs reference")
        else:
            drivers.append(f"{feature}={value:.3g} (percentile {percentile:.2f})")
    return drivers


def build_axis_report(
    feature_row: pd.Series,
    reference_features: pd.DataFrame,
    antibody_name: str | None = None,
    network_projection_row: pd.Series | None = None,
) -> pd.DataFrame:
    """Build a mechanistic developability report for one antibody feature row."""
    rows = []
    antibody_name = antibody_name or str(feature_row.name)

    for axis_def in AXIS_DEFINITIONS:
        score, percentiles = weighted_axis_score(
            feature_row,
            reference_features,
            axis_def.feature_weights,
        )
        rows.append(
            {
                "antibody": antibody_name,
                "axis": axis_def.axis,
                "risk_score": display_score(score, axis_def.assign_risk),
                "risk_label": risk_label(score) if axis_def.assign_risk else "not assigned",
                "main_features": "; ".join(top_feature_drivers(percentiles, feature_row)),
                "suggested_assay": axis_def.suggested_assay,
                "confidence": axis_def.confidence,
                "evidence_strength": axis_def.evidence_strength,
                "benchmark_support": axis_def.benchmark_support,
                "recommended_experiment": axis_def.suggested_assay,
                "mutation_direction": axis_def.mutation_direction,
                "mechanistic_note": axis_def.mechanistic_note,
            }
        )

    if network_projection_row is not None:
        network_score = float(network_projection_row.get("network_risk_score", np.nan))
        mapped_regime = network_projection_row.get("mapped_regime", "unknown regime")
        drivers = [
            f"network_risk_score={network_score:.3g}",
            f"mapped_regime={mapped_regime}",
        ]
        for column in [
            "mapped_largest_cluster_fraction",
            "mapped_percolation_probability",
            "mapped_mean_degree",
        ]:
            if column in network_projection_row and pd.notna(network_projection_row[column]):
                drivers.append(f"{column}={network_projection_row[column]:.3g}")

        rows.insert(
            3,
            {
                "antibody": antibody_name,
                "axis": "Bulk network / viscosity",
                "risk_score": network_score,
                "risk_label": risk_label(network_score),
                "main_features": "; ".join(drivers),
                "suggested_assay": "HAC / high-concentration viscosity",
                "confidence": "low-medium",
                "evidence_strength": "suggestive",
                "benchmark_support": (
                    "HAC shows network-like residual structure in archetype diagnostics, but the "
                    "strongest cluster-preserving permutation control did not pass."
                ),
                "recommended_experiment": "HAC / high-concentration viscosity / DLS-SAXS if formulation risk matters",
                "mutation_direction": (
                    "Treat as a formulation hypothesis; inspect charge architecture and validate at "
                    "high concentration before mutation decisions."
                ),
                "mechanistic_note": (
                    "Literature-calibrated solution-regime projection; HAC network signal is "
                    "suggestive but failed the strongest cluster-preserving permutation null."
                ),
            },
        )
    else:
        rows.insert(
            3,
            {
                "antibody": antibody_name,
                "axis": "Bulk network / viscosity",
                "risk_score": np.nan,
                "risk_label": "unknown",
                "main_features": "network projection not supplied",
                "suggested_assay": "HAC / high-concentration viscosity",
                "confidence": "low",
                "evidence_strength": "low",
                "benchmark_support": "Network projection not supplied.",
                "recommended_experiment": "HAC / high-concentration viscosity",
                "mutation_direction": "No network-axis mutation recommendation without projection.",
                "mechanistic_note": "Needs literature-calibrated network projection.",
            },
        )

    return pd.DataFrame(rows)
