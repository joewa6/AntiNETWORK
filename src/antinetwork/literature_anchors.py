from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd



NUMERIC_ANCHOR_COLUMNS = [
    "anchor_id",
    "paper_key",
    "system_id",
    "antibody_label",
    "condition_id",
    "pH",
    "salt_type",
    "salt_mM",
    "additive_type",
    "additive_concentration",
    "protein_concentration_mg_ml",
    "reported_mechanism",
    "network_regime_target",
    "target_score_0_1_2",
    "confidence",
    "use_for",
]


@dataclass(frozen=True)
class LiteratureMappingCoefficients:
    """Global condition-to-network coefficients for literature-anchor calibration."""

    concentration_to_attraction: float
    salt_screening_to_attraction: float
    additive_to_mitigation: float
    salt_to_repulsion: float
    mechanism_to_attraction: float = 0.0
    hydrophobic_to_attraction: float = 0.0
    mechanism_to_valence: float = 0.0


def load_numeric_literature_anchors(path: str) -> pd.DataFrame:
    """Load condition-level literature anchors and preserve traceability columns."""
    df = pd.read_csv(path)
    missing = sorted(set(NUMERIC_ANCHOR_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required literature-anchor columns: {missing}")
    return df


def numeric_values(value: object) -> list[float]:
    """Extract numeric tokens from hand-curated literature text fields."""
    if pd.isna(value):
        return []
    text = str(value)
    if text.lower().strip() in {"none", "nan", "not_used_for_v0_numeric_calibration"}:
        return []
    values = []
    for token in re.findall(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?", text):
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def numeric_first(value: object, default: float = np.nan) -> float:
    values = numeric_values(value)
    return values[0] if values else default


def numeric_max(value: object, default: float = np.nan) -> float:
    values = numeric_values(value)
    return max(values) if values else default


def calibration_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with usable 0/1/2 target scores, excluding theory-only anchors."""
    rows = df[df["target_score_0_1_2"].notna()].copy()
    return rows[rows["target_score_0_1_2"].isin([0, 1, 2])].copy()


def _text_contains(frame: pd.DataFrame, columns: list[str], pattern: str) -> pd.Series:
    text = frame[columns].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    return text.str.contains(pattern, regex=True).astype(float)


def build_literature_condition_features(
    anchors: pd.DataFrame,
    include_mechanism: bool = False,
) -> pd.DataFrame:
    """Build condition-level features for literature-anchor network calibration.

    The condition-only feature set uses formulation variables and excludes
    mechanism labels. The mechanism-aware version adds coarse physical tags from
    reported mechanisms, but still excludes target labels.
    """
    rows = anchors.copy()
    features = pd.DataFrame(index=rows.index)
    features["pH_numeric"] = rows["pH"].map(numeric_first)
    features["salt_mM_numeric"] = rows["salt_mM"].map(numeric_max).fillna(0.0)
    features["protein_concentration_mg_ml_numeric"] = (
        rows["protein_concentration_mg_ml"].map(numeric_max).fillna(0.0)
    )
    features["temperature_C_numeric"] = rows["temperature_C"].map(numeric_first).fillna(25.0)

    salt_text = rows[["salt_type", "salt_mM", "buffer"]].fillna("").astype(str).agg(" ".join, axis=1)
    additive_text = rows[["additive_type", "additive_concentration"]].fillna("").astype(str).agg(
        " ".join, axis=1
    )
    features["has_salt"] = (features["salt_mM_numeric"] > 0).astype(float)
    features["salt_screening_factor"] = np.sqrt(50.0 / (50.0 + features["salt_mM_numeric"]))
    features["salt_load"] = 1.0 - features["salt_screening_factor"]
    features["low_salt_condition"] = (
        (features["salt_mM_numeric"] <= 50.0) & (features["has_salt"] > 0)
    ).astype(float)
    features["high_salt_condition"] = (features["salt_mM_numeric"] >= 100.0).astype(float)
    features["arginine_or_excipient"] = (
        salt_text.str.contains("arg", case=False, regex=False)
        | additive_text.str.contains("arg", case=False, regex=False)
        | additive_text.str.contains("trehalose|excipient|additive", case=False, regex=True)
    ).astype(float)
    features["concentration_scaled"] = (
        features["protein_concentration_mg_ml_numeric"] / 200.0
    ).clip(0.0, 1.5)
    features["high_concentration"] = (
        features["protein_concentration_mg_ml_numeric"] >= 100.0
    ).astype(float)
    features["low_pH_charge_boost"] = (7.4 - features["pH_numeric"].fillna(7.4)).clip(0.0, 2.0) / 2.0

    if include_mechanism:
        mechanism_cols = ["reported_mechanism", "self_association_measure"]
        features["charge_mechanism_signal"] = _text_contains(
            rows,
            mechanism_cols,
            "charge|electrostatic|fab-fc|fab fc|cation|anion|salt|pi",
        )
        features["hydrophobic_mechanism_signal"] = _text_contains(
            rows,
            mechanism_cols,
            "hydrophobic|aromatic|short-range",
        )
        features["specific_interface_signal"] = _text_contains(
            rows,
            mechanism_cols,
            "fab-fab|fab fab|fab-fc|fab fc|fv|patch|anisotropic",
        )
    else:
        features["charge_mechanism_signal"] = 0.0
        features["hydrophobic_mechanism_signal"] = 0.0
        features["specific_interface_signal"] = 0.0

    return features


def literature_mapping_coordinates(
    features: pd.DataFrame,
    coefficients: LiteratureMappingCoefficients,
) -> pd.DataFrame:
    """Map literature condition features onto patchy-network coordinates."""
    out = pd.DataFrame(index=features.index)
    mitigation = coefficients.additive_to_mitigation * features["arginine_or_excipient"]
    attraction = (
        coefficients.concentration_to_attraction * features["concentration_scaled"]
        + coefficients.salt_screening_to_attraction * features["salt_screening_factor"]
        + coefficients.mechanism_to_attraction * features["charge_mechanism_signal"]
        + coefficients.hydrophobic_to_attraction * features["hydrophobic_mechanism_signal"]
        + 0.4 * features["low_pH_charge_boost"]
        - mitigation
    )
    repulsion = (
        coefficients.salt_to_repulsion * features["salt_load"]
        + mitigation
        + 0.25 * features["high_salt_condition"]
    )
    valence_latent = (
        features["high_concentration"]
        + features["low_salt_condition"]
        + coefficients.mechanism_to_valence * features["specific_interface_signal"]
    )

    out["fv_net_charge"] = repulsion
    out["fv_cdr_net_charge"] = attraction
    out["vh_vl_charge_imbalance"] = valence_latent
    out["fv_cdr_hydrophobic_fraction"] = 0.28 + 0.18 * features["hydrophobic_mechanism_signal"]
    out["attraction_strength"] = attraction.clip(0.0, 4.0)
    out["repulsion_strength"] = repulsion.clip(0.0, 1.5)
    out["patch_valence"] = np.select(
        [valence_latent < 0.75, valence_latent < 1.5],
        [1, 3],
        default=5,
    ).astype(int)
    return out


def project_literature_conditions(
    features: pd.DataFrame,
    regime_map: pd.DataFrame,
    coefficients: LiteratureMappingCoefficients,
) -> pd.DataFrame:
    """Project literature conditions onto the nearest simulated regime-map point."""
    coordinates = literature_mapping_coordinates(features, coefficients)
    rows = []
    for idx, row in coordinates.iterrows():
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
        rows.append(pd.Series(projected, name=idx))

    projection = pd.DataFrame(rows)
    projection["network_risk_score"] = (
        projection["mapped_largest_cluster_fraction"]
        + projection["mapped_percolation_probability"]
        + 0.25 * projection["mapped_mean_degree"]
        - 0.25 * projection["mapped_edge_turnover"]
    )
    return projection


def spearman_rank(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 2 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(valid.iloc[:, 0].rank().corr(valid.iloc[:, 1].rank()))


def calibrate_literature_mapping(
    features: pd.DataFrame,
    target: pd.Series,
    regime_map: pd.DataFrame,
    coefficient_grid: pd.DataFrame,
) -> pd.DataFrame:
    """Grid-search global coefficients against literature target-score ranks."""
    rows = []
    for _, candidate in coefficient_grid.iterrows():
        coefficients = LiteratureMappingCoefficients(**candidate.to_dict())
        projected = project_literature_conditions(features, regime_map, coefficients)
        rho = spearman_rank(projected["network_risk_score"], target)
        rows.append(
            {
                **candidate.to_dict(),
                "training_spearman": rho,
                "training_abs_spearman": abs(rho) if pd.notna(rho) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("training_abs_spearman", ascending=False)


def leave_one_paper_out_literature_fit(
    anchors: pd.DataFrame,
    features: pd.DataFrame,
    regime_map: pd.DataFrame,
    coefficient_grid: pd.DataFrame,
) -> pd.DataFrame:
    """Fit on all but one paper and evaluate held-out rank alignment."""
    rows = []
    target = anchors["target_score_0_1_2"]
    for paper_key in sorted(anchors["paper_key"].dropna().unique()):
        train = anchors["paper_key"] != paper_key
        test = anchors["paper_key"] == paper_key
        if test.sum() < 2 or target.loc[test].nunique() < 2:
            continue
        calibration = calibrate_literature_mapping(
            features.loc[train],
            target.loc[train],
            regime_map,
            coefficient_grid,
        )
        best = LiteratureMappingCoefficients(**calibration.iloc[0][coefficient_grid.columns].to_dict())
        projected = project_literature_conditions(features.loc[test], regime_map, best)
        rho = spearman_rank(projected["network_risk_score"], target.loc[test])
        binary_target = (target.loc[test] >= 2).astype(int)
        binary_score = projected["network_risk_score"].rank()
        binary_rho = spearman_rank(binary_score, binary_target)
        rows.append(
            {
                "held_out_paper": paper_key,
                "n_test": int(test.sum()),
                "target_values": ",".join(map(str, sorted(target.loc[test].dropna().unique()))),
                "heldout_spearman": rho,
                "heldout_binary_high_risk_spearman": binary_rho,
                "best_training_spearman": calibration.iloc[0]["training_spearman"],
                **calibration.iloc[0][coefficient_grid.columns].to_dict(),
            }
        )
    return pd.DataFrame(rows)
