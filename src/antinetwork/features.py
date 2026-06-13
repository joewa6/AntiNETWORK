from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

POSITIVE = set("KRH")
NEGATIVE = set("DE")
AROMATIC = set("FWY")
HYDROPHOBIC = set("AILMFWYVC")
ALIPHATIC_INDEX_WEIGHTS = {"A": 1.0, "V": 2.9, "I": 3.9, "L": 3.9}
HYDROPATHY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}
CANONICAL_AA = set(HYDROPATHY)


@dataclass(frozen=True)
class Region:
    name: str
    start: int
    stop: int


# Heuristic AHo-alignment windows. These are intentionally crude kill-test
# proxies, not a replacement for curated antibody numbering.
HEAVY_CDR_REGIONS = [
    Region("cdr_h1", 25, 42),
    Region("cdr_h2", 56, 76),
    Region("cdr_h3", 105, 138),
]
LIGHT_CDR_REGIONS = [
    Region("cdr_l1", 24, 42),
    Region("cdr_l2", 56, 72),
    Region("cdr_l3", 104, 130),
]


def clean_sequence(sequence: str) -> str:
    """Remove alignment gaps, separators, and stop codons."""
    return str(sequence).replace("-", "").replace("|", "").replace("*", "")


def _max_run(seq: str, residues: set[str]) -> int:
    longest = 0
    current = 0
    for residue in seq:
        if residue in residues:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _shannon_entropy(seq: str) -> float:
    counts = np.array([seq.count(residue) for residue in CANONICAL_AA], dtype=float)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return np.nan
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _opposite_charge_adjacencies(seq: str) -> int:
    pairs = zip(seq, seq[1:], strict=False)
    return sum((a in POSITIVE and b in NEGATIVE) or (a in NEGATIVE and b in POSITIVE) for a, b in pairs)


def residue_features(sequence: str, prefix: str) -> dict[str, float]:
    """Compute simple composition, charge, and stability-proxy descriptors."""
    seq = clean_sequence(sequence)
    length = len(seq)
    if length == 0:
        return {
            f"{prefix}_length": 0.0,
            f"{prefix}_net_charge": 0.0,
            f"{prefix}_positive_count": 0.0,
            f"{prefix}_negative_count": 0.0,
            f"{prefix}_aromatic_count": 0.0,
            f"{prefix}_hydrophobic_count": 0.0,
            f"{prefix}_aliphatic_count": 0.0,
            f"{prefix}_proline_count": 0.0,
            f"{prefix}_glycine_count": 0.0,
            f"{prefix}_cysteine_count": 0.0,
            f"{prefix}_predicted_disulfide_pairs": 0.0,
            f"{prefix}_lys_arg_asymmetry": 0.0,
            f"{prefix}_acid_base_imbalance": 0.0,
            f"{prefix}_absolute_charge_fraction": np.nan,
            f"{prefix}_charge_density": np.nan,
            f"{prefix}_hydrophobic_fraction": np.nan,
            f"{prefix}_aromatic_fraction": np.nan,
            f"{prefix}_aliphatic_fraction": np.nan,
            f"{prefix}_proline_fraction": np.nan,
            f"{prefix}_glycine_fraction": np.nan,
            f"{prefix}_cysteine_fraction": np.nan,
            f"{prefix}_proline_minus_glycine_fraction": np.nan,
            f"{prefix}_gravy": np.nan,
            f"{prefix}_aliphatic_index": np.nan,
            f"{prefix}_composition_entropy": np.nan,
            f"{prefix}_max_hydrophobic_run_fraction": np.nan,
            f"{prefix}_max_flexible_run_fraction": np.nan,
            f"{prefix}_opposite_charge_adjacency_fraction": np.nan,
        }

    positive = sum(residue in POSITIVE for residue in seq)
    negative = sum(residue in NEGATIVE for residue in seq)
    aromatic = sum(residue in AROMATIC for residue in seq)
    hydrophobic = sum(residue in HYDROPHOBIC for residue in seq)
    aliphatic = sum(residue in ALIPHATIC_INDEX_WEIGHTS for residue in seq)
    proline = seq.count("P")
    glycine = seq.count("G")
    cysteine = seq.count("C")
    lys = seq.count("K")
    arg = seq.count("R")
    hydropathy_values = [HYDROPATHY[residue] for residue in seq if residue in HYDROPATHY]
    aliphatic_index = 100.0 * sum(
        ALIPHATIC_INDEX_WEIGHTS.get(residue, 0.0) for residue in seq
    ) / length

    return {
        f"{prefix}_length": float(length),
        f"{prefix}_net_charge": float(positive - negative),
        f"{prefix}_positive_count": float(positive),
        f"{prefix}_negative_count": float(negative),
        f"{prefix}_aromatic_count": float(aromatic),
        f"{prefix}_hydrophobic_count": float(hydrophobic),
        f"{prefix}_aliphatic_count": float(aliphatic),
        f"{prefix}_proline_count": float(proline),
        f"{prefix}_glycine_count": float(glycine),
        f"{prefix}_cysteine_count": float(cysteine),
        f"{prefix}_predicted_disulfide_pairs": float(cysteine // 2),
        f"{prefix}_lys_arg_asymmetry": float(lys - arg),
        f"{prefix}_acid_base_imbalance": float(abs(positive - negative)),
        f"{prefix}_absolute_charge_fraction": (positive + negative) / length,
        f"{prefix}_charge_density": (positive - negative) / length,
        f"{prefix}_hydrophobic_fraction": hydrophobic / length,
        f"{prefix}_aromatic_fraction": aromatic / length,
        f"{prefix}_aliphatic_fraction": aliphatic / length,
        f"{prefix}_proline_fraction": proline / length,
        f"{prefix}_glycine_fraction": glycine / length,
        f"{prefix}_cysteine_fraction": cysteine / length,
        f"{prefix}_proline_minus_glycine_fraction": (proline - glycine) / length,
        f"{prefix}_gravy": float(np.mean(hydropathy_values)) if hydropathy_values else np.nan,
        f"{prefix}_aliphatic_index": float(aliphatic_index),
        f"{prefix}_composition_entropy": _shannon_entropy(seq),
        f"{prefix}_max_hydrophobic_run_fraction": _max_run(seq, HYDROPHOBIC) / length,
        f"{prefix}_max_flexible_run_fraction": _max_run(seq, {"G", "P"}) / length,
        f"{prefix}_opposite_charge_adjacency_fraction": _opposite_charge_adjacencies(seq)
        / max(length - 1, 1),
    }


def region_sequence(aligned_sequence: str, region: Region) -> str:
    return str(aligned_sequence)[region.start : region.stop]


def build_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build crude physically interpretable sequence features from GDPa1 rows."""
    rows: list[dict[str, float]] = []

    for _, row in df.iterrows():
        features: dict[str, float] = {}

        vh = row["vh_protein_sequence"]
        vl = row["vl_protein_sequence"]
        heavy_aligned = row["heavy_aligned_aho"]
        light_aligned = row["light_aligned_aho"]

        features.update(residue_features(vh, "vh"))
        features.update(residue_features(vl, "vl"))

        heavy_cdr = "".join(region_sequence(heavy_aligned, region) for region in HEAVY_CDR_REGIONS)
        light_cdr = "".join(region_sequence(light_aligned, region) for region in LIGHT_CDR_REGIONS)
        features.update(residue_features(heavy_cdr, "heavy_cdr"))
        features.update(residue_features(light_cdr, "light_cdr"))
        features.update(residue_features(heavy_cdr + light_cdr, "fv_cdr"))

        cdr_h3 = region_sequence(heavy_aligned, HEAVY_CDR_REGIONS[-1])
        features["cdr_h3_length"] = float(len(clean_sequence(cdr_h3)))
        features["fv_net_charge"] = features["vh_net_charge"] + features["vl_net_charge"]
        features["vh_vl_charge_imbalance"] = abs(features["vh_net_charge"] - features["vl_net_charge"])
        features["vh_vl_hydrophobicity_imbalance"] = abs(
            features["vh_hydrophobic_fraction"] - features["vl_hydrophobic_fraction"]
        )
        features["vh_vl_aliphatic_index_imbalance"] = abs(
            features["vh_aliphatic_index"] - features["vl_aliphatic_index"]
        )
        features["vh_vl_gravy_imbalance"] = abs(features["vh_gravy"] - features["vl_gravy"])
        features["vh_vl_length_imbalance"] = abs(features["vh_length"] - features["vl_length"])
        features["cdr_charge_fraction_of_fv"] = features["fv_cdr_net_charge"] / max(
            abs(features["fv_net_charge"]), 1.0
        )
        features["fv_predicted_disulfide_pairs"] = (
            features["vh_predicted_disulfide_pairs"] + features["vl_predicted_disulfide_pairs"]
        )
        features["fv_stability_composition_index"] = (
            features["vh_aliphatic_index"]
            + features["vl_aliphatic_index"]
            + 100.0 * features["fv_cdr_proline_minus_glycine_fraction"]
            + 25.0 * features["fv_cdr_aromatic_fraction"]
            - 25.0 * features["fv_cdr_max_flexible_run_fraction"]
        )

        rows.append(features)

    return pd.DataFrame(rows, index=df.index)


def aligned_identity(seq_a: str, seq_b: str) -> float:
    """Compute identity between two aligned sequences, ignoring gap-gap/gap-residue sites."""
    comparable = [(a, b) for a, b in zip(str(seq_a), str(seq_b)) if a != "-" and b != "-"]
    if not comparable:
        return np.nan
    return float(sum(a == b for a, b in comparable) / len(comparable))


def combined_aligned_sequences(df: pd.DataFrame) -> pd.Series:
    return df["heavy_aligned_aho"].astype(str) + "|" + df["light_aligned_aho"].astype(str)


def pairwise_aligned_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dense pairwise identity matrix from heavy+light AHo alignments."""
    combined = combined_aligned_sequences(df).reset_index(drop=True)
    values = np.eye(len(combined), dtype=float)
    for i, seq_i in enumerate(combined):
        for j in range(i + 1, len(combined)):
            identity = aligned_identity(seq_i, combined.iloc[j])
            values[i, j] = identity
            values[j, i] = identity
    return pd.DataFrame(values, index=df.index, columns=df.index)
