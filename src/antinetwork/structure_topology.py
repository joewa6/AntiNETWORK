from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

from antinetwork.features import HEAVY_CDR_REGIONS, LIGHT_CDR_REGIONS, Region

POSITIVE_RESIDUES = {"LYS", "ARG", "HIS"}
NEGATIVE_RESIDUES = {"ASP", "GLU"}
THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def residue_charge(resname: str, ph: float = 7.4) -> float:
    """Approximate residue charge for crude topology descriptors."""
    if resname in {"LYS", "ARG"}:
        return 1.0
    if resname == "HIS":
        return 0.5 if ph <= 6.0 else 0.05
    if resname in NEGATIVE_RESIDUES:
        return -1.0
    return 0.0


def residue_coordinate(residue) -> np.ndarray | None:
    """Use CB as a side-chain proxy, falling back to CA."""
    if "CB" in residue:
        return residue["CB"].coord.astype(float)
    if "CA" in residue:
        return residue["CA"].coord.astype(float)
    atoms = list(residue.get_atoms())
    if not atoms:
        return None
    return np.mean([atom.coord for atom in atoms], axis=0).astype(float)


def aligned_region_indices(aligned_sequence: str, regions: list[Region]) -> set[int]:
    """Map aligned CDR windows to ungapped sequence indices."""
    cdr_aligned_positions = set()
    for region in regions:
        cdr_aligned_positions.update(range(region.start, region.stop))

    ungapped_idx = -1
    indices = set()
    for aligned_idx, residue in enumerate(str(aligned_sequence)):
        if residue == "-":
            continue
        ungapped_idx += 1
        if aligned_idx in cdr_aligned_positions:
            indices.add(ungapped_idx)
    return indices


def parse_structure_residues(
    pdb_path: Path,
    heavy_aligned_aho: str,
    light_aligned_aho: str,
) -> pd.DataFrame:
    """Parse heavy/light PDB residues into coordinate and CDR annotations."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = structure[0]
    heavy_cdr = aligned_region_indices(heavy_aligned_aho, HEAVY_CDR_REGIONS)
    light_cdr = aligned_region_indices(light_aligned_aho, LIGHT_CDR_REGIONS)

    rows = []
    for chain_id, cdr_indices in [("H", heavy_cdr), ("L", light_cdr)]:
        if chain_id not in model:
            continue
        sequence_index = -1
        for residue in model[chain_id]:
            if residue.id[0] != " ":
                continue
            resname = residue.resname.strip()
            one = THREE_TO_ONE.get(resname)
            if one is None:
                continue
            coord = residue_coordinate(residue)
            if coord is None:
                continue
            sequence_index += 1
            rows.append(
                {
                    "chain": chain_id,
                    "sequence_index": sequence_index,
                    "resname": resname,
                    "aa": one,
                    "x": coord[0],
                    "y": coord[1],
                    "z": coord[2],
                    "is_cdr": sequence_index in cdr_indices,
                }
            )
    return pd.DataFrame(rows)


def surface_mask(coords: np.ndarray, radius: float = 10.0, quantile: float = 0.45) -> np.ndarray:
    """Approximate surface residues by low local CA/CB neighbor density."""
    if len(coords) == 0:
        return np.array([], dtype=bool)
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    neighbor_counts = ((distances < radius) & (distances > 0)).sum(axis=1)
    threshold = np.quantile(neighbor_counts, quantile)
    return neighbor_counts <= threshold


def patch_components(
    residues: pd.DataFrame,
    sign: str,
    ph: float,
    distance_cutoff: float = 10.0,
) -> list[set[int]]:
    charges = residues["resname"].map(lambda name: residue_charge(name, ph)).to_numpy()
    if sign == "positive":
        selected = residues.index[charges > 0].tolist()
    elif sign == "negative":
        selected = residues.index[charges < 0].tolist()
    else:
        raise ValueError(f"Unknown sign: {sign}")

    graph = nx.Graph()
    graph.add_nodes_from(selected)
    coords = residues.loc[selected, ["x", "y", "z"]].to_numpy(dtype=float)
    for i, source in enumerate(selected):
        for j, target in enumerate(selected[i + 1 :], start=i + 1):
            if np.linalg.norm(coords[i] - coords[j]) <= distance_cutoff:
                graph.add_edge(source, target)
    return [set(component) for component in nx.connected_components(graph)]


def patch_summary(residues: pd.DataFrame, sign: str, ph: float) -> dict[str, float]:
    components = patch_components(residues, sign=sign, ph=ph)
    if not components:
        return {
            f"{sign}_patch_count": 0.0,
            f"largest_{sign}_patch_size": 0.0,
            f"largest_{sign}_patch_cdr_fraction": np.nan,
            f"{sign}_patch_mean_size": 0.0,
        }
    sizes = np.array([len(component) for component in components], dtype=float)
    largest = components[int(np.argmax(sizes))]
    cdr_fraction = residues.loc[list(largest), "is_cdr"].mean()
    return {
        f"{sign}_patch_count": float(len(components)),
        f"largest_{sign}_patch_size": float(max(sizes)),
        f"largest_{sign}_patch_cdr_fraction": float(cdr_fraction),
        f"{sign}_patch_mean_size": float(sizes.mean()),
    }


def charge_topology_features_for_structure(
    pdb_path: Path,
    heavy_aligned_aho: str,
    light_aligned_aho: str,
    ph_values: tuple[float, ...] = (6.0, 7.4),
) -> dict[str, float]:
    """Compute crude structure-derived charge topology features for one Fv PDB."""
    parsed = parse_structure_residues(pdb_path, heavy_aligned_aho, light_aligned_aho)
    if parsed.empty:
        return {"structure_available": 0.0}

    coords = parsed[["x", "y", "z"]].to_numpy(dtype=float)
    parsed = parsed.copy()
    parsed["is_surface_proxy"] = surface_mask(coords)
    surface = parsed[parsed["is_surface_proxy"]].copy()
    if surface.empty:
        surface = parsed.copy()

    out: dict[str, float] = {
        "structure_available": 1.0,
        "structure_residue_count": float(len(parsed)),
        "surface_proxy_residue_count": float(len(surface)),
        "surface_proxy_fraction": float(len(surface) / len(parsed)),
    }

    center = surface[["x", "y", "z"]].to_numpy(dtype=float).mean(axis=0)
    for ph in ph_values:
        key = str(ph).replace(".", "p")
        charges = surface["resname"].map(lambda name: residue_charge(name, ph)).to_numpy(dtype=float)
        charge_coords = surface[["x", "y", "z"]].to_numpy(dtype=float)
        abs_charge = np.abs(charges)
        charged = abs_charge > 0
        positive = charges > 0
        negative = charges < 0

        out[f"pH{key}_surface_net_charge"] = float(charges.sum())
        out[f"pH{key}_surface_positive_charge"] = float(charges[positive].sum())
        out[f"pH{key}_surface_negative_charge"] = float(charges[negative].sum())
        out[f"pH{key}_surface_abs_charge"] = float(abs_charge.sum())
        out[f"pH{key}_surface_cdr_charge_fraction"] = float(
            abs_charge[surface["is_cdr"].to_numpy(dtype=bool)].sum() / max(abs_charge.sum(), 1.0)
        )

        if positive.any() and negative.any():
            pos_centroid = np.average(charge_coords[positive], axis=0, weights=charges[positive])
            neg_centroid = np.average(charge_coords[negative], axis=0, weights=np.abs(charges[negative]))
            out[f"pH{key}_positive_negative_centroid_distance"] = float(
                np.linalg.norm(pos_centroid - neg_centroid)
            )
        else:
            out[f"pH{key}_positive_negative_centroid_distance"] = np.nan

        if charged.any():
            weighted_vectors = charges[:, None] * (charge_coords - center)
            dipole = weighted_vectors.sum(axis=0) / max(abs_charge.sum(), 1.0)
            radial = np.linalg.norm(charge_coords[charged] - center, axis=1)
            out[f"pH{key}_charge_dipole_proxy"] = float(np.linalg.norm(dipole))
            out[f"pH{key}_charge_radial_variance_proxy"] = float(np.var(radial))
        else:
            out[f"pH{key}_charge_dipole_proxy"] = np.nan
            out[f"pH{key}_charge_radial_variance_proxy"] = np.nan

        for sign in ["positive", "negative"]:
            summary = patch_summary(surface, sign=sign, ph=ph)
            out.update({f"pH{key}_{name}": value for name, value in summary.items()})

    return out


def structure_path_for_antibody(structure_dir: Path, antibody_name: str) -> Path:
    return structure_dir / f"{str(antibody_name).lower()}.pdb"


def build_charge_topology_features(
    df: pd.DataFrame,
    structure_dir: str | Path = "data/raw/hf_snapshot/structures",
) -> pd.DataFrame:
    """Build charge topology features for all antibodies with local PDBs."""
    structure_dir = Path(structure_dir)
    rows = []
    for _, row in df.iterrows():
        pdb_path = structure_path_for_antibody(structure_dir, row["antibody_name"])
        if not pdb_path.exists():
            rows.append(
                {
                    "antibody_name": row["antibody_name"],
                    "pdb_path": None,
                    "structure_available": 0.0,
                }
            )
            continue
        features = charge_topology_features_for_structure(
            pdb_path,
            row["heavy_aligned_aho"],
            row["light_aligned_aho"],
        )
        features["antibody_name"] = row["antibody_name"]
        features["pdb_path"] = str(pdb_path)
        rows.append(features)

    return pd.DataFrame(rows, index=df.index)


def topology_feature_groups(columns: pd.Index | list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        name = str(column)
        if name in {"antibody_name", "pdb_path", "structure_available"}:
            continue
        if "pH6p0" in name:
            ph = "pH6.0"
        elif "pH7p4" in name:
            ph = "pH7.4"
        else:
            ph = "global"

        if "patch" in name:
            family = "patch"
        elif "dipole" in name or "radial" in name or "centroid_distance" in name:
            family = "topology_asymmetry"
        elif "surface" in name or "charge" in name:
            family = "surface_charge"
        else:
            family = "structure_quality"

        rows.append({"feature": name, "ph": ph, "topology_family": family})
    return pd.DataFrame(rows).set_index("feature")
