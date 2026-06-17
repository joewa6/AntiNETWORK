from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

from antinetwork.features import (
    AROMATIC,
    HEAVY_CDR_REGIONS,
    HYDROPATHY,
    HYDROPHOBIC,
    LIGHT_CDR_REGIONS,
    Region,
    clean_sequence,
    region_sequence,
)

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


def parse_structure_atoms(
    pdb_path: Path,
    heavy_aligned_aho: str,
    light_aligned_aho: str,
) -> pd.DataFrame:
    """Parse heavy/light atoms with residue-level CDR annotations."""
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
            sequence_index += 1
            for atom in residue.get_atoms():
                element = (atom.element or atom.name[0]).strip().upper()
                rows.append(
                    {
                        "chain": chain_id,
                        "sequence_index": sequence_index,
                        "resname": resname,
                        "aa": one,
                        "atom_name": atom.name.strip(),
                        "element": element,
                        "x": float(atom.coord[0]),
                        "y": float(atom.coord[1]),
                        "z": float(atom.coord[2]),
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


def residue_neighbor_counts(coords: np.ndarray, radius: float = 8.0) -> np.ndarray:
    """Count local residue-coordinate neighbors for packing-density proxies."""
    if len(coords) == 0:
        return np.array([], dtype=float)
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    return ((distances <= radius) & (distances > 0)).sum(axis=1).astype(float)


def _charged_mask(residues: pd.DataFrame, sign: str, ph: float = 7.4) -> np.ndarray:
    charges = residues["resname"].map(lambda name: residue_charge(name, ph)).to_numpy(dtype=float)
    if sign == "positive":
        return charges > 0
    if sign == "negative":
        return charges < 0
    if sign == "charged":
        return np.abs(charges) > 0
    raise ValueError(f"Unknown charge sign: {sign}")


def _component_sizes(
    residues: pd.DataFrame,
    mask: np.ndarray,
    distance_cutoff: float = 10.0,
) -> list[int]:
    selected = np.flatnonzero(mask)
    if len(selected) == 0:
        return []
    graph = nx.Graph()
    graph.add_nodes_from(selected.tolist())
    coords = residues.iloc[selected][["x", "y", "z"]].to_numpy(dtype=float)
    for local_i, source in enumerate(selected):
        for local_j, target in enumerate(selected[local_i + 1 :], start=local_i + 1):
            if np.linalg.norm(coords[local_i] - coords[local_j]) <= distance_cutoff:
                graph.add_edge(int(source), int(target))
    return [len(component) for component in nx.connected_components(graph)]


def _patch_size_summary(
    residues: pd.DataFrame,
    mask: np.ndarray,
    prefix: str,
    distance_cutoff: float = 10.0,
) -> dict[str, float]:
    sizes = _component_sizes(residues, mask, distance_cutoff=distance_cutoff)
    if not sizes:
        return {
            f"{prefix}_patch_count": 0.0,
            f"{prefix}_largest_patch_size": 0.0,
            f"{prefix}_mean_patch_size": 0.0,
        }
    values = np.array(sizes, dtype=float)
    return {
        f"{prefix}_patch_count": float(len(values)),
        f"{prefix}_largest_patch_size": float(values.max()),
        f"{prefix}_mean_patch_size": float(values.mean()),
    }


def _contact_pairs(
    left: pd.DataFrame,
    right: pd.DataFrame,
    distance_cutoff: float,
) -> list[tuple[int, int, float]]:
    if left.empty or right.empty:
        return []
    left_coords = left[["x", "y", "z"]].to_numpy(dtype=float)
    right_coords = right[["x", "y", "z"]].to_numpy(dtype=float)
    distances = np.linalg.norm(left_coords[:, None, :] - right_coords[None, :, :], axis=2)
    pairs = np.argwhere(distances <= distance_cutoff)
    left_index = left.index.to_numpy()
    right_index = right.index.to_numpy()
    return [(int(left_index[i]), int(right_index[j]), float(distances[i, j])) for i, j in pairs]


def _interface_features(residues: pd.DataFrame, distance_cutoff: float = 8.0) -> dict[str, float]:
    heavy = residues[residues["chain"] == "H"]
    light = residues[residues["chain"] == "L"]
    pairs = _contact_pairs(heavy, light, distance_cutoff=distance_cutoff)
    interface_nodes = sorted({idx for pair in pairs for idx in pair[:2]})
    interface = residues.loc[interface_nodes] if interface_nodes else residues.iloc[[]]
    pair_resnames = [(residues.loc[i, "resname"], residues.loc[j, "resname"]) for i, j, _ in pairs]
    pair_aas = [(residues.loc[i, "aa"], residues.loc[j, "aa"]) for i, j, _ in pairs]

    hydrophobic_contacts = sum(a in HYDROPHOBIC and b in HYDROPHOBIC for a, b in pair_aas)
    aromatic_contacts = sum(a in AROMATIC and b in AROMATIC for a, b in pair_aas)
    salt_bridges = sum(
        (a in POSITIVE_RESIDUES and b in NEGATIVE_RESIDUES)
        or (a in NEGATIVE_RESIDUES and b in POSITIVE_RESIDUES)
        for a, b in pair_resnames
    )
    charge_pairs = sum(
        (a in POSITIVE_RESIDUES | NEGATIVE_RESIDUES) and (b in POSITIVE_RESIDUES | NEGATIVE_RESIDUES)
        for a, b in pair_resnames
    )

    if interface.empty:
        degrees = np.array([], dtype=float)
    else:
        degrees = residue_neighbor_counts(interface[["x", "y", "z"]].to_numpy(dtype=float), radius=distance_cutoff)

    return {
        "interface_contact_count": float(len(pairs)),
        "interface_residue_count": float(len(interface)),
        "interface_heavy_residue_count": float((interface["chain"] == "H").sum()) if len(interface) else 0.0,
        "interface_light_residue_count": float((interface["chain"] == "L").sum()) if len(interface) else 0.0,
        "interface_hydrophobic_contact_count": float(hydrophobic_contacts),
        "interface_aromatic_contact_count": float(aromatic_contacts),
        "interface_salt_bridge_count": float(salt_bridges),
        "interface_charge_pair_count": float(charge_pairs),
        "interface_hydrophobic_fraction": float(interface["aa"].isin(HYDROPHOBIC).mean())
        if len(interface)
        else np.nan,
        "interface_aromatic_fraction": float(interface["aa"].isin(AROMATIC).mean()) if len(interface) else np.nan,
        "interface_mean_packing_degree": float(degrees.mean()) if len(degrees) else np.nan,
        "interface_min_packing_degree": float(degrees.min()) if len(degrees) else np.nan,
        "interface_max_packing_degree": float(degrees.max()) if len(degrees) else np.nan,
    }


def _interface_graph_features(residues: pd.DataFrame, distance_cutoff: float = 8.0) -> dict[str, float]:
    pairs = _contact_pairs(
        residues[residues["chain"] == "H"],
        residues[residues["chain"] == "L"],
        distance_cutoff=distance_cutoff,
    )
    nodes = sorted({idx for pair in pairs for idx in pair[:2]})
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((left, right) for left, right, _ in pairs)
    if graph.number_of_nodes() == 0:
        return {
            "interface_graph_node_count": 0.0,
            "interface_graph_edge_count": 0.0,
            "interface_graph_mean_degree": np.nan,
            "interface_graph_max_degree": np.nan,
            "interface_graph_mean_betweenness": np.nan,
            "interface_graph_max_betweenness": np.nan,
            "interface_graph_mean_clustering": np.nan,
            "interface_graph_diameter": np.nan,
        }

    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    betweenness = np.array(list(nx.betweenness_centrality(graph).values()), dtype=float)
    clustering = np.array(list(nx.clustering(graph).values()), dtype=float)
    if nx.is_connected(graph):
        diameter = float(nx.diameter(graph))
    else:
        component = max(nx.connected_components(graph), key=len)
        diameter = float(nx.diameter(graph.subgraph(component))) if len(component) > 1 else 0.0
    return {
        "interface_graph_node_count": float(graph.number_of_nodes()),
        "interface_graph_edge_count": float(graph.number_of_edges()),
        "interface_graph_mean_degree": float(degrees.mean()),
        "interface_graph_max_degree": float(degrees.max()),
        "interface_graph_mean_betweenness": float(betweenness.mean()),
        "interface_graph_max_betweenness": float(betweenness.max()),
        "interface_graph_mean_clustering": float(clustering.mean()),
        "interface_graph_diameter": diameter,
    }


def _surface_stability_features(residues: pd.DataFrame) -> dict[str, float]:
    coords = residues[["x", "y", "z"]].to_numpy(dtype=float)
    residues = residues.copy()
    residues["is_surface_proxy"] = surface_mask(coords)
    surface = residues[residues["is_surface_proxy"]].copy()
    if surface.empty:
        surface = residues.copy()

    hydropathy = surface["aa"].map(HYDROPATHY).fillna(0.0).to_numpy(dtype=float)
    hydrophobic = surface["aa"].isin(HYDROPHOBIC).to_numpy()
    aromatic = surface["aa"].isin(AROMATIC).to_numpy()
    charges = surface["resname"].map(lambda name: residue_charge(name, 7.4)).to_numpy(dtype=float)
    charge_coords = surface[["x", "y", "z"]].to_numpy(dtype=float)
    center = charge_coords.mean(axis=0)
    abs_charge = np.abs(charges)
    dipole = (charges[:, None] * (charge_coords - center)).sum(axis=0) / max(abs_charge.sum(), 1.0)

    out = {
        "surface_proxy_residue_count": float(len(surface)),
        "surface_hydrophobic_fraction": float(hydrophobic.mean()) if len(surface) else np.nan,
        "surface_aromatic_fraction": float(aromatic.mean()) if len(surface) else np.nan,
        "surface_hydrophobicity_sum": float(np.clip(hydropathy, 0.0, None).sum()),
        "surface_hydrophobicity_mean": float(hydropathy.mean()) if len(surface) else np.nan,
        "surface_aromatic_count": float(aromatic.sum()),
        "surface_net_charge": float(charges.sum()),
        "surface_abs_charge": float(abs_charge.sum()),
        "surface_charge_variance": float(np.var(charges)) if len(charges) else np.nan,
        "surface_charge_dipole_proxy": float(np.linalg.norm(dipole)),
        "surface_charge_asymmetry": float(abs(charges.sum()) / max(abs_charge.sum(), 1.0)),
    }
    out.update(_patch_size_summary(surface, hydrophobic, "hydrophobic"))
    out.update(_patch_size_summary(surface, aromatic, "aromatic"))
    out.update(_patch_size_summary(surface, _charged_mask(surface, "positive"), "positive"))
    out.update(_patch_size_summary(surface, _charged_mask(surface, "negative"), "negative"))
    return out


def _cdr_flexibility_features(
    heavy_aligned_aho: str,
    light_aligned_aho: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for chain_prefix, aligned, regions in [
        ("h", heavy_aligned_aho, HEAVY_CDR_REGIONS),
        ("l", light_aligned_aho, LIGHT_CDR_REGIONS),
    ]:
        for region in regions:
            seq = clean_sequence(region_sequence(aligned, region))
            key = region.name.replace("cdr_", "")
            out[f"cdr_{key}_length"] = float(len(seq))
            out[f"cdr_{key}_glycine_count"] = float(seq.count("G"))
            out[f"cdr_{key}_proline_count"] = float(seq.count("P"))
            out[f"cdr_{key}_glycine_fraction"] = seq.count("G") / len(seq) if seq else np.nan
            out[f"cdr_{key}_proline_fraction"] = seq.count("P") / len(seq) if seq else np.nan
            out[f"cdr_{key}_loop_entropy_proxy"] = (seq.count("G") - seq.count("P")) / len(seq) if seq else np.nan
        chain_cdr = clean_sequence("".join(region_sequence(aligned, region) for region in regions))
        out[f"{chain_prefix}_cdr_loop_entropy_proxy"] = (
            (chain_cdr.count("G") - chain_cdr.count("P")) / len(chain_cdr) if chain_cdr else np.nan
        )
    return out


def _weak_region_features(residues: pd.DataFrame) -> dict[str, float]:
    coords = residues[["x", "y", "z"]].to_numpy(dtype=float)
    packing = residue_neighbor_counts(coords, radius=8.0)
    residues = residues.copy()
    residues["packing_degree"] = packing
    out = {
        "packing_mean_degree": float(np.mean(packing)) if len(packing) else np.nan,
        "packing_min_degree": float(np.min(packing)) if len(packing) else np.nan,
        "packing_worst10_mean_degree": float(np.mean(np.sort(packing)[:10])) if len(packing) else np.nan,
        "packing_worst20_mean_degree": float(np.mean(np.sort(packing)[:20])) if len(packing) else np.nan,
    }
    for region_name, region in {"cdr": residues[residues["is_cdr"]], "framework": residues[~residues["is_cdr"]]}.items():
        values = region["packing_degree"].to_numpy(dtype=float)
        out[f"{region_name}_packing_min_degree"] = float(values.min()) if len(values) else np.nan
        out[f"{region_name}_packing_worst10_mean_degree"] = (
            float(np.mean(np.sort(values)[:10])) if len(values) else np.nan
        )
    return out


def _buried_residue_mask(residues: pd.DataFrame) -> np.ndarray:
    coords = residues[["x", "y", "z"]].to_numpy(dtype=float)
    surface = surface_mask(coords)
    return ~surface


def _atom_neighbor_counts(atoms: pd.DataFrame, radius: float = 4.5) -> np.ndarray:
    if atoms.empty:
        return np.array([], dtype=float)
    coords = atoms[["x", "y", "z"]].to_numpy(dtype=float)
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    return ((distances <= radius) & (distances > 0)).sum(axis=1).astype(float)


def _unsatisfied_polar_proxy_count(
    atoms: pd.DataFrame,
    buried_atom_mask: np.ndarray,
    hbond_cutoff: float = 3.5,
) -> float:
    atoms = atoms.reset_index(drop=True)
    polar = atoms["element"].isin(["N", "O"]).to_numpy()
    selected = np.flatnonzero(polar & buried_atom_mask)
    if len(selected) == 0:
        return 0.0
    polar_coords = atoms.loc[polar, ["x", "y", "z"]].to_numpy(dtype=float)
    polar_index = atoms.index[polar].to_numpy()
    out = 0
    for atom_idx in selected:
        coord = atoms.loc[atom_idx, ["x", "y", "z"]].to_numpy(dtype=float)
        distances = np.linalg.norm(polar_coords - coord, axis=1)
        partners = polar_index[(distances <= hbond_cutoff) & (distances > 0.1)]
        if len(partners) == 0:
            out += 1
    return float(out)


def _defect_features(
    residues: pd.DataFrame,
    atoms: pd.DataFrame,
    heavy_aligned_aho: str,
) -> dict[str, float]:
    residues = residues.copy()
    buried = _buried_residue_mask(residues)
    coords = residues[["x", "y", "z"]].to_numpy(dtype=float)
    packing = residue_neighbor_counts(coords, radius=8.0)
    residues["is_buried_proxy"] = buried
    residues["packing_degree"] = packing
    buried_residues = residues[buried]

    atom_counts = _atom_neighbor_counts(atoms)
    if len(atom_counts):
        buried_atom_threshold = np.quantile(atom_counts, 0.75)
        buried_atoms = atom_counts >= buried_atom_threshold
    else:
        buried_atoms = np.array([], dtype=bool)

    h3_indices = aligned_region_indices(heavy_aligned_aho, [HEAVY_CDR_REGIONS[-1]])
    h3 = residues[(residues["chain"] == "H") & (residues["sequence_index"].isin(h3_indices))]
    h3_buried = h3[h3["is_buried_proxy"]]
    h3_hydrophobic = h3["aa"].isin(HYDROPHOBIC).to_numpy()
    h3_packing = h3["packing_degree"].to_numpy(dtype=float)

    interface_pairs = _contact_pairs(
        residues[residues["chain"] == "H"],
        residues[residues["chain"] == "L"],
        distance_cutoff=8.0,
    )
    interface_nodes = sorted({idx for pair in interface_pairs for idx in pair[:2]})
    interface = residues.loc[interface_nodes] if interface_nodes else residues.iloc[[]]
    if interface.empty:
        interface_buried_charged = 0.0
        interface_unsat = 0.0
        interface_worst5 = np.nan
    else:
        interface_buried_charged = float(
            (interface["is_buried_proxy"] & interface["resname"].isin(POSITIVE_RESIDUES | NEGATIVE_RESIDUES)).sum()
        )
        interface_keys = set(zip(interface["chain"], interface["sequence_index"], strict=False))
        atom_interface = atoms[
            [key in interface_keys for key in zip(atoms["chain"], atoms["sequence_index"], strict=False)]
        ]
        interface_atom_counts = _atom_neighbor_counts(atom_interface)
        if len(interface_atom_counts):
            interface_buried_atoms = interface_atom_counts >= np.quantile(interface_atom_counts, 0.75)
            interface_unsat = _unsatisfied_polar_proxy_count(atom_interface, interface_buried_atoms)
        else:
            interface_unsat = 0.0
        interface_worst5 = float(np.mean(np.sort(interface["packing_degree"].to_numpy(dtype=float))[:5]))

    out = {
        "defect_buried_charged_residue_count": float(
            buried_residues["resname"].isin(POSITIVE_RESIDUES | NEGATIVE_RESIDUES).sum()
        ),
        "defect_buried_positive_residue_count": float(buried_residues["resname"].isin(POSITIVE_RESIDUES).sum()),
        "defect_buried_negative_residue_count": float(buried_residues["resname"].isin(NEGATIVE_RESIDUES).sum()),
        "defect_unsatisfied_polar_atom_proxy_count": _unsatisfied_polar_proxy_count(atoms, buried_atoms),
        "defect_worst5_packing_mean_degree": float(np.mean(np.sort(packing)[:5])) if len(packing) else np.nan,
        "defect_worst10_packing_mean_degree": float(np.mean(np.sort(packing)[:10])) if len(packing) else np.nan,
        "defect_h3_buried_glycine_count": float((h3_buried["aa"] == "G").sum()),
        "defect_h3_buried_hydrophobic_count": float(h3_buried["aa"].isin(HYDROPHOBIC).sum()),
        "defect_h3_hydrophobic_fraction": float(h3_hydrophobic.mean()) if len(h3_hydrophobic) else np.nan,
        "defect_h3_contact_density": float(h3_packing.mean()) if len(h3_packing) else np.nan,
        "defect_h3_min_packing_degree": float(h3_packing.min()) if len(h3_packing) else np.nan,
        "defect_interface_buried_charged_residue_count": interface_buried_charged,
        "defect_interface_unsatisfied_polar_atom_proxy_count": interface_unsat,
        "defect_interface_worst5_packing_mean_degree": interface_worst5,
    }
    return out


def structure_stability_features_for_structure(
    pdb_path: Path,
    heavy_aligned_aho: str,
    light_aligned_aho: str,
) -> dict[str, float]:
    """Build structure-derived Fab stability proxies for Tm-style experiments."""
    residues = parse_structure_residues(pdb_path, heavy_aligned_aho, light_aligned_aho)
    if residues.empty:
        return {"structure_available": 0.0}
    atoms = parse_structure_atoms(pdb_path, heavy_aligned_aho, light_aligned_aho)

    out: dict[str, float] = {
        "structure_available": 1.0,
        "structure_residue_count": float(len(residues)),
    }
    out.update(_interface_features(residues))
    out.update(_surface_stability_features(residues))
    out.update(_cdr_flexibility_features(heavy_aligned_aho, light_aligned_aho))
    out.update(_weak_region_features(residues))
    out.update(_interface_graph_features(residues))
    out.update(_defect_features(residues, atoms, heavy_aligned_aho))
    return out


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
    name = str(antibody_name)
    candidates = [
        structure_dir / f"{name}.pdb",
        structure_dir / f"{name.lower()}.pdb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    target = f"{name.casefold()}.pdb"
    for candidate in structure_dir.glob("*.pdb"):
        if candidate.name.casefold() == target:
            return candidate
    return candidates[-1]


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


def build_structure_stability_features(
    df: pd.DataFrame,
    structure_dir: str | Path = "data/raw/hf_snapshot/structures",
) -> pd.DataFrame:
    """Build structure-derived stability/interface features for all local PDBs."""
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
        features = structure_stability_features_for_structure(
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
