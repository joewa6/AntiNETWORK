#!/usr/bin/env python
"""Sequence-identity kNN baseline for external HIC, on the GDPa3 held-out set.

This is the memorization control behind the README's headline HIC claim: it asks
how much of the external HIC rank-order is recoverable from pure sequence
similarity to the GDPa1 training antibodies, with no physicochemical features.

For each GDPa3 antibody it finds its nearest GDPa1 neighbours by local
(Smith-Waterman) VH and light-chain identity, predicts HIC as the neighbour mean,
and reports Spearman against the GDPa3 HIC labels. The physics ridge model scores
~0.49 on the same split; this baseline scores ~0.33 at k=8 (VH+VL). Physics beats
memorization by ~0.16, but ~0.33 is sequence-recoverable — see the README.

The script reads the gated GDPa1/GDPa3 data locally and emits only a Spearman
number (a statistic). It requires data obtained from source (see download_gdpa.py
for GDPa1; GDPa3 is gated/external and not redistributed here).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from scipy.stats import spearmanr


def clean(seq: str) -> str:
    return str(seq).replace("-", "").replace("|", "").replace("*", "").strip()


def make_aligner() -> PairwiseAligner:
    """Local aligner with match=1/mismatch=0, so score ~= identical residues."""
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1
    return aligner


def identity(aligner: PairwiseAligner, a: str, b: str) -> float:
    return aligner.score(a, b) / max(min(len(a), len(b)), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdpa1-csv", type=Path, default=Path("data/raw/GDPa1_v1.2_20250814.csv"))
    parser.add_argument(
        "--gdpa3-xlsx",
        type=Path,
        default=Path("data/GinkgoComp/GDPa3_20260106_full.xlsx"),
        help="Gated/external GDPa3 workbook (not redistributed in this repo).",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[5, 8, 10])
    parser.add_argument("--physics-external-spearman", type=float, default=0.49)
    args = parser.parse_args()

    g1 = pd.read_csv(args.gdpa1_csv).dropna(subset=["HIC"]).reset_index(drop=True)
    seqs3 = pd.read_excel(args.gdpa3_xlsx, sheet_name="Sequences")
    avg3 = pd.read_excel(args.gdpa3_xlsx, sheet_name="Assay Data - average")
    g3 = (
        seqs3.merge(avg3[["antibody_id", "hic_rt_avg"]], on="antibody_id", how="left")
        .dropna(subset=["hic_rt_avg"])
        .reset_index(drop=True)
    )
    print(f"GDPa1 with HIC: {len(g1)}   GDPa3 with hic_rt_avg: {len(g3)}")

    g1_vh = [clean(s) for s in g1["vh_protein_sequence"]]
    g1_vl = [clean(s) for s in g1["vl_protein_sequence"]]
    g3_vh = [clean(s) for s in g3["vh_protein_sequence"]]
    g3_lc = [clean(s) for s in g3["lc_protein_sequence"]]  # full light chain
    y1 = g1["HIC"].to_numpy(float)
    y3 = g3["hic_rt_avg"].to_numpy(float)

    aligner = make_aligner()
    n3, n1 = len(g3), len(g1)
    vh_id = np.zeros((n3, n1))
    vl_id = np.zeros((n3, n1))
    for i in range(n3):
        for j in range(n1):
            vh_id[i, j] = identity(aligner, g3_vh[i], g1_vh[j])
            vl_id[i, j] = identity(aligner, g3_lc[i], g1_vl[j])  # local finds VL within LC
    combined = 0.5 * (vh_id + vl_id)

    def knn_spearman(sim: np.ndarray, k: int) -> float:
        preds = np.array([np.nanmean(y1[np.argsort(sim[i])[::-1][:k]]) for i in range(n3)])
        return float(spearmanr(preds, y3).statistic)

    print(f"\nphysics ridge (external HIC, reference) = {args.physics_external_spearman:.2f}")
    for k in args.k:
        rho = knn_spearman(combined, k)
        delta = args.physics_external_spearman - rho
        print(f"identity-kNN VH+VL k={k:2d}  external HIC Spearman = {rho:.3f}   (physics - kNN = {delta:+.3f})")


if __name__ == "__main__":
    main()
