#!/usr/bin/env python
"""Plot held-out HIC: our physics model against the competition's official baselines.

Reads only the aggregate table results/external_hic_vs_field_baselines.csv (per
method, no per-antibody data). Shows that simple sequence physics sits upper-pack
on the blinded GDPa3 held-out set — above every sequence-based ML baseline — so
the internal->external collapse on other assays is not a feature-quality failure.

Output: results/hic_vs_field_baselines.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results")
WINNER = 0.708  # best of 113 competition teams (outcomes paper); predictions not in repo

DISPLAY = {
    "our_physics_descriptors": "our physics (sequence descriptors)",
    "saprot_vh_vl": "SaProt (structure-aware LM)",
    "aggrescan3d": "Aggrescan3D (structure)",
    "moe_baseline": "MOE descriptors",
    "esm2_tap_ridge": "ESM-2 + TAP ridge",
    "piggen": "PiGGen",
    "deepsp_ridge": "DeepSP ridge",
    "esm2_ridge": "ESM-2 ridge",
    "ablang2_elastic_net": "AbLang2 elastic net",
    "tap_linear": "TAP linear",
    "esm2_tap_rf": "ESM-2 + TAP RF",
    "onehot_ridge": "one-hot ridge",
    "esm2_tap_xgb": "ESM-2 + TAP XGB",
    "tap_single_features": "TAP single feature",
    "random_predictor": "random predictor",
}


def main() -> None:
    df = pd.read_csv(RESULTS / "external_hic_vs_field_baselines.csv")
    df = df.sort_values("heldout_hic_spearman", ascending=True).reset_index(drop=True)
    field_median = float(df.loc[~df["is_ours"], "heldout_hic_spearman"].median())

    labels = [DISPLAY.get(m, m) for m in df["method"]]
    colors = ["#2e7d32" if o else "#bdbdbd" for o in df["is_ours"]]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    y = range(len(df))
    ax.barh(list(y), df["heldout_hic_spearman"], color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5)

    ax.axvline(field_median, color="#888888", linestyle=":", linewidth=1.1)
    ax.text(field_median, len(df) - 0.4, f"  field-baseline median {field_median:.2f}",
            fontsize=8, color="#666666", va="top")
    ax.axvline(WINNER, color="black", linestyle="--", linewidth=1.1)
    ax.text(WINNER, 0.2, f"competition winner {WINNER:.2f}  ", fontsize=8, color="black",
            ha="right", va="bottom")

    ours_i = df.index[df["is_ours"]][0]
    ax.text(df.loc[ours_i, "heldout_hic_spearman"] + 0.008, ours_i,
            f"{df.loc[ours_i, 'heldout_hic_spearman']:.2f}", va="center", fontsize=8.5,
            color="#2e7d32", fontweight="bold")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("GDPa3 held-out HIC — Spearman")
    ax.set_title(
        "Held-out HIC: simple sequence physics is upper-pack among field baselines",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color="#ececec", linewidth=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = RESULTS / "hic_vs_field_baselines.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
