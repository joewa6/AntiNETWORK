#!/usr/bin/env python
"""Plot the headline result: internal GDPa1 CV vs external GDPa3, per assay.

Reads only the aggregate summary tables in results/ (no per-antibody data). The
figure shows that honest external evaluation collapses developability signal for
every assay except HIC, with the published competition field ceiling marked.

Output: results/headline_collapse.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results")

LABELS = {
    "HIC": "HIC\n(hydrophobicity)",
    "PR_CHO": "PR_CHO\n(polyreactivity)",
    "AC-SINS_pH7.4": "AC-SINS\n(self-assoc.)",
    "Titer": "Titer\n(expression)",
    "Tm2": "Tm2\n(stability)",
}


def main() -> None:
    collapse = pd.read_csv(RESULTS / "internal_vs_external_triage_comparison.csv")
    ceiling = pd.read_csv(RESULTS / "external_vs_field_ceiling.csv")[
        ["assay", "published_top_spearman"]
    ]
    df = collapse.merge(ceiling, on="assay", how="left")
    df = df.sort_values("external_spearman", ascending=False).reset_index(drop=True)

    x = range(len(df))
    width = 0.38
    internal_color = "#bdbdbd"
    external_color = "#4a76a8"
    survivor_color = "#2e7d32"

    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    ax.bar(
        [i - width / 2 for i in x],
        df["internal_spearman"],
        width,
        color=internal_color,
        label="Internal CV (GDPa1)",
    )
    ext_colors = [survivor_color if a == "HIC" else external_color for a in df["assay"]]
    ax.bar(
        [i + width / 2 for i in x],
        df["external_spearman"],
        width,
        color=ext_colors,
        label="External held-out (GDPa3)",
    )

    # field ceiling: best of 113 competition teams, as a short marker per assay
    for i, top in zip(x, df["published_top_spearman"]):
        if pd.notna(top):
            ax.hlines(top, i - width, i + width, color="black", linewidth=1.4)
    ax.hlines([], [], [], color="black", linewidth=1.4, label="Field ceiling (best of 113 teams)")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS.get(a, a) for a in df["assay"]], fontsize=9)
    ax.set_ylabel("Spearman rank correlation")
    ax.set_title(
        "Honest external evaluation collapses developability signal — only HIC survives",
        fontsize=11,
    )

    # mark the survivor plainly
    hic_i = list(df["assay"]).index("HIC")
    ax.annotate(
        "holds out of distribution",
        xy=(hic_i + width / 2, df.loc[hic_i, "external_spearman"]),
        xytext=(hic_i + 0.1, df.loc[hic_i, "external_spearman"] + 0.13),
        fontsize=8.5,
        ha="left",
        arrowprops=dict(arrowstyle="-", color="#2e7d32", linewidth=0.8),
        color="#2e7d32",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    fig.tight_layout()
    out = RESULTS / "headline_collapse.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
