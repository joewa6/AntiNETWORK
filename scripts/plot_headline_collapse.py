#!/usr/bin/env python
"""Plot the headline result: internal GDPa1 CV vs external GDPa3, per assay.

Reads only aggregate summary tables in results/ (no per-antibody data). The
figure shows that honest external evaluation collapses most developability
signal. HIC survives as a physicochemical signal; raw Tm2 is partially
recoverable but is best explained by framework/subtype proxies rather than a
clean Fab-stability endpoint.

Output: results/headline_collapse.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
    tm2 = pd.read_csv(RESULTS / "tm2_structure_tier_summary.csv")
    tm2_raw = tm2[(tm2["assay"] == "Tm2") & (tm2["feature_set"] == "tier2_framework_proxy")].iloc[0]

    collapse = collapse.copy()
    tm2_mask = collapse["assay"] == "Tm2"
    collapse.loc[tm2_mask, "feature_set"] = tm2_raw["feature_set"]
    collapse.loc[tm2_mask, "model"] = tm2_raw["model"]
    collapse.loc[tm2_mask, "internal_spearman"] = tm2_raw["internal_spearman"]
    collapse.loc[tm2_mask, "external_spearman"] = tm2_raw["external_spearman"]
    collapse.loc[tm2_mask, "external_worst_20_auroc"] = tm2_raw["external_worst_20_auroc"]

    ceiling = pd.read_csv(RESULTS / "external_vs_field_ceiling.csv")[
        ["assay", "published_top_spearman"]
    ]
    df = collapse.merge(ceiling, on="assay", how="left")
    df = df.sort_values("external_spearman", ascending=False).reset_index(drop=True)

    x = range(len(df))
    width = 0.38
    internal_color = "#bdbdbd"
    other_external = "#4a76a8"
    hic_external = "#2e7d32"
    tm2_external = "#8e5ea2"

    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    ax.bar(
        [i - width / 2 for i in x],
        df["internal_spearman"],
        width,
        color=internal_color,
    )
    ext_colors = [
        hic_external if assay == "HIC" else tm2_external if assay == "Tm2" else other_external
        for assay in df["assay"]
    ]
    ax.bar([i + width / 2 for i in x], df["external_spearman"], width, color=ext_colors)

    for i, top in zip(x, df["published_top_spearman"], strict=False):
        if pd.notna(top):
            ax.hlines(top, i - width, i + width, color="black", linewidth=1.4)

    for i, ext in enumerate(df["external_spearman"]):
        if abs(ext) < 0.035:
            ax.text(
                i + width / 2,
                0.012,
                "≈0",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#555555",
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS.get(a, a) for a in df["assay"]], fontsize=9)
    ax.set_ylabel("Spearman rank correlation")
    ax.set_title(
        "External evaluation: HIC survives; Tm2 is mostly framework/subtype signal",
        fontsize=11,
    )

    hic_i = list(df["assay"]).index("HIC")
    ax.annotate(
        "holds out",
        xy=(hic_i + width / 2, df.loc[hic_i, "external_spearman"]),
        xytext=(hic_i + 0.15, df.loc[hic_i, "external_spearman"] + 0.12),
        fontsize=8.5,
        ha="left",
        arrowprops=dict(arrowstyle="-", color=hic_external, linewidth=0.8),
        color=hic_external,
    )

    tm2_i = list(df["assay"]).index("Tm2")
    ax.annotate(
        "framework/subtype proxy",
        xy=(tm2_i + width / 2, df.loc[tm2_i, "external_spearman"]),
        xytext=(tm2_i + 0.10, df.loc[tm2_i, "external_spearman"] + 0.10),
        fontsize=8.5,
        ha="left",
        arrowprops=dict(arrowstyle="-", color=tm2_external, linewidth=0.8),
        color=tm2_external,
    )

    legend_items = [
        Line2D([0], [0], color="black", linewidth=1.4, label="Field ceiling (best of 113 teams)"),
        Patch(facecolor=internal_color, label="Internal CV (GDPa1)"),
        Patch(facecolor=hic_external, label="External HIC"),
        Patch(facecolor=tm2_external, label="External Tm2"),
        Patch(facecolor=other_external, label="External other assays"),
    ]
    ax.legend(handles=legend_items, frameon=False, fontsize=8.2, loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = RESULTS / "headline_collapse.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
