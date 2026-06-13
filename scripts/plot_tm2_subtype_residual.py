#!/usr/bin/env python
"""Plot the Tm2 subtype/framework signal versus the subtype residual.

Reads only aggregate results/tm2_structure_tier_summary.csv. The figure shows
that raw Tm2 is partly predictable from framework/subtype proxies, while the
remaining subtype-residual signal is weak.

Output: results/tm2_subtype_residual.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results")


def main() -> None:
    df = pd.read_csv(RESULTS / "tm2_structure_tier_summary.csv")
    rows = pd.DataFrame(
        [
            {
                "label": "Raw Tm2\nframework proxy",
                "spearman": _value(df, "Tm2", "tier2_framework_proxy", "external_spearman"),
                "auroc": _value(df, "Tm2", "tier2_framework_proxy", "external_worst_20_auroc"),
                "color": "#8e5ea2",
            },
            {
                "label": "Raw Tm2\nstructure-rich",
                "spearman": _value(df, "Tm2", "sequence_plus_structure_all", "external_spearman"),
                "auroc": _value(df, "Tm2", "sequence_plus_structure_all", "external_worst_20_auroc"),
                "color": "#4a76a8",
            },
            {
                "label": "Subtype residual\nbest selected",
                "spearman": _value(
                    df,
                    "Tm2_subtype_residual",
                    "tier1_cdr_flexibility",
                    "external_spearman",
                ),
                "auroc": _value(
                    df,
                    "Tm2_subtype_residual",
                    "tier1_cdr_flexibility",
                    "external_worst_20_auroc",
                ),
                "color": "#bdbdbd",
            },
        ]
    )

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = range(len(rows))
    ax.bar(x, rows["spearman"], color=rows["color"])
    for i, row in rows.iterrows():
        ax.text(
            i,
            row["spearman"] + 0.015,
            f"{row['spearman']:.2f}\nAUROC {row['auroc']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.392, color="black", linestyle="--", linewidth=1.0)
    ax.text(2.45, 0.392, " field ceiling 0.392", va="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(rows["label"], fontsize=9)
    ax.set_ylabel("GDPa3 Spearman")
    ax.set_title("Tm2 signal largely follows framework/subtype; residual is weak", fontsize=11)
    ax.set_ylim(-0.02, 0.46)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#ececec", linewidth=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = RESULTS / "tm2_subtype_residual.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def _value(df: pd.DataFrame, assay: str, feature_set: str, column: str) -> float:
    row = df[(df["assay"] == assay) & (df["feature_set"] == feature_set)]
    if row.empty:
        raise ValueError(f"Missing {assay} / {feature_set} in Tm2 summary.")
    return float(row.iloc[0][column])


if __name__ == "__main__":
    main()
