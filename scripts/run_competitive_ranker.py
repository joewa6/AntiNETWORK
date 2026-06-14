from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from antinetwork.competitive_ranker import (
    DEFAULT_COMPETITIVE_MODELS,
    evaluate_competitive_rankers,
    select_best_rankers,
)
from antinetwork.gdpa import MAIN_CSV
from antinetwork.topology_falsification import (
    DEFAULT_GATE_ASSAYS,
    build_feature_blocks_with_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate competitive developability rankers on current feature blocks."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/competitive_ranker"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-scrambles", type=int, default=1)
    parser.add_argument("--cv-kind", choices=["cluster", "random"], default="cluster")
    parser.add_argument("--models", nargs="+", default=DEFAULT_COMPETITIVE_MODELS)
    parser.add_argument("--assays", nargs="+", default=DEFAULT_GATE_ASSAYS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.raw_dir / MAIN_CSV)
    groups = df["hierarchical_cluster_IgG_isotype_stratified_fold"]
    blocks = build_feature_blocks_with_diagnostics(
        df,
        n_scrambles=args.n_scrambles,
        random_state=args.seed,
    ).feature_blocks

    feature_sets = {
        "global_composition": blocks["global_composition"],
        "surface_composition": blocks["surface_composition"],
        "real_topology": blocks["real_topology"],
    }
    feature_sets["combined_physical_surface_topology"] = pd.concat(
        [
            feature_sets["global_composition"].add_prefix("global__"),
            feature_sets["surface_composition"].add_prefix("surface__"),
            feature_sets["real_topology"].add_prefix("topology__"),
        ],
        axis=1,
    )

    dataset = f"GDPa1_{args.cv_kind}_cv_current_features"
    result = evaluate_competitive_rankers(
        df,
        feature_sets=feature_sets,
        assays=args.assays,
        groups=groups if args.cv_kind == "cluster" else None,
        model_names=args.models,
        cv_kind=args.cv_kind,
        random_state=args.seed,
        dataset=dataset,
    )
    best = select_best_rankers(result.metrics, primary_metric="spearman")

    stem = f"current_feature_ranker_{args.cv_kind}_cv"
    result.metrics.to_csv(args.out_dir / f"{stem}_metrics.csv", index=False)
    result.predictions.to_csv(args.out_dir / f"{stem}_oof_predictions.csv", index=False)
    best.to_csv(args.out_dir / f"{stem}_best_by_spearman.csv", index=False)

    columns = [
        "assay",
        "feature_set",
        "model",
        "n",
        "spearman",
        "pearson",
        "r2",
        "worst_20_recall",
        "worst_20_auroc",
        "rmse_over_sd",
    ]
    print(best[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
