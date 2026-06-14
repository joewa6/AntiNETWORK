from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from antinetwork.competitive_ranker import (
    DEFAULT_COMPETITIVE_MODELS,
    evaluate_competitive_rankers,
    evaluate_external_ranker_predictions,
    select_best_rankers,
    train_full_ranker_predictions,
)
from antinetwork.features import build_physical_features
from antinetwork.gdpa import MAIN_CSV
from antinetwork.official_benchmark import (
    OFFICIAL_ASSAYS,
    load_ginkgo_workbook,
    make_ginkgo_external_dataset,
)


AUDIT_VERSION = "external_audit_v9_triage_ranker"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen GDPa1-trained developability triage audit on GDPa3."
    )
    parser.add_argument("--comp-dir", type=Path, default=Path("data/GinkgoComp"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/external_triage_audit"))
    parser.add_argument("--gdpa1-workbook", default="GDPa1_v1.3_20251027_full.xlsx")
    parser.add_argument("--gdpa3-workbook", default="GDPa3_20260106_full.xlsx")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--models", nargs="+", default=DEFAULT_COMPETITIVE_MODELS)
    parser.add_argument("--assays", nargs="+", default=OFFICIAL_ASSAYS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gdpa1 = load_ginkgo_workbook(args.comp_dir / args.gdpa1_workbook)
    train = make_ginkgo_external_dataset(gdpa1, split="train")
    folds = pd.read_csv(args.raw_dir / MAIN_CSV)[
        ["antibody_name", "hierarchical_cluster_IgG_isotype_stratified_fold"]
    ]
    train = train.merge(folds, on="antibody_name", how="left", validate="one_to_one")
    train_groups = train["hierarchical_cluster_IgG_isotype_stratified_fold"]
    if train_groups.isna().any():
        missing = int(train_groups.isna().sum())
        raise ValueError(f"{missing} GDPa1 rows are missing cluster-fold labels.")
    train_features = build_physical_features(train)
    feature_sets = {"global_composition": train_features}

    internal = evaluate_competitive_rankers(
        train,
        feature_sets=feature_sets,
        assays=args.assays,
        groups=train_groups,
        model_names=args.models,
        cv_kind="cluster",
        random_state=args.seed,
        dataset=f"{AUDIT_VERSION}_GDPa1_cluster_cv",
    )
    frozen = select_best_rankers(internal.metrics, primary_metric="spearman")
    model_by_assay = dict(zip(frozen["assay"], frozen["model"], strict=False))

    spec = {
        "audit_version": AUDIT_VERSION,
        "selection_dataset": "GDPa1",
        "external_dataset": "GDPa3",
        "feature_set": "global_composition",
        "feature_builder": "antinetwork.features.build_physical_features",
        "cv_kind": "cluster",
        "primary_selection_metric": "spearman",
        "models_considered": list(args.models),
        "assays": list(args.assays),
        "model_by_assay": model_by_assay,
        "random_seed": args.seed,
        "note": "GDPa3 labels are loaded only after this frozen specification is selected from GDPa1 CV.",
    }
    (args.out_dir / "frozen_external_triage_spec_v9.json").write_text(
        json.dumps(spec, indent=2) + "\n"
    )

    gdpa3 = load_ginkgo_workbook(args.comp_dir / args.gdpa3_workbook)
    test = make_ginkgo_external_dataset(gdpa3, split="test")
    test_features = build_physical_features(test)

    predictions = train_full_ranker_predictions(
        train,
        train_features,
        test,
        test_features,
        assays=args.assays,
        model_by_assay=model_by_assay,
        random_state=args.seed,
    )
    external = evaluate_external_ranker_predictions(
        test,
        predictions,
        assays=args.assays,
        model_by_assay=model_by_assay,
        feature_set="global_composition",
        dataset=f"{AUDIT_VERSION}_GDPa3",
    )

    comparison = (
        frozen[
            [
                "assay",
                "feature_set",
                "model",
                "n",
                "spearman",
                "pearson",
                "r2",
                "worst_20_recall",
                "worst_20_auroc",
            ]
        ]
        .rename(
            columns={
                "n": "internal_n",
                "spearman": "internal_spearman",
                "pearson": "internal_pearson",
                "r2": "internal_r2",
                "worst_20_recall": "internal_worst_20_recall",
                "worst_20_auroc": "internal_worst_20_auroc",
            }
        )
        .merge(
            external[
                [
                    "assay",
                    "n",
                    "spearman",
                    "pearson",
                    "r2",
                    "worst_20_recall",
                    "worst_20_auroc",
                ]
            ].rename(
                columns={
                    "n": "external_n",
                    "spearman": "external_spearman",
                    "pearson": "external_pearson",
                    "r2": "external_r2",
                    "worst_20_recall": "external_worst_20_recall",
                    "worst_20_auroc": "external_worst_20_auroc",
                }
            ),
            on="assay",
            how="left",
        )
    )
    comparison["spearman_delta_external_minus_internal"] = (
        comparison["external_spearman"] - comparison["internal_spearman"]
    )
    comparison["worst20_auroc_delta_external_minus_internal"] = (
        comparison["external_worst_20_auroc"] - comparison["internal_worst_20_auroc"]
    )
    comparison["verdict"] = comparison.apply(_verdict, axis=1)

    internal.metrics.to_csv(args.out_dir / "GDPa1_internal_triage_metrics_v9.csv", index=False)
    frozen.to_csv(args.out_dir / "GDPa1_frozen_model_selection_v9.csv", index=False)
    predictions.to_csv(args.out_dir / "GDPa3_external_triage_predictions_v9.csv", index=False)
    external.to_csv(args.out_dir / "GDPa3_external_triage_metrics_v9.csv", index=False)
    comparison.to_csv(args.out_dir / "GDPa1_vs_GDPa3_triage_comparison_v9.csv", index=False)

    columns = [
        "assay",
        "model",
        "internal_spearman",
        "external_spearman",
        "internal_worst_20_auroc",
        "external_worst_20_auroc",
        "verdict",
    ]
    print(comparison[columns].round(3).to_string(index=False))


def _verdict(row: pd.Series) -> str:
    external_auc = row["external_worst_20_auroc"]
    external_spearman = row["external_spearman"]
    if pd.isna(external_auc) or pd.isna(external_spearman):
        return "NO EXTERNAL SCORE"
    if external_auc >= 0.75 and external_spearman >= 0.30:
        return "EXTERNAL TRIAGE HOLDS"
    if external_auc >= 0.65 and external_spearman >= 0.15:
        return "WEAK EXTERNAL TRIAGE"
    return "EXTERNAL TRIAGE FAIL"


if __name__ == "__main__":
    main()
