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
from antinetwork.plm_embeddings import (
    DEFAULT_PLM_MODELS,
    build_multi_plm_feature_sets,
    load_chain_embeddings,
)


AUDIT_VERSION = "external_audit_v10_frozen_plm_ranker"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GDPa1-selected frozen PLM ranker audit on GDPa3."
    )
    parser.add_argument("--comp-dir", type=Path, default=Path("data/GinkgoComp"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("reports/plm_embeddings"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/plm_ranker_audit"))
    parser.add_argument("--gdpa1-workbook", default="GDPa1_v1.3_20251027_full.xlsx")
    parser.add_argument("--gdpa3-workbook", default="GDPa3_20260106_full.xlsx")
    parser.add_argument("--plm-models", nargs="+", default=sorted(DEFAULT_PLM_MODELS))
    parser.add_argument("--models", nargs="+", default=DEFAULT_COMPETITIVE_MODELS)
    parser.add_argument("--assays", nargs="+", default=OFFICIAL_ASSAYS)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = _load_train(args)
    test = _load_test(args)
    train_groups = train["hierarchical_cluster_IgG_isotype_stratified_fold"]

    train_physics = build_physical_features(train)
    test_physics = build_physical_features(test)
    train_embeddings = _load_embeddings(args.embedding_dir, "gdpa1", args.plm_models)
    test_embeddings = _load_embeddings(args.embedding_dir, "gdpa3", args.plm_models)

    train_feature_sets = build_multi_plm_feature_sets(
        train_embeddings,
        train_physics,
        model_names=args.plm_models,
    )
    test_feature_sets = build_multi_plm_feature_sets(
        test_embeddings,
        test_physics,
        model_names=args.plm_models,
    )

    internal = evaluate_competitive_rankers(
        train,
        feature_sets=train_feature_sets,
        assays=args.assays,
        groups=train_groups,
        model_names=args.models,
        cv_kind="cluster",
        random_state=args.seed,
        dataset=f"{AUDIT_VERSION}_GDPa1_cluster_cv",
        pca_components=args.pca_components,
    )

    comparisons = []
    external_frames = []
    prediction_frames = []
    frozen_rows = []
    for feature_set, train_features in train_feature_sets.items():
        feature_metrics = internal.metrics[internal.metrics["feature_set"] == feature_set]
        frozen = select_best_rankers(feature_metrics, primary_metric="spearman")
        model_by_assay = dict(zip(frozen["assay"], frozen["model"], strict=False))
        test_features = test_feature_sets[feature_set]
        predictions = train_full_ranker_predictions(
            train,
            train_features,
            test,
            test_features,
            assays=args.assays,
            model_by_assay=model_by_assay,
            random_state=args.seed,
            pca_components=args.pca_components,
        )
        predictions["feature_set"] = feature_set
        prediction_frames.append(predictions)
        external = evaluate_external_ranker_predictions(
            test,
            predictions,
            assays=args.assays,
            model_by_assay=model_by_assay,
            feature_set=feature_set,
            dataset=f"{AUDIT_VERSION}_GDPa3",
        )
        external_frames.append(external)
        frozen_rows.append(frozen)
        comparisons.append(_comparison_table(frozen, external))

    frozen_selection = pd.concat(frozen_rows, ignore_index=True)
    external_metrics = pd.concat(external_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    comparison = pd.concat(comparisons, ignore_index=True)
    comparison["externally_useful"] = (
        (comparison["external_spearman"] >= 0.30)
        & (comparison["external_worst_20_auroc"] >= 0.70)
    )

    spec = {
        "audit_version": AUDIT_VERSION,
        "feature_blocks": list(train_feature_sets),
        "plm_models": list(args.plm_models),
        "models_considered": list(args.models),
        "assays": list(args.assays),
        "pca_components_inside_cv_and_full_train": args.pca_components,
        "selection_dataset": "GDPa1",
        "external_dataset": "GDPa3",
        "primary_selection_metric": "spearman",
        "external_useful_threshold": {
            "spearman": 0.30,
            "worst_20_auroc": 0.70,
        },
        "random_seed": args.seed,
    }
    (args.out_dir / "frozen_plm_ranker_spec_v10.json").write_text(json.dumps(spec, indent=2) + "\n")
    internal.metrics.to_csv(args.out_dir / "GDPa1_plm_internal_metrics_v10.csv", index=False)
    frozen_selection.to_csv(args.out_dir / "GDPa1_plm_frozen_selection_v10.csv", index=False)
    predictions.to_csv(args.out_dir / "GDPa3_plm_external_predictions_v10.csv", index=False)
    external_metrics.to_csv(args.out_dir / "GDPa3_plm_external_metrics_v10.csv", index=False)
    comparison.to_csv(args.out_dir / "GDPa1_vs_GDPa3_plm_triage_comparison_v10.csv", index=False)

    best = (
        comparison.sort_values(
            ["assay", "externally_useful", "external_worst_20_auroc", "external_spearman"],
            ascending=[True, False, False, False],
        )
        .groupby("assay", as_index=False)
        .first()
    )
    best.to_csv(args.out_dir / "GDPa3_plm_best_by_external_triage_v10.csv", index=False)
    print(
        best[
            [
                "assay",
                "feature_set",
                "model",
                "internal_spearman",
                "external_spearman",
                "internal_worst_20_auroc",
                "external_worst_20_auroc",
                "externally_useful",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


def _load_train(args: argparse.Namespace) -> pd.DataFrame:
    workbook = load_ginkgo_workbook(args.comp_dir / args.gdpa1_workbook)
    train = make_ginkgo_external_dataset(workbook, split="train")
    folds = pd.read_csv(args.raw_dir / MAIN_CSV)[
        ["antibody_name", "hierarchical_cluster_IgG_isotype_stratified_fold"]
    ]
    train = train.merge(folds, on="antibody_name", how="left", validate="one_to_one")
    if train["hierarchical_cluster_IgG_isotype_stratified_fold"].isna().any():
        raise ValueError("Some GDPa1 rows are missing cluster-fold labels.")
    return train


def _load_test(args: argparse.Namespace) -> pd.DataFrame:
    workbook = load_ginkgo_workbook(args.comp_dir / args.gdpa3_workbook)
    return make_ginkgo_external_dataset(workbook, split="test")


def _load_embeddings(
    embedding_dir: Path,
    dataset: str,
    model_names: list[str],
) -> pd.DataFrame:
    frames = []
    missing = []
    for model_name in model_names:
        path = embedding_dir / f"{dataset}_{model_name}_chain_embeddings.parquet"
        if not path.exists():
            missing.append(str(path))
            continue
        frames.append(load_chain_embeddings(path))
    if missing:
        raise FileNotFoundError(
            "Missing PLM embedding parquet files. Build them first with scripts/build_plm_embeddings.py:\n"
            + "\n".join(missing)
        )
    return pd.concat(frames, ignore_index=True)


def _comparison_table(internal: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    left = internal[
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
    ].rename(
        columns={
            "n": "internal_n",
            "spearman": "internal_spearman",
            "pearson": "internal_pearson",
            "r2": "internal_r2",
            "worst_20_recall": "internal_worst_20_recall",
            "worst_20_auroc": "internal_worst_20_auroc",
        }
    )
    right = external[
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
    )
    return left.merge(right, on="assay", how="left")


if __name__ == "__main__":
    main()
