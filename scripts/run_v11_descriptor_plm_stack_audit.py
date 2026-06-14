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
from antinetwork.official_benchmark import load_ginkgo_workbook, make_ginkgo_external_dataset
from antinetwork.plm_embeddings import (
    build_antibody_embedding_features,
    load_chain_embeddings,
)


AUDIT_VERSION = "external_audit_v11_descriptor_plm_stack"
TARGET_ASSAYS = ["HIC", "PR_CHO"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="v11 HIC/PR_CHO descriptor + PLM stack audit under the GDPa3 firewall."
    )
    parser.add_argument("--comp-dir", type=Path, default=Path("data/GinkgoComp"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("reports/plm_embeddings"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/v11_descriptor_plm_stack"))
    parser.add_argument("--gdpa1-workbook", default="GDPa1_v1.3_20251027_full.xlsx")
    parser.add_argument("--gdpa3-workbook", default="GDPa3_20260106_full.xlsx")
    parser.add_argument(
        "--moe-train",
        type=Path,
        default=Path("data/GinkgoComp/features/processed_features/GDPa1/MOE_properties.csv"),
    )
    parser.add_argument(
        "--moe-test",
        type=Path,
        default=Path("data/GinkgoComp/features/processed_features/heldout_test/MOE_properties.csv"),
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_COMPETITIVE_MODELS)
    parser.add_argument("--assays", nargs="+", default=TARGET_ASSAYS)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--allow-missing-moe",
        action="store_true",
        help="Run only non-MOE blocks if MOE_properties.csv files are unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = _load_train(args)
    test = _load_test(args)
    train_groups = train["hierarchical_cluster_IgG_isotype_stratified_fold"]

    train_physics = build_physical_features(train)
    test_physics = build_physical_features(test)
    train_embeddings = _load_required_embeddings(args.embedding_dir, "gdpa1")
    test_embeddings = _load_required_embeddings(args.embedding_dir, "gdpa3")
    train_esm2 = build_antibody_embedding_features(train_embeddings, "esm2_t12_35M")
    test_esm2 = build_antibody_embedding_features(test_embeddings, "esm2_t12_35M")
    train_ab_roberta = build_antibody_embedding_features(train_embeddings, "ab_roberta")
    test_ab_roberta = build_antibody_embedding_features(test_embeddings, "ab_roberta")

    moe_available = args.moe_train.exists() and args.moe_test.exists()
    if moe_available:
        train_moe = load_moe_features(args.moe_train).reindex(train["antibody_name"].to_numpy())
        test_moe = load_moe_features(
            args.moe_test,
            test_names=test["antibody_name"].to_numpy(),
            heldout_sequences=args.comp_dir / "seqs" / "heldout-set-sequences.csv",
            gdpa3_sequences=test,
        ).reindex(test["antibody_name"].to_numpy())
        train_moe.index = train.index
        test_moe.index = test.index
    elif args.allow_missing_moe:
        train_moe = pd.DataFrame(index=train.index)
        test_moe = pd.DataFrame(index=test.index)
    else:
        missing = [str(path) for path in [args.moe_train, args.moe_test] if not path.exists()]
        raise FileNotFoundError(
            "v11 requires row-level MOE feature CSVs. Restore these files or pass explicit paths:\n"
            + "\n".join(missing)
        )

    train_feature_sets = _build_feature_sets(
        train_physics,
        train_moe,
        train_esm2,
        train_ab_roberta,
        include_moe=moe_available,
    )
    test_feature_sets = _build_feature_sets(
        test_physics,
        test_moe,
        test_esm2,
        test_ab_roberta,
        include_moe=moe_available,
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

    external_frames = []
    comparison_frames = []
    prediction_frames = []
    selected_frames = []
    for feature_set, train_features in train_feature_sets.items():
        selected = select_best_rankers(
            internal.metrics[internal.metrics["feature_set"] == feature_set],
            primary_metric="spearman",
        )
        model_by_assay = dict(zip(selected["assay"], selected["model"], strict=False))
        predictions = train_full_ranker_predictions(
            train,
            train_features,
            test,
            test_feature_sets[feature_set],
            assays=args.assays,
            model_by_assay=model_by_assay,
            random_state=args.seed,
            pca_components=args.pca_components,
        )
        predictions["feature_set"] = feature_set
        external = evaluate_external_ranker_predictions(
            test,
            predictions,
            assays=args.assays,
            model_by_assay=model_by_assay,
            feature_set=feature_set,
            dataset=f"{AUDIT_VERSION}_GDPa3",
        )
        selected_frames.append(selected)
        prediction_frames.append(predictions)
        external_frames.append(external)
        comparison_frames.append(_comparison_table(selected, external))

    selected_all = pd.concat(selected_frames, ignore_index=True)
    external_all = pd.concat(external_frames, ignore_index=True)
    predictions_all = pd.concat(prediction_frames, ignore_index=True)
    comparison = pd.concat(comparison_frames, ignore_index=True)
    comparison["externally_useful"] = (
        (comparison["external_spearman"] >= 0.25)
        & (comparison["external_worst_20_auroc"] >= 0.70)
    )
    comparison["beats_current_anchor"] = _beats_anchor(comparison)

    spec = {
        "audit_version": AUDIT_VERSION,
        "assays": list(args.assays),
        "feature_blocks": list(train_feature_sets),
        "models_considered": list(args.models),
        "pca_components_inside_cv_and_full_train": args.pca_components,
        "selection_dataset": "GDPa1",
        "external_dataset": "GDPa3",
        "primary_selection_metric": "spearman",
        "success_conditions": {
            "HIC": "external_worst_20_auroc > 0.787",
            "PR_CHO": "external_worst_20_auroc >= 0.70 and external_spearman >= 0.25",
        },
        "moe_feature_paths": {
            "train": str(args.moe_train),
            "test": str(args.moe_test),
            "available": moe_available,
        },
        "random_seed": args.seed,
    }
    (args.out_dir / "v11_frozen_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    internal.metrics.to_csv(args.out_dir / "v11_internal_selection_metrics.csv", index=False)
    selected_all.to_csv(args.out_dir / "v11_selected_models.csv", index=False)
    predictions_all.to_csv(args.out_dir / "v11_external_predictions.csv", index=False)
    external_all.to_csv(args.out_dir / "v11_external_firewall_metrics.csv", index=False)
    comparison.to_csv(args.out_dir / "v11_internal_external_comparison.csv", index=False)

    best = (
        comparison.sort_values(
            ["assay", "beats_current_anchor", "externally_useful", "external_worst_20_auroc"],
            ascending=[True, False, False, False],
        )
        .groupby("assay", as_index=False)
        .first()
    )
    best.to_csv(args.out_dir / "v11_best_by_external_triage.csv", index=False)
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
                "beats_current_anchor",
                "externally_useful",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


def load_moe_features(
    path: Path,
    test_names: pd.Index | list[str] | None = None,
    heldout_sequences: Path | None = None,
    gdpa3_sequences: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load official MOE_properties.csv and index rows by local antibody_name."""
    frame = pd.read_csv(path)
    frame = frame.drop(columns=[c for c in ["antibody_id", "mseq", "file_name"] if c in frame.columns])
    if "antibody_name" not in frame.columns:
        raise ValueError(f"{path} is missing antibody_name.")
    frame = frame.copy()
    if test_names is not None and not frame["antibody_name"].astype(str).str.startswith("GDPa3-").all():
        frame["antibody_name"] = _map_heldout_names_to_gdpa3(
            frame["antibody_name"],
            heldout_sequences=heldout_sequences,
            gdpa3_sequences=gdpa3_sequences,
            test_names=pd.Index(test_names),
        )
    frame = frame.set_index("antibody_name")
    numeric = frame.select_dtypes(include="number")
    return numeric.sort_index(axis=1)


def _build_feature_sets(
    physics: pd.DataFrame,
    moe: pd.DataFrame,
    esm2: pd.DataFrame,
    ab_roberta: pd.DataFrame,
    include_moe: bool,
) -> dict[str, pd.DataFrame]:
    feature_sets: dict[str, pd.DataFrame] = {
        "physics_global": physics,
        "esm2_35M_plus_physics": _concat(esm2, physics.add_prefix("physics__")),
        "ab_roberta_plus_physics": _concat(ab_roberta, physics.add_prefix("physics__")),
    }
    if include_moe:
        feature_sets.update(
            {
                "MOE": moe,
                "MOE_plus_physics": _concat(moe, physics.add_prefix("physics__")),
                "MOE_plus_physics_plus_esm2_35M": _concat(
                    moe,
                    physics.add_prefix("physics__"),
                    esm2.add_prefix("esm2__"),
                ),
                "MOE_plus_physics_plus_ab_roberta": _concat(
                    moe,
                    physics.add_prefix("physics__"),
                    ab_roberta.add_prefix("ab_roberta__"),
                ),
            }
        )
    return feature_sets


def _concat(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(frames, axis=1)


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


def _load_required_embeddings(embedding_dir: Path, dataset: str) -> pd.DataFrame:
    paths = [
        embedding_dir / f"{dataset}_esm2_t12_35M_chain_embeddings.parquet",
        embedding_dir / f"{dataset}_ab_roberta_chain_embeddings.parquet",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing PLM embedding files:\n" + "\n".join(missing))
    return pd.concat([load_chain_embeddings(path) for path in paths], ignore_index=True)


def _map_heldout_names_to_gdpa3(
    names: pd.Series,
    heldout_sequences: Path | None,
    gdpa3_sequences: pd.DataFrame | None,
    test_names: pd.Index,
) -> pd.Series:
    if heldout_sequences is None or gdpa3_sequences is None:
        raise ValueError("Heldout MOE remapping requires heldout and GDPa3 sequence tables.")
    heldout = pd.read_csv(heldout_sequences)
    heldout["_sequence_key"] = (
        heldout["vh_protein_sequence"].astype(str) + "|" + heldout["vl_protein_sequence"].astype(str)
    )
    gdpa3 = gdpa3_sequences.copy()
    gdpa3["_sequence_key"] = (
        gdpa3["vh_protein_sequence"].astype(str) + "|" + gdpa3["vl_protein_sequence"].astype(str)
    )
    mapping = heldout[["antibody_name", "_sequence_key"]].merge(
        gdpa3[["antibody_name", "_sequence_key"]],
        on="_sequence_key",
        how="left",
        suffixes=("_heldout", "_gdpa3"),
        validate="one_to_one",
    )
    lookup = mapping.set_index("antibody_name_heldout")["antibody_name_gdpa3"]
    mapped = names.map(lookup)
    if mapped.isna().any():
        missing = int(mapped.isna().sum())
        if len(names) == len(test_names):
            return pd.Series(test_names.to_numpy(), index=names.index)
        raise ValueError(f"{missing} heldout MOE rows could not be mapped to GDPa3 IDs.")
    return mapped


def _comparison_table(internal: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    left = internal[
        [
            "assay",
            "feature_set",
            "model",
            "n",
            "spearman",
            "worst_20_recall",
            "worst_20_auroc",
        ]
    ].rename(
        columns={
            "n": "internal_n",
            "spearman": "internal_spearman",
            "worst_20_recall": "internal_worst_20_recall",
            "worst_20_auroc": "internal_worst_20_auroc",
        }
    )
    right = external[
        [
            "assay",
            "n",
            "spearman",
            "worst_20_recall",
            "worst_20_auroc",
        ]
    ].rename(
        columns={
            "n": "external_n",
            "spearman": "external_spearman",
            "worst_20_recall": "external_worst_20_recall",
            "worst_20_auroc": "external_worst_20_auroc",
        }
    )
    return left.merge(right, on="assay", how="left")


def _beats_anchor(comparison: pd.DataFrame) -> pd.Series:
    hic = (comparison["assay"] == "HIC") & (comparison["external_worst_20_auroc"] > 0.787)
    pr = (
        (comparison["assay"] == "PR_CHO")
        & (comparison["external_worst_20_auroc"] >= 0.70)
        & (comparison["external_spearman"] >= 0.25)
    )
    return hic | pr


if __name__ == "__main__":
    main()
