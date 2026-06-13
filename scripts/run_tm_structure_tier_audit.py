from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from antinetwork.competitive_ranker import (
    TM_STRUCTURE_MODELS,
    evaluate_competitive_rankers,
    evaluate_external_ranker_predictions,
    select_best_rankers,
    train_full_ranker_predictions,
)
from antinetwork.features import build_physical_features
from antinetwork.gdpa import MAIN_CSV
from antinetwork.official_benchmark import load_ginkgo_workbook, make_ginkgo_external_dataset
from antinetwork.structure_topology import build_structure_stability_features


AUDIT_VERSION = "tm_structure_tier_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tm2 physical-mechanism tier audit without PLMs or neural nets."
    )
    parser.add_argument("--comp-dir", type=Path, default=Path("data/GinkgoComp"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/tm_structure_tier_audit"))
    parser.add_argument("--gdpa1-workbook", default="GDPa1_v1.3_20251027_full.xlsx")
    parser.add_argument("--gdpa3-workbook", default="GDPa3_20260106_full.xlsx")
    parser.add_argument("--structure-train-dir", type=Path, default=Path("data/structures/GDPa1"))
    parser.add_argument("--structure-test-dir", type=Path, default=Path("data/structures/GDPa3"))
    parser.add_argument("--summary-out", type=Path, default=Path("results/tm2_structure_tier_summary.csv"))
    parser.add_argument("--assays", nargs="+", default=["Tm2", "Tm2_subtype_residual"])
    parser.add_argument("--models", nargs="+", default=TM_STRUCTURE_MODELS)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = _load_train(args)
    test = _load_test(args)
    _add_subtype_residual_assay(train, test, assay="Tm2")
    train_groups = train["hierarchical_cluster_IgG_isotype_stratified_fold"]

    train_physics = build_physical_features(train)
    test_physics = build_physical_features(test)
    train_structure = build_structure_stability_features(train, args.structure_train_dir).select_dtypes(
        include="number"
    )
    test_for_structures = _with_heldout_structure_names(
        test,
        heldout_sequences=args.comp_dir / "seqs" / "heldout-set-sequences.csv",
    )
    test_structure = build_structure_stability_features(
        test_for_structures,
        args.structure_test_dir,
    ).select_dtypes(include="number")
    train_structure, test_structure = train_structure.align(test_structure, join="outer", axis=1)

    train_feature_sets = build_tm_tier_feature_sets(train_physics, train_structure, train)
    test_feature_sets = build_tm_tier_feature_sets(test_physics, test_structure, test)
    train_feature_sets, test_feature_sets = _align_feature_sets(train_feature_sets, test_feature_sets)

    internal = evaluate_competitive_rankers(
        train,
        feature_sets=train_feature_sets,
        assays=args.assays,
        groups=train_groups,
        model_names=args.models,
        cv_kind="cluster",
        random_state=args.seed,
        dataset=f"{AUDIT_VERSION}_GDPa1_cluster_cv",
    )

    external_frames = []
    all_model_external_frames = []
    comparison_frames = []
    all_model_comparison_frames = []
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
        for model_name in args.models:
            model_lookup = {assay: model_name for assay in args.assays}
            all_predictions = train_full_ranker_predictions(
                train,
                train_features,
                test,
                test_feature_sets[feature_set],
                assays=args.assays,
                model_by_assay=model_lookup,
                random_state=args.seed,
            )
            all_external = evaluate_external_ranker_predictions(
                test,
                all_predictions,
                assays=args.assays,
                model_by_assay=model_lookup,
                feature_set=feature_set,
                dataset=f"{AUDIT_VERSION}_GDPa3_all_models",
            )
            internal_row = internal.metrics[
                (internal.metrics["feature_set"] == feature_set)
                & (internal.metrics["model"] == model_name)
            ]
            all_model_external_frames.append(all_external)
            all_model_comparison_frames.append(_comparison_table(internal_row, all_external))

    selected_all = pd.concat(selected_frames, ignore_index=True)
    external_all = pd.concat(external_frames, ignore_index=True)
    all_model_external = pd.concat(all_model_external_frames, ignore_index=True)
    predictions_all = pd.concat(prediction_frames, ignore_index=True)
    comparison = pd.concat(comparison_frames, ignore_index=True)
    all_model_comparison = pd.concat(all_model_comparison_frames, ignore_index=True)
    comparison["externally_useful"] = (
        (comparison["external_spearman"] >= 0.20)
        & (comparison["external_worst_20_auroc"] >= 0.60)
    )
    all_model_comparison["externally_useful"] = (
        (all_model_comparison["external_spearman"] >= 0.20)
        & (all_model_comparison["external_worst_20_auroc"] >= 0.60)
    )

    best_by_external = comparison.sort_values(
        ["external_spearman", "external_worst_20_auroc"],
        ascending=[False, False],
    )

    spec = {
        "audit_version": AUDIT_VERSION,
        "assays": list(args.assays),
        "models_considered": list(args.models),
        "feature_sets": list(train_feature_sets),
        "selection_dataset": "GDPa1",
        "external_dataset": "GDPa3",
        "primary_selection_metric": "spearman",
        "success_condition": "external_spearman >= 0.20 and external_worst_20_auroc >= 0.60",
        "note": (
            "No PLM/ESM/transformer feature blocks. Tm2_subtype_residual subtracts "
            "GDPa1-estimated E[Tm2 | hc_subtype + lc_subtype]. GDPa3 labels are used "
            "only after GDPa1 model selection and residual baseline freezing."
        ),
    }
    (args.out_dir / "tm_structure_tier_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    internal.metrics.to_csv(args.out_dir / "tm_structure_tier_internal_metrics.csv", index=False)
    selected_all.to_csv(args.out_dir / "tm_structure_tier_selected_models.csv", index=False)
    predictions_all.to_csv(args.out_dir / "tm_structure_tier_external_predictions.csv", index=False)
    external_all.to_csv(args.out_dir / "tm_structure_tier_external_metrics.csv", index=False)
    all_model_external.to_csv(
        args.out_dir / "tm_structure_tier_external_all_model_metrics.csv",
        index=False,
    )
    comparison.to_csv(args.out_dir / "tm_structure_tier_internal_external_comparison.csv", index=False)
    all_model_comparison.to_csv(
        args.out_dir / "tm_structure_tier_internal_external_all_model_comparison.csv",
        index=False,
    )
    best_by_external.to_csv(args.out_dir / "tm_structure_tier_best_by_external.csv", index=False)
    _write_public_summary(comparison, args.summary_out)

    print(
        best_by_external[
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
        .head(12)
        .to_string(index=False)
    )


def build_tm_tier_feature_sets(
    physics: pd.DataFrame,
    structure: pd.DataFrame,
    metadata: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    tier1_interface = _select_columns(structure, ["interface_"], exclude=["interface_graph_"])
    tier1_hydrophobicity = _select_columns(structure, ["hydrophobic", "aromatic", "surface_proxy"])
    tier1_electrostatics = _select_columns(
        structure,
        ["positive_", "negative_", "charge", "dipole", "asymmetry"],
    )
    tier1_cdr_flexibility = _select_columns(
        structure,
        ["cdr_", "loop_entropy"],
        exclude=["cdr_packing_"],
    )
    tier2_frustration = _select_columns(structure, ["packing_", "_packing_"])
    tier2_defects = _select_columns(structure, ["defect_"])
    tier2_framework = _framework_proxy_features(metadata)
    tier3_graph = _select_columns(structure, ["interface_graph_"])
    tier1_all = pd.concat(
        [
            tier1_interface.add_prefix("interface__"),
            tier1_hydrophobicity.add_prefix("hydrophobic__"),
            tier1_electrostatics.add_prefix("electrostatic__"),
            tier1_cdr_flexibility.add_prefix("flexibility__"),
        ],
        axis=1,
    )
    tier2_all = pd.concat(
        [
            tier2_frustration.add_prefix("frustration__"),
            tier2_defects.add_prefix("defect__"),
            tier2_framework.add_prefix("framework__"),
        ],
        axis=1,
    )
    structure_all = pd.concat(
        [
            tier1_all.add_prefix("tier1__"),
            tier2_all.add_prefix("tier2__"),
            tier3_graph.add_prefix("tier3_graph__"),
        ],
        axis=1,
    )
    return {
        "tier1_interface": tier1_interface,
        "tier1_hydrophobicity": tier1_hydrophobicity,
        "tier1_electrostatics": tier1_electrostatics,
        "tier1_cdr_flexibility": tier1_cdr_flexibility,
        "tier1_all": tier1_all,
        "tier2_frustration": tier2_frustration,
        "tier2_defects": tier2_defects,
        "tier2_framework_proxy": tier2_framework,
        "tier2_all": tier2_all,
        "tier3_interface_graph": tier3_graph,
        "structure_all_tiers": structure_all,
        "sequence_physics": physics,
        "sequence_plus_structure_all": pd.concat(
            [physics.add_prefix("sequence__"), structure_all.add_prefix("structure__")],
            axis=1,
        ),
    }


def _select_columns(
    frame: pd.DataFrame,
    include: list[str],
    exclude: list[str] | None = None,
) -> pd.DataFrame:
    exclude = exclude or []
    columns = [
        column
        for column in frame.columns
        if any(pattern in column for pattern in include)
        and not any(pattern in column for pattern in exclude)
    ]
    if not columns:
        raise ValueError(f"No columns matched include={include} exclude={exclude}")
    return frame[columns].copy()


def _framework_proxy_features(metadata: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ["hc_subtype", "lc_subtype"] if column in metadata.columns]
    if not columns:
        return pd.DataFrame(index=metadata.index)
    return pd.get_dummies(metadata[columns].astype(str), prefix=columns, dtype=float)


def _align_feature_sets(
    train_sets: dict[str, pd.DataFrame],
    test_sets: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    aligned_train = {}
    aligned_test = {}
    for name, train in train_sets.items():
        test = test_sets[name]
        aligned_train[name], aligned_test[name] = train.align(test, join="outer", axis=1)
    return aligned_train, aligned_test


def _add_subtype_residual_assay(
    train: pd.DataFrame,
    test: pd.DataFrame,
    assay: str,
) -> None:
    """Add residual labels after subtracting GDPa1-estimated subtype expectation."""
    residual = f"{assay}_subtype_residual"
    controls = [column for column in ["hc_subtype", "lc_subtype"] if column in train.columns and column in test.columns]
    if not controls:
        train[residual] = train[assay] - train[assay].mean()
        test[residual] = test[assay] - train[assay].mean()
        return

    train_keys = _subtype_keys(train, controls)
    test_keys = _subtype_keys(test, controls)
    means = train.groupby(train_keys, dropna=False)[assay].mean()
    global_mean = float(train[assay].mean())

    train_expected = train_keys.map(means).fillna(global_mean)
    test_expected = test_keys.map(means).fillna(global_mean)
    train[residual] = train[assay] - train_expected.to_numpy(dtype=float)
    test[residual] = test[assay] - test_expected.to_numpy(dtype=float)


def _subtype_keys(frame: pd.DataFrame, controls: list[str]) -> pd.Series:
    return frame[controls].astype(str).agg("|".join, axis=1)


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


def _with_heldout_structure_names(
    gdpa3_sequences: pd.DataFrame,
    heldout_sequences: Path,
) -> pd.DataFrame:
    if not heldout_sequences.exists():
        return gdpa3_sequences
    heldout = pd.read_csv(heldout_sequences)
    heldout["_sequence_key"] = (
        heldout["vh_protein_sequence"].astype(str) + "|" + heldout["vl_protein_sequence"].astype(str)
    )
    gdpa3 = gdpa3_sequences.copy()
    gdpa3["_sequence_key"] = (
        gdpa3["vh_protein_sequence"].astype(str) + "|" + gdpa3["vl_protein_sequence"].astype(str)
    )
    mapping = gdpa3[["_sequence_key"]].merge(
        heldout[["antibody_name", "_sequence_key"]],
        on="_sequence_key",
        how="left",
        validate="one_to_one",
    )
    if mapping["antibody_name"].notna().all():
        gdpa3["antibody_name"] = mapping["antibody_name"].to_numpy()
    return gdpa3.drop(columns=["_sequence_key"])


def _comparison_table(internal: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    left = internal[
        [
            "assay",
            "feature_set",
            "model",
            "n",
            "spearman",
            "worst_20_auroc",
        ]
    ].rename(
        columns={
            "n": "internal_n",
            "spearman": "internal_spearman",
            "worst_20_auroc": "internal_worst_20_auroc",
        }
    )
    right = external[["assay", "n", "spearman", "worst_20_auroc"]].rename(
        columns={
            "n": "external_n",
            "spearman": "external_spearman",
            "worst_20_auroc": "external_worst_20_auroc",
        }
    )
    return left.merge(right, on="assay", how="left")


def _write_public_summary(comparison: pd.DataFrame, path: Path) -> None:
    """Write aggregate rows used by README figures, with no per-antibody data."""
    selections = [
        ("Tm2", "tier2_framework_proxy", "raw Tm2 is best explained by framework/subtype proxy"),
        ("Tm2", "sequence_plus_structure_all", "best structure-rich selected model"),
        (
            "Tm2",
            "tier2_defects",
            "defect features move raw Tm2 but do not clear Spearman threshold",
        ),
        (
            "Tm2_subtype_residual",
            "tier1_cdr_flexibility",
            "best GDPa1-selected residual model after subtracting subtype expectation",
        ),
    ]
    rows = []
    for assay, feature_set, note in selections:
        row = comparison[(comparison["assay"] == assay) & (comparison["feature_set"] == feature_set)]
        if row.empty:
            continue
        record = row.iloc[0][
            [
                "assay",
                "feature_set",
                "model",
                "internal_spearman",
                "external_spearman",
                "external_worst_20_auroc",
                "externally_useful",
            ]
        ].to_dict()
        record["note"] = note
        rows.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
