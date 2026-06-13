from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from antinetwork.competitive_ranker import evaluate_external_ranker_predictions
from antinetwork.official_benchmark import (
    OFFICIAL_ASSAYS,
    load_ginkgo_workbook,
    load_official_prediction_tables,
    make_ginkgo_external_dataset,
    map_sequence_named_predictions_to_external_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score frozen GDPa3 prediction tables with external triage metrics."
    )
    parser.add_argument("--comp-dir", type=Path, default=Path("data/GinkgoComp"))
    parser.add_argument("--gdpa3-workbook", default="GDPa3_20260106_full.xlsx")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/external_prediction_triage"))
    parser.add_argument(
        "--official-prediction-root",
        type=Path,
        default=Path("/private/tmp/abdev-benchmark/outputs/predictions/heldout_test"),
    )
    parser.add_argument("--assays", nargs="+", default=OFFICIAL_ASSAYS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gdpa3 = load_ginkgo_workbook(args.comp_dir / args.gdpa3_workbook)
    truth = make_ginkgo_external_dataset(gdpa3, split="test")
    external_sequences = gdpa3["sequences"]

    frames = []
    missing = []
    frames.extend(_score_available_repo_predictions(args.comp_dir, truth, args.assays))
    official_frames, official_missing = _score_official_prediction_root(
        args.official_prediction_root,
        truth,
        external_sequences,
        args.assays,
    )
    frames.extend(official_frames)
    missing.extend(official_missing)

    if frames:
        metrics = pd.concat(frames, ignore_index=True)
        metrics = metrics.sort_values(["assay", "worst_20_auroc", "spearman"], ascending=[True, False, False])
        best = metrics.groupby("assay", as_index=False).first().sort_values("assay")
    else:
        metrics = pd.DataFrame()
        best = pd.DataFrame()

    missing_df = pd.DataFrame(missing)
    metrics.to_csv(args.out_dir / "GDPa3_external_prediction_triage_metrics.csv", index=False)
    best.to_csv(args.out_dir / "GDPa3_external_prediction_triage_best_by_worst20_auroc.csv", index=False)
    missing_df.to_csv(args.out_dir / "missing_row_level_prediction_sources.csv", index=False)

    if not best.empty:
        columns = ["assay", "model", "feature_set", "n", "spearman", "worst_20_auroc", "worst_20_recall"]
        print(best[columns].round(3).to_string(index=False))
    if not missing_df.empty:
        print("\nMissing row-level prediction sources:")
        print(missing_df.to_string(index=False))


def _score_available_repo_predictions(
    comp_dir: Path,
    truth: pd.DataFrame,
    assays: list[str],
) -> list[pd.DataFrame]:
    frames = []
    labelled_paths = [
        (
            "physical_v1",
            "physical_sequence",
            comp_dir / "outputs" / "external_GDPa3" / "GDPa3_predictions.csv",
        ),
        (
            "v9_physical_ranker",
            "global_composition",
            Path("reports/external_triage_audit/GDPa3_external_triage_predictions_v9.csv"),
        ),
    ]
    for model, feature_set, path in labelled_paths:
        if not path.exists():
            continue
        predictions = pd.read_csv(path)
        frames.append(
            evaluate_external_ranker_predictions(
                truth,
                predictions,
                assays=[assay for assay in assays if assay in predictions.columns],
                model_by_assay={assay: model for assay in assays},
                feature_set=feature_set,
                dataset="GDPa3_external_prediction_triage",
            )
        )

    plus_path = comp_dir / "outputs" / "abdev_baseline_reproduction" / "official_plus_physical_GDPa3_predictions.csv"
    if plus_path.exists():
        table = pd.read_csv(plus_path)
        for model, predictions in table.groupby("model", sort=False):
            frames.append(
                evaluate_external_ranker_predictions(
                    truth,
                    predictions,
                    assays=[assay for assay in assays if assay in predictions.columns],
                    model_by_assay={assay: str(model) for assay in assays},
                    feature_set="official_processed_or_plus_physical",
                    dataset="GDPa3_external_prediction_triage",
                )
            )
    return frames


def _score_official_prediction_root(
    prediction_root: Path,
    truth: pd.DataFrame,
    external_sequences: pd.DataFrame,
    assays: list[str],
) -> tuple[list[pd.DataFrame], list[dict[str, str]]]:
    frames = []
    missing = []
    try:
        tables = load_official_prediction_tables(prediction_root)
    except FileNotFoundError:
        return frames, [{"source": str(prediction_root), "reason": "no */predictions.csv files found"}]

    for model, table in tables.items():
        try:
            mapped = _ensure_external_ids(table, external_sequences)
        except ValueError as exc:
            missing.append({"source": model, "reason": str(exc)})
            continue
        available_assays = [assay for assay in assays if assay in mapped.columns]
        if not available_assays:
            missing.append({"source": model, "reason": "no requested assay prediction columns"})
            continue
        frames.append(
            evaluate_external_ranker_predictions(
                truth,
                mapped,
                assays=available_assays,
                model_by_assay={assay: model for assay in assays},
                feature_set="official_row_level_prediction",
                dataset="GDPa3_external_prediction_triage",
            )
        )
    return frames, missing


def _ensure_external_ids(
    predictions: pd.DataFrame,
    external_sequences: pd.DataFrame,
) -> pd.DataFrame:
    if predictions["antibody_name"].astype(str).str.startswith("GDPa3-").all():
        return predictions
    return map_sequence_named_predictions_to_external_ids(predictions, external_sequences)


if __name__ == "__main__":
    main()
