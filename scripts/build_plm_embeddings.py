from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from antinetwork.gdpa import MAIN_CSV
from antinetwork.official_benchmark import load_ginkgo_workbook, make_ginkgo_external_dataset
from antinetwork.plm_embeddings import (
    DEFAULT_PLM_MODELS,
    PLMEmbeddingSpec,
    build_chain_embedding_table,
    save_chain_embeddings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen VH/VL PLM embeddings.")
    parser.add_argument("--dataset", choices=["gdpa1", "gdpa3", "raw-gdpa1"], required=True)
    parser.add_argument("--model-name", choices=sorted(DEFAULT_PLM_MODELS), required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/plm_embeddings"))
    parser.add_argument("--comp-dir", type=Path, default=Path("data/GinkgoComp"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--gdpa1-workbook", default="GDPa1_v1.3_20251027_full.xlsx")
    parser.add_argument("--gdpa3-workbook", default="GDPa3_20260106_full.xlsx")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--allow-duplicate-openmp",
        action="store_true",
        help="Set KMP_DUPLICATE_LIB_OK=TRUE before importing torch if this local env needs it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.allow_duplicate_openmp:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataset(args)
    values = DEFAULT_PLM_MODELS[args.model_name]
    spec = PLMEmbeddingSpec(model_name=args.model_name, **values)
    embeddings = build_chain_embedding_table(
        df,
        spec,
        batch_size=args.batch_size,
        device=args.device,
    )

    path = args.out_dir / f"{args.dataset}_{args.model_name}_chain_embeddings.parquet"
    save_chain_embeddings(embeddings, path)
    print(f"Saved {embeddings.shape[0]} chain embeddings to {path}")


def _load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    if args.dataset == "raw-gdpa1":
        return pd.read_csv(args.raw_dir / MAIN_CSV)
    if args.dataset == "gdpa1":
        workbook = load_ginkgo_workbook(args.comp_dir / args.gdpa1_workbook)
        return make_ginkgo_external_dataset(workbook, split="train")
    if args.dataset == "gdpa3":
        workbook = load_ginkgo_workbook(args.comp_dir / args.gdpa3_workbook)
        return make_ginkgo_external_dataset(workbook, split="test")
    raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main()
