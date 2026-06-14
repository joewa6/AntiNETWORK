#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from huggingface_hub import snapshot_download

from antinetwork.gdpa import DATASET_ID, DEFAULT_HF_CSV_URI, MAIN_CSV, load_gdpa_csv


def download_csv(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    df = load_gdpa_csv(DEFAULT_HF_CSV_URI)
    df.to_csv(output, index=False)
    print(f"Saved {len(df):,} rows and {len(df.columns):,} columns to {output}")


def download_snapshot(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=output_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Saved dataset snapshot to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download GDPa1 data from Hugging Face.")
    parser.add_argument(
        "--method",
        choices=["csv", "snapshot"],
        default="csv",
        help="Download only the main CSV or mirror the full dataset repository.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw") / MAIN_CSV,
        help="CSV output path when --method csv is used.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("data/raw/hf_snapshot"),
        help="Output directory when --method snapshot is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.method == "csv":
            download_csv(args.output)
        else:
            download_snapshot(args.snapshot_dir)
    except (GatedRepoError, HfHubHTTPError) as exc:
        print(
            "Could not access GDPa1 on Hugging Face. Accept the dataset terms at "
            "https://huggingface.co/datasets/ginkgo-datapoints/GDPa1, then run "
            "`hf auth login` inside the `anti` environment.",
            file=sys.stderr,
        )
        print(f"Original error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
