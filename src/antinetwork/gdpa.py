from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
from datasets import load_dataset

DATASET_ID = "ginkgo-datapoints/GDPa1"
MAIN_CSV = "GDPa1_v1.2_20250814.csv"
DEFAULT_HF_CSV_URI = f"hf://datasets/{DATASET_ID}/{MAIN_CSV}"

FOLD_COLUMNS = [
    "random_fold",
    "hierarchical_cluster_fold",
    "hierarchical_cluster_IgG_isotype_stratified_fold",
]


def load_gdpa_csv(path_or_uri: str | Path | None = None, **read_csv_kwargs) -> pd.DataFrame:
    """Load the main GDPa1 CSV using pandas.

    The default Hugging Face URI requires that the user has accepted the dataset
    terms and has authenticated with `hf auth login`.
    """
    return pd.read_csv(path_or_uri or DEFAULT_HF_CSV_URI, **read_csv_kwargs)


def load_gdpa_dataset(split: str = "train") -> pd.DataFrame:
    """Load GDPa1 through Hugging Face Datasets and return a pandas DataFrame."""
    dataset_dict = load_dataset(DATASET_ID)
    return dataset_dict[split].to_pandas()


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return count and percentage of missing values by column."""
    missing_count = df.isna().sum()
    report = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_fraction": missing_count / len(df),
            "dtype": df.dtypes.astype(str),
        }
    )
    return report.sort_values(["missing_fraction", "missing_count"], ascending=False)


def fold_summary(df: pd.DataFrame, fold_columns: list[str] | None = None) -> dict[str, pd.Series]:
    """Count observations in each available cross-validation fold column."""
    columns = fold_columns or FOLD_COLUMNS
    return {
        column: df[column].value_counts(dropna=False).sort_index()
        for column in columns
        if column in df.columns
    }


def numeric_assay_columns(df: pd.DataFrame, exclude_folds: bool = True) -> list[str]:
    """Infer numeric assay-like columns from a GDPa1 table."""
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if exclude_folds:
        numeric_columns = [column for column in numeric_columns if column not in FOLD_COLUMNS]
    return numeric_columns


def network_from_correlations(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    min_abs_corr: float = 0.65,
    method: str = "spearman",
) -> nx.Graph:
    """Build an undirected assay-correlation graph.

    Nodes are numeric assay columns. Edges connect assays with an absolute
    pairwise correlation greater than or equal to `min_abs_corr`.
    """
    if not 0 <= min_abs_corr <= 1:
        raise ValueError("min_abs_corr must be between 0 and 1.")

    selected_columns = columns or numeric_assay_columns(df)
    corr = df[selected_columns].corr(method=method, min_periods=3)

    graph = nx.Graph()
    for column in selected_columns:
        graph.add_node(column)

    for i, source in enumerate(selected_columns):
        for target in selected_columns[i + 1 :]:
            value = corr.loc[source, target]
            if pd.notna(value) and abs(value) >= min_abs_corr:
                graph.add_edge(source, target, correlation=float(value), abs_correlation=float(abs(value)))

    return graph
