from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from antinetwork.features import clean_sequence


DEFAULT_PLM_MODELS = {
    "esm2_t12_35M": {
        "hf_model_id": "facebook/esm2_t12_35M_UR50D",
        "tokenizer_mode": "raw",
    },
    "ab_roberta": {
        "hf_model_id": "mogam-ai/Ab-RoBERTa",
        "tokenizer_mode": "spaced",
    },
}


@dataclass(frozen=True)
class PLMEmbeddingSpec:
    """A frozen embedding model specification."""

    model_name: str
    hf_model_id: str
    tokenizer_mode: str = "raw"


def default_plm_specs() -> list[PLMEmbeddingSpec]:
    """Return the boring first-pass PLM baselines."""
    return [
        PLMEmbeddingSpec(model_name=name, **values)
        for name, values in DEFAULT_PLM_MODELS.items()
    ]


def build_chain_embedding_table(
    df: pd.DataFrame,
    spec: PLMEmbeddingSpec,
    batch_size: int = 8,
    device: str | None = None,
) -> pd.DataFrame:
    """Embed VH and VL chains with a frozen Hugging Face masked-language model."""
    transformers, torch = _import_embedding_dependencies()
    tokenizer = transformers.AutoTokenizer.from_pretrained(spec.hf_model_id)
    model = transformers.AutoModel.from_pretrained(spec.hf_model_id)
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    sequence_rows = []
    for chain, column in [("vh", "vh_protein_sequence"), ("vl", "vl_protein_sequence")]:
        for idx, row in df.iterrows():
            sequence_rows.append(
                {
                    "row_index": idx,
                    "antibody_name": row["antibody_name"],
                    "model_name": spec.model_name,
                    "chain": chain,
                    "sequence": _format_sequence(row[column], spec.tokenizer_mode),
                }
            )

    vectors = []
    for start in range(0, len(sequence_rows), batch_size):
        batch = sequence_rows[start : start + batch_size]
        encoded = tokenizer(
            [row["sequence"] for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
        pooled = _mean_pool_hidden(output.last_hidden_state, encoded["attention_mask"])
        vectors.append(pooled.cpu().numpy())

    embeddings = np.vstack(vectors) if vectors else np.empty((0, 0))
    meta = pd.DataFrame(sequence_rows).drop(columns=["sequence"])
    values = pd.DataFrame(
        embeddings,
        columns=[f"embedding_{i}" for i in range(embeddings.shape[1])],
    )
    return pd.concat([meta.reset_index(drop=True), values], axis=1)


def build_antibody_embedding_features(
    chain_embeddings: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Convert chain-level VH/VL embeddings into antibody-level interaction features."""
    subset = chain_embeddings[chain_embeddings["model_name"] == model_name].copy()
    if subset.empty:
        raise ValueError(f"No chain embeddings found for model_name={model_name!r}.")

    embedding_cols = _embedding_columns(subset)
    vh = _chain_frame(subset, "vh", embedding_cols)
    vl = _chain_frame(subset, "vl", embedding_cols)
    common = vh.index.intersection(vl.index)
    vh = vh.loc[common]
    vl = vl.loc[common]

    parts = [
        vh.add_prefix(f"{model_name}__vh__"),
        vl.add_prefix(f"{model_name}__vl__"),
        ((vh + vl) / 2.0).add_prefix(f"{model_name}__mean__"),
        (vh - vl).add_prefix(f"{model_name}__diff__"),
        (vh - vl).abs().add_prefix(f"{model_name}__absdiff__"),
        (vh * vl).add_prefix(f"{model_name}__prod__"),
    ]
    return pd.concat(parts, axis=1).sort_index()


def build_multi_plm_feature_sets(
    chain_embeddings: pd.DataFrame,
    physical_features: pd.DataFrame,
    model_names: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Create the predeclared PLM/physics feature blocks for the audit."""
    model_names = model_names or sorted(chain_embeddings["model_name"].unique())
    plm_features = {
        model_name: build_antibody_embedding_features(chain_embeddings, model_name)
        for model_name in model_names
    }
    feature_sets: dict[str, pd.DataFrame] = {
        "physics_global": physical_features,
    }
    for model_name, features in plm_features.items():
        feature_sets[model_name] = features
        feature_sets[f"{model_name}_plus_physics"] = pd.concat(
            [features, physical_features.add_prefix("physics__")],
            axis=1,
        )
    if len(model_names) >= 2:
        combined_parts = [plm_features[name] for name in model_names]
        combined_parts.append(physical_features.add_prefix("physics__"))
        feature_sets["all_plm_plus_physics"] = pd.concat(combined_parts, axis=1)
    return feature_sets


def save_chain_embeddings(frame: pd.DataFrame, path: str | Path) -> None:
    """Save chain embeddings as parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def load_chain_embeddings(path: str | Path) -> pd.DataFrame:
    """Load chain embeddings saved by save_chain_embeddings."""
    return pd.read_parquet(path)


def _format_sequence(sequence: str, tokenizer_mode: str) -> str:
    seq = clean_sequence(sequence)
    if tokenizer_mode == "raw":
        return seq
    if tokenizer_mode == "spaced":
        return " ".join(seq)
    raise ValueError(f"Unknown tokenizer_mode: {tokenizer_mode}")


def _mean_pool_hidden(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(hidden)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def _embedding_columns(frame: pd.DataFrame) -> list[str]:
    cols = [
        col
        for col in frame.columns
        if col.startswith("embedding_") and frame[col].notna().any()
    ]
    if not cols:
        raise ValueError("No embedding_* columns found.")
    return cols


def _chain_frame(
    frame: pd.DataFrame,
    chain: str,
    embedding_cols: list[str],
) -> pd.DataFrame:
    out = frame[frame["chain"] == chain].set_index("row_index")
    if out.index.has_duplicates:
        raise ValueError(f"Duplicate {chain} embeddings for at least one row_index.")
    return out[embedding_cols].astype(float)


def _import_embedding_dependencies():
    try:
        import torch
        import transformers
    except Exception as exc:
        raise RuntimeError(
            "PLM embedding generation requires torch and transformers. "
            "Install them in the anti environment, then rerun the embedding script."
        ) from exc
    return transformers, torch
