import numpy as np
import pandas as pd

from antinetwork.competitive_ranker import make_numeric_regression_pipeline
from antinetwork.plm_embeddings import (
    PLMEmbeddingSpec,
    build_antibody_embedding_features,
    build_multi_plm_feature_sets,
    default_plm_specs,
)


def test_build_antibody_embedding_features_adds_chain_interactions():
    rows = []
    for idx, name in enumerate(["ab1", "ab2"]):
        for chain, offset in [("vh", 0.0), ("vl", 10.0)]:
            rows.append(
                {
                    "row_index": idx,
                    "antibody_name": name,
                    "model_name": "toy_plm",
                    "chain": chain,
                    "embedding_0": idx + offset,
                    "embedding_1": idx + offset + 1.0,
                }
            )
    chain_embeddings = pd.DataFrame(rows)

    features = build_antibody_embedding_features(chain_embeddings, "toy_plm")

    assert features.shape == (2, 12)
    assert features.loc[0, "toy_plm__mean__embedding_0"] == 5.0
    assert features.loc[0, "toy_plm__diff__embedding_0"] == -10.0
    assert features.loc[0, "toy_plm__absdiff__embedding_0"] == 10.0
    assert features.loc[1, "toy_plm__prod__embedding_1"] == 24.0


def test_build_multi_plm_feature_sets_creates_predeclared_blocks():
    rows = []
    for model in ["esm2_t12_35M", "ab_roberta"]:
        for idx in range(3):
            for chain in ["vh", "vl"]:
                rows.append(
                    {
                        "row_index": idx,
                        "antibody_name": f"ab{idx}",
                        "model_name": model,
                        "chain": chain,
                        "embedding_0": float(idx),
                    }
                )
    physical = pd.DataFrame({"charge": [1.0, 2.0, 3.0]})

    feature_sets = build_multi_plm_feature_sets(pd.DataFrame(rows), physical)

    assert set(feature_sets) == {
        "physics_global",
        "ab_roberta",
        "ab_roberta_plus_physics",
        "esm2_t12_35M",
        "esm2_t12_35M_plus_physics",
        "all_plm_plus_physics",
    }
    assert "physics__charge" in feature_sets["all_plm_plus_physics"]


def test_adaptive_pca_pipeline_caps_components_inside_fit():
    x = pd.DataFrame(np.random.default_rng(7).normal(size=(12, 30)))
    y = np.linspace(0.0, 1.0, 12)

    pipeline = make_numeric_regression_pipeline("ridge", pca_components=64)
    pipeline.fit(x, y)

    assert pipeline.named_steps["pca"].n_components_ == 11


def test_default_plm_specs_are_boring_public_baselines():
    specs = {spec.model_name: spec for spec in default_plm_specs()}

    assert isinstance(specs["esm2_t12_35M"], PLMEmbeddingSpec)
    assert specs["esm2_t12_35M"].hf_model_id == "facebook/esm2_t12_35M_UR50D"
    assert specs["ab_roberta"].hf_model_id == "mogam-ai/Ab-RoBERTa"
