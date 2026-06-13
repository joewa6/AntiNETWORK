from __future__ import annotations

import numpy as np
import pandas as pd

from antinetwork.official_benchmark import (
    compare_to_competition_targets,
    compare_external_to_competition_targets,
    evaluate_external_predictions,
    evaluate_official_cv_predictions,
    ginkgo_column_mapping_table,
    make_ginkgo_external_dataset,
    map_sequence_named_predictions_to_external_ids,
    normalize_ginkgo_sequences,
    official_fold_model_predictions,
    official_property_target_table,
    recall_at_top_fraction,
    residual_charge_after_official_predictions,
    train_full_model_predictions,
)


def test_recall_at_top_fraction_flips_lower_is_better_assays():
    y_true = np.array([1.0, 2.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 100.0])
    good_for_low_hic = np.array([1.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0])
    bad_for_low_hic = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

    assert recall_at_top_fraction(y_true, good_for_low_hic, "HIC") == 1.0
    assert recall_at_top_fraction(y_true, bad_for_low_hic, "HIC") == 0.0


def test_evaluate_official_cv_predictions_returns_average_and_aggregated_rows():
    truth = pd.DataFrame(
        {
            "antibody_name": [f"ab{i}" for i in range(20)],
            "hierarchical_cluster_IgG_isotype_stratified_fold": [i % 5 for i in range(20)],
            "HIC": np.arange(20, dtype=float),
        }
    )
    preds = truth[["antibody_name"]].copy()
    preds["HIC"] = truth["HIC"]

    metrics = evaluate_official_cv_predictions(truth, preds, assays=["HIC"], model="toy")

    assert set(metrics["fold"]) == {"0", "1", "2", "3", "4", "average", "aggregated"}
    assert metrics.loc[metrics["fold"] == "average", "spearman"].iloc[0] == 1.0


def test_official_fold_model_predictions_uses_held_out_folds():
    df = pd.DataFrame(
        {
            "antibody_name": [f"ab{i}" for i in range(25)],
            "hierarchical_cluster_IgG_isotype_stratified_fold": [i % 5 for i in range(25)],
            "HIC": np.linspace(0.0, 1.0, 25),
        }
    )
    features = pd.DataFrame({"charge": np.linspace(0.0, 1.0, 25)}, index=df.index)

    preds = official_fold_model_predictions(df, features, assays=["HIC"])

    assert preds["HIC"].notna().all()
    assert preds["antibody_name"].tolist() == df["antibody_name"].tolist()


def test_residual_charge_after_official_predictions_reports_delta():
    n = 50
    charge = np.linspace(-1.0, 1.0, n)
    official = np.linspace(0.0, 1.0, n)
    target = official + charge
    df = pd.DataFrame(
        {
            "antibody_name": [f"ab{i}" for i in range(n)],
            "hierarchical_cluster_IgG_isotype_stratified_fold": [i % 5 for i in range(n)],
            "HIC": target,
        }
    )
    features = pd.DataFrame({"charge": charge}, index=df.index)
    official_preds = {
        "official": pd.DataFrame(
            {
                "antibody_name": df["antibody_name"],
                "HIC": official,
            }
        )
    }

    result = residual_charge_after_official_predictions(
        df,
        features,
        official_preds,
        assays=["HIC"],
    )

    assert result.loc[0, "official_model"] == "official"
    assert result.loc[0, "plus_charge_spearman"] >= result.loc[0, "official_spearman"]


def test_official_property_target_comparison_reports_status():
    targets = official_property_target_table()
    metrics = pd.DataFrame(
        {
            "fold": ["average", "average"],
            "assay": ["HIC", "PR_CHO"],
            "model": ["hic_model", "pr_model"],
            "spearman": [0.71, 0.20],
        }
    )

    comparison = compare_to_competition_targets(metrics)

    assert set(targets["assay"]) == {"HIC", "Tm2", "Titer", "PR_CHO", "AC-SINS_pH7.4"}
    assert comparison.loc[comparison["assay"] == "HIC", "status"].iloc[0] == "meets_or_beats_top"
    assert comparison.loc[comparison["assay"] == "PR_CHO", "status"].iloc[0] == "below_bar"


def test_train_full_model_predictions_predicts_new_rows():
    train = pd.DataFrame(
        {
            "antibody_name": [f"train{i}" for i in range(12)],
            "HIC": np.linspace(0.0, 1.0, 12),
        }
    )
    predict = pd.DataFrame({"antibody_name": ["held0", "held1"]})
    train_features = pd.DataFrame({"feature": np.linspace(0.0, 1.0, 12)})
    predict_features = pd.DataFrame({"feature": [0.25, 0.75]})

    predictions = train_full_model_predictions(
        train,
        train_features,
        predict,
        predict_features,
        assays=["HIC"],
    )

    assert predictions["antibody_name"].tolist() == ["held0", "held1"]
    assert predictions["HIC"].notna().all()


def test_normalize_ginkgo_sequences_uses_id_and_lc_sequence_fallbacks():
    sequences = pd.DataFrame(
        {
            "antibody_id": ["GDPa3-001"],
            "vh_protein_sequence": ["EVQ"],
            "lc_protein_sequence": ["SYE"],
        }
    )

    normalized = normalize_ginkgo_sequences(sequences)

    assert normalized.loc[0, "antibody_name"] == "GDPa3-001"
    assert normalized.loc[0, "vl_protein_sequence"] == "SYE"
    assert normalized.loc[0, "heavy_aligned_aho"] == "EVQ"
    assert normalized.loc[0, "light_aligned_aho"] == "SYE"


def test_normalize_ginkgo_sequences_tolerates_vhh_sequence_for_inspection():
    sequences = pd.DataFrame(
        {
            "antibody_id": ["GDPa2-001"],
            "protein_sequence": ["EVQLVESGGG"],
        }
    )

    normalized = normalize_ginkgo_sequences(sequences)

    assert normalized.loc[0, "vh_protein_sequence"] == "EVQLVESGGG"
    assert normalized.loc[0, "vl_protein_sequence"] == ""


def test_ginkgo_external_dataset_maps_train_and_test_targets():
    workbook = {
        "sequences": normalize_ginkgo_sequences(
            pd.DataFrame(
                {
                    "antibody_id": ["ab1", "ab2"],
                    "vh_protein_sequence": ["EVQ", "EVR"],
                    "lc_protein_sequence": ["SYE", "SYQ"],
                }
            )
        ),
        "assay_average": pd.DataFrame(
            {
                "antibody_id": ["ab1", "ab2"],
                "hic_rt_avg": [1.0, 2.0],
                "tm2_nanodsf_avg": [80.0, 81.0],
                "polyreactivity_prscore_cho_avg": [0.1, 0.2],
                "acsins_dLmax_ph7.4_avg": [3.0, 4.0],
                "titer_avg": [100.0, 200.0],
                "sec_%monomer_avg": [95.0, 96.0],
            }
        ),
        "assay_tidy": pd.DataFrame(),
    }

    mapping = ginkgo_column_mapping_table(workbook["assay_average"], workbook["assay_average"])
    test = make_ginkgo_external_dataset(workbook, split="test")

    assert mapping.loc[mapping["assay"] == "HIC", "test_present"].iloc[0]
    assert test["HIC"].tolist() == [1.0, 2.0]
    assert test["Titer"].tolist() == [100.0, 200.0]


def test_evaluate_external_predictions_and_compare_targets():
    truth = pd.DataFrame(
        {
            "antibody_name": [f"ab{i}" for i in range(10)],
            "HIC": np.arange(10, dtype=float),
        }
    )
    predictions = pd.DataFrame(
        {
            "antibody_name": truth["antibody_name"],
            "HIC": truth["HIC"],
        }
    )

    metrics = evaluate_external_predictions(truth, predictions, assays=["HIC"], model="toy")
    comparison = compare_external_to_competition_targets(metrics)

    assert metrics.loc[0, "dataset"] == "GDPa3_external"
    assert np.isclose(metrics.loc[0, "spearman"], 1.0)
    assert comparison.loc[comparison["assay"] == "HIC", "status"].iloc[0] == "meets_or_beats_top"


def test_map_sequence_named_predictions_to_external_ids_uses_exact_sequence_keys():
    predictions = pd.DataFrame(
        {
            "antibody_name": ["legacy_b", "legacy_a"],
            "vh_protein_sequence": ["VH2", "VH1"],
            "vl_protein_sequence": ["VL2", "VL1"],
            "HIC": [2.0, 1.0],
        }
    )
    external_sequences = pd.DataFrame(
        {
            "antibody_id": ["GDPa3-001", "GDPa3-002"],
            "vh_protein_sequence": ["VH1", "VH2"],
            "lc_protein_sequence": ["VL1", "VL2"],
        }
    )

    mapped = map_sequence_named_predictions_to_external_ids(predictions, external_sequences)

    assert mapped["antibody_name"].tolist() == ["GDPa3-002", "GDPa3-001"]
    assert mapped["legacy_antibody_name"].tolist() == ["legacy_b", "legacy_a"]
