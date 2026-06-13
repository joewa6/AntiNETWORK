# GDPa3 Firewall

> Note: the analysis notebooks referenced below are not included in this public
> repository — their saved outputs embed per-antibody benchmark data, which is
> not redistributed here. They are listed as the project's audit log.

## Rule

**GDPa3 is a sequestered audit set. It may not be used for any decision.**

This is an immutable, project-wide rule. It applies to every notebook, script, and analysis in this repository.

---

## Permitted use by dataset

| Dataset | Permitted use |
|---------|--------------|
| GDPa1   | Training, cross-validation, model selection, feature engineering, descriptor family selection, hyperparameter tuning, threshold setting, normalisation, everything development |
| GDPa2.1 | Optional VHH format-shift sanity check only |
| GDPa3   | External audit only — single, named, frozen evaluations |

---

## What counts as destroying GDPa3

These are forbidden:

```
try threshold A → check GDPa3
try threshold B → check GDPa3
try threshold C → check GDPa3
choose C because GDPa3 was best        ← test-set tuning
```

Also forbidden:
- Adding or removing descriptor families based on GDPa3 deltas
- Choosing a model configuration because it looked good on GDPa3
- Setting risk thresholds using GDPa3 percentiles
- Normalising features using GDPa1 + GDPa3 combined
- Looking at GDPa3 assay correlations before freezing all modelling choices

---

## Notebook protocol

Every new analysis that produces a GDPa3 score must be split into three stages:

### Stage 1 — Feature construction
Build features for GDPa1 and GDPa3. No evaluation. No assay labels loaded for GDPa3.

Example: `20_build_ab_structures.ipynb`

### Stage 2 — GDPa1-only model selection
Use GDPa1 CV to select:
- descriptor families
- model family and regularisation
- feature scaling and missing-value strategy
- any thresholds (patch radius, RSA cutoff, etc.)

Record the frozen specification. **Do not load GDPa3 labels in this notebook.**

Example: `21_structure_patch_GDPa1_model_selection.ipynb`

### Stage 3 — GDPa3 audit
Load the frozen specification from Stage 2. Train on full GDPa1. Predict GDPa3. Evaluate once. Write the result. Do not change anything after seeing it.

Example: `22_structure_patch_GDPa3_audit.ipynb`

---

## Audit history

Each entry is a named, versioned GDPa3 audit. Adding a version here means the evaluation has been run and the result is final.

| Version | Notebook | Description | Date |
|---------|----------|-------------|------|
| v1 | 16_external_GDPa3_benchmark.ipynb | Physical-only profiler baseline | — |
| v2 | 17_embedding_boosted_profiler_benchmark.ipynb | Embedding ridge | — |
| v3 | 18_abdev_benchmark_baseline_reproduction.ipynb | Official baseline reproduction | — |
| v4 | 19_physical_descriptor_additive_value.ipynb | Physical descriptor family additivity | — |
| v5 | 22_structure_patch_GDPa3_audit.ipynb | ABodyBuilder2 surface patch descriptors — HIC 0.516, PR 0.156, AC-SINS 0.091. Patches hurt: falsification, not failure. | 2026-06-03 |
| v6 | 24_rank_ensemble_GDPa3_audit.ipynb | Rank-average ensemble of official model predictions, selected by GDPa1 CV | pending |
| v7 | 27_winner_style_GDPa3_audit.ipynb | Winner-style stacking with robust GDPa1 selection (mean - 2×std - H3-stress penalty) | pending |
| v8 | 30_deepsp_GDPa3_audit_v8.ipynb | Local DeepSP features + MOE + physical, robust selection. Caveat: local DeepSP does not fully reproduce official deepsp_ridge (PR gap -0.19, Tm2 gap -0.22). HIC/AC-SINS candidate only. | pending |
| v9 | scripts/run_external_triage_audit.py | Physical sequence triage ranker, selected by GDPa1 cluster CV and audited once on GDPa3. HIC holds externally (Spearman 0.491, worst-20 AUROC 0.760); PR_CHO has moderate Spearman but weak tail triage; AC-SINS pH7.4, Titer, and Tm2 fail triage. | 2026-06-08 |
| v10 | scripts/run_plm_ranker_audit.py | Frozen ESM-2 35M and Ab-RoBERTa embeddings with PCA(64), selected by GDPa1 cluster CV and audited once on GDPa3. HIC remains the only useful external triage endpoint; ESM-2 improves HIC Spearman to 0.520 but lowers worst-20 AUROC to 0.715 versus physics-only 0.760. PR_CHO, AC-SINS pH7.4, Titer, and Tm2 do not pass the external usefulness threshold. | 2026-06-08 |

### v7 pre-registered pass criteria (set before audit, cannot be changed after)

**Primary pass** (any one): PR_CHO ≥ 0.33 OR HIC ≥ 0.60  
**Secondary pass** (any one): AC-SINS ≥ 0.25 OR Titer ≥ 0.20  
**Hard constraint**: neither HIC nor PR_CHO may fall below v4 (HIC 0.561, PR_CHO 0.307)

If v7 fails primary pass: stop stacking official predictions. Next move is DeepSP/AbLang-style feature reproduction, not more ensembles.

### v8 pre-registered pass criteria (set before audit, cannot be changed after)

**Useful** (any one): HIC > 0.561 OR AC-SINS > 0.215  
**Exciting** (any one): HIC ≥ 0.60 OR AC-SINS ≥ 0.25  
**Caveat**: local DeepSP does not reproduce official deepsp_ridge for PR/Tm2/Titer. v8 is a HIC/AC-SINS candidate only.  
If both HIC and AC-SINS drop below v4, local DeepSP does not help and next move is obtaining official DeepSP features or AbLang2 embeddings.

---

## Why this matters

The benchmark paper that defines this project explicitly showed that GDPa1 cross-validation overestimated GDPa3 held-out performance. GDPa3 is the only honest external signal the project has.

If structure-patch thresholds, descriptor families, or model choices are tuned against GDPa3, the v5 audit result is meaningless. The only thing it would say is "I made the test score go up after seeing the test set" — which is not science.

GDPa3 must remain a surprise. That is its only value.

---

## Enforcement

Every notebook must state in its opening cell whether it loads GDPa3 assay labels. If it does, it must be a named audit notebook that locks all modelling choices before that load.

No notebook may load GDPa3 assay labels and also make any modelling choice (feature selection, hyperparameter choice, threshold choice) in the same session.
