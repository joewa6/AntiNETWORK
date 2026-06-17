# AntiNETWORK

Pushing simple antibody developability models until they break.

Most antibody developability models look better than they are.

This project asks a deliberately unfashionable question:

**How far can simple, physically interpretable antibody features go before
external validation kills them?**

The aim is not to build the biggest predictor, stack embeddings, or win a
leaderboard by accident. The aim is to test whether crude but meaningful
descriptors - charge, hydrophobicity, aromaticity, VH/VL balance, CDR
composition, framework/subtype effects, and simple topology proxies - carry real
assay signal when sequence-family leakage is controlled.

Some do.

Most do not.

That is the point.

AntiNETWORK is a reproducible audit of antibody developability prediction on the
Ginkgo Datapoints AbDev benchmark: GDPa1 public training data, 246 antibodies;
GDPa3 blinded held-out data, 80 antibodies. The central test is simple: if a
signal looks good in cross-validation but collapses on GDPa3, it was probably
not a robust developability model. It may still be biologically interesting, but
it is not something to trust.

Small models. Physical assumptions. Honest failure.

```mermaid
flowchart LR
    A[Antibody sequences] --> B[Simple physical features]
    B --> C[Grouped CV on GDPa1]
    B --> D[One-time GDPa3 firewall]
    C --> E[Internal signal]
    D --> F[External signal]
    E --> G{Does it survive?}
    F --> G
    G -->|Yes| H[Possible real signal]
    G -->|No| I[Likely leakage / artefact / weak endpoint]
```

This is not a model zoo. It is a failure-testing machine.

## The result in one picture

![Internal versus external validation collapse](results/headline_collapse.png)

Most endpoints look better under internal GDPa1 validation than they do on the
held-out GDPa3 split. HIC is the useful exception: the simple physicochemical
model holds externally. Tm2 also holds numerically, but the audit suggests it is
mostly framework/subtype signal rather than a clean Fab-stability descriptor.

## What survived

The cleanest signal is **HIC**.

Simple physicochemical descriptors reach roughly 0.49 Spearman on the GDPa3
held-out split, close to the internal grouped-CV estimate and above a
sequence-identity kNN baseline. That suggests the HIC signal is not just
sequence-neighbour memorisation. It is partly encoded in basic molecular
properties.

This is the useful result.

The second apparent signal is **Tm2**, but it is less clean. Raw Tm2 is partly
predictable, but the best simple signal looks more like a framework/subtype
proxy than a direct Fab-stability model. After subtracting the GDPa1-estimated
subtype expectation, the residual Tm2 signal is weak.

So the honest interpretation is:

- HIC: real physicochemical signal.
- Tm2: partly predictable, but mostly through framework/subtype structure in the
  dataset.
- PR_CHO, AC-SINS, Titer: weak or unstable under external validation.
- PLM embeddings: improve some internal scores but do not improve the held-out
  story.
- Topology simulator: built and tested, but does not beat the null, so it is
  not used in the headline result.

## The useful positive result: HIC

![HIC against published field baselines](results/hic_vs_field_baselines.png)

On held-out HIC, the simple physicochemical model reaches roughly 0.49 Spearman.
That places it in the same range as stronger published baselines and above the
sequence-only baselines reported in the competition. The interpretation is
narrow: HIC contains real physicochemical signal, but the field ceiling remains
higher, and structure appears to be the missing ingredient.

## What failed

The negative results matter because a simple model is useful when it fails
clearly.

PR_CHO, AC-SINS, and Titer either collapse or remain too weak under the GDPa3
firewall to treat as robust developability predictors. Frozen protein-language
model embeddings improve some internal validation scores, but they do not rescue
the external story. The structure-derived charge-topology features and
patchy-particle interaction-network simulator are implemented and unit-tested,
but they do not beat a composition-matched scramble null.

That is not an embarrassment. It is the instrument doing its job.

## Benchmark context

The Ginkgo AbDev competition ran from 8 September to 18 November 2025 with 113
teams. Its results are published, so the held-out numbers reported here are
comparable against a fixed benchmark: the field ceilings below are the best
scores achieved by those teams on the same GDPa3 held-out set.

- Competition leaderboard: https://huggingface.co/spaces/ginkgo-datapoints/abdev-leaderboard
- Outcomes paper: "2025 Ginkgo Datapoints Antibody Developability Competition
  outcomes: limited model performance and a call for data standardization,"
  *mAbs* (2026), https://doi.org/10.1080/19420862.2026.2634216

## Results table

Spearman rank correlation per assay. Internal is sequence-cluster grouped
cross-validation on GDPa1; external is the one-time GDPa3 held-out evaluation;
the field ceiling is the best of 113 competition teams on the same held-out set.

| Assay   | Internal grouped-CV | External GDPa3 | Field ceiling (best of 113) |
|---------|--------------------:|---------------:|----------------------------:|
| HIC     | ~0.47               | ~0.49 (held)   | 0.708                       |
| PR_CHO  | 0.52                | 0.28           | 0.356                       |
| AC-SINS | 0.33                | 0.13           | 0.337                       |
| Titer   | 0.21                | 0.00           | 0.310                       |
| Tm2     | 0.29                | 0.33           | 0.392                       |

HIC is the one assay where the internal number holds out of distribution. On the
same held-out split, a sequence-identity kNN baseline reaches 0.33. So physics
beats nearest-neighbour memorization by roughly 0.16 Spearman externally. The
HIC signal is physicochemical and real, not just memorization, but kNN alone
also recovers a large fraction of the achievable rank-order. That bound is part
of the finding.

## Method

The project is built around boring-but-necessary scientific hygiene:

- Sequence-cluster grouped cross-validation.
- A one-time external GDPa3 firewall.
- Sequence-identity nearest-neighbour baselines.
- Label-shuffle and composition-matched scramble nulls.
- Deliberately simple physicochemical models.
- Negative results kept in the repo instead of being quietly buried.

Feature and model families:

- Crude per-chain and per-CDR physicochemical sequence features: charge,
  hydrophobicity, aromatic content, VH/VL imbalances.
- Structure-derived Fab stability features for the Tm2 follow-up: VH/VL
  interface contacts, surface hydrophobic/electrostatic patches, CDR
  flexibility, weak-region packing, defect proxies, and interface graph
  descriptors.
- Ridge, elastic-net, linear/RBF SVM, spline-ridge GAM proxy, random forest, and
  hist-gradient-boosting models.
- Structure-derived charge-topology features and frozen protein-language-model
  embeddings are implemented but are not load-bearing for the headline result.

For exact input files, assay column mappings, patch definitions, and feature
calculations, see
[`docs/input_data_and_feature_dictionary.md`](docs/input_data_and_feature_dictionary.md).

## Additional audits

### Tm2 framework/subtype audit

**Tm2 is not clean Fab Tm.** The best Tm2 follow-up result is a
framework/subtype proxy with a spline-ridge GAM-like model: GDPa1 grouped-CV
Spearman 0.29 and GDPa3 Spearman 0.33, close to the published field ceiling of
0.392. That does not mean the Fab-stability mechanism is solved.

The AbDev outcomes paper notes that Tm2 reflects overlapping Fab/CH3 unfolding
transitions in intact IgG measurements, and the assay often cannot cleanly
deconvolve Fab, CH2, and CH3 transitions. When we subtract the GDPa1-estimated
expectation `E[Tm2 | hc_subtype + lc_subtype]`, the best selected residual model
falls to 0.10 external Spearman. The supported interpretation is: Tm2 contains a
real framework/isotype component, plus only a weak and unstable residual
Fab-structure signal.

![Raw Tm2 is partly predictable from framework/subtype and structure-rich features, but the subtype residual is weak.](results/tm2_subtype_residual.png)

Regenerate with `python scripts/run_tm_structure_tier_audit.py` and
`python scripts/plot_tm2_subtype_residual.py`.

### Phase 2 model/feature sweep

A broader non-neural sweep tests mechanism blocks for HIC/SMAC
hydrophobic-charge behavior, PR CDR stickiness, AC-SINS patch anisotropy, and Tm
framework defects across ridge, elastic-net, SVMs, a spline GAM proxy, random
forest, and hist-gradient boosting. The firewall-safe primary selection still
uses GDPa1 grouped CV before touching GDPa3.

That selection keeps HIC strong at 0.51 external Spearman but exposes target
instability: GDPa1-selected PR_CHO, AC-SINS, Titer, and Tm2 feature/model choices
do not reliably hold up externally. The post-hoc mechanism diagnostics are more
encouraging but should not be treated as selected models: best mechanism-family
rows reach HIC 0.55, PR_CHO 0.34, AC-SINS 0.25, and Tm2 0.29 external Spearman.

![Phase 2 external Spearman heatmap by target and feature block for GDPa1-selected models.](results/phase2_target_feature_heatmap.png)

Each cell scores the GDPa1-selected model for one target and feature block on
GDPa3. Aggregate summary rows are in
`results/phase2_feature_model_summary.csv`; all-model GDPa3 rankings live under
ignored `reports/phase2_feature_model_audit/` and are diagnostic/post-hoc only.

### Preregistered mechanism firewall check

To test the main Phase 2 hypotheses without choosing feature blocks from GDPa3,
`scripts/run_phase2_preregistered_firewall.py` locks two mechanism choices in
advance: AC-SINS uses the patch-anisotropy block, and Tm2 uses a
subtype-residual target with framework/subtype columns removed from the defect
block. GDPa1 grouped CV then selects only the model class inside each fixed
feature block.

The clean AC-SINS patch test reaches 0.164 external Spearman and 0.612 worst-20
AUROC, below the predefined usefulness threshold; the Tm2 residual defect test
is negative at -0.133 external Spearman. So the post-hoc AC-SINS 0.25 heatmap
cell is best treated as a hypothesis for better patch/self-association
descriptors, not as a confirmed firewall result.

## What the audit establishes

Under firewalled evaluation, simple physicochemical features land in the same
range as far heavier methods on the one assay where a clean signal exists. That
does not mean simple models are magic. It means HIC contains accessible
physicochemical signal, and larger sequence models do not automatically buy
generalisation.

For the other assays, the external evaluation is the story: internal validation
overstates performance, target-specific mechanism choices are fragile at this
sample size, and the honest negative result is more useful than a prettier
internal score.

## Status / scope of implementation

Implemented and load-bearing for the result above:

- Physicochemical per-chain/per-CDR features.
- Sequence-cluster grouped cross-validation.
- External GDPa3 firewall.
- Falsification controls: composition-matched scramble null, label-shuffle null,
  sequence-identity kNN baseline.
- Tm2 post-hoc tier audit: framework/subtype baseline, structure-rich stability
  features, defect proxies, and subtype-residual evaluation.
- Phase 2 feature/model audit over headline targets with target-specific
  mechanism blocks and expanded non-neural model classes.

Implemented but **not** load-bearing:

- Structure-derived charge-topology features and the patchy-particle
  interaction-network simulator are built and unit-tested, but they do not beat a
  composition-matched scramble null. They are not used in the headline result.
- PLM embeddings (ESM-2 35M, Ab-RoBERTa) are implemented but reduce external
  performance, so they are not used in the headline result.

## Firewall hygiene

For the original external triage and HIC result, GDPa3 held-out labels were used
only for one-time external evaluation and a single audit-time kNN baseline. The
Tm2 tier/residual analysis is explicitly a post-hoc mechanistic audit motivated
by the AbDev assay paper and the observed subtype signal; within that audit,
model selection is still performed on GDPa1 grouped CV before GDPa3 scoring.

## Data availability

The GDPa1/GDPa3 benchmark data (Ginkgo Datapoints) is gated and carries no stated
redistribution license, so it is **not** included in this repository. Obtain it
from the source under Ginkgo's access terms: GDPa1 is on Hugging Face
(`ginkgo-datapoints/GDPa1`); fetch it with `scripts/download_gdpa.py` after
accepting the dataset terms and running `hf auth login`. GDPa3 is the
gated/external held-out set and is not redistributed here. The only data-derived
files committed are aggregate summary statistics under `results/` - per-assay
metric tables with no per-antibody rows, sequences, or raw assay values.

## Reproduction

```bash
conda env create -f environment.yml
conda activate anti
python scripts/download_gdpa.py        # GDPa1 from Hugging Face, after `hf auth login`
python -m pytest                       # no data or network required
```

### Reproduce the headline numbers

With GDPa1 and GDPa3 obtained from source, each headline result is regenerated by
one entry point. GDPa3 is gated/external; outputs here are aggregate tables under
`results/`.

| Headline | Command | Output |
|----------|---------|--------|
| HIC external ~0.49; PLMs help internal, hurt external | `python scripts/run_v11_descriptor_plm_stack_audit.py` | `results/descriptor_plm_internal_vs_external.csv`, `results/external_vs_field_ceiling.csv` |
| Internal to external collapse, including PR_CHO 0.52 to 0.28 | `python scripts/run_external_triage_audit.py` | `results/internal_vs_external_triage_comparison.csv` |
| Tm2 framework/subtype signal and subtype-residual collapse | `python scripts/run_tm_structure_tier_audit.py` | `results/tm2_structure_tier_summary.csv` |
| Phase 2 feature/model sweep over headline targets | `python scripts/run_phase2_feature_model_audit.py` | `results/phase2_feature_model_summary.csv`, `results/phase2_target_feature_heatmap.png` |
| Preregistered AC-SINS patch and Tm2 residual mechanism tests | `python scripts/run_phase2_preregistered_firewall.py` | `results/phase2_preregistered_firewall_summary.csv` |
| HIC physics 0.49 beats identity-kNN 0.33 | `python scripts/knn_baseline_hic.py` | prints the comparison |
| Topology does not beat a composition-matched null | `antinetwork.topology_falsification.run_topology_falsification_gate` (exercised by `tests/test_topology_falsification.py`) | `results/topology_gate_verdicts.csv` |
