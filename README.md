# AntiNETWORK

A leakage-controlled audit of antibody developability assays on the Ginkgo
AbDev benchmark (GDPa1 public training set, 246 antibodies; GDPa3 blinded
held-out set, 80 antibodies). The question it asks is which developability
signals survive honest external evaluation once sequence-family leakage is
controlled — not "can we build a predictor." The headline outcome is
negative-leaning, and reporting that accurately is the point of the project.

![Internal GDPa1 cross-validation versus external GDPa3 held-out Spearman per assay; HIC survives as a physicochemical signal, while the Tm2 signal is mostly framework/subtype-associated.](results/headline_collapse.png)

*Per assay: internal sequence-cluster grouped cross-validation on GDPa1 versus a
single prediction on the blinded GDPa3 held-out set. HIC survives as the cleanest
physicochemical signal. A post-hoc Tm2 structure/framework audit reaches 0.33
external Spearman, but the signal is largely explained by heavy/light subtype
rather than a clean Fab-stability measurement. Black bars mark the field ceiling
— the best per-assay score across the 113 competition teams. Regenerate with
`python scripts/plot_headline_collapse.py`.*

## Benchmark

The Ginkgo AbDev competition (GDPa1 training set, GDPa3 held-out set) ran from
8 September to 18 November 2025 with 113 teams. Its results are published, so the
held-out numbers reported here are comparable against a fixed, citable benchmark
— the field ceilings below are the best scores achieved by those teams on the
same held-out set.

- Competition leaderboard: https://huggingface.co/spaces/ginkgo-datapoints/abdev-leaderboard
- Outcomes paper: "2025 Ginkgo Datapoints Antibody Developability Competition
  outcomes: limited model performance and a call for data standardization,"
  *mAbs* (2026), https://doi.org/10.1080/19420862.2026.2634216

## Method

- Crude per-chain and per-CDR physicochemical sequence features: charge,
  hydrophobicity, aromatic content, VH/VL imbalances.
- Structure-derived Fab stability features for the Tm2 follow-up: VH/VL
  interface contacts, surface hydrophobic/electrostatic patches, CDR flexibility,
  weak-region packing, defect proxies, and interface graph descriptors.
- Models: ridge, elastic-net, linear/RBF SVM, spline-ridge GAM proxy, random
  forest, hist-gradient-boosting.
- Sequence-cluster grouped cross-validation as the internal evaluation standard.
- An external GDPa3 firewall: train on GDPa1, predict the blinded held-out set
  once.
- Falsification controls: composition-matched scramble null, label-shuffle null,
  and a sequence-identity kNN baseline.
- Structure-derived charge-topology features and frozen protein-language-model
  (PLM) embeddings are also implemented (see Status).

## Results

Spearman rank correlation per assay. Internal is grouped-CV on GDPa1; external is
the one-time GDPa3 held-out evaluation; the field ceiling is the best of 113
competition teams on the same held-out set.

| Assay   | Internal grouped-CV | External GDPa3 | Field ceiling (best of 113) |
|---------|--------------------:|---------------:|----------------------------:|
| HIC     | ≈ 0.47              | ≈ 0.49 (held)  | 0.708                       |
| PR_CHO  | 0.52                | 0.28           | 0.356                       |
| AC-SINS | 0.33                | 0.13           | 0.337                       |
| Titer   | 0.21                | 0.00           | 0.310                       |
| Tm2     | 0.29                | 0.33           | 0.392                       |

HIC is the one assay where the internal number holds out of distribution. On the
same held-out split, a sequence-identity kNN baseline reaches 0.33. So physics
beats nearest-neighbour memorization by roughly 0.16 Spearman externally — the
HIC signal is physicochemical and real, not memorization — but kNN alone reaches
0.33, meaning roughly two-thirds of the achievable HIC rank-order is also
recoverable from sequence similarity. That bound is part of the finding.

**Tm2 is not clean Fab Tm.** The best Tm2 follow-up result is a
framework/subtype proxy with a spline-ridge GAM-like model: GDPa1 grouped-CV
Spearman 0.29 and GDPa3 Spearman 0.33, close to the published field ceiling of
0.392. That does not mean the Fab-stability mechanism is solved. The AbDev
outcomes paper notes that Tm2 reflects overlapping Fab/CH3 unfolding transitions
in intact IgG measurements, and the assay often cannot cleanly deconvolve Fab,
CH2, and CH3 transitions. When we subtract the GDPa1-estimated expectation
`E[Tm2 | hc_subtype + lc_subtype]`, the best selected residual model falls to
0.10 external Spearman. So the supported interpretation is: Tm2 contains a real
framework/isotype component, plus only a weak and unstable residual Fab-structure
signal.

![Raw Tm2 is partly predictable from framework/subtype and structure-rich features, but the subtype residual is weak.](results/tm2_subtype_residual.png)

*Post-hoc Tm2 tier audit. The raw Tm2 framework/subtype proxy reaches 0.33
external Spearman; a sequence+structure model reaches 0.29; after subtracting
the GDPa1-estimated subtype expectation, the best selected residual model reaches
only 0.10. Regenerate with `python scripts/run_tm_structure_tier_audit.py` and
`python scripts/plot_tm2_subtype_residual.py`.*

**Against the field's own methods.** On the same held-out HIC split, our physics
model (0.49) ranks 4th of 15 official competition baselines — level with the MOE
descriptor baseline, above every sequence-based ML method (ESM-2 0.40, AbLang2
0.36, DeepSP 0.40), and above the field-baseline median (0.38). The only methods
clearly above it use 3D structure (SaProt 0.54, Aggrescan3D 0.54), consistent
with structure being the missing ingredient for HIC; the competition winner
reached 0.71. So where the other assays collapse from internal CV to held-out,
that reflects honest evaluation, not weak features: on the assay where signal
exists, these crude descriptors are competitive with the field's deep-learning
models out of distribution.

![Held-out HIC Spearman for our physics model against the competition's official baselines; our model ranks 4th of 15, above every sequence-based ML method and the field-baseline median, below only two structure-based methods and the competition winner.](results/hic_vs_field_baselines.png)

*Our physics model versus the competition's 14 official baselines on the blinded GDPa3 held-out HIC split (data: `results/external_hic_vs_field_baselines.csv`; regenerate with `python scripts/plot_hic_vs_field_baselines.py`). The competition winner's 0.71 used predictions not released in this repository.*

## What the audit establishes

Under firewalled evaluation, simple physicochemical features land in the same
range as far heavier methods, because the heavier methods do not generalize
either: the PLM embeddings improve internal cross-validation but reduce external
performance. HIC is the cleanest leakage-robust, beyond-memorization signal.
PR_CHO is partial, AC-SINS/Titer remain weak, and Tm2 is best treated as a mixed
endpoint with a large framework/subtype component. This independently
reproduces, in this pipeline, the conclusion the AbDev competition organizers
published from their own 113-team results — that cross-validation overstates
performance and out-of-distribution generalization is limited.

## Status / scope of implementation

Implemented and load-bearing for the result above:

- Physicochemical per-chain/per-CDR features.
- Sequence-cluster grouped cross-validation.
- External GDPa3 firewall.
- Falsification controls: composition-matched scramble null, label-shuffle null,
  sequence-identity kNN baseline.
- Tm2 post-hoc tier audit: framework/subtype baseline, structure-rich stability
  features, defect proxies, and subtype-residual evaluation.

Implemented but NOT load-bearing:

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
files committed are aggregate summary statistics under `results/` — per-assay
metric tables with no per-antibody rows, sequences, or raw assay values.

## Reproduction

```bash
conda env create -f environment.yml
conda activate anti
python scripts/download_gdpa.py        # GDPa1 from Hugging Face, after `hf auth login`
python -m pytest                       # 59 tests, no data or network required
```

### Reproduce the headline numbers

With GDPa1 and GDPa3 obtained from source (GDPa3 is gated/external), each headline
result is regenerated by one entry point; outputs are the aggregate tables under
`results/`.

| Headline | Command | Output |
|----------|---------|--------|
| HIC external ≈ 0.49; PLMs help internal, hurt external | `python scripts/run_v11_descriptor_plm_stack_audit.py` | `results/descriptor_plm_internal_vs_external.csv`, `results/external_vs_field_ceiling.csv` |
| Internal→external collapse (PR_CHO 0.52→0.28, etc.) | `python scripts/run_external_triage_audit.py` | `results/internal_vs_external_triage_comparison.csv` |
| Tm2 framework/subtype signal and subtype-residual collapse | `python scripts/run_tm_structure_tier_audit.py` | `results/tm2_structure_tier_summary.csv` |
| HIC physics 0.49 beats identity-kNN 0.33 | `python scripts/knn_baseline_hic.py` | prints the comparison |
| Topology does not beat a composition-matched null | `antinetwork.topology_falsification.run_topology_falsification_gate` (exercised by `tests/test_topology_falsification.py`) | `results/topology_gate_verdicts.csv` |
