# results/

Aggregate summary statistics only. No per-antibody rows, no sequences, no raw
assay values — these are computed statistics, not the underlying GDPa1/GDPa3
benchmark data, which is not redistributed here (see the top-level README).

| File | Granularity | What it shows |
|------|-------------|---------------|
| `internal_vs_external_triage_comparison.csv` | per assay | Grouped-CV vs external GDPa3 Spearman/AUROC; the internal→external collapse |
| `external_vs_field_ceiling.csv` | per assay | External GDPa3 Spearman vs the published competition field ceiling |
| `descriptor_plm_internal_vs_external.csv` | per assay × feature set | Physics vs PLM stacks, internal vs external (PLMs help internal, hurt external) |
| `tm2_structure_tier_summary.csv` | per Tm2 audit row | Framework/subtype, structure-rich, defect, and subtype-residual Tm2 aggregate results |
| `topology_gate_verdicts.csv` | per assay | Real topology vs composition-matched scramble and label-shuffle nulls |
| `positive_control_summary.csv` | per control | Falsification detector calibration on synthetic labels |
| `scramble_diagnostics_summary.csv` | per diagnostic | Quality of the composition-matched scramble null |
