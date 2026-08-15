---
name: model-card
description: Model card for TinySepsis, following the Mitchell et al. (2019) model card format adapted for a clinical risk-prediction model, aligned with FDA/Health Canada/MHRA Good Machine Learning Practice (GMLP) Principle 9 ("users are provided clear, essential information").
---

# Model Card: TinySepsis

**Version:** research prototype, not released for clinical use.
**Date:** see `results/checkpoints/tinysepsis_meta.json` for the exact training run this card describes; regenerate this card whenever the model is retrained (Section "Versioning" below).

## 1. Model details

| | |
|---|---|
| Developer | Elian Alfonso López Preciado, independent researcher |
| Model type | 2-layer GRU (gated recurrent unit), <200K parameters |
| Input | Last 24h of 34 vital-sign/lab channels (value, missingness mask, time-since-last-measurement, first difference) + age, sex |
| Output | Calibrated probability of sepsis onset within 6 hours; secondary 4h/8h horizons |
| Training data | PhysioNet/Computing in Cardiology Challenge 2019, Hospital A only (14,235 patients) |
| License | MIT (code); dataset under PhysioNet's own open-data license, not redistributed |
| Intended use | See `regulatory/intended_use_statement.md` |
| Not intended for | Any autonomous diagnostic or treatment decision; pediatric, obstetric, or non-ICU populations (untested); any deployment without a locally-run prospective validation (Section 3) |

## 2. Intended use (summary; full statement in `intended_use_statement.md`)

TinySepsis is a **research prototype clinical decision-support tool**. It estimates a calibrated probability that an adult ICU patient will meet Sepsis-3 criteria within the next 4, 6, or 8 hours, using vital signs and laboratory values already collected as part of routine care. It is designed to be reviewed by a clinician alongside the underlying vitals/labs (via the top-contributing-factors output), not acted on as a standalone signal. It is **not** a diagnostic device and has not been authorized by the FDA or any other regulator.

## 3. Training and evaluation data

- **Source:** PhysioNet/CinC Challenge 2019 (Reyna et al., 2019), open access, no credentialing required.
- **Cohort:** Two hospital systems ("Hospital A", "Hospital B"), 40,336 adult ICU admissions total.
- **Training split:** Hospital A only, patient-level 70/15/15 train/val/test (14,235/3,049/3,052 patients).
- **Cross-institution evaluation:** Hospital B (20,000 patients), never used in training, validation, or calibration -- but part of the same original data-collection effort as Hospital A, not an independent cohort (see `regulatory/clinical_validation_protocol.md` Section 2 for why this matters and what a real external validation would require).
- **Demographics available:** age, sex only. No race, ethnicity, insurance status, or facility-type fields -- Section 6 (Fairness) is limited by this.
- **Known distribution shift documented:** sepsis prevalence differs between the two hospitals (1.37% vs. 0.90% of patient-hours), and model performance is reported separately for each (paper Tables 1-2).

## 4. Performance

Full numbers with confidence characterization (multi-seed replication, statistical tests) are in `paper/main.pdf` (Sections 8-9) and machine-readable in `results/tables/main_results.csv`, `results/tables/external_results.csv`, `results/tables/multiseed_summary.csv`, `results/tables/multiseed_stats.json`. This card intentionally does not duplicate point estimates here -- they change on every retrain, and duplicated numbers rot. What does not change across retrains:

- Both internal (Hospital A) and cross-institution (Hospital B) discrimination, calibration, and clinical-utility metrics are reported side by side; the Hospital-B numbers are the ones that matter for judging real-world reliability.
- Performance is reported after post-hoc calibration (temperature scaling + isotonic regression) as well as raw, because raw sigmoid outputs from this model are **not** meaningful probabilities (see calibration table).
- A false-alarm-rate budget (conformal risk control, target 10%) is reported with its *realized* rate on both the internal and cross-institution splits, not just the target.
- Subgroup performance (age band, sex) is reported in `results/tables/subgroup_analysis.csv`.

## 5. Limitations (see also `paper/main.pdf` Section "Limitations" for the full list)

1. **Retrospective only.** No prospective validation has been performed. Retrospective performance does not establish that using this tool changes clinician behavior or patient outcomes.
2. **Single hospital pair.** All cross-institution results come from one pair of hospitals within one data-collection effort. Generalization to a genuinely independent institution (different EHR vendor, different population, different lab assay methods) is unverified.
3. **Not tested outside the ICU**, not tested in pediatric or obstetric populations, not tested against non-Sepsis-3 sepsis definitions.
4. **LOINC mapping for EHR integration is unverified** for several lab channels (see `src/tinysepsis/integration/fhir_mapping.py`, codes flagged `needs_verification`) -- using this integration against real FHIR data before a clinical-informatics review could silently feed the wrong variable into the model.
5. **A known architecture bug was found and fixed** during development (the model was, for a period, effectively ignoring recent observations for any patient with fewer than 24 hours of ICU data -- see git history / `tests/test_model.py::test_prediction_reads_the_last_position_not_a_length_based_gather`). This is disclosed here as a reminder that a single retrospective AUROC number can hide implementation errors; independent re-implementation or code review is recommended before any high-stakes use.

## 6. Fairness / subgroup performance

Reported by age band and sex (`results/tables/subgroup_analysis.csv`) as a **risk-surfacing exercise**, not a certification of fairness. No race/ethnicity data was available in the training set to audit for the disparities documented in other clinical-AI systems (e.g., Obermeyer et al., *Science*, 2019). A production deployment must re-run this analysis on the receiving institution's own, more complete demographic data before go-live.

## 7. Ethical and regulatory considerations

Not a validated medical device. Has not undergone FDA clearance, CE marking, or equivalent review. See `regulatory/intended_use_statement.md`, `regulatory/risk_management_plan.md`, and `paper/main.pdf` Section "Ethical Considerations" for the full treatment, including why this may or may not require FDA premarket review depending on how it is deployed and marketed (21st Century Cures Act non-device CDS criteria).

## 8. Versioning

Regenerate this card's Section 4 pointer whenever `scripts/train_model.py` produces a new `tinysepsis_best.pt`. Under a future Predetermined Change Control Plan (`regulatory/predetermined_change_control_plan.md`), retraining on new data or recalibrating would be a pre-specified, bounded change; changing the architecture, input features, or intended use would not, and would require a new model card and a new validation cycle.
