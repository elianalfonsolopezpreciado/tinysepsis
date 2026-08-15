---
name: risk-management-plan
description: Risk management plan for TinySepsis following the ISO 14971 (medical device risk management) structure -- hazard identification, risk analysis, risk control -- adapted for an AI/ML clinical decision-support tool.
---

# Risk Management Plan: TinySepsis

Structured per ISO 14971 (Application of risk management to medical devices), the standard FDA expects to see referenced in a 510(k)/De Novo submission's risk analysis, regardless of whether TinySepsis is ultimately regulated as a device or deployed under the non-device CDS exemption (see `intended_use_statement.md`) -- the hazard analysis below is good practice either way, since the clinical consequences of getting a sepsis alert wrong don't depend on which FDA pathway applies.

This is a **draft**, produced by the development team without an independent clinical safety reviewer. A real risk management file requires review by a qualified clinician not involved in model development and, for a regulated submission, a Class of Safety review by a certified quality/regulatory professional.

## 1. Scope

Covers TinySepsis as scoped in `intended_use_statement.md`: an ICU sepsis early-warning score, reviewed by a healthcare professional, not used autonomously.

## 2. Hazard identification and risk analysis

| # | Hazard | Cause | Harm | Severity | Probability (pre-mitigation) | Risk (pre-mitigation) |
|---|---|---|---|---|---|---|
| H1 | False negative (missed sepsis) | Model fails to flag a genuinely deteriorating patient | Delayed antibiotics/fluids, increased morbidity/mortality | Critical | Non-trivial -- Table "main_results" test-split sensitivity at fixed specificity is well below 100% by design (no model is) | High |
| H2 | False positive / alarm fatigue | Model over-alarms, clinicians habituate and start ignoring alerts (incl. true ones) | Delayed recognition of a *future* true positive due to habituation; clinician time burden | Serious | Measured directly: alarms-per-1000-patient-hours (paper Table 1-2) | High if alarm rate not actively managed |
| H3 | Miscalibrated probability | Raw model output is not a true probability (paper: raw ECE ~0.35-0.40) | Clinician misjudges actual risk if raw score is shown instead of calibrated score | Serious | Certain if raw scores are ever surfaced instead of calibrated ones | High -- **mitigated by design**, see control C3 |
| H4 | Silent performance degradation under distribution shift | Model deployed at an institution whose case-mix/equipment/documentation differs from training data (exactly what Section 8-9 of the paper measures for Hospital A->B) | Any of H1-H3, worse than validation numbers suggest, without warning | Critical | Directly measured: cross-institution AUROC drop is non-zero for every model tested, including TinySepsis | High |
| H5 | LOINC/terminology mapping error in EHR integration | Wrong FHIR Observation mapped to wrong model feature (`fhir_mapping.py` "needs_verification" codes) | Model silently scores on wrong/garbled input | Critical (silent) | Not yet formally verified -- 8 of 34 codes flagged unverified as of this writing | High until terminology review complete |
| H6 | Missing-data misinterpretation | A feature that was never measured (e.g., lactate not drawn) is treated identically to a normal value, or the missingness signal itself is over-weighted | Under- or over-estimation of risk for patients with sparse monitoring | Moderate | Explicitly modeled (mask + time-since-last-measurement channels) but ablation shows this materially changes performance (paper Table "ablations"), i.e. the model *is* sensitive to it, which cuts both ways | Moderate |
| H7 | Subgroup performance disparity | Model trained on a population that may not reflect deployment population; only age/sex available for audit | Systematically worse detection for an under-represented subgroup | Serious | Partially measured (age band, sex); race/ethnicity/insurance not measurable in current data | Moderate-High, unknown magnitude |
| H8 | Automation bias / over-reliance | Clinician defers to the score instead of independent assessment, contrary to intended use | Missed clinical signs not captured by the model's 34 input channels | Serious | Well-documented general phenomenon in clinical-AI human-factors literature; not separately measured here | Moderate |
| H9 | Cybersecurity: model/API compromise or data interception | CDS Hooks service exposed without TLS/auth, or ONNX artifact tampered with | Wrong scores served at scale; PHI interception in transit | Critical if PHI involved | Not yet addressed -- reference implementation has no auth (see `cds_hooks_app.py` docstring) | High until deployment hardening (C9) |
| H10 | Post-market drift undetected | Real-world case mix shifts over time (new patient population, new equipment, seasonal illness patterns) without anyone monitoring for it | Same as H4 but emerging gradually, post-deployment | Critical | Not yet instrumented -- see `post_market_surveillance_plan.md` | High until C10 in place |

## 3. Risk controls

| Hazard | Control | Status |
|---|---|---|
| H1, H2 | Conformal risk control gives an explicit, auditable false-alarm-rate target (not a black-box threshold); operating point is tunable per deployment site rather than fixed | Implemented (`scripts/calibrate_and_conformal.py`); **not yet re-fit per real deployment site**, only per this dataset's own splits |
| H3 | Isotonic + temperature calibration mandatory before any score is surfaced; raw sigmoid output never exposed via the demo API or CDS Hooks card | Implemented |
| H4, H10 | Explicit cross-institution evaluation built into the validation methodology from the start, not an afterthought; documented performance-monitoring plan for post-deployment drift | Cross-institution eval implemented; post-market monitoring plan is a *plan*, not yet running against real data (C10 below) |
| H5 | LOINC mapping table with explicit per-code confidence flags (`verified`/`standard`/`needs_verification`); CDS Hooks card includes a dev-note when unverified codes are in play | Implemented as a *flag*, not a fix -- clinical-informatics review of the 8 unverified codes is an open action item |
| H6 | Explicit mask + time-since-last-measurement channels rather than naive imputation; ablation study quantifies their contribution | Implemented and measured |
| H7 | Subgroup reporting by age/sex as standard output, not optional | Implemented; race/ethnicity/insurance auditing is an open gap requiring richer data |
| H8 | Intended Use Statement explicitly requires independent clinician review; CDS Hooks card always includes the underlying contributing factors, never a bare score; card language explicitly says "do not act on this score alone" | Implemented at the labeling/UX level; true mitigation requires clinician training, outside this project's scope |
| H9 | Reference implementation clearly marked "not for clinical use" / no auth; production deployment requires TLS, OAuth2 (SMART on FHIR's standard auth flow), and a signed/verified model artifact before go-live | **Open** -- explicitly out of scope for the current reference implementation, called out in `cds_hooks_app.py`'s docstring |
| H10 | Performance-monitoring plan defined (`post_market_surveillance_plan.md`) specifying what to track and trigger thresholds for re-validation | Plan exists; **no live monitoring instrumented**, since there is no live deployment |

## 4. Residual risk

After the controls above, the residual risk profile is dominated by H4/H5/H7/H9/H10 -- the ones marked "open" or "plan not yet running." **None of these residual risks should be interpreted as acceptable for an uncontrolled clinical deployment.** They define the gap between "research prototype with good engineering hygiene" (where this project is today) and "deployable clinical tool" (which requires the closed items above plus the prospective validation in `clinical_validation_protocol.md`).

## 5. Review and update triggers

This risk file must be re-reviewed whenever: (a) the model architecture, training data, or feature set changes; (b) a new deployment site is onboarded (H4/H5/H7 are all site-specific); (c) post-market monitoring (H10) flags a performance drop; (d) any incident (near-miss or actual harm) is reported.
