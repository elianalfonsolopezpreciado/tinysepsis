---
name: predetermined-change-control-plan
description: Draft Predetermined Change Control Plan (PCCP) for TinySepsis, following the structure of FDA's final guidance "Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions" (December 4, 2024).
---

# Predetermined Change Control Plan (Draft): TinySepsis

**Status: draft, illustrative.** A real PCCP is submitted *with* an initial marketing application and reviewed by FDA as part of that submission -- it cannot exist meaningfully before there is an authorized device to attach it to. This document exists so that (a) the modification process is designed with FDA's PCCP framework in mind from the start rather than retrofitted later, and (b) it doubles as an internal change-management policy even under the non-device CDS pathway, where no submission is required but disciplined change control is still good engineering practice.

Per the December 2024 final guidance, a PCCP has three required components: (1) a description of the planned modifications, (2) the methodology to develop, validate, and implement each modification, (3) an impact assessment. Structured below accordingly.

## 1. Planned modifications (pre-specified, in scope for this PCCP)

| Modification type | Description | In scope? |
|---|---|---|
| Periodic retraining on newly accumulated data from the *same* deployment site(s) already validated | Refresh model weights using the same architecture, feature set, and label definition, on an expanded or rolled-forward training window | Yes |
| Recalibration (temperature scaling / isotonic regression) | Re-fit the calibration layer only, without changing the underlying GRU weights, in response to a post-market-surveillance trigger (`post_market_surveillance_plan.md` Section 4) | Yes |
| Conformal threshold re-fit | Re-fit the alarm threshold $\tau$ for a new $\alpha$ target or a new deployment site's validation split | Yes |
| Adding a new deployment site (after that site's own local validation cohort is run per `clinical_validation_protocol.md`) | Extending intended use to a new institution using the existing model, contingent on that site's local validation meeting pre-specified acceptance criteria (Section 3) | Yes, contingent |

## 2. Explicitly OUT of scope for this PCCP (require a new submission / full revalidation)

- Changing the model architecture (e.g., GRU -> Transformer) or the input feature set (adding/removing clinical variables).
- Changing the prediction horizon (4h/6h/8h) used as the primary claim, or the underlying sepsis-onset label definition.
- Expanding intended use to a new population (pediatric, non-ICU, etc.).
- Any change to the false-alarm-budget philosophy itself (e.g., moving from conformal risk control to an unvalidated alternative).
- Retraining on data from a population meaningfully different from the validated population (e.g., a different country's ICU case mix) without treating it as a new-site validation first.

## 3. Methodology and acceptance criteria for in-scope modifications

For **periodic retraining** and **recalibration**:
1. Run the modified model against the site's held-out validation cohort (never used in the retrain/recalibration itself).
2. Compare against the pre-specified acceptance envelope: AUROC must not drop more than 0.03 vs. the currently-deployed model's last-validated AUROC at that site; ECE must remain below 0.02; realized false-alarm rate must remain within 5 percentage points of the $\alpha$ target.
3. If any criterion fails, the modification is rejected and the currently-deployed model/calibration remains active; the failure is logged as a post-market-surveillance trigger event.
4. If all criteria pass, the modification is deployed and the model card (`model_card.md`) and post-market monitoring baseline are updated to reflect it.

For **new-site onboarding**:
1. Run the existing (unmodified) model against the new site's local validation cohort, collected per `clinical_validation_protocol.md`.
2. The same acceptance envelope as above applies, but benchmarked against the *original* validated performance (Hospital A/B in this project), not a moving target.
3. If the new site's performance falls outside the envelope, this PCCP does not authorize deployment there; it requires a site-specific validation and, if the gap is large, potentially triggers the "architecture change" path (out of scope, Section 2) if the underlying reason is a genuine feature-set or population mismatch this model cannot handle.

## 4. Impact assessment

- **Benefit of allowing pre-specified changes without a new full review cycle:** keeps the model from silently going stale (the single biggest risk this project's own cross-institution results demonstrate) without requiring a multi-month regulatory cycle for routine maintenance.
- **Risk of allowing pre-specified changes:** a poorly-specified acceptance envelope could let a genuinely degraded model continue serving alerts. The envelope in Section 3 is a first draft, deliberately conservative (small deltas), and should be tightened or loosened only with clinical stakeholder input, not a unilateral engineering decision.
- **Traceability:** every modification under this PCCP must be logged (date, type, validation results, decision) in a change log alongside the model card, so that "what changed and when" is always reconstructable -- this is also a GMLP Principle 9 requirement (users informed of device modifications and updates).

## 5. Relationship to the current state of the project

None of the "in scope" modifications above have an operational validation pipeline running against live data yet -- `scripts/train_model.py`, `scripts/calibrate_and_conformal.py`, and `scripts/run_multiseed.py` implement the *mechanics* (retrain, recalibrate, validate) but currently only against the fixed research dataset, not a live feed with a rolling acceptance-criteria check. Building that rolling check is the concrete engineering task this PCCP implies as a prerequisite for actually using it.
