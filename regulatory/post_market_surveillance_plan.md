---
name: post-market-surveillance-plan
description: Post-market performance monitoring plan for TinySepsis, aligned with GMLP Principle 9 (real-world performance monitoring) and the kind of monitoring plan an FDA PCCP submission must describe.
---

# Post-Market Surveillance / Performance Monitoring Plan: TinySepsis

**Status: plan only.** There is no live deployment to monitor. This document specifies what *would* be tracked if TinySepsis were deployed at a real site, so that (a) a future PCCP submission has this ready, and (b) anyone piloting this at a hospital knows what instrumentation to stand up before go-live, not after.

## 1. Why this matters here specifically

This project's own headline finding is that model performance degrades when moving to a new institution (paper Sections 8-9), and that the *degree* of that degradation varies by model and axis (AUROC ranking vs. calibrated alarm behavior -- see the conformal-threshold-utility finding in paper Section 9.3-9.4). A model that behaves this differently between two hospitals in the *same original dataset* should not be assumed stable over *time* at a single site either (new equipment, new case mix, seasonal effects, EHR upgrades). Post-market monitoring is not a formality for this specific tool -- it is answering the same question the paper asks, but continuously, instead of once at deployment.

## 2. Metrics tracked, and why

| Metric | Cadence | Trigger for review |
|---|---|---|
| AUROC / AUPRC (rolling window, e.g. trailing 90 days) | Monthly | Drop of >0.03 AUROC vs. deployment-time baseline |
| Expected Calibration Error (rolling) | Monthly | ECE increase such that the "isotonic-calibrated" claim no longer holds (e.g. >0.02, vs. ~0.001-0.007 measured pre-deployment) |
| Realized false-alarm rate vs. the conformal target ($\alpha$) | Weekly | Realized rate exceeds target by >5 percentage points (this project's own cross-institution result showed realized FPR roughly doubling the target -- treat that as the reference magnitude for "this can happen") |
| Alarms per 1000 patient-hours | Weekly | Sustained rise, proxy for creeping alarm fatigue |
| Sepsis prevalence in the deployed population | Monthly | Meaningful shift (as small as the 1.37%->0.90% shift between the two hospitals in this project's own data was enough to change results) triggers recalibration review |
| Subgroup AUROC (age band, sex, and any additional demographics available at the deployment site) | Quarterly | Any subgroup falling >0.05 AUROC below the overall population |
| Clinician override / dismissal rate on alerts | Monthly | Rising dismissal rate is an early behavioral signal of alarm fatigue, ahead of any accuracy metric moving |
| Time-to-recognition (model alert timestamp vs. clinical Sepsis-3 recognition timestamp) | Quarterly, requires chart review sample | The clinical outcome the whole system exists to move; not derivable from the model's own logs alone |

## 3. Data pipeline required (not built)

Monitoring requires: (a) logging every score served with a de-identified patient/encounter key, (b) a downstream join against actual Sepsis-3 outcomes (from the EHR, with a delay for outcome ascertainment), (c) a dashboard or scheduled report, (d) an alerting mechanism when a trigger in Section 2 fires. None of this exists yet; it is infrastructure a real pilot site would need to build, ideally before go-live, using the CDS Hooks service's card `uuid` as the join key back to the originating recommendation.

## 4. Recalibration protocol

If a trigger fires: (1) freeze the alarm threshold at its last-known-good conformal value rather than letting it silently drift; (2) pull the trailing window of scored cases with adjudicated outcomes; (3) re-run `scripts/calibrate_and_conformal.py`-equivalent logic on that local data; (4) if recalibration alone does not resolve the trigger, treat it as a signal that retraining (not just recalibration) may be needed, which is a materially bigger change (see `predetermined_change_control_plan.md` for which of these two are "pre-authorized, bounded changes" vs. which require a fresh validation cycle).

## 5. Incident reporting

Any suspected false negative that contributed to a delayed sepsis recognition, or any false positive that led to an unnecessary intervention, should be logged with enough detail to feed back into the risk file (`risk_management_plan.md`) and, if TinySepsis is ever regulated as a device, would need to be assessed against FDA Medical Device Reporting (MDR) obligations (21 CFR Part 803) by qualified regulatory staff -- not a determination this document makes.
