---
name: clinical-validation-protocol
description: Draft prospective, multi-center clinical validation study protocol for TinySepsis, structured for submission to an Institutional Review Board (IRB) and, eventually, an FDA marketing application -- modeled on the design used by Prenosis's Sepsis ImmunoScore (the first FDA-authorized AI/ML sepsis diagnostic, De Novo, April 2024).
---

# Clinical Validation Protocol (Draft): TinySepsis Early Warning System

**Status: draft protocol, not yet reviewed by an IRB, biostatistician, or clinical collaborator beyond the project's own author.** This document is written so that the collaborating physician has a concrete starting point to bring to a hospital research office or IRB, not as a finished, submittable protocol. Every section marked `[NEEDS PI INPUT]` requires a clinical Principal Investigator's judgment, not an engineering decision.

## 1. Background and rationale

TinySepsis (`paper/main.pdf`) is a compact recurrent neural network trained and retrospectively evaluated on the open-access PhysioNet/CinC Challenge 2019 dataset. Its central finding -- that the model's discrimination degrades less than tabular baselines when moving between two hospitals in that dataset -- is retrospective, single-hospital-pair evidence (see the paper's own Limitations section). It does **not** establish that the model works prospectively, in a live clinical workflow, at a new institution, or that using it changes clinician behavior or patient outcomes. This protocol exists to generate that evidence, following the same general study architecture (derivation / internal validation / external validation) that FDA accepted for the first FDA-authorized AI sepsis diagnostic, Prenosis's Sepsis ImmunoScore (De Novo authorization, April 2024): a prospective, observational, multi-center design with derivation, internal-validation, and external-validation cohorts from different hospitals.

## 2. Objectives

**Primary objective:** Prospectively evaluate TinySepsis's discrimination (AUROC) for 6-hour sepsis onset in a real-world ICU population, at a site not used to train, validate, or calibrate the model.

**Secondary objectives:**
- Calibration (ECE, Brier score) of the deployed model's output against real-world outcomes.
- Realized false-alarm rate against the pre-specified conformal target, in a live population (directly testing whether the paper's own finding -- that this rate roughly doubles under cross-institution shift -- replicates prospectively).
- Clinical workflow metrics: time from alert to clinician acknowledgment, alert dismissal rate, and (where feasible without altering care) time from alert to first sepsis-directed intervention, compared against standard-of-care recognition time.
- Subgroup performance (age, sex, and, at sites where available, race/ethnicity and insurance status -- an explicit improvement over the retrospective study's data limitations).

**Explicitly not an objective of this phase:** demonstrating a mortality or length-of-stay benefit. That requires an interventional (not observational) design and is out of scope until Phase 1 (below) succeeds.

## 3. Study design

**Phase 0 -- Silent/shadow mode (recommended starting phase).** TinySepsis runs in real time against live EHR data (via the CDS Hooks integration, `src/tinysepsis/integration/cds_hooks_app.py`) but its output is **not shown to clinicians** and does not influence care in any way. Scores are logged and compared, after the fact, against actual clinical course and Sepsis-3 adjudication. This is the lowest-risk possible design (no patient is exposed to any model-driven decision) and is the appropriate first step before Phase 1.

**Phase 1 -- Visible, non-interventional CDS.** Contingent on Phase 0 meeting pre-specified performance thresholds `[NEEDS PI INPUT: exact thresholds]`, the alert becomes visible to clinicians (via the CDS Hooks card) as decision support, with explicit instruction that it does not mandate any action. Outcomes are compared against a concurrent or historical control period/unit, observationally -- clinicians remain free to act on their own judgment. This phase produces the human-AI-team evidence GMLP Principle 7 calls for (`gmlp_self_assessment.md`), which Phase 0 cannot.

**Phase 2 -- Interventional trial (future, out of scope for this protocol).** Only after Phase 1, and only with a dedicated protocol, would a randomized or stepped-wedge design testing an actual care-pathway change (e.g., automatic sepsis-bundle order-set trigger) be appropriate.

This document specifies **Phase 0** in detail and sketches Phase 1's design for planning purposes.

## 4. Study population

**Derivation cohort:** already complete -- PhysioNet Challenge 2019, Hospital A (`paper/main.pdf` Section 5).

**Internal validation cohort:** already complete -- PhysioNet Challenge 2019, Hospital B, used as a cross-institution (not fully independent-cohort) check (`paper/main.pdf` Section 8).

**External validation cohort (this protocol's focus):** adult patients ($\ge$18 years `[NEEDS PI INPUT: confirm]`) admitted to the ICU at one or more collaborating hospitals **not** part of the PhysioNet Challenge 2019 collection, over a defined enrollment window.

**Inclusion criteria (draft):**
- Admitted to a participating ICU during the enrollment window.
- Sufficient EHR data density to compute the model's inputs (at minimum: some vital signs recorded within the observation window) `[NEEDS PI INPUT: minimum data-completeness threshold]`.

**Exclusion criteria (draft):**
- Meets Sepsis-3 criteria (or has already received sepsis-directed treatment) at or before ICU admission -- the model is an *early warning* tool; patients already recognized as septic at admission are not the target population, consistent with the pre-suspicion-window design in `paper/main.pdf` Section 3.
- Comfort-care-only / hospice status at admission `[NEEDS PI INPUT: confirm this exclusion and its justification]`.
- Age <18, pregnant, or otherwise outside the validated intended-use population (`intended_use_statement.md`).

## 5. Sites

At least one external site, ideally 2+ for the "external validation" claim to carry real weight (a single new site is still only one hospital pair, same limitation the retrospective study has -- see `risk_management_plan.md` H4/H7 and the paper's own Limitations section on this exact point). `[NEEDS PI INPUT: candidate sites, IRB-of-record arrangement if multi-site]`.

## 6. Sample size

**Rough guidance, not a substitute for a formal power calculation** `[NEEDS BIOSTATISTICIAN INPUT]`: to detect an AUROC of 0.65 as significantly different from a null of 0.50 with 80% power and alpha=0.05, using the method of Hanley & McNeil (1982) for AUROC standard error under an expected sepsis prevalence of ~1-3% (matching PhysioNet's own prevalence), on the order of several hundred to a few thousand septic-patient encounters would typically be needed depending on the exact prevalence and effect size at the new site -- meaning total ICU-patient enrollment likely in the tens of thousands of patient-days given how rare the event is. **This must be replaced with a formal calculation before submission**; it is included here only so the scale of the undertaking (a real, multi-month-to-multi-year prospective study, not a weekend project) is visible upfront.

## 7. Data collection

Via the CDS Hooks integration against the site's live FHIR-compliant EHR (Epic, Oracle Health/Cerner, or other SMART-on-FHIR-compliant system). Requires, before any patient data flows: (a) a signed Business Associate Agreement (BAA) or equivalent, (b) the site's own LOINC mapping validated by their clinical informatics team against `src/tinysepsis/integration/fhir_mapping.py` (closing the H5 risk in `risk_management_plan.md`), (c) IRB approval, (d) a data use/retention agreement consistent with the site's policies and applicable law (HIPAA in the US).

## 8. Primary and secondary endpoints

| Endpoint | Type | Definition |
|---|---|---|
| AUROC for 6h sepsis onset | Primary | Model score vs. adjudicated Sepsis-3 onset, at the pre-suspicion censoring point defined in `paper/main.pdf` Section 3 |
| Expected Calibration Error | Secondary | Same definition as `src/tinysepsis/eval/metrics.py::expected_calibration_error`, computed fresh on this cohort |
| Realized false-alarm rate vs. target $\alpha$ | Secondary | Same definition as `src/tinysepsis/eval/conformal.py`, using the threshold carried over from the retrospective study (not re-fit on this cohort, to test true prospective transfer) |
| Subgroup AUROC | Secondary | By age band, sex, and any additional demographics available at the site |
| Alert acknowledgment / dismissal rate (Phase 1 only) | Secondary | From CDS Hooks card interaction logging |

## 9. Statistical analysis plan

`[NEEDS BIOSTATISTICIAN INPUT]`. At minimum: point estimates with 95% CIs (bootstrap, patient-level clustering respected) for all endpoints above; pre-specified comparison against the retrospective Hospital-B numbers as a benchmark for "did prospective performance meet, exceed, or fall short of the retrospective cross-institution estimate"; a pre-specified statistical stopping/success threshold agreed with the PI and site before data collection begins, to avoid post-hoc threshold selection.

## 10. Ethical considerations

`[NEEDS PI / IRB INPUT]`. Phase 0 (shadow mode, no care impact) is likely eligible for a waiver of individual informed consent as minimal-risk observational research, but this determination is the IRB's to make, not this document's. Phase 1 (visible alerts) raises different considerations (clinician and potentially patient notification) that should be discussed with the IRB separately. Data minimization, de-identification for any secondary analysis, and a clear data retention/destruction policy should be specified in the IRB submission.

## 11. Relationship to FDA pathway

A successful Phase 0/Phase 1 at 1-2 external sites, combined with the retrospective results already in hand, would materially strengthen (but likely not alone complete) the clinical-validation component of either a 510(k) or De Novo submission (`intended_use_statement.md`). It would not, on its own, satisfy the Quality Management System (ISO 13485/QMSR) or full Good Machine Learning Practice requirements (`gmlp_self_assessment.md`), which are organizational, not study-design, gaps.

## 12. Next steps for the collaborating physician

1. Review and fill in every `[NEEDS PI INPUT]` marker above.
2. Identify a candidate external site (ideally one where the physician has an existing relationship) and a local co-investigator with IRB access.
3. Engage a biostatistician for Section 6 and 9 before drafting an actual IRB submission.
4. Treat Phase 0 (shadow mode) as the realistic near-term goal -- it requires no clinical workflow change and no patient risk, which is by far the fastest path to real, prospective, external evidence.
