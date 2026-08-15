---
name: intended-use-statement
description: Formal intended-use statement for TinySepsis, in the format FDA reviewers expect for a Clinical Decision Support software function assessment (21st Century Cures Act Section 3060 / 21 U.S.C. 360j(o)(1)(E)).
---

# Intended Use Statement: TinySepsis

## Statement of intended use

TinySepsis is intended to provide healthcare professionals caring for adult patients in an intensive care unit with a supplementary, calibrated estimate of the probability that a patient will meet Sepsis-3 clinical criteria within the next 4, 6, or 8 hours, computed from vital signs and laboratory values already present in the patient's electronic health record. The output is intended to be reviewed alongside the underlying clinical data it is based on (surfaced via the "top contributing factors" field) and is intended to support, not replace, the independent clinical judgment of the treating healthcare professional.

TinySepsis is **not** intended to:
- Diagnose sepsis or any other condition.
- Be used as the sole basis for a clinical decision (initiating antibiotics, ordering additional tests, escalating care, or any other action).
- Be used in populations outside adult ICU patients (pediatric, obstetric, outpatient, emergency-department-only, or post-acute settings are out of scope and untested).
- Operate without a healthcare professional able to independently review the basis for the score.

## Why this framing matters: the Cures Act non-device CDS pathway

Under the 21st Century Cures Act (Section 3060, amending 21 U.S.C. § 360j(o)(1)(E)) and FDA's 2022/2026 Clinical Decision Support Software guidance, software that performs all four of the following functions is excluded from the definition of a "device" and therefore from FDA premarket review:

1. Does not acquire, process, or analyze a medical image, an in vitro diagnostic signal, or a pattern/signal from a signal-acquisition system.
2. Displays, analyzes, or prints medical information about a patient or other medical information.
3. Is intended for the purpose of supporting or providing recommendations to a healthcare professional about prevention, diagnosis, or treatment.
4. Is intended for the purpose of enabling the healthcare professional to independently review the basis for the recommendations, so that the professional does not rely primarily on the software's output to make a clinical decision.

**TinySepsis, as scoped by this Intended Use Statement, is designed to satisfy criteria 1-4:** it processes only already-adjudicated numeric vitals/labs (criterion 1), displays a risk score and its basis (criterion 2), is explicitly framed as decision support for a healthcare professional (criterion 3), and surfaces the specific contributing vitals/labs so the reviewing clinician has an independent basis to agree or disagree (criterion 4, implemented via the `top_contributing_factors` field in both the demo API and the CDS Hooks integration).

**This is a design intent, not a legal determination.** Whether a specific deployment actually qualifies as non-device CDS depends on the final labeling, marketing claims, and workflow -- e.g., marketing this as "diagnoses sepsis" or removing the contributing-factors explanation would likely make it fail criterion 4 and pull it back into FDA's device jurisdiction, the same way the FDA's 2022 draft CDS guidance narrowed the exemption specifically for software making time-critical, high-acuity predictions (of which sepsis prediction is a canonical example the FDA has previously called out as likely still requiring premarket review even when nominally "CDS"). **A qualified regulatory consultant or attorney should confirm the exemption applies before any commercial deployment; this document is preparation for that conversation, not a substitute for it.**

## Alternative pathway: regulated SaMD

If the intended use is expanded to include a diagnostic claim, autonomous alerting without clinician review, or removal of the explanatory output, TinySepsis would need to pursue FDA marketing authorization as Software as a Medical Device (SaMD), most likely:

- **510(k)** if a predicate device is claimed (e.g., AlgoDx's NAVOY CDS, 510(k)-cleared 2024, is a plausible predicate for an ICU sepsis early-warning CDS tool), or
- **De Novo** if no adequate predicate exists and the device is low-to-moderate risk (the pathway Prenosis's Sepsis ImmunoScore used in April 2024, the first FDA-authorized AI sepsis diagnostic).

This pathway requires the elements documented in `regulatory/risk_management_plan.md`, `regulatory/predetermined_change_control_plan.md`, `regulatory/clinical_validation_protocol.md`, and a certified Quality Management System (ISO 13485, incorporated into FDA's QMSR as of February 2026) -- none of which are in place today. See `regulatory/gmlp_self_assessment.md` for a gap analysis against FDA/Health Canada/MHRA's Good Machine Learning Practice principles either way, since GMLP is good practice regardless of which pathway is pursued.

## User

Licensed healthcare professionals (physicians, advanced practice providers, nurses) caring for adult ICU patients, viewing the output within their existing clinical workflow (EHR-embedded via CDS Hooks, `src/tinysepsis/integration/cds_hooks_app.py`) or a standalone research interface (`src/tinysepsis/demo/app.py`).

## Explicitly out of scope

- Pediatric and neonatal patients.
- Obstetric patients.
- Non-ICU settings (ED triage, general ward, outpatient) -- the training data and label definition (pre-suspicion window, Sepsis-3 in an ICU context) do not transfer to these settings without separate validation.
- Any use where the clinician cannot review the underlying vitals/labs the score is based on.
- Any jurisdiction outside the one(s) in which appropriate regulatory clearance/exemption has been confirmed.
