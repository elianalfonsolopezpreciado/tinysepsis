# Regulatory & clinical-deployment documentation

This directory exists because "the physician said improve it to FDA level" is not a
one-line task -- it decomposes into specific, mostly-independent artifacts, listed
here roughly in the order a real deployment conversation would need them.

| Document | Answers |
|---|---|
| [`intended_use_statement.md`](intended_use_statement.md) | What is this for, who uses it, and does it even need FDA clearance? (21st Century Cures Act non-device CDS exemption analysis) |
| [`model_card.md`](model_card.md) | What is the model, what data trained it, what are its limits? |
| [`risk_management_plan.md`](risk_management_plan.md) | What can go wrong (ISO 14971-style hazard analysis), and what mitigates it? |
| [`gmlp_self_assessment.md`](gmlp_self_assessment.md) | Honest scorecard against FDA/Health Canada/MHRA's 10 Good Machine Learning Practice principles |
| [`post_market_surveillance_plan.md`](post_market_surveillance_plan.md) | How would we know if it degrades after deployment? |
| [`predetermined_change_control_plan.md`](predetermined_change_control_plan.md) | Which future changes (retraining, recalibration, new sites) are pre-authorized vs. need a full new review? |
| [`clinical_validation_protocol.md`](clinical_validation_protocol.md) | The actual prospective study design needed to generate real (not retrospective) evidence -- the document to bring to an IRB or a hospital research office. |
| [`model_improvement_roadmap.md`](model_improvement_roadmap.md) | Given what other 2024-2025 sepsis-prediction models do (Transformers, Neural CDEs, uncertainty-aware attention), what's actually worth building next, prioritized by impact on *this project's* own evaluation axes, not benchmark chasing. |
| [`mexico_market_roadmap.md`](mexico_market_roadmap.md) | For Pear Labs specifically: COFEPRIS regulatory path, who to sell to first, pricing, and a phased timeline from today to a paid hospital deployment in Mexico. |

## What this is, and is not

These are **drafts written by the development team**, not certifications, not legal
advice, and not a substitute for the people who actually have to sign off on each of
these in a real regulatory process: a clinician co-investigator, a biostatistician, a
regulatory affairs professional, and eventually an IRB and/or FDA reviewer. Every
document says so explicitly and marks open items.

What they *do* accomplish: they turn "we should get this to FDA level someday" into
a concrete, reviewable set of documents that name exactly what is done, what is
drafted-but-unreviewed, and what hasn't been started -- so the next conversation with
the collaborating physician (or any hospital, or any regulatory consultant) starts
from a real gap analysis instead of a blank page.

## The honest one-paragraph summary

TinySepsis has good engineering hygiene (tested code, reproducible pipeline,
documented limitations, an honest multi-seed statistical replication of its central
claim) and now has a first draft of the regulatory scaffolding a real submission
would eventually need. It has **zero** prospective validation, **zero** independent
clinical review, **zero** certified quality system, and **zero** live deployment. The
gap between "well-engineered research prototype" and "sellable to a hospital" is not
mainly more code -- it is the clinical validation study (`clinical_validation_protocol.md`),
the multi-disciplinary review these documents keep flagging as missing, and, if a
regulated pathway is chosen, capital and time on the order of what Prenosis or AlgoDx
spent to reach their 2024 FDA authorizations. This project can meaningfully de-risk
and prepare for that; it cannot substitute for it.
