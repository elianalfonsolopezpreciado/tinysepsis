---
name: gmlp-self-assessment
description: Honest self-assessment of TinySepsis against the 10 Good Machine Learning Practice (GMLP) guiding principles jointly published by FDA, Health Canada, and the UK MHRA (October 2021).
---

# GMLP Self-Assessment: TinySepsis

Ten principles, jointly issued by FDA/Health Canada/MHRA (Oct 2021) to guide safe, effective, high-quality AI/ML medical device development. This is a **self-assessment**, done by the development team, not an independent audit. Status legend: ✅ addressed, 🟡 partially addressed, ❌ not addressed.

| # | Principle (paraphrased) | Status | Evidence / gap |
|---|---|---|---|
| 1 | Multi-disciplinary expertise used throughout the product lifecycle | ❌ | Single-developer project to date. No clinician, biostatistician, or regulatory professional has reviewed the model, the labels, or the risk file. **This is the single most important gap to close before any real deployment conversation** -- exactly what the collaborating physician's involvement should formalize next. |
| 2 | Good software engineering and security practices | 🟡 | Version-controlled, tested (60+ unit tests, CI-style pytest suite), reproducible pipeline with fixed seeds where possible. No security review, no auth on the reference CDS Hooks service, no SBOM, no penetration testing. |
| 3 | Clinical study participants and data sets are representative of the intended patient population | 🟡 | Two US-hospital ICU populations, adults only, matches the stated intended-use population directionally, but is not necessarily representative of a *specific* future deployment site's population, and has no race/ethnicity data to check representativeness on that axis at all. |
| 4 | Training data sets are independent of test sets | ✅ | Patient-level (not row-level) splitting enforced and unit-tested (`tests/test_splits.py`); Hospital B held out entirely from training/validation/calibration. |
| 5 | Selected reference datasets are based on best available methods | 🟡 | PhysioNet Challenge 2019 is a recognized, peer-reviewed benchmark (Reyna et al., *Critical Care Medicine*, 2019) with a defined Sepsis-3-based label. Not a clinician-adjudicated gold-standard label specific to this project. |
| 6 | Model design tailored to the available data and reflects the intended use | ✅ | Explicit missingness encoding (informative-missingness literature-motivated), sequence model matched to the temporal nature of the task, compact architecture matched to the stated low-resource deployment goal. |
| 7 | Focus on the performance of the human-AI TEAM, not just the model in isolation | ❌ | No human-factors study, no clinician usability testing, no measurement of how the score changes clinician behavior. The CDS Hooks card design (score + contributing factors + explicit "don't act alone" language) is informed by GMLP intent but untested with real users. |
| 8 | Testing demonstrates device performance during clinically relevant conditions | 🟡 | Cross-institution stress test (the paper's core contribution) is exactly this kind of testing, and the multi-seed replication addresses one form of robustness. Not tested under real EHR data-quality conditions (missing/mistimed/duplicate real-world observations), not tested prospectively. |
| 9 | Users are given clear, essential information (performance, training data characteristics, subgroup performance, update history) | ✅ (as documentation) / ❌ (as UX) | `model_card.md` covers this in writing. The actual CDS Hooks card shown to a clinician does not yet surface subgroup caveats or a link to full performance documentation -- a real deployment should add a "more info" link from the card itself. |
| 10 | Deployed models are monitored for real-world performance, and re-training/updates are managed | 🟡 | Plan exists (`post_market_surveillance_plan.md`, `predetermined_change_control_plan.md`). No live monitoring, because there is no live deployment. |

## Net assessment

**3 of 10 fully addressed, 5 partially, 2 not addressed.** The two "not addressed" items (#1 multi-disciplinary expertise, #7 human-AI team performance) are not things a solo technical project can close by writing more code -- they require the collaborating physician (and ideally additional clinical/human-factors expertise) to be actively involved in model review, label validation, and eventually a usability study, not just an endorsement. That is the honest bottleneck between "good research prototype" and "credible hospital pilot," more than any remaining engineering work.
