---
name: mexico-market-roadmap
description: Go-to-market roadmap for Pear Labs to bring TinySepsis from research prototype to a paid deployment in a Mexican hospital, grounded in COFEPRIS regulation, Mexican sepsis epidemiology, hospital market structure, and healthcare-SaaS pricing norms.
---

# Roadmap: TinySepsis in Mexican Hospitals (Pear Labs)

**Read this first:** every number below that isn't from this project's own results is from a web search done while writing this document, not verified against a primary regulatory or market-research source by a lawyer, regulatory consultant, or business analyst. Treat this as a structured starting hypothesis for Pear Labs to pressure-test with real experts, not a business plan ready to execute.

## 0. Why this is a real opportunity, not just a nice paper

- Mexican ICU studies report sepsis mortality around **30%**, with per-case treatment costs of **600,000-1,870,000 MXN** (roughly USD 35,000-110,000 at typical exchange rates). Even a modest reduction in late recognition -- the exact failure mode this project targets -- has real economic value per prevented case, which is the foundation of any pricing conversation with a hospital's finance committee, not just its clinical staff.
- **87% of documented sepsis cases in Mexico occur in public-sector ICUs** (IMSS, ISSSTE, Secretaría de Salud), only 13% in private ones. This inverts the usual first-instinct go-to-market (private hospitals are easier to sell to, but are a minority of the actual disease burden) -- addressed directly in Section 3.
- Mexico's digital-health sector is growing fast (healthtech investment in Latin America was ~USD 253M in 2024, up 36.6% year over year; Mexico's own digital-health market was projected near USD 500M by 2025) -- a receptive investor and partner environment for a company like Pear Labs to raise around, separate from the hospital sales motion itself.

## 1. Where Pear Labs is today

A validated-on-open-data research prototype (`paper/main.pdf`) with: a working model and calibration pipeline, a genuine (if single-hospital-pair) cross-institution robustness result with statistical backing, a working EHR integration reference implementation (CDS Hooks/FHIR), and draft regulatory scaffolding (`regulatory/`). **Zero** prospective validation, **zero** Mexican regulatory filing, **zero** paying customer, **zero** clinical or regulatory hire beyond the collaborating physician. This is Day 0 of a multi-year process, not Day 300.

## 2. Regulatory path: COFEPRIS

COFEPRIS (Comisión Federal para la Protección contra Riesgos Sanitarios) is Mexico's FDA-equivalent. Under **NOM-241-SSA1-2021**, software intended for diagnosis, monitoring, or treatment assistance is regulated as a medical device, classified into Risk Class I/II/III much like the FDA's system -- TinySepsis, as scoped in `intended_use_statement.md`, would most plausibly land in Class II (moderate risk, decision-support, not autonomous), the same class every FDA-authorized AI/ML SaMD to date has landed in.

**The single most important regulatory fact for Pear Labs' strategy**: as of **September 2025**, COFEPRIS operates an **Abbreviated Regulatory Pathway** that accepts existing authorizations from FDA, Health Canada, and other IMDRF/MDSAP-recognized bodies, with a **target 30-day review window** -- dramatically faster than a full independent Mexican submission built from scratch. This means the fastest realistic path to a Mexican sanitary registration is:

**Pursue FDA authorization (or another IMDRF-recognized market's) first, then leverage the abbreviated COFEPRIS pathway** -- not attempt a from-scratch Mexican filing as the first regulatory event. This reframes Section 11's US regulatory groundwork (`regulatory/README.md`) as directly load-bearing for the Mexican go-to-market, not a separate track.

Alternatively, if speed-to-Mexico matters more than a US launch, a direct Class II COFEPRIS submission (with its own clinical evidence package, likely built from the `clinical_validation_protocol.md` Phase 0/1 study run at a Mexican site) is the fallback, at the cost of losing the abbreviated-pathway speed advantage.

## 3. Who to sell to first: public vs.\ private, and why

| | Share of sepsis burden | Procurement speed | Price sensitivity | Practical entry point |
|---|---|---|---|---|
| Public (IMSS, ISSSTE, Secretaría de Salud) | ~87% | Slow -- formal tenders via CompraNet/IMSS's own procurement portal, multi-month to multi-year cycles, budget-cycle-dependent | Very high | Realistic only *after* private-sector validation; likely requires a local systems-integrator or distributor relationship, not a direct sale |
| Private (Christus Muguerza, Grupo Ángeles, TecSalud/Zambrano Hellion, Star Médica, Médica Sur, H+, Puerta de Hierro, ABC) | ~13% | Faster -- department-level pilots possible without a full national tender | Lower, outcomes-justifiable | **The correct Phase-1 pilot target** |

**Recommendation:** the clinical validation protocol's Phase 0 (silent/shadow mode, `clinical_validation_protocol.md` Section 3) should run at **one private hospital with an existing academic/research orientation** -- TecSalud (affiliated with Tecnológico de Monterrey, likely to have both IRB infrastructure and a receptive innovation culture) or Médica Sur are the most plausible starting candidates given their ranking and academic ties, though this requires an actual relationship, not a cold outreach, to move quickly. A successful Phase 0/1 there becomes the case study used to approach either (a) other private groups, or (b) a public-sector pilot via a state-level Secretaría de Salud (state health ministries often have more procurement latitude than the federal IMSS/ISSSTE tender process).

## 4. Data privacy and compliance (separate from, and required alongside, COFEPRIS)

- **LFPDPPP** (Ley Federal de Protección de Datos Personales en Posesión de los Particulares): health data is legally "sensitive personal data," the highest protection tier -- requires express written consent (or a valid legal basis under the hospital's own patient-data governance, since the hospital -- not Pear Labs -- is typically the data controller in a CDS deployment), a clear privacy notice, and documented administrative/technical/physical safeguards.
- **NOM-024-SSA3-2012**: the specific technical standard for electronic health record systems in Mexico -- governs how clinical data systems must handle confidentiality, integrity, and interoperability. Any FHIR/CDS Hooks integration deployed in Mexico should be reviewed against this standard specifically, not assumed equivalent to HIPAA compliance.
- **Practical implication:** Pear Labs will very likely operate as a data *processor* under a hospital-controlled data-processing agreement, not as a data controller -- the contract structure matters as much as the technical compliance, and should be drafted by counsel familiar with Mexican health-data law specifically, not adapted from a US HIPAA BAA template.

## 5. Pricing strategy

US healthcare-SaaS benchmarks (per-bed licensing in the style of Epic; enterprise deals commonly USD 100K-500K+/year; volume discounts of 15-25% past 250 beds) **do not transfer directly to Mexico** -- Mexican hospital budgets, even at large private groups, are a fraction of comparable US systems'. Two pricing directions worth piloting, not a single fixed number:

1. **Per-ICU-bed subscription, scaled to local benchmarks.** Most Mexican private hospitals are small (89% of private hospitals have under 25 beds total, per the sector data found in this research); pricing per *ICU* bed rather than per hospital bed, and scaled well below US per-bed norms, is more realistic for an initial pilot-to-paid conversion.
2. **Outcomes/value-based framing anchored to the cost-per-case data in Section 0.** A pricing conversation framed as "a small percentage of the documented ~600K-1.9M MXN cost of a single missed or late-recognized sepsis case" is a far stronger opening position with a hospital's CFO than a generic per-seat SaaS number, *provided* the Phase 0/1 clinical validation (Section 6) actually produces a defensible local effect-size estimate first -- this pricing model should not be pitched before that evidence exists.

Neither number should be fixed publicly (including in the paper) before Pear Labs has run an actual pilot and a real pricing conversation; this section exists to structure that conversation, not to pre-commit to a figure.

## 6. Concrete phased roadmap

| Phase | What | Rough timeline | Exit criteria |
|---|---|---|---|
| 0 (now) | Research prototype + regulatory groundwork (this repository) | Done | Reproducible, tested, documented |
| 1 | Recruit a Mexican clinical co-investigator/site (beyond the current collaborating physician) at a candidate private hospital (Section 3); formalize the Phase 0 shadow-mode protocol (`clinical_validation_protocol.md`) with that site's IRB | 2-4 months | Signed site agreement + IRB approval for shadow mode |
| 2 | Run Phase 0 (silent mode, no care impact) at the pilot site; instrument post-market-style monitoring even though it's pre-market (`post_market_surveillance_plan.md` metrics, applied prospectively) | 6-12 months (needs enough septic-patient encounters for a meaningful estimate, per the rough power calculation in `clinical_validation_protocol.md` Section 6) | Local AUROC/calibration/alarm-rate estimates that meet or exceed the retrospective Hospital-B numbers |
| 3 | Pursue FDA authorization (or another IMDRF-recognized-market pathway) in parallel with Phase 2, using the Phase 0 data as part of the clinical evidence package alongside the retrospective PhysioNet results | Overlapping with Phase 2; FDA De Novo/510(k) review itself typically runs several months to over a year after submission, on top of pre-submission QMS work (`regulatory/README.md`) | Marketing authorization (US or other IMDRF market) |
| 4 | File the COFEPRIS Abbreviated Pathway submission, referencing the Phase 3 authorization | ~30-day target review per COFEPRIS's own stated window, once Phase 3 is complete | Mexican sanitary registration (Class II) |
| 5 | Phase 1 clinical study (visible alerts, non-interventional) at the same pilot site, now under Mexican registration | 6-12 months | Human-AI-team evidence (GMLP Principle 7), real alert-dismissal/workflow data |
| 6 | Convert the pilot site from research relationship to paid deployment; use as reference for a second private-hospital-group sale | Sales cycle for enterprise health-tech is commonly 6-18 months even with a strong reference site | First paying contract signed |
| 7 | Use 2-3 private-sector reference deployments to approach a state-level Secretaría de Salud or a public-sector pilot | 12+ months after Phase 6 | First public-sector engagement |

**Total, realistically: 3-5 years from today to a mature public-sector presence**, with the first paid private-sector contract plausible in **18-30 months** if Phases 1-2 go well and Pear Labs has the clinical/regulatory team in place to run them -- which it does not yet have (Section 7).

## 7. What Pear Labs needs to hire or contract before Phase 1 can start

This is the same gap `regulatory/gmlp_self_assessment.md` identifies (multi-disciplinary expertise, Principle 1) applied to the specific Mexican go-to-market:

- A **Mexican regulatory affairs consultant** (COFEPRIS-specific, ideally with SaMD experience) -- do not attempt the filing without one.
- A **clinical co-investigator with IRB access** at the target pilot hospital, distinct from (though possibly in addition to) the current collaborating physician.
- A **health-data privacy counsel** familiar with LFPDPPP and NOM-024-SSA3-2012 specifically.
- A **biostatistician** for the clinical validation protocol's sample-size and analysis plan (already flagged as a gap in `clinical_validation_protocol.md`).
- Eventually, a **hospital sales/business-development hire** with existing relationships in the Mexican private hospital sector -- this kind of relationship-driven enterprise sale is unlikely to close cold.

## 8. The honest risk list

- COFEPRIS's abbreviated pathway is new (Sept 2025) -- its real-world processing time and edge cases are not yet well-documented; do not treat the 30-day figure as guaranteed.
- Mexican hospital procurement, especially public-sector, is materially slower and more relationship-dependent than this document's timeline can fully capture from a desk review -- local partners will revise this timeline once real conversations start.
- A single-hospital-pair retrospective result (this project's current evidence) is not sufficient grounds for a health-economic pricing claim; Section 5's outcomes-based pricing direction depends entirely on Phase 2 producing real local evidence first.
- "Really save lives" requires the interventional (Phase 2, `clinical_validation_protocol.md`) evidence this roadmap doesn't yet have -- everything before that is de-risking, not proof of clinical benefit.
