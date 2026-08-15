# EHR integration: CDS Hooks + FHIR

Real hospitals don't want a standalone app; they want an alert that shows up inside
the EHR a clinician is already using. The industry-standard way to do that is
[CDS Hooks](https://cds-hooks.org/) (an HL7 spec) plus
[SMART on FHIR](https://docs.smarthealthit.org/) for data access and auth. This
directory implements TinySepsis as a real CDS Hooks service.

## What's here

- `fhir_mapping.py` -- LOINC code <-> TinySepsis feature name table. **Read the
  module docstring before using this against real data**: several codes are flagged
  `needs_verification` and have not been confirmed against loinc.org or a
  terminology service.
- `fhir_adapter.py` -- turns a FHIR `Bundle` of `Observation` resources (what an EHR
  sends as a CDS Hooks `prefetch`) into the hourly-binned sequence format the model
  consumes.
- `cds_hooks_app.py` -- the FastAPI service itself: `GET /cds-services` (discovery)
  and `POST /cds-services/tinysepsis-sepsis-risk` (the `patient-view` hook).

## Try it without a real EHR

```bash
python scripts/simulate_cds_hooks_call.py
```

Builds a synthetic FHIR bundle for a deteriorating patient (rising HR/RR, falling
SBP, rising lactate) and a stable control, calls the hook exactly as an EHR would,
and prints the resulting alert cards.

## Run it as a real service

```bash
uvicorn tinysepsis.integration.cds_hooks_app:app --port 8421
```

## What's needed before this touches a real patient

1. **Clinical-informatics review of `fhir_mapping.py`.** Every code marked
   `needs_verification` must be confirmed (or corrected) by someone with LOINC/FHIR
   terminology expertise, ideally against the receiving institution's own dictionary.
2. **Authentication.** The reference service has none. A real deployment needs
   SMART on FHIR's standard OAuth2 flow (the EHR authenticates the call) and TLS.
   See `regulatory/risk_management_plan.md` hazard H9.
3. **A real sandbox test**, not just the synthetic simulation above --
   [SMART Health IT's sandbox](https://launch.smarthealthit.org/) or a vendor
   sandbox (Epic, Oracle Health) is the natural next step, using this service's
   existing discovery/hook contract unchanged.
4. Everything in `regulatory/` -- this is the technical half of "integrated with a
   hospital's EHR"; the regulatory, clinical-validation, and quality-system halves
   are separate, larger undertakings documented there.

## Why `patient-view`

CDS Hooks defines several trigger points (`patient-view`, `order-select`,
`order-sign`, `encounter-start`, ...). `patient-view` -- fired when a clinician opens
a patient's chart -- is the standard choice for a continuously-relevant risk score
like this one (the same pattern used in published sepsis-alert CDS Hooks
implementations). A production deployment might also register for periodic
re-evaluation as new vitals/labs arrive, which CDS Hooks does not natively support
as a "background" trigger -- that would require either polling or an
EHR-vendor-specific extension, a design question left open here.
