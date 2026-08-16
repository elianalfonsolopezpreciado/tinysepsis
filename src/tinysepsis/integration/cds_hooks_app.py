"""TinySepsis as a CDS Hooks service (HL7 CDS Hooks 1.1 / STU2-compatible).

This is what "integrated into the EHR" concretely means: an EHR (Epic,
Oracle Health/Cerner, or any SMART-on-FHIR-compliant system) calls this
service's `patient-view` hook whenever a clinician opens a patient's chart,
passing FHIR Observation resources it already prefetched per the
`prefetch` template declared at discovery. The service returns a "card"
that the EHR renders inline in the clinician's own workflow -- no separate
app, no separate login.

NOT FOR CLINICAL USE. Prototype integration reference only, exercised here
against synthetic FHIR payloads (scripts/simulate_cds_hooks_call.py), not a
live EHR. Wiring this against a real sandbox (SMART Health IT, Logica, or
a vendor's own sandbox) and a clinical-informatics review of
fhir_mapping.py's LOINC codes are the next real steps, not done here.

Run:
    uvicorn tinysepsis.integration.cds_hooks_app:app --port 8421
"""
import uuid
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from tinysepsis.demo.app import score_and_explain
from tinysepsis.integration.fhir_adapter import build_predict_request
from tinysepsis.integration.fhir_mapping import UNVERIFIED_CODES

app = FastAPI(title="TinySepsis CDS Hooks Service")

SERVICE_ID = "tinysepsis-sepsis-risk"
HOOK = "patient-view"

CDS_SERVICES = {
    "services": [
        {
            "hook": HOOK,
            "title": "TinySepsis Early Warning (research prototype)",
            "description": (
                "Estimates 6-hour sepsis risk from the last 24h of vitals/labs using a "
                "calibrated, conformal-risk-controlled model. NOT a validated medical "
                "device; not for clinical use. See paper/main.pdf for validation status."
            ),
            "id": SERVICE_ID,
            "prefetch": {
                "patient": "Patient/{{context.patientId}}",
                "vitals": (
                    "Observation?patient={{context.patientId}}"
                    "&category=vital-signs&_sort=-date&_count=100"
                ),
                "labs": (
                    "Observation?patient={{context.patientId}}"
                    "&category=laboratory&_sort=-date&_count=100"
                ),
            },
        }
    ]
}


class CdsHooksRequest(BaseModel):
    hookInstance: str
    hook: str
    context: dict[str, Any] = {}
    prefetch: dict[str, Any] | None = None


def _merge_bundles(*bundles: dict | None) -> dict:
    entries = []
    for b in bundles:
        if not b:
            continue
        if b.get("resourceType") == "Bundle":
            entries.extend(b.get("entry", []))
        elif b.get("resourceType") == "Observation":
            entries.append({"resource": b})
    return {"resourceType": "Bundle", "entry": entries}


def _extract_age_gender(patient_resource: dict | None) -> tuple[float, int]:
    if not patient_resource:
        return 60.0, 0
    age = 60.0
    birth_date = patient_resource.get("birthDate")
    if birth_date:
        from datetime import date
        y, m, d = (int(x) for x in birth_date.split("-"))
        today = date.today()
        age = float(today.year - y - ((today.month, today.day) < (m, d)))
    gender = 1 if patient_resource.get("gender") == "male" else 0
    return age, gender


@app.get("/cds-services")
def discovery():
    return CDS_SERVICES


@app.post(f"/cds-services/{SERVICE_ID}")
def sepsis_risk_hook(req: CdsHooksRequest):
    prefetch = req.prefetch or {}
    patient = prefetch.get("patient")
    obs_bundle = _merge_bundles(prefetch.get("vitals"), prefetch.get("labs"))
    age, gender = _extract_age_gender(patient)

    predict_req = build_predict_request(obs_bundle, age=age, gender=gender)

    if not predict_req.observations:
        return {
            "cards": [{
                "uuid": str(uuid.uuid4()),
                "summary": "TinySepsis: insufficient data",
                "indicator": "info",
                "detail": "No mapped vital-sign/lab observations were found in the prefetched bundle.",
                "source": {"label": "TinySepsis (research prototype)"},
            }]
        }

    result = score_and_explain(predict_req)
    prob = result["risk_probability"]
    tau = result["conformal_threshold"]
    alarm = result["alarm_raised"]
    top_factors = result["top_contributing_factors"]

    indicator = "warning" if alarm else "info"
    summary = (
        f"TinySepsis: elevated 6h sepsis risk ({prob:.0%})"
        if alarm else
        f"TinySepsis: 6h sepsis risk {prob:.0%} (below alert threshold)"
    )
    detail = (
        f"Calibrated risk probability: {prob:.3f} (conformal alarm threshold: {tau:.3f}). "
        f"Top contributing signals: {', '.join(top_factors) if top_factors else 'none'}. "
        "Research prototype -- not a validated medical device, not for clinical use. "
        "Independently assess the patient; do not act on this score alone."
    )

    card: dict[str, Any] = {
        "uuid": str(uuid.uuid4()),
        "summary": summary,
        "indicator": indicator,
        "detail": detail,
        "source": {
            "label": "TinySepsis (research prototype)",
            "topic": {"system": "http://tinysepsis.research", "code": "sepsis-risk"},
        },
    }
    if UNVERIFIED_CODES:
        card["detail"] += (
            f" [dev note: {len(UNVERIFIED_CODES)} LOINC codes in this deployment's mapping "
            "are unverified and pending clinical-informatics review]"
        )

    return {"cards": [card]}
