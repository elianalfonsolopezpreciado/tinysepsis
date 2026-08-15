"""Simulate a full CDS Hooks round-trip: build a synthetic FHIR prefetch
bundle for a deteriorating patient (rising HR/RR, falling SBP, rising
lactate over 6 hours -- a textbook sepsis trajectory), call the
`patient-view` hook exactly as an EHR would, and print the resulting card.

This does NOT talk to a real EHR or FHIR server. It demonstrates the
service's request/response contract end-to-end so the integration can be
inspected and reviewed before anyone wires it to a real sandbox. Next real
step: point this at a SMART Health IT (https://launch.smarthealthit.org/)
or vendor sandbox instead of the synthetic bundle below.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402
from tinysepsis.integration.cds_hooks_app import app  # noqa: E402
from tinysepsis.integration.fhir_mapping import LOINC_MAP  # noqa: E402

client = TestClient(app)


def make_observation(feature: str, value: float, hours_ago: float, patient_id: str = "example-patient-1"):
    entry = LOINC_MAP[feature]
    eff = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    return {
        "resource": {
            "resourceType": "Observation",
            "status": "final",
            "code": {"coding": [{"system": "http://loinc.org", "code": entry.code, "display": entry.display}]},
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": eff,
            "valueQuantity": {"value": value, "unit": entry.unit_hint},
        }
    }


def deteriorating_patient_bundle():
    """6 hours of a patient trending toward sepsis: HR and RR climbing,
    SBP falling, lactate rising -- then wraps into a FHIR searchset Bundle."""
    entries = []
    for h in range(6, 0, -1):
        t = 6 - h  # 0..5, time progressing
        entries.append(make_observation("HR", 80 + t * 8, hours_ago=h))
        entries.append(make_observation("Resp", 16 + t * 2, hours_ago=h))
        entries.append(make_observation("SBP", 118 - t * 6, hours_ago=h))
        entries.append(make_observation("Temp", 37.0 + t * 0.3, hours_ago=h))
        entries.append(make_observation("O2Sat", 97 - t * 1.5, hours_ago=h))
    entries.append(make_observation("Lactate", 3.4, hours_ago=1))
    entries.append(make_observation("WBC", 15.2, hours_ago=2))
    return {"resourceType": "Bundle", "type": "searchset", "entry": entries}


def main():
    print("1) Discovery:", flush=True)
    disco = client.get("/cds-services").json()
    print(disco, flush=True)

    print("\n2) Simulated hook call (deteriorating patient, sepsis-like trajectory):", flush=True)
    bundle = deteriorating_patient_bundle()
    request_body = {
        "hookInstance": "d1577c69-dfbe-44ad-ba6d-3e05e953b2ea",
        "hook": "patient-view",
        "context": {"patientId": "example-patient-1", "userId": "Practitioner/example"},
        "prefetch": {
            "patient": {"resourceType": "Patient", "id": "example-patient-1",
                        "birthDate": "1958-03-14", "gender": "male"},
            "vitals": bundle,
            "labs": bundle,
        },
    }
    resp = client.post("/cds-services/tinysepsis-sepsis-risk", json=request_body)
    print(f"status: {resp.status_code}", flush=True)
    print(resp.json(), flush=True)

    print("\n3) Same call, healthy/stable patient (control):", flush=True)
    stable_entries = [
        make_observation("HR", 76, hours_ago=1),
        make_observation("Resp", 15, hours_ago=1),
        make_observation("SBP", 122, hours_ago=1),
        make_observation("Temp", 36.8, hours_ago=1),
        make_observation("O2Sat", 98, hours_ago=1),
    ]
    stable_bundle = {"resourceType": "Bundle", "type": "searchset", "entry": stable_entries}
    request_body["prefetch"]["vitals"] = stable_bundle
    request_body["prefetch"]["labs"] = {"resourceType": "Bundle", "type": "searchset", "entry": []}
    resp2 = client.post("/cds-services/tinysepsis-sepsis-risk", json=request_body)
    print(resp2.json(), flush=True)


if __name__ == "__main__":
    main()
