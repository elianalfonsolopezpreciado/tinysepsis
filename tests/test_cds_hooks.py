from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from tinysepsis.integration.cds_hooks_app import app, CDS_SERVICES, SERVICE_ID
from tinysepsis.integration.fhir_mapping import LOINC_MAP

client = TestClient(app)
NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _obs(feature, value, hours_ago):
    entry = LOINC_MAP[feature]
    eff = (NOW - timedelta(hours=hours_ago)).isoformat()
    return {
        "resource": {
            "resourceType": "Observation",
            "code": {"coding": [{"system": "http://loinc.org", "code": entry.code}]},
            "effectiveDateTime": eff,
            "valueQuantity": {"value": value},
        }
    }


def _hook_request(vitals_entries, labs_entries=None, birth_date="1958-03-14", gender="male"):
    return {
        "hookInstance": "test-instance",
        "hook": "patient-view",
        "context": {"patientId": "p1"},
        "prefetch": {
            "patient": {"resourceType": "Patient", "id": "p1", "birthDate": birth_date, "gender": gender},
            "vitals": {"resourceType": "Bundle", "entry": vitals_entries},
            "labs": {"resourceType": "Bundle", "entry": labs_entries or []},
        },
    }


def test_discovery_lists_the_patient_view_hook():
    resp = client.get("/cds-services")
    assert resp.status_code == 200
    services = resp.json()["services"]
    assert len(services) == 1
    assert services[0]["hook"] == "patient-view"
    assert services[0]["id"] == SERVICE_ID
    assert "prefetch" in services[0]


def test_discovery_matches_the_module_level_constant():
    resp = client.get("/cds-services")
    assert resp.json() == CDS_SERVICES


def test_hook_call_with_no_observations_returns_info_card():
    body = _hook_request(vitals_entries=[])
    resp = client.post(f"/cds-services/{SERVICE_ID}", json=body)
    assert resp.status_code == 200
    cards = resp.json()["cards"]
    assert len(cards) == 1
    assert cards[0]["indicator"] == "info"
    assert "insufficient data" in cards[0]["summary"].lower()


def test_hook_call_returns_a_valid_card_shape():
    body = _hook_request(vitals_entries=[_obs("HR", 90, hours_ago=1), _obs("Resp", 18, hours_ago=1)])
    resp = client.post(f"/cds-services/{SERVICE_ID}", json=body)
    assert resp.status_code == 200
    card = resp.json()["cards"][0]
    for key in ("uuid", "summary", "indicator", "detail", "source"):
        assert key in card
    assert card["indicator"] in ("info", "warning", "critical")
    assert "not for clinical use" in card["detail"].lower()


def test_deteriorating_patient_scores_higher_than_stable_patient():
    """The bug this whole integration surfaced: a deteriorating patient
    and a stable one must NOT get the same risk score. This is the
    end-to-end version of tests/test_model.py's unit-level regression test."""
    stable = [
        _obs("HR", 76, hours_ago=1), _obs("Resp", 15, hours_ago=1),
        _obs("SBP", 122, hours_ago=1), _obs("Temp", 36.8, hours_ago=1),
        _obs("O2Sat", 98, hours_ago=1),
    ]
    deteriorating = [
        _obs("HR", 135, hours_ago=1), _obs("Resp", 32, hours_ago=1),
        _obs("SBP", 78, hours_ago=1), _obs("Temp", 39.3, hours_ago=1),
        _obs("O2Sat", 86, hours_ago=1), _obs("Lactate", 4.8, hours_ago=1),
    ]
    resp_stable = client.post(f"/cds-services/{SERVICE_ID}", json=_hook_request(stable)).json()
    resp_sick = client.post(f"/cds-services/{SERVICE_ID}", json=_hook_request(deteriorating)).json()

    detail_stable = resp_stable["cards"][0]["detail"]
    detail_sick = resp_sick["cards"][0]["detail"]
    prob_stable = float(detail_stable.split("risk probability: ")[1].split(" ")[0])
    prob_sick = float(detail_sick.split("risk probability: ")[1].split(" ")[0])

    assert prob_sick > prob_stable
