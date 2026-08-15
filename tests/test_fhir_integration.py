from datetime import datetime, timedelta, timezone

from tinysepsis.integration.fhir_adapter import build_predict_request, observations_from_bundle
from tinysepsis.integration.fhir_mapping import LOINC_MAP, LOINC_TO_FEATURE

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)  # fixed reference, no wall-clock jitter


def _obs(feature, value, hours_ago, patient_id="p1"):
    entry = LOINC_MAP[feature]
    eff = (NOW - timedelta(hours=hours_ago)).isoformat()
    return {
        "resource": {
            "resourceType": "Observation",
            "code": {"coding": [{"system": "http://loinc.org", "code": entry.code}]},
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": eff,
            "valueQuantity": {"value": value},
        }
    }


def test_loinc_to_feature_is_a_clean_bijection_per_code():
    # every LOINC code maps back to exactly the feature that produced it
    for feature, entry in LOINC_MAP.items():
        assert LOINC_TO_FEATURE[entry.code] == feature


def test_observations_from_bundle_ignores_unmapped_codes():
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            _obs("HR", 88, hours_ago=1),
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "99999-9"}]},
                    "effectiveDateTime": datetime.now(timezone.utc).isoformat(),
                    "valueQuantity": {"value": 1.0},
                }
            },
        ],
    }
    parsed = observations_from_bundle(bundle)
    assert len(parsed) == 1
    assert parsed[0]["feature"] == "HR"


def test_observations_from_bundle_ignores_non_observation_resources():
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p1"}},
            _obs("HR", 88, hours_ago=1),
        ],
    }
    parsed = observations_from_bundle(bundle)
    assert len(parsed) == 1


def test_build_predict_request_bins_observations_into_hours_most_recent_last():
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            _obs("HR", 80, hours_ago=3),
            _obs("HR", 90, hours_ago=2),
            _obs("HR", 100, hours_ago=1),
        ],
    }
    req = build_predict_request(bundle, age=55, gender=1)
    assert len(req.observations) == 3
    hours = [o.hour for o in req.observations]
    assert hours == sorted(hours)  # ascending, i.e. chronological
    # the most recent observation (1h ago) must land in the LAST bin
    assert req.observations[-1].values["HR"] == 100
    assert req.observations[0].values["HR"] == 80


def test_build_predict_request_collapses_same_hour_observations_keeping_latest():
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            _obs("HR", 70, hours_ago=1.9),
            _obs("HR", 72, hours_ago=1.1),  # same hour bucket (both floor to 1h ago), later wins
        ],
    }
    req = build_predict_request(bundle, age=55, gender=1)
    assert len(req.observations) == 1
    assert req.observations[0].values["HR"] == 72


def test_build_predict_request_empty_bundle_returns_no_observations():
    req = build_predict_request({"resourceType": "Bundle", "entry": []}, age=40, gender=0)
    assert req.observations == []


def test_build_predict_request_truncates_to_seq_len():
    entries = [_obs("HR", 70 + h, hours_ago=h) for h in range(1, 40)]
    bundle = {"resourceType": "Bundle", "entry": entries}
    req = build_predict_request(bundle, age=40, gender=0, seq_len=24)
    assert len(req.observations) == 24
    # kept the 24 MOST RECENT hours (smallest hours_ago), last one is the very latest
    assert req.observations[-1].values["HR"] == 71  # hours_ago=1 -> value 70+1
