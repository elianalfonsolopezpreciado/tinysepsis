"""Convert a FHIR R4 Bundle of Observation resources (as an EHR would send
via a CDS Hooks 'prefetch') into the HourlyObservation sequence TinySepsis's
model-serving code already consumes (tinysepsis.demo.app.PredictRequest).

Hour binning: PhysioNet Challenge 2019 (the data TinySepsis was trained on)
is hourly-binned, so we bucket FHIR observations the same way -- floor each
observation's age, in hours, relative to the most recent observation in the
bundle ("now"). Multiple observations of the same LOINC code within one
hour bucket keep only the most recent value, matching how the training
pipeline treats same-hour duplicates.
"""
from __future__ import annotations

from datetime import datetime, timezone

from tinysepsis.demo.app import HourlyObservation, PredictRequest
from tinysepsis.integration.fhir_mapping import LOINC_TO_FEATURE


def _parse_datetime(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _extract_loinc_code(observation: dict) -> str | None:
    for coding in observation.get("code", {}).get("coding", []):
        if coding.get("system") in ("http://loinc.org", "https://loinc.org"):
            return coding.get("code")
    return None


def _extract_value(observation: dict) -> float | None:
    vq = observation.get("valueQuantity")
    if vq and "value" in vq:
        return float(vq["value"])
    return None


def observations_from_bundle(bundle: dict) -> list[dict]:
    """Flatten a FHIR Bundle (searchset of Observations) into raw entries."""
    entries = bundle.get("entry", [])
    out = []
    for e in entries:
        resource = e.get("resource", {})
        if resource.get("resourceType") != "Observation":
            continue
        code = _extract_loinc_code(resource)
        value = _extract_value(resource)
        eff = resource.get("effectiveDateTime") or resource.get("issued")
        if code is None or value is None or eff is None:
            continue
        feature = LOINC_TO_FEATURE.get(code)
        if feature is None:
            continue  # unmapped code -- ignored, not an error
        out.append({"feature": feature, "value": value, "time": _parse_datetime(eff)})
    return out


def build_predict_request(
    observation_bundle: dict,
    age: float,
    gender: int,
    seq_len: int = 24,
) -> PredictRequest:
    raw = observations_from_bundle(observation_bundle)
    if not raw:
        return PredictRequest(age=age, gender=gender, observations=[])

    t_max = max(r["time"] for r in raw)
    hourly: dict[int, dict[str, float]] = {}
    for r in raw:
        hour = -int((t_max - r["time"]).total_seconds() // 3600)  # <=0, 0 = most recent
        hourly.setdefault(hour, {})[r["feature"]] = r["value"]

    min_hour = min(hourly.keys())
    shifted = {h - min_hour: v for h, v in hourly.items()}  # rebase to start at 0

    observations = [
        HourlyObservation(hour=h, values=v)
        for h, v in sorted(shifted.items())
    ][-seq_len:]

    return PredictRequest(age=age, gender=gender, observations=observations)
