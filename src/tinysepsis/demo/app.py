"""TinySepsis local research demo (FastAPI + ONNX Runtime).

NOT FOR CLINICAL USE. Prototype decision-support research tool only. Runs
fully offline on CPU; no data leaves the machine.

Usage:
    uvicorn tinysepsis.demo.app:app --reload --port 8420
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent.parent.parent
ONNX_PATH = ROOT / "results" / "checkpoints" / "tinysepsis.onnx"
NORM_STATS_PATH = ROOT / "data" / "processed" / "norm_stats.json"
CALIBRATORS_PATH = ROOT / "results" / "calibration" / "calibrators.json"

from tinysepsis.data.schema import NUMERIC_FEATURES  # noqa: E402

app = FastAPI(
    title="TinySepsis Research Demo",
    description="Prototype decision-support research tool. NOT a medical device. Not for clinical use.",
)

_session = None
_norm_stats = None
_calibrators = None


def _lazy_load():
    global _session, _norm_stats, _calibrators
    if _session is None:
        _session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        _norm_stats = json.loads(NORM_STATS_PATH.read_text())
        _calibrators = json.loads(CALIBRATORS_PATH.read_text()) if CALIBRATORS_PATH.exists() else {"temperature": 1.0, "conformal_tau": 0.5}
    return _session, _norm_stats, _calibrators


class HourlyObservation(BaseModel):
    hour: int
    values: dict[str, float | None] = Field(default_factory=dict, description="feature name -> value, omit/null if not measured")


class PredictRequest(BaseModel):
    age: float
    gender: int
    observations: list[HourlyObservation]


def _build_sequence(req: PredictRequest, seq_len: int = 24):
    stats = _lazy_load()[1]
    n_feat = len(NUMERIC_FEATURES)
    seq = np.zeros((seq_len, n_feat * 4), dtype=np.float32)
    last_val = {c: None for c in NUMERIC_FEATURES}
    last_hour_seen = {c: None for c in NUMERIC_FEATURES}

    obs_sorted = sorted(req.observations, key=lambda o: o.hour)[-seq_len:]
    n_real = len(obs_sorted)
    offset = seq_len - n_real

    contributions = {c: 0.0 for c in NUMERIC_FEATURES}

    for i, obs in enumerate(obs_sorted):
        row_pos = offset + i
        for j, feat in enumerate(NUMERIC_FEATURES):
            v = obs.values.get(feat)
            mask = 1.0 if v is not None else 0.0
            if v is not None:
                last_val[feat] = v
                last_hour_seen[feat] = obs.hour
            val = last_val[feat]
            mean, std = stats[feat]["mean"], stats[feat]["std"]
            val_z = ((val if val is not None else mean) - mean) / std

            tslm = 0.0 if last_hour_seen[feat] is None else min(obs.hour - last_hour_seen[feat], 48)
            tslm_mean, tslm_std = stats[f"{feat}__tslm"]["mean"], stats[f"{feat}__tslm"]["std"]
            tslm_z = (tslm - tslm_mean) / tslm_std

            seq[row_pos, j] = val_z
            seq[row_pos, n_feat + j] = mask
            seq[row_pos, 2 * n_feat + j] = tslm_z
            seq[row_pos, 3 * n_feat + j] = 0.0  # delta1 omitted in the simplified demo input

            if mask == 1.0:
                contributions[feat] = abs(val_z)

    pad_mask = np.zeros((seq_len,), dtype=np.float32)
    pad_mask[offset:] = 1.0

    age_mean, age_std = stats["Age"]["mean"], stats["Age"]["std"]
    static = np.array([(req.age - age_mean) / age_std, float(req.gender)], dtype=np.float32)

    return seq[None, :, :], pad_mask[None, :], static[None, :], contributions


@app.get("/health")
def health():
    return {"status": "ok", "disclaimer": "Prototype research tool. Not for clinical use."}


@app.post("/predict")
def predict(req: PredictRequest):
    session, stats, calibrators = _lazy_load()
    seq, pad_mask, static, contributions = _build_sequence(req)

    (logit,) = session.run(None, {"seq": seq, "pad_mask": pad_mask, "static": static})
    logit = float(logit.squeeze())

    T = calibrators.get("temperature", 1.0)
    prob = 1.0 / (1.0 + np.exp(-logit / T))
    tau = calibrators.get("conformal_tau", 0.5)
    alarm = bool(prob >= tau)

    top_factors = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)[:5]

    return {
        "risk_probability": round(float(prob), 4),
        "alarm_raised": alarm,
        "conformal_threshold": tau,
        "top_contributing_factors": [f for f, _ in top_factors if _ > 0],
        "disclaimer": "Prototype decision-support research tool. NOT a medical device. Not for clinical use.",
    }
