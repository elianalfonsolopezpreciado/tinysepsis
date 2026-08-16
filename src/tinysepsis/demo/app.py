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
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent.parent.parent
ONNX_PATH = ROOT / "results" / "checkpoints" / "tinysepsis.onnx"
TORCH_CKPT_PATH = ROOT / "results" / "checkpoints" / "tinysepsis_best.pt"
NORM_STATS_PATH = ROOT / "data" / "processed" / "norm_stats.json"
CALIBRATORS_PATH = ROOT / "results" / "calibration" / "calibrators.json"

from tinysepsis.data.schema import NUMERIC_FEATURES, STATIC_FEATURES  # noqa: E402
from tinysepsis.eval.explain import aggregate_feature_attributions, integrated_gradients  # noqa: E402
from tinysepsis.models.tiny_sepsis import TinySepsisModel  # noqa: E402

app = FastAPI(
    title="TinySepsis Research Demo",
    description="Prototype decision-support research tool. NOT a medical device. Not for clinical use.",
)

_session = None
_norm_stats = None
_calibrators = None
_torch_model = None


def _lazy_load():
    """ONNX Runtime serves the actual risk score (the fast, portable
    inference path this project exports for deployment); the PyTorch
    checkpoint that produced that ONNX graph is loaded separately, only to
    compute Integrated Gradients explanations, which need real gradients
    that the ONNX graph doesn't expose. The two were verified numerically
    equivalent at export time (scripts/export_onnx.py's parity check), so
    using PyTorch for explanations and ONNX for the score is consistent,
    not a second, possibly-diverging model.
    """
    global _session, _norm_stats, _calibrators, _torch_model
    if _session is None:
        _session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        _norm_stats = json.loads(NORM_STATS_PATH.read_text())
        _calibrators = json.loads(CALIBRATORS_PATH.read_text()) if CALIBRATORS_PATH.exists() else {"temperature": 1.0, "conformal_tau": 0.5}

        ckpt = torch.load(TORCH_CKPT_PATH, map_location="cpu", weights_only=False)
        model_args = ckpt["args"]
        _torch_model = TinySepsisModel(
            num_dynamic_features=ckpt["num_dynamic"],
            num_static_features=ckpt["num_static"],
            hidden_size=model_args["hidden_size"],
            num_layers=model_args["num_layers"],
        )
        _torch_model.load_state_dict(ckpt["model_state"])
        _torch_model.eval()
    return _session, _norm_stats, _calibrators


def explain(seq: np.ndarray, static: np.ndarray) -> dict[str, float]:
    """Integrated Gradients attribution for one (seq, static) example,
    aggregated to one |attribution| value per clinical feature (dynamic and
    static). Baseline is all-zeros -- see tinysepsis.eval.explain for what
    that represents for this input encoding."""
    _lazy_load()
    seq_t = torch.from_numpy(seq).float()
    static_t = torch.from_numpy(static).float()
    seq_attr, static_attr = integrated_gradients(_torch_model, seq_t, static_t, steps=50)
    return aggregate_feature_attributions(seq_attr, static_attr, NUMERIC_FEATURES, STATIC_FEATURES)


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

    pad_mask = np.zeros((seq_len,), dtype=np.float32)
    pad_mask[offset:] = 1.0

    age_mean, age_std = stats["Age"]["mean"], stats["Age"]["std"]
    static = np.array([(req.age - age_mean) / age_std, float(req.gender)], dtype=np.float32)

    return seq[None, :, :], pad_mask[None, :], static[None, :]


@app.get("/health")
def health():
    return {"status": "ok", "disclaimer": "Prototype research tool. Not for clinical use."}


def score_and_explain(req: PredictRequest) -> dict:
    """Shared by the /predict route and the CDS Hooks integration: ONNX
    gives the calibrated risk score, Integrated Gradients (via the sibling
    PyTorch checkpoint, see _lazy_load) gives the feature attributions that
    justify it."""
    session, stats, calibrators = _lazy_load()
    seq, pad_mask, static = _build_sequence(req)  # pad_mask unused by the ONNX graph (see tiny_sepsis.py)

    (logit,) = session.run(None, {"seq": seq, "static": static})
    logit = float(logit.squeeze())

    T = calibrators.get("temperature", 1.0)
    prob = 1.0 / (1.0 + np.exp(-logit / T))
    tau = calibrators.get("conformal_tau", 0.5)
    alarm = bool(prob >= tau)

    attributions = explain(seq, static)
    top_factors = sorted(attributions.items(), key=lambda kv: kv[1], reverse=True)[:5]

    return {
        "risk_probability": round(float(prob), 4),
        "alarm_raised": alarm,
        "conformal_threshold": tau,
        "top_contributing_factors": [f for f, v in top_factors if v > 0],
        "disclaimer": "Prototype decision-support research tool. NOT a medical device. Not for clinical use.",
    }


@app.post("/predict")
def predict(req: PredictRequest):
    return score_and_explain(req)
