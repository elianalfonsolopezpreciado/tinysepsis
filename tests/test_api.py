from fastapi.testclient import TestClient

import tinysepsis.demo.app as app_module
from tinysepsis.demo.app import app, PredictRequest, _build_sequence, explain
from tinysepsis.data.schema import NUMERIC_FEATURES, STATIC_FEATURES
from tinysepsis.models.tiny_sepsis import TinySepsisModel

client = TestClient(app)


def test_health_endpoint_has_disclaimer():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "not for clinical use" in body["disclaimer"].lower()


def _fake_stats():
    stats = {}
    for c in NUMERIC_FEATURES:
        stats[c] = {"mean": 0.0, "std": 1.0}
        stats[f"{c}__tslm"] = {"mean": 0.0, "std": 1.0}
    stats["Age"] = {"mean": 50.0, "std": 20.0}
    return stats


def test_build_sequence_shapes_and_padding(monkeypatch):
    monkeypatch.setattr(app_module, "_norm_stats", _fake_stats())
    monkeypatch.setattr(app_module, "_session", object())  # bypass lazy load
    monkeypatch.setattr(app_module, "_calibrators", {"temperature": 1.0, "conformal_tau": 0.5})

    req = PredictRequest(
        age=65, gender=1,
        observations=[
            {"hour": 1, "values": {"HR": 90.0, "Lactate": 2.1}},
            {"hour": 2, "values": {"HR": 95.0}},
        ],
    )
    seq, pad_mask, static = _build_sequence(req, seq_len=24)
    assert seq.shape == (1, 24, len(NUMERIC_FEATURES) * 4)
    assert pad_mask.shape == (1, 24)
    assert pad_mask.sum() == 2  # only 2 real hours
    assert static.shape == (1, 2)


def test_build_sequence_truncates_to_seq_len(monkeypatch):
    monkeypatch.setattr(app_module, "_norm_stats", _fake_stats())
    monkeypatch.setattr(app_module, "_session", object())
    monkeypatch.setattr(app_module, "_calibrators", {"temperature": 1.0, "conformal_tau": 0.5})

    obs = [{"hour": h, "values": {"HR": 80.0}} for h in range(1, 40)]
    req = PredictRequest(age=40, gender=0, observations=obs)
    seq, pad_mask, static = _build_sequence(req, seq_len=24)
    assert pad_mask.sum() == 24  # only the last 24 hours kept


def test_explain_returns_one_attribution_per_feature(monkeypatch):
    n_feat = len(NUMERIC_FEATURES)
    fake_model = TinySepsisModel(num_dynamic_features=n_feat * 4, num_static_features=2, hidden_size=8, num_layers=1)
    fake_model.eval()
    monkeypatch.setattr(app_module, "_torch_model", fake_model)
    monkeypatch.setattr(app_module, "_session", object())
    monkeypatch.setattr(app_module, "_norm_stats", _fake_stats())
    monkeypatch.setattr(app_module, "_calibrators", {"temperature": 1.0, "conformal_tau": 0.5})

    req = PredictRequest(
        age=65, gender=1,
        observations=[{"hour": 1, "values": {"HR": 90.0, "Lactate": 2.1}}],
    )
    seq, pad_mask, static = _build_sequence(req, seq_len=24)
    attributions = explain(seq, static)

    assert set(attributions.keys()) == set(NUMERIC_FEATURES) | set(STATIC_FEATURES)
    assert all(v >= 0 for v in attributions.values())  # aggregated as |attribution|
