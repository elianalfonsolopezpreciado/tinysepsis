"""Post-hoc calibration (temperature scaling + isotonic) and conformal false-
alarm-rate control for the primary TinySepsis GRU model.

Fits everything on the VAL split, freezes it, and reports calibrated /
risk-controlled performance on test (internal, same-hospital) and
external_test (hospital B, different institution) without ever touching
those splits during fitting.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tinysepsis.eval.calibration import TemperatureScaler, IsotonicCalibrator  # noqa: E402
from tinysepsis.eval.conformal import calibrate_threshold, empirical_fpr, empirical_tpr  # noqa: E402
from tinysepsis.eval.metrics import (  # noqa: E402
    auroc, auprc, brier, expected_calibration_error, calibration_curve,
)

PRED_DIR = ROOT / "results" / "predictions"
OUT_DIR = ROOT / "results" / "calibration"
TAG = "tinysepsis"
ALARM_BUDGET_ALPHA = 0.10  # target: false-alarm rate <= 10% on non-septic hours


def to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    val = pl.read_parquet(PRED_DIR / f"{TAG}__val.parquet")
    test = pl.read_parquet(PRED_DIR / f"{TAG}__test.parquet")
    ext = pl.read_parquet(PRED_DIR / f"{TAG}__external_test.parquet")

    val_logit = to_logit(val["y_prob"].to_numpy())
    val_y = val["y_true"].to_numpy()

    print("Fitting temperature scaling on val...", flush=True)
    ts = TemperatureScaler().fit(val_logit, val_y.astype(np.float32))
    print(f"  T = {ts.T:.4f}", flush=True)

    val_prob_ts = ts.transform(val_logit)
    print("Fitting isotonic regression on val (post-temperature)...", flush=True)
    iso = IsotonicCalibrator().fit(val_prob_ts, val_y)

    results = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in [("val", val), ("test", test), ("external_test", ext)]:
        y = split_df["y_true"].to_numpy()
        p_raw = split_df["y_prob"].to_numpy()
        p_ts = ts.transform(to_logit(p_raw))
        p_iso = iso.transform(p_ts)

        row = {}
        for name, p in [("raw", p_raw), ("temperature", p_ts), ("isotonic", p_iso)]:
            row[name] = {
                "auroc": auroc(y, p),
                "auprc": auprc(y, p),
                "brier": brier(y, p),
                "ece": expected_calibration_error(y, p),
            }
        results[split_name] = row

        centers, obs, counts = calibration_curve(y, p_iso)
        np.savez(OUT_DIR / f"calcurve_{split_name}.npz", centers=centers, obs=obs, counts=counts)

    # --- Conformal risk control for the false-alarm budget, fit on val (isotonic-calibrated) ---
    val_p_final = iso.transform(ts.transform(val_logit))
    tau = calibrate_threshold(val_p_final, val_y, ALARM_BUDGET_ALPHA)
    print(f"Conformal threshold for alpha={ALARM_BUDGET_ALPHA}: tau={tau:.4f}", flush=True)

    conformal_results = {}
    for split_name, split_df in [("val", val), ("test", test), ("external_test", ext)]:
        y = split_df["y_true"].to_numpy()
        p_raw = split_df["y_prob"].to_numpy()
        p_final = iso.transform(ts.transform(to_logit(p_raw)))
        conformal_results[split_name] = {
            "tau": tau,
            "empirical_fpr": empirical_fpr(p_final, y, tau),
            "empirical_tpr": empirical_tpr(p_final, y, tau),
            "target_alpha": ALARM_BUDGET_ALPHA,
        }

    with open(OUT_DIR / "calibration_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / "conformal_results.json", "w") as f:
        json.dump(conformal_results, f, indent=2)
    with open(OUT_DIR / "calibrators.json", "w") as f:
        json.dump({"temperature": ts.T, "conformal_tau": tau, "alpha": ALARM_BUDGET_ALPHA}, f, indent=2)

    print(json.dumps(results, indent=2))
    print(json.dumps(conformal_results, indent=2))


if __name__ == "__main__":
    main()
