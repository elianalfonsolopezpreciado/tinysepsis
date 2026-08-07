"""Report clinical utility and alarm burden at the conformal-calibrated
threshold for ALL SIX models, not just the fixed 85%-sensitivity threshold
used in Tables 1-2. The fixed-sensitivity threshold exists purely to make
Tables 1-2 comparable across models on a shared operating rule; it is not a
deployment recommendation. This script answers the natural follow-up
question -- reviewers correctly asked for it -- of whether utility looks
less catastrophic at the threshold this paper actually proposes for
deployment (Section 6's conformal risk control, alpha=0.10), fit
per-model on the validation split exactly as in scripts/calibrate_and_conformal.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tinysepsis.eval.calibration import TemperatureScaler, IsotonicCalibrator  # noqa: E402
from tinysepsis.eval.conformal import calibrate_threshold, empirical_fpr  # noqa: E402
from tinysepsis.eval.utility import patient_utility, normalized_utility  # noqa: E402
from tinysepsis.eval.metrics import alarms_per_1000_patient_hours  # noqa: E402

PRED_DIR = ROOT / "results" / "predictions"
DATA_PATH = ROOT / "data" / "processed" / "enriched.parquet"
TABLE_DIR = ROOT / "results" / "tables"
OUT_DIR = ROOT / "results" / "calibration"

MODELS = ["qsofa", "news2", "logreg", "xgboost", "lightgbm", "tinysepsis"]
ALARM_BUDGET_ALPHA = 0.10


def to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def compute_utility(pred_df, y_bin, patient_meta_d):
    pred_pd = pred_df.with_columns(pl.Series("y_bin", y_bin))
    total_u, total_best, total_inaction = 0.0, 0.0, 0.0
    for pid, grp in pred_pd.group_by("patient_id", maintain_order=True):
        pid = pid[0] if isinstance(pid, tuple) else pid
        meta = patient_meta_d.get(pid)
        if meta is None:
            continue
        t_susp = meta["t_susp"]
        is_septic = t_susp is not None
        grp_sorted = grp.sort("ICULOS")
        yb = grp_sorted["y_bin"].to_numpy()
        hours = grp_sorted["ICULOS"].to_numpy()
        if is_septic:
            tau_arr = hours.astype(np.float64) - float(t_susp)
            total_u += patient_utility(yb, True, tau_arr)
            total_best += patient_utility(np.ones_like(yb), True, tau_arr)
            total_inaction += patient_utility(np.zeros_like(yb), True, tau_arr)
        else:
            total_u += patient_utility(yb, False)
            total_best += patient_utility(np.zeros_like(yb), False)
            total_inaction += patient_utility(np.zeros_like(yb), False)
    return normalized_utility(total_u, total_best, total_inaction)


def main():
    df_full = pl.read_parquet(DATA_PATH)
    patient_meta = (
        df_full.group_by("patient_id")
        .agg(pl.col("t_susp").first())
    )
    patient_meta_d = {row["patient_id"]: row for row in patient_meta.to_dicts()}

    rows = []
    per_model_calibrators = {}
    for model in MODELS:
        val_path = PRED_DIR / f"{model}__val.parquet"
        test_path = PRED_DIR / f"{model}__test.parquet"
        ext_path = PRED_DIR / f"{model}__external_test.parquet"
        if not (val_path.exists() and test_path.exists() and ext_path.exists()):
            print(f"skip {model}: missing prediction files", flush=True)
            continue

        val = pl.read_parquet(val_path)
        val_y = val["y_true"].to_numpy()
        val_logit = to_logit(val["y_prob"].to_numpy())

        ts = TemperatureScaler().fit(val_logit, val_y.astype(np.float32))
        val_p_ts = ts.transform(val_logit)
        iso = IsotonicCalibrator().fit(val_p_ts, val_y)
        val_p_final = iso.transform(val_p_ts)

        tau = calibrate_threshold(val_p_final, val_y, ALARM_BUDGET_ALPHA)
        per_model_calibrators[model] = {"temperature": ts.T, "conformal_tau": tau, "alpha": ALARM_BUDGET_ALPHA}
        print(f"{model}: T={ts.T:.3f} tau={tau:.4f}", flush=True)

        for split_name, path in [("test", test_path), ("external_test", ext_path)]:
            pred = pl.read_parquet(path)
            y = pred["y_true"].to_numpy()
            p_final = iso.transform(ts.transform(to_logit(pred["y_prob"].to_numpy())))
            y_bin = (p_final >= tau).astype(int)

            fpr = empirical_fpr(p_final, y, tau)
            alarms = alarms_per_1000_patient_hours(y_bin, len(y))
            util = compute_utility(pred, y_bin, patient_meta_d)

            rows.append({
                "model": model, "split": split_name, "tau": tau,
                "empirical_fpr": fpr, "alarms_per_1000h": alarms,
                "normalized_utility": util,
            })

    out_df = pl.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(TABLE_DIR / "utility_at_conformal_threshold.csv")
    print(out_df, flush=True)

    with open(OUT_DIR / "per_model_calibrators.json", "w") as f:
        json.dump(per_model_calibrators, f, indent=2)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
