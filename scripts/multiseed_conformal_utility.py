"""Tier 1 item 3 of regulatory/model_improvement_roadmap.md: the paper's
utility-at-conformal-threshold finding (Section 9.4 -- TinySepsis's
calibrated threshold transfers worse under distribution shift than its
raw ranking does) was originally a SINGLE-RUN result, same as the
hyperparameter choice Priority 1 fixed. This is the mechanical extension
promised there: refit calibration + conformal risk control independently
per seed (val-only, exactly as scripts/calibrate_and_conformal.py does for
the single run) on all 5 of scripts/run_multiseed.py's already-trained
TinySepsis seeds, and test whether the internal-to-external utility drop
is a consistent, statistically-supported effect or an artifact of the one
seed the paper originally reported.

No retraining: reuses tinysepsis_seed{0-4} predictions already on disk.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tinysepsis.eval.calibration import TemperatureScaler, IsotonicCalibrator  # noqa: E402
from tinysepsis.eval.conformal import calibrate_threshold, empirical_fpr  # noqa: E402
from tinysepsis.eval.utility import patient_utility, normalized_utility  # noqa: E402
from tinysepsis.eval.metrics import alarms_per_1000_patient_hours, auroc  # noqa: E402

PRED_DIR = ROOT / "results" / "predictions"
DATA_PATH = ROOT / "data" / "processed" / "enriched.parquet"
TABLE_DIR = ROOT / "results" / "tables"

SEEDS = [0, 1, 2, 3, 4]
ALARM_BUDGET_ALPHA = 0.10


def to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def cohens_d(a, b):
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    if pooled_std < 1e-12:
        return float("inf") if a.mean() != b.mean() else 0.0
    return float((a.mean() - b.mean()) / pooled_std)


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
    patient_meta = df_full.group_by("patient_id").agg(pl.col("t_susp").first())
    patient_meta_d = {row["patient_id"]: row for row in patient_meta.to_dicts()}

    rows = []
    for seed in SEEDS:
        tag = f"tinysepsis_seed{seed}"
        val_path = PRED_DIR / f"{tag}__val.parquet"
        test_path = PRED_DIR / f"{tag}__test.parquet"
        ext_path = PRED_DIR / f"{tag}__external_test.parquet"
        if not (val_path.exists() and test_path.exists() and ext_path.exists()):
            print(f"MISSING: {tag}", flush=True)
            continue

        val = pl.read_parquet(val_path)
        val_y = val["y_true"].to_numpy()
        val_logit = to_logit(val["y_prob"].to_numpy())

        ts = TemperatureScaler().fit(val_logit, val_y.astype(np.float32))
        val_p_ts = ts.transform(val_logit)
        iso = IsotonicCalibrator().fit(val_p_ts, val_y)
        val_p_final = iso.transform(val_p_ts)

        tau = calibrate_threshold(val_p_final, val_y, ALARM_BUDGET_ALPHA)
        print(f"seed={seed}: T={ts.T:.3f} tau={tau:.4f}", flush=True)

        for split_name, path in [("test", test_path), ("external_test", ext_path)]:
            pred = pl.read_parquet(path)
            y = pred["y_true"].to_numpy()
            p_raw = pred["y_prob"].to_numpy()
            p_final = iso.transform(ts.transform(to_logit(p_raw)))
            y_bin = (p_final >= tau).astype(int)

            fpr = empirical_fpr(p_final, y, tau)
            alarms = alarms_per_1000_patient_hours(y_bin, len(y))
            util = compute_utility(pred, y_bin, patient_meta_d)
            raw_auroc = auroc(y, p_raw)

            rows.append({
                "seed": seed, "split": split_name, "tau": tau, "temperature": ts.T,
                "raw_auroc": raw_auroc, "empirical_fpr": fpr,
                "alarms_per_1000h": alarms, "normalized_utility": util,
            })

    df = pl.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.write_csv(TABLE_DIR / "multiseed_conformal_utility_raw.csv")
    print(df, flush=True)

    summary = (
        df.group_by("split")
        .agg(
            pl.col("raw_auroc").mean().alias("auroc_mean"), pl.col("raw_auroc").std().alias("auroc_std"),
            pl.col("empirical_fpr").mean().alias("fpr_mean"), pl.col("empirical_fpr").std().alias("fpr_std"),
            pl.col("alarms_per_1000h").mean().alias("alarms_mean"), pl.col("alarms_per_1000h").std().alias("alarms_std"),
            pl.col("normalized_utility").mean().alias("utility_mean"), pl.col("normalized_utility").std().alias("utility_std"),
            pl.col("seed").count().alias("n_seeds"),
        )
        .sort("split")
    )
    summary.write_csv(TABLE_DIR / "multiseed_conformal_utility_summary.csv")
    print(summary, flush=True)

    test_util = df.filter(pl.col("split") == "test")["normalized_utility"].to_numpy()
    ext_util = df.filter(pl.col("split") == "external_test")["normalized_utility"].to_numpy()
    test_fpr = df.filter(pl.col("split") == "test")["empirical_fpr"].to_numpy()
    ext_fpr = df.filter(pl.col("split") == "external_test")["empirical_fpr"].to_numpy()

    stats_out = {}
    if len(test_util) == len(ext_util) and len(test_util) > 1:
        t_u, p_u = stats.ttest_rel(ext_util, test_util)
        t_f, p_f = stats.ttest_rel(ext_fpr, test_fpr)
        stats_out = {
            "utility_drop": {
                "test_utility": test_util.tolist(), "external_utility": ext_util.tolist(),
                "paired_t": float(t_u), "paired_p": float(p_u), "cohens_d": cohens_d(ext_util, test_util),
            },
            "fpr_shift": {
                "test_fpr": test_fpr.tolist(), "external_fpr": ext_fpr.tolist(),
                "paired_t": float(t_f), "paired_p": float(p_f), "cohens_d": cohens_d(ext_fpr, test_fpr),
            },
            "alarm_budget_alpha": ALARM_BUDGET_ALPHA,
        }
        with open(TABLE_DIR / "multiseed_conformal_utility_stats.json", "w") as f:
            json.dump(stats_out, f, indent=2)
        print(json.dumps(stats_out, indent=2), flush=True)

    print("Multi-seed conformal utility analysis complete.", flush=True)


if __name__ == "__main__":
    main()
