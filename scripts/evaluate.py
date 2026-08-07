"""Consolidate all baseline + TinySepsis predictions into final results tables
and figures: discrimination, calibration, clinical utility, alarms/1000h,
lead time, subgroup breakdown, decision curves.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tinysepsis.eval.metrics import (  # noqa: E402
    auroc, auprc, brier, expected_calibration_error, sensitivity_at_specificity,
    specificity_at_sensitivity, alarms_per_1000_patient_hours, decision_curve_net_benefit,
    lead_time_hours, calibration_curve,
)
from tinysepsis.eval.utility import patient_utility, normalized_utility  # noqa: E402

PRED_DIR = ROOT / "results" / "predictions"
DATA_PATH = ROOT / "data" / "processed" / "enriched.parquet"
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["qsofa", "news2", "logreg", "xgboost", "lightgbm", "tinysepsis"]
SPLITS = ["test", "external_test"]
SENS_TARGETS = [0.80, 0.85, 0.90]
SPEC_TARGETS = [0.90, 0.95]


def load_pred(model, split):
    path = PRED_DIR / f"{model}__{split}.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


def main():
    df_full = pl.read_parquet(DATA_PATH)
    patient_meta = (
        df_full.group_by("patient_id")
        .agg(
            pl.col("t_susp").first(),
            pl.col("hospital").first(),
            pl.col("Age").first(),
            pl.col("Gender").first(),
        )
    )
    patient_meta_d = {row["patient_id"]: row for row in patient_meta.to_dicts()}

    summary_rows = []
    for model in MODELS:
        for split in SPLITS:
            pred = load_pred(model, split)
            if pred is None:
                print(f"skip {model}/{split}: no predictions", flush=True)
                continue
            y = pred["y_true"].to_numpy()
            p = pred["y_prob"].to_numpy()
            if len(np.unique(y)) < 2:
                print(f"skip {model}/{split}: single class", flush=True)
                continue

            row = {
                "model": model, "split": split,
                "n": len(y), "prevalence": float(y.mean()),
                "auroc": auroc(y, p), "auprc": auprc(y, p),
                "brier": brier(y, p), "ece": expected_calibration_error(y, p),
            }
            for s in SENS_TARGETS:
                sens, thr = sensitivity_at_specificity(y, p, s)
                row[f"sens_at_spec{int(s*100)}"] = sens
            for s in SPEC_TARGETS:
                spec, thr = specificity_at_sensitivity(y, p, s)
                row[f"spec_at_sens{int(s*100)}"] = spec

            # Alarms/1000 patient-hours + utility + lead time at the
            # threshold achieving 85% sensitivity (a clinically-motivated
            # fixed operating point, consistent across models for comparison).
            spec_at_sens85, thr_at_sens85 = specificity_at_sensitivity(y, p, 0.85)
            y_bin = (p >= thr_at_sens85).astype(int)
            row["alarms_per_1000h"] = alarms_per_1000_patient_hours(y_bin, len(y))

            # Utility + lead time, computed per patient
            pred_pd = pred.with_columns(pl.Series("y_bin", y_bin))
            total_u, total_best, total_inaction, lead_times = 0.0, 0.0, 0.0, []
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
                    best_yb = np.ones_like(yb)
                    inaction_yb = np.zeros_like(yb)
                    total_best += patient_utility(best_yb, True, tau_arr)
                    total_inaction += patient_utility(inaction_yb, True, tau_arr)
                    alarm_hours = hours[yb == 1].tolist()
                    lt = lead_time_hours(alarm_hours, int(t_susp))
                    if lt is not None:
                        lead_times.append(lt)
                else:
                    total_u += patient_utility(yb, False)
                    total_best += patient_utility(np.zeros_like(yb), False)
                    total_inaction += patient_utility(np.zeros_like(yb), False)

            row["normalized_utility"] = normalized_utility(total_u, total_best, total_inaction)
            row["mean_lead_time_h"] = float(np.mean(lead_times)) if lead_times else None
            row["n_alarmed_septic_patients"] = len(lead_times)

            summary_rows.append(row)

    summary_df = pl.DataFrame(summary_rows)
    summary_df.write_parquet(TABLE_DIR / "main_results.parquet")
    summary_df.write_csv(TABLE_DIR / "main_results.csv")
    print(summary_df, flush=True)

    # --- Calibration curves figure (test split, all models) ---
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    for model in MODELS:
        pred = load_pred(model, "test")
        if pred is None:
            continue
        y, p = pred["y_true"].to_numpy(), pred["y_prob"].to_numpy()
        centers, obs, counts = calibration_curve(y, p)
        if len(centers) > 0:
            ax.plot(centers, obs, marker="o", markersize=3, label=model)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration (test, internal hospital A)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration_curves.png", dpi=150)
    plt.close(fig)

    # --- Decision curve (main model vs baselines, test split) ---
    fig, ax = plt.subplots(figsize=(5, 5))
    thresholds = np.linspace(0.01, 0.5, 50)
    for model in MODELS:
        pred = load_pred(model, "test")
        if pred is None:
            continue
        y, p = pred["y_true"].to_numpy(), pred["y_prob"].to_numpy()
        dc = decision_curve_net_benefit(y, p, thresholds)
        ax.plot(thresholds, dc["net_benefit_model"], label=model)
    ax.plot(thresholds, dc["net_benefit_all"], "k--", label="treat all")
    ax.plot(thresholds, dc["net_benefit_none"], "k:", label="treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis (test)")
    ax.legend(fontsize=7)
    ax.set_ylim(-0.05, max(0.1, float(np.nanmax(dc["net_benefit_model"])) + 0.02))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "decision_curve.png", dpi=150)
    plt.close(fig)

    # --- Subgroup analysis (age, sex) for the main model ---
    pred = load_pred("tinysepsis", "test")
    if pred is not None:
        pred_meta = pred.with_columns(
            pl.col("patient_id").map_elements(lambda pid: patient_meta_d.get(pid, {}).get("Age"), return_dtype=pl.Float64).alias("Age"),
            pl.col("patient_id").map_elements(lambda pid: patient_meta_d.get(pid, {}).get("Gender"), return_dtype=pl.Float64).alias("Gender"),
        )
        subgroup_rows = []
        age_bins = [(0, 50), (50, 65), (65, 80), (80, 200)]
        for lo, hi in age_bins:
            sub = pred_meta.filter((pl.col("Age") >= lo) & (pl.col("Age") < hi))
            y, p = sub["y_true"].to_numpy(), sub["y_prob"].to_numpy()
            if len(np.unique(y)) < 2:
                continue
            subgroup_rows.append({"subgroup": f"age_{lo}_{hi}", "n": len(y), "auroc": auroc(y, p), "prevalence": float(y.mean())})
        for g in [0, 1]:
            sub = pred_meta.filter(pl.col("Gender") == g)
            y, p = sub["y_true"].to_numpy(), sub["y_prob"].to_numpy()
            if len(np.unique(y)) < 2:
                continue
            subgroup_rows.append({"subgroup": f"gender_{g}", "n": len(y), "auroc": auroc(y, p), "prevalence": float(y.mean())})
        subgroup_df = pl.DataFrame(subgroup_rows)
        subgroup_df.write_csv(TABLE_DIR / "subgroup_analysis.csv")
        print(subgroup_df, flush=True)

    print("Evaluation complete.", flush=True)


if __name__ == "__main__":
    main()
