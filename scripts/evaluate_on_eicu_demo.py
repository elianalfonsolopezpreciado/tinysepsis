"""Zero-shot evaluation of the already-trained models (GRU/attention/CDE) on
data/processed/eicu_demo_enriched.parquet's "eicu_demo_external" split --
2,520 ICU stays across 186 distinct hospitals (per patient.csv's own
hospitalid field; see scripts/build_eicu_demo_dataset.py's module docstring
for how this dataset was built), none of which overlap the PhysioNet
Challenge 2019 Hospital A/B this project trains and validates on.

IMPORTANT LABEL CAVEAT: only 1 of 2,520 patients meets this project's
simplified Sepsis-3 criteria (6 positive hours total), because eICU
Demo's microLab (microbiology culture) table is extremely sparse -- only
64/2,520 patients have any culture record at all, and the overlap between
those and antibiotic-treated patients (561/2,520) is just 1 patient. This
is a genuine property of the DEMO subsample (the full, credentialed eICU
reports ~4.6% Sepsis-3 prevalence in the literature, so this is not
expected to be a problem once real eICU access is obtained), not a bug in
the labeling pipeline. Consequently: AUROC/AUPRC on this split is
NOT a meaningful sensitivity estimate (n=1 positive patient) and is
reported here only for completeness, clearly flagged -- do not quote it
as a generalization result. What IS meaningful with this data: whether
the ALREADY-CALIBRATED false-alarm rate (fit on Challenge 2019's val
split, per scripts/calibrate_and_conformal.py's alpha=0.10 budget) holds
up on the ~2,519 negative-labeled patients spread across 186 hospitals
never seen during training or calibration -- a real specificity/alarm-
burden generalization test, independent of the sepsis-detection question.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tinysepsis.data.dataset import TinySepsisDataset  # noqa: E402
from tinysepsis.models.tiny_sepsis import TinySepsisModel  # noqa: E402
from tinysepsis.models.tiny_sepsis_attn import TinySepsisAttnModel  # noqa: E402
from tinysepsis.models.tiny_sepsis_cde import TinySepsisCDEModel  # noqa: E402
from tinysepsis.eval.calibration import TemperatureScaler, IsotonicCalibrator  # noqa: E402
from tinysepsis.eval.conformal import calibrate_threshold, empirical_fpr  # noqa: E402
from tinysepsis.eval.metrics import auroc, auprc, alarms_per_1000_patient_hours  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "eicu_demo_enriched.parquet"
CKPT_DIR = ROOT / "results" / "checkpoints"
PRED_DIR = ROOT / "results" / "predictions"
TABLE_DIR = ROOT / "results" / "tables"
ALARM_BUDGET_ALPHA = 0.10


def to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def load_gru(device):
    ckpt = torch.load(CKPT_DIR / "tinysepsis_best.pt", map_location=device, weights_only=False)
    args = ckpt["args"]
    model = TinySepsisModel(num_dynamic_features=ckpt["num_dynamic"], num_static_features=ckpt["num_static"],
                             hidden_size=args["hidden_size"], num_layers=args["num_layers"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_attn(device):
    path = CKPT_DIR / "tinysepsis_attn_seed0_best.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = TinySepsisAttnModel(num_dynamic_features=ckpt["num_dynamic"], num_static_features=ckpt["num_static"],
                                 d_model=args["d_model"], nhead=args["nhead"], num_layers=args["num_layers"],
                                 dim_feedforward=args["dim_feedforward"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_cde(device):
    path = CKPT_DIR / "cde_seed0_best.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = TinySepsisCDEModel(num_dynamic_features=ckpt["num_dynamic"], num_static_features=ckpt["num_static"],
                                hidden_size=args["hidden_size"], mlp_hidden=args["mlp_hidden"],
                                solver=args["solver"], solver_options={"step_size": args["solver_step_size"]}).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def run_inference(model, loader, device):
    all_probs, all_labels, all_pids = [], [], []
    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            static = batch["static"].to(device)
            logits = model(seq, static)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch["label"].numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    return y_true, y_prob


def fit_calibration(tag: str):
    val = pl.read_parquet(PRED_DIR / f"{tag}__val.parquet")
    val_y = val["y_true"].to_numpy()
    val_logit = to_logit(val["y_prob"].to_numpy())
    ts = TemperatureScaler().fit(val_logit, val_y.astype(np.float32))
    val_p_ts = ts.transform(val_logit)
    iso = IsotonicCalibrator().fit(val_p_ts, val_y)
    val_p_final = iso.transform(val_p_ts)
    tau = calibrate_threshold(val_p_final, val_y, ALARM_BUDGET_ALPHA)
    return ts, iso, tau


def evaluate(name, model, val_tag, device):
    print(f"=== {name} ===", flush=True)
    ds = TinySepsisDataset(DATA_PATH, "eicu_demo_external", seq_len=24, label_col="label_6h")
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    y_true, y_prob_raw = run_inference(model, loader, device)

    ts, iso, tau = fit_calibration(val_tag)
    p_final = iso.transform(ts.transform(to_logit(y_prob_raw)))
    y_bin = (p_final >= tau).astype(int)

    n_pos_rows = int(y_true.sum())
    result = {
        "model": name,
        "n_rows": len(y_true),
        "n_patients": len(ds.patients),
        "n_positive_rows": n_pos_rows,
        "raw_auroc": auroc(y_true, y_prob_raw) if n_pos_rows > 0 else None,
        "raw_auprc": auprc(y_true, y_prob_raw) if n_pos_rows > 0 else None,
        "conformal_tau": float(tau),
        "empirical_fpr": empirical_fpr(p_final, y_true, tau),
        "alarms_per_1000h": alarms_per_1000_patient_hours(y_bin, len(y_true)),
        "mean_calibrated_prob": float(p_final.mean()),
    }
    print(json.dumps(result, indent=2), flush=True)

    pids = [ds.index[i][0] for i in range(len(ds))]
    hosp_ids = [pid.split("_")[1] for pid in pids]
    per_hosp = pl.DataFrame({"hospitalid": hosp_ids, "alarm": y_bin}).group_by("hospitalid").agg(
        pl.col("alarm").mean().alias("alarm_rate"), pl.len().alias("n_hours")
    )
    result["per_hospital_alarm_rate_std"] = float(per_hosp["alarm_rate"].std())
    result["per_hospital_alarm_rate_mean"] = float(per_hosp["alarm_rate"].mean())
    result["n_hospitals"] = per_hosp.height
    return result, per_hosp


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    results = []
    gru = load_gru(device)
    r, per_hosp = evaluate("tinysepsis_gru", gru, "tinysepsis", device)
    results.append(r)
    per_hosp.write_csv(TABLE_DIR / "eicu_demo_gru_per_hospital.csv")

    attn = load_attn(device)
    if attn is not None:
        r2, per_hosp2 = evaluate("tinysepsis_attn", attn, "tinysepsis_attn_seed0", device)
        results.append(r2)
        per_hosp2.write_csv(TABLE_DIR / "eicu_demo_attn_per_hospital.csv")

    cde = load_cde(device)
    if cde is not None:
        r3, per_hosp3 = evaluate("tinysepsis_cde", cde, "cde_seed0", device)
        results.append(r3)
        per_hosp3.write_csv(TABLE_DIR / "eicu_demo_cde_per_hospital.csv")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLE_DIR / "eicu_demo_zeroshot_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote", TABLE_DIR / "eicu_demo_zeroshot_eval.json", flush=True)


if __name__ == "__main__":
    main()
