"""Train zero-shot clinical scores + Logistic Regression + XGBoost + LightGBM
baselines on the tabular (current-hour) feature representation, for the
label_6h primary task. Saves predictions (probabilities) for every split to
Parquet so downstream evaluation/calibration/plots can reuse them.
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tinysepsis.data.schema import NUMERIC_FEATURES  # noqa: E402
from tinysepsis.models.clinical_scores import qsofa_lite, news2_lite  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "enriched.parquet"
OUT_DIR = ROOT / "results" / "predictions"
CKPT_DIR = ROOT / "results" / "checkpoints"
LABEL = "label_6h"

TABULAR_COLS = (
    [f"{c}__z" for c in NUMERIC_FEATURES]
    + [f"{c}__mask" for c in NUMERIC_FEATURES]
    + [f"{c}__tslm__z" for c in NUMERIC_FEATURES]
    + [f"{c}__delta1__z" for c in NUMERIC_FEATURES]
    + ["Age__z", "Gender", "hours_since_admission"]
)


def load_split(df: pl.DataFrame, split: str):
    sub = df.filter(pl.col("split") == split)
    X = sub.select(TABULAR_COLS).to_numpy().astype(np.float32)
    y = sub[LABEL].to_numpy().astype(np.int32)
    pid = sub["patient_id"].to_numpy()
    hour = sub["ICULOS"].to_numpy()
    return X, y, pid, hour, sub


def save_predictions(name, split, pid, hour, y, prob, sub):
    out = pl.DataFrame(
        {
            "patient_id": pid,
            "ICULOS": hour,
            "y_true": y,
            "y_prob": prob.astype(np.float32),
        }
    )
    out_path = OUT_DIR / f"{name}__{split}.parquet"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    print(f"  saved {out_path} ({len(out)} rows)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42,
                    help="random_state for LogReg/XGBoost/LightGBM (qSOFA/NEWS2 are deterministic scores, unaffected)")
    p.add_argument("--tag-suffix", default="",
                    help="appended to model names in output filenames, e.g. '_seed1' for multi-seed runs")
    p.add_argument("--skip-clinical-scores", action="store_true",
                    help="skip qSOFA/NEWS2 (deterministic; redundant across seeds in a multi-seed sweep)")
    p.add_argument("--only", nargs="*", default=None, choices=["logreg", "xgboost", "lightgbm"],
                    help="train only these ML models (default: all three)")
    args = p.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading enriched.parquet...", flush=True)
    df = pl.read_parquet(DATA_PATH)
    df = df.with_columns(qsofa_lite(df), news2_lite(df))
    # Pre-first-observation hours have no vitals yet (forward-fill can't
    # look backward), so the score is null there; treat "not yet measured"
    # as "no concerning finding" (score 0), consistent with how leading
    # nulls are handled elsewhere in the pipeline (normalize.py).
    df = df.with_columns(pl.col("qsofa_lite").fill_null(0), pl.col("news2_lite").fill_null(0))

    splits = ["train", "val", "test", "external_test"]
    data = {s: load_split(df, s) for s in splits}

    if not args.skip_clinical_scores:
        # --- Clinical scores (no training needed; score itself = "risk") ---
        for name, col in [("qsofa", "qsofa_lite"), ("news2", "news2_lite")]:
            print(f"Scoring baseline: {name}", flush=True)
            for split in splits:
                sub = df.filter(pl.col("split") == split)
                prob = (sub[col].to_numpy().astype(np.float32))
                prob = prob / max(prob.max(), 1.0)  # crude [0,1] normalization for AUROC-safe scoring
                save_predictions(name, split, sub["patient_id"].to_numpy(), sub["ICULOS"].to_numpy(),
                                  sub[LABEL].to_numpy(), prob, sub)

    X_train, y_train, pid_train, hour_train, _ = data["train"]
    X_val, y_val, pid_val, hour_val, _ = data["val"]

    pos_rate = y_train.mean()
    print(f"Train positive rate ({LABEL}): {pos_rate:.4f}", flush=True)
    scale_pos_weight = (1 - pos_rate) / pos_rate

    models_to_run = args.only if args.only else ["logreg", "xgboost", "lightgbm"]

    if "logreg" in models_to_run:
        print("Training Logistic Regression...", flush=True)
        logreg = LogisticRegression(max_iter=500, class_weight="balanced", C=1.0, random_state=args.seed)
        logreg.fit(X_train, y_train)
        for split in splits:
            X, y, pid, hour, sub = data[split]
            prob = logreg.predict_proba(X)[:, 1]
            save_predictions(f"logreg{args.tag_suffix}", split, pid, hour, y, prob, sub)

    if "xgboost" in models_to_run:
        print("Training XGBoost...", flush=True)
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            n_jobs=-1,
            tree_method="hist",
            random_state=args.seed,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        for split in splits:
            X, y, pid, hour, sub = data[split]
            prob = xgb_model.predict_proba(X)[:, 1]
            save_predictions(f"xgboost{args.tag_suffix}", split, pid, hour, y, prob, sub)
        if not args.tag_suffix:
            xgb_model.save_model(str(CKPT_DIR / "xgboost.json"))

    if "lightgbm" in models_to_run:
        print("Training LightGBM...", flush=True)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            verbosity=-1,
            random_state=args.seed,
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        for split in splits:
            X, y, pid, hour, sub = data[split]
            prob = lgb_model.predict_proba(X)[:, 1]
            save_predictions(f"lightgbm{args.tag_suffix}", split, pid, hour, y, prob, sub)
        if not args.tag_suffix:
            lgb_model.booster_.save_model(str(CKPT_DIR / "lightgbm.txt"))

    print("All baselines trained and predictions saved.", flush=True)


if __name__ == "__main__":
    main()
