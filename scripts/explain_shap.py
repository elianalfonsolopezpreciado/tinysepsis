"""SHAP explainability analysis for the XGBoost baseline (label_6h task).

Produces a summary bar plot of mean |SHAP value| per feature group (value,
mask, time-since-last-measurement, delta) aggregated across the 34 raw
clinical channels, on a random sample of the internal test split, plus the
raw per-feature SHAP values as a Parquet for further analysis.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tinysepsis.data.schema import NUMERIC_FEATURES  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "enriched.parquet"
MODEL_PATH = ROOT / "results" / "checkpoints" / "xgboost.json"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

TABULAR_COLS = (
    [f"{c}__z" for c in NUMERIC_FEATURES]
    + [f"{c}__mask" for c in NUMERIC_FEATURES]
    + [f"{c}__tslm__z" for c in NUMERIC_FEATURES]
    + [f"{c}__delta1__z" for c in NUMERIC_FEATURES]
    + ["Age__z", "Gender", "hours_since_admission"]
)
N_SAMPLE = 3000
SEED = 42


def channel_group(col: str) -> str:
    if col.endswith("__mask"):
        return "mask"
    if "__tslm" in col:
        return "time_since_last_measurement"
    if "__delta1" in col:
        return "delta_1h"
    if col in ("Age__z", "Gender", "hours_since_admission"):
        return "static/derived"
    return "value"


def main():
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    df = pl.read_parquet(DATA_PATH).filter(pl.col("split") == "test")
    n = min(N_SAMPLE, df.height)
    df_sample = df.sample(n=n, seed=SEED)
    X = df_sample.select(TABULAR_COLS).to_numpy().astype(np.float32)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs = np.abs(shap_values).mean(axis=0)
    group_importance = {}
    for col, val in zip(TABULAR_COLS, mean_abs):
        g = channel_group(col)
        group_importance[g] = group_importance.get(g, 0.0) + float(val)

    groups = sorted(group_importance, key=group_importance.get, reverse=True)
    values = [group_importance[g] for g in groups]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.barh(groups, values)
    ax.set_xlabel("Summed mean |SHAP value| across 34 channels")
    ax.set_title("Feature-group importance (XGBoost, test split)")
    ax.invert_yaxis()
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "shap_group_importance.png", dpi=150)
    plt.close(fig)

    top_idx = np.argsort(mean_abs)[::-1][:15]
    top_features = pl.DataFrame({
        "feature": [TABULAR_COLS[i] for i in top_idx],
        "mean_abs_shap": [float(mean_abs[i]) for i in top_idx],
    })
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    top_features.write_csv(TABLE_DIR / "shap_top_features.csv")
    print(top_features, flush=True)
    print("wrote results/figures/shap_group_importance.png and results/tables/shap_top_features.csv", flush=True)


if __name__ == "__main__":
    main()
