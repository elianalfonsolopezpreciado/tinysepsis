"""Statistical significance of the Priority 1 hyperparameter sweep result
(scripts/hyperparam_sweep.py must be run first): is the winning candidate
(dropout=0.3) actually different from baseline (dropout=0.1), or within
noise? Same Welch's t-test / Mann-Whitney U / Cohen's d protocol as
scripts/multiseed_stats.py, applied to the 5-seed Stage B confirmation
results, on both external AUROC and the internal-to-external gap (the
sweep's actual optimization target).
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = ROOT / "results" / "tables"


def cohens_d(a, b):
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    if pooled_std < 1e-12:
        return float("inf") if a.mean() != b.mean() else 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def compare(df, metric, winner, baseline="baseline"):
    a = df.filter(pl.col("config") == winner)[metric].to_numpy()
    b = df.filter(pl.col("config") == baseline)[metric].to_numpy()
    alt = "greater" if metric == "external_auroc" else "less"  # winner should have higher AUROC, lower gap
    t, p_welch = stats.ttest_ind(a, b, equal_var=False)
    u, p_mw = stats.mannwhitneyu(a, b, alternative=alt)
    return {
        f"{winner}_{metric}": a.tolist(),
        f"{baseline}_{metric}": b.tolist(),
        "welch_t": float(t),
        "welch_p": float(p_welch),
        "mannwhitney_u": float(u),
        "mannwhitney_p": float(p_mw),
        "cohens_d": cohens_d(a, b),
    }


def main():
    df = pl.read_csv(TABLE_DIR / "hyperparam_confirm_raw.csv")
    winner = "dropout03"

    results = {
        "winner": winner,
        "external_auroc": compare(df, "external_auroc", winner),
        "gap": compare(df, "gap", winner),
    }

    out_path = TABLE_DIR / "hyperparam_sweep_stats.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
