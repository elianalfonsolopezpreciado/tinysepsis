"""Statistical tests on the multi-seed internal-to-external AUROC gap
(scripts/run_multiseed.py must be run first). Persists Welch's t-test,
Mann-Whitney U, and Cohen's d for TinySepsis vs. each tabular baseline to
results/tables/multiseed_stats.json, so the significance numbers quoted in
the paper are reproducible from a script rather than a one-off console check.
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


def main():
    df = pl.read_csv(TABLE_DIR / "multiseed_raw.csv")
    tiny = df.filter(pl.col("model") == "tinysepsis")["gap"].to_numpy()

    results = {"tinysepsis_gaps": tiny.tolist(), "n_seeds": len(tiny)}
    baselines = ["xgboost", "lightgbm", "logreg"]
    pooled = []

    for name in baselines:
        other = df.filter(pl.col("model") == name)["gap"].to_numpy()
        pooled.append(other)
        t, p_welch = stats.ttest_ind(tiny, other, equal_var=False)
        u, p_mw = stats.mannwhitneyu(tiny, other, alternative="less")
        results[name] = {
            "gaps": other.tolist(),
            "welch_t": float(t),
            "welch_p": float(p_welch),
            "mannwhitney_u": float(u),
            "mannwhitney_p": float(p_mw),
            "cohens_d": cohens_d(tiny, other),
        }

    pooled_all = np.concatenate(pooled)
    t, p_welch = stats.ttest_ind(tiny, pooled_all, equal_var=False)
    results["pooled_tabular"] = {
        "n": len(pooled_all),
        "welch_t": float(t),
        "welch_p": float(p_welch),
        "cohens_d": cohens_d(tiny, pooled_all),
    }

    n_comparisons = len(baselines)
    results["bonferroni_alpha"] = 0.05 / n_comparisons

    out_path = TABLE_DIR / "multiseed_stats.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
