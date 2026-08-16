"""Multi-seed comparison of the Neural CDE encoder (Priority 2 of
regulatory/model_improvement_roadmap.md) against the GRU (TinySepsis),
on the exact same 5-seed protocol scripts/run_multiseed.py already used
for the GRU-vs-tabular-baselines comparison, so the CDE-vs-GRU numbers
are directly comparable rather than a one-off single-seed anecdote.

Requires scripts/run_multiseed.py to have already produced
results/tables/multiseed_raw.csv (for the GRU/"tinysepsis" seed rows) --
this script does not retrain the GRU, only the CDE.
"""
import json
import subprocess
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tinysepsis.eval.metrics import auroc  # noqa: E402

PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
PRED_DIR = ROOT / "results" / "predictions"
TABLE_DIR = ROOT / "results" / "tables"

SEEDS = [0, 1, 2, 3, 4]


def cohens_d(a, b):
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    if pooled_std < 1e-12:
        return float("inf") if a.mean() != b.mean() else 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def run(cmd, label):
    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print(result.stdout[-2000:], flush=True)
    if result.returncode != 0:
        print(result.stderr[-2000:], flush=True)
        print(f"note: {label} exited {result.returncode}; will verify via output files below", flush=True)


def main():
    for seed in SEEDS:
        tag = f"cde_seed{seed}"
        test_path = PRED_DIR / f"{tag}__test.parquet"
        ext_path = PRED_DIR / f"{tag}__external_test.parquet"
        if test_path.exists() and ext_path.exists():
            print(f"{tag}: already on disk, skipping", flush=True)
            continue
        run(
            [PY, str(ROOT / "scripts" / "train_model_cde.py"), "--tag", tag, "--seed", str(seed)],
            tag,
        )

    rows = []
    for seed in SEEDS:
        tag = f"cde_seed{seed}"
        test_path = PRED_DIR / f"{tag}__test.parquet"
        ext_path = PRED_DIR / f"{tag}__external_test.parquet"
        if not test_path.exists() or not ext_path.exists():
            print(f"MISSING: {tag}", flush=True)
            continue
        test_pred = pl.read_parquet(test_path)
        ext_pred = pl.read_parquet(ext_path)
        test_auroc = auroc(test_pred["y_true"].to_numpy(), test_pred["y_prob"].to_numpy())
        ext_auroc = auroc(ext_pred["y_true"].to_numpy(), ext_pred["y_prob"].to_numpy())
        rows.append({"model": "cde", "seed": seed, "test_auroc": test_auroc,
                     "external_auroc": ext_auroc, "gap": test_auroc - ext_auroc})

    cde_df = pl.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cde_df.write_csv(TABLE_DIR / "multiseed_cde_raw.csv")
    print(cde_df, flush=True)

    gru_raw_path = TABLE_DIR / "multiseed_raw.csv"
    if not gru_raw_path.exists():
        print(f"WARNING: {gru_raw_path} not found -- cannot compare against GRU. Run scripts/run_multiseed.py first.", flush=True)
        return

    gru_df = pl.read_csv(gru_raw_path).filter(pl.col("model") == "tinysepsis")
    combined = pl.concat([gru_df, cde_df.select(gru_df.columns)])
    combined.write_csv(TABLE_DIR / "multiseed_cde_vs_gru_raw.csv")

    summary = (
        combined.group_by("model")
        .agg(
            pl.col("test_auroc").mean().alias("test_mean"), pl.col("test_auroc").std().alias("test_std"),
            pl.col("external_auroc").mean().alias("external_mean"), pl.col("external_auroc").std().alias("external_std"),
            pl.col("gap").mean().alias("gap_mean"), pl.col("gap").std().alias("gap_std"),
            pl.col("seed").count().alias("n_seeds"),
        )
        .sort("model")
    )
    summary.write_csv(TABLE_DIR / "multiseed_cde_vs_gru_summary.csv")
    print(summary, flush=True)

    gru_gap = gru_df["gap"].to_numpy()
    cde_gap = cde_df["gap"].to_numpy()
    gru_ext = gru_df["external_auroc"].to_numpy()
    cde_ext = cde_df["external_auroc"].to_numpy()

    if len(gru_gap) == len(cde_gap) and len(gru_gap) > 0:
        t_gap, p_gap = stats.ttest_ind(cde_gap, gru_gap, equal_var=False)
        u_gap, pmw_gap = stats.mannwhitneyu(cde_gap, gru_gap, alternative="less")
        t_ext, p_ext = stats.ttest_ind(cde_ext, gru_ext, equal_var=False)
        u_ext, pmw_ext = stats.mannwhitneyu(cde_ext, gru_ext, alternative="greater")
        stats_out = {
            "gap": {"welch_t": float(t_gap), "welch_p": float(p_gap),
                    "mannwhitney_u": float(u_gap), "mannwhitney_p": float(pmw_gap),
                    "cohens_d": cohens_d(cde_gap, gru_gap)},
            "external_auroc": {"welch_t": float(t_ext), "welch_p": float(p_ext),
                                "mannwhitney_u": float(u_ext), "mannwhitney_p": float(pmw_ext),
                                "cohens_d": cohens_d(cde_ext, gru_ext)},
        }
        with open(TABLE_DIR / "multiseed_cde_vs_gru_stats.json", "w") as f:
            json.dump(stats_out, f, indent=2)
        print(json.dumps(stats_out, indent=2), flush=True)

    print("CDE multi-seed comparison complete.", flush=True)


if __name__ == "__main__":
    main()
