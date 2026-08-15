"""Multi-seed robustness check: retrain TinySepsis and the ML baselines
(LogReg, XGBoost, LightGBM) under N different random seeds, then report
mean +/- std internal (test) and external (external_test) AUROC per model,
and the internal-to-external gap, so the paper's central claim ("TinySepsis
degrades less under distribution shift than tabular baselines") rests on
more than a single representative run per model.

qSOFA/NEWS2 are deterministic (no training), so they are excluded here --
their numbers are identical every run and already reported once in
Table 1/2.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tinysepsis.eval.metrics import auroc  # noqa: E402

PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
PRED_DIR = ROOT / "results" / "predictions"
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"

SEEDS = [0, 1, 2, 3, 4]
BASELINE_MODELS = ["logreg", "xgboost", "lightgbm"]


def run(cmd, label):
    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print(result.stdout[-2000:], flush=True)
    if result.returncode != 0:
        print(result.stderr[-2000:], flush=True)
        print(f"note: {label} exited {result.returncode}; will verify via output files below", flush=True)


def main():
    for seed in SEEDS:
        baselines_done = all(
            (PRED_DIR / f"{m}_seed{seed}__external_test.parquet").exists() for m in BASELINE_MODELS
        )
        if baselines_done:
            print(f"baselines seed={seed}: predictions already on disk, skipping retrain", flush=True)
        else:
            run(
                [PY, str(ROOT / "scripts" / "train_baselines.py"),
                 "--seed", str(seed), "--tag-suffix", f"_seed{seed}", "--skip-clinical-scores"],
                f"baselines seed={seed}",
            )
        run(
            [PY, str(ROOT / "scripts" / "train_model.py"),
             "--tag", f"tinysepsis_seed{seed}", "--seed", str(seed)],
            f"tinysepsis seed={seed}",
        )

    rows = []
    for seed in SEEDS:
        for model in BASELINE_MODELS + ["tinysepsis"]:
            tag = f"{model}_seed{seed}"
            test_path = PRED_DIR / f"{tag}__test.parquet"
            ext_path = PRED_DIR / f"{tag}__external_test.parquet"
            if not test_path.exists() or not ext_path.exists():
                print(f"MISSING: {tag} (test_exists={test_path.exists()}, ext_exists={ext_path.exists()})", flush=True)
                continue
            test_pred = pl.read_parquet(test_path)
            ext_pred = pl.read_parquet(ext_path)
            test_auroc = auroc(test_pred["y_true"].to_numpy(), test_pred["y_prob"].to_numpy())
            ext_auroc = auroc(ext_pred["y_true"].to_numpy(), ext_pred["y_prob"].to_numpy())
            rows.append({
                "model": model, "seed": seed,
                "test_auroc": test_auroc, "external_auroc": ext_auroc,
                "gap": test_auroc - ext_auroc,
            })

    df = pl.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.write_csv(TABLE_DIR / "multiseed_raw.csv")
    print(df, flush=True)

    summary = (
        df.group_by("model")
        .agg(
            pl.col("test_auroc").mean().alias("test_mean"),
            pl.col("test_auroc").std().alias("test_std"),
            pl.col("external_auroc").mean().alias("external_mean"),
            pl.col("external_auroc").std().alias("external_std"),
            pl.col("gap").mean().alias("gap_mean"),
            pl.col("gap").std().alias("gap_std"),
            pl.col("seed").count().alias("n_seeds"),
        )
        .sort("model")
    )
    summary.write_csv(TABLE_DIR / "multiseed_summary.csv")
    print(summary, flush=True)

    # --- Figure: strip plot of external-vs-internal AUROC gap per seed, per model ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_order = ["logreg", "xgboost", "lightgbm", "tinysepsis"]
    model_labels = {"logreg": "LogReg", "xgboost": "XGBoost", "lightgbm": "LightGBM", "tinysepsis": "TinySepsis"}

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for i, model in enumerate(model_order):
        sub = df.filter(pl.col("model") == model)
        if sub.height == 0:
            continue
        xs = np.full(sub.height, i) + np.random.default_rng(0).uniform(-0.08, 0.08, sub.height)
        axes[0].scatter(xs, sub["test_auroc"].to_numpy(), alpha=0.7, s=25)
        axes[0].scatter([i], [sub["test_auroc"].mean()], color="black", marker="_", s=300, zorder=5)
        axes[1].scatter(xs, sub["external_auroc"].to_numpy(), alpha=0.7, s=25)
        axes[1].scatter([i], [sub["external_auroc"].mean()], color="black", marker="_", s=300, zorder=5)

    for ax, title in [(axes[0], "Internal test (Hospital A)"), (axes[1], "External test (Hospital B)")]:
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels([model_labels[m] for m in model_order], rotation=20)
        ax.set_ylabel("AUROC")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "multiseed_auroc.pdf")
    plt.close(fig)

    print("Multi-seed sweep complete.", flush=True)


if __name__ == "__main__":
    main()
