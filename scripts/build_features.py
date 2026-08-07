"""Full feature pipeline: raw_long.parquet -> enriched, split, normalized parquet.

Run after scripts/ingest.py. Memory-conscious: single pass with Polars,
writes one final Parquet file plus a normalization-stats JSON.
"""
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tinysepsis.data.features import add_missingness_and_dynamics  # noqa: E402
from tinysepsis.data.labels import add_early_warning_labels  # noqa: E402
from tinysepsis.data.splits import assign_splits  # noqa: E402
from tinysepsis.data.normalize import fit_stats, save_stats, apply_stats  # noqa: E402

IN_PATH = ROOT / "data" / "processed" / "raw_long.parquet"
OUT_PATH = ROOT / "data" / "processed" / "enriched.parquet"
STATS_PATH = ROOT / "data" / "processed" / "norm_stats.json"


def main():
    print("Loading raw_long.parquet...", flush=True)
    df = pl.read_parquet(IN_PATH)
    print(f"{df.height} rows, {df['patient_id'].n_unique()} patients", flush=True)

    print("Adding early-warning labels (t_susp reconstruction, censoring)...", flush=True)
    df = add_early_warning_labels(df)
    print(f"{df.height} rows after censoring at t_susp", flush=True)

    print("Adding missingness masks, time-since-last-measurement, deltas...", flush=True)
    df = add_missingness_and_dynamics(df)

    print("Assigning patient-level splits (A: train/val/test, B: external_test)...", flush=True)
    df = assign_splits(df)
    print(df.group_by("split").agg(pl.col("patient_id").n_unique()), flush=True)

    print("Fitting normalization stats on train split...", flush=True)
    stats = fit_stats(df)
    save_stats(stats, STATS_PATH)

    print("Applying normalization...", flush=True)
    df = apply_stats(df, stats)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH, compression="zstd")
    print(f"Wrote {df.height} rows -> {OUT_PATH}", flush=True)

    prevalence = (
        df.group_by("split")
        .agg(
            pl.col("label_6h").mean().alias("pos_rate_6h"),
            pl.col("patient_id").n_unique().alias("n_patients"),
        )
    )
    print(prevalence, flush=True)


if __name__ == "__main__":
    main()
