"""Ingest raw PhysioNet Challenge 2019 .psv files into a single long-format Parquet.

Memory-conscious: streams one small file at a time with Polars, never loads
the raw CSVs into pandas. Output is one row per (patient, hour).
"""
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "processed" / "raw_long.parquet"

sys.path.insert(0, str(ROOT / "src"))
from tinysepsis.data.schema import ALL_RAW_COLUMNS, PLAUSIBLE_RANGE  # noqa: E402


def read_one(path: Path, hospital: str) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        separator="|",
        null_values=["NaN"],
        schema_overrides={c: pl.Float64 for c in ALL_RAW_COLUMNS},
    )
    patient_id = f"{hospital}_{path.stem}"
    df = df.with_columns(
        pl.lit(patient_id).alias("patient_id"),
        pl.lit(hospital).alias("hospital"),
    )
    return df


def apply_plausible_ranges(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for col, (lo, hi) in PLAUSIBLE_RANGE.items():
        exprs.append(
            pl.when((pl.col(col) < lo) | (pl.col(col) > hi))
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )
    return df.with_columns(exprs)


def main():
    frames = []
    n_read = 0
    for hospital, subdir in [("A", "training_setA"), ("B", "training_setB")]:
        files = sorted((RAW_DIR / subdir).glob("*.psv"))
        print(f"Reading {len(files)} files from {subdir}...", flush=True)
        for i, path in enumerate(files):
            frames.append(read_one(path, hospital))
            n_read += 1
            if n_read % 5000 == 0:
                print(f"  {n_read} files read", flush=True)

    print("Concatenating...", flush=True)
    full = pl.concat(frames, how="vertical")
    del frames

    print(f"Total rows before cleaning: {full.height}", flush=True)
    full = apply_plausible_ranges(full)

    # Drop patients with implausibly short stays (<4h) - too little signal.
    stay_len = full.group_by("patient_id").agg(pl.len().alias("n_hours"))
    short_patients = stay_len.filter(pl.col("n_hours") < 4)["patient_id"]
    n_before = full["patient_id"].n_unique()
    full = full.filter(~pl.col("patient_id").is_in(short_patients))
    n_after = full["patient_id"].n_unique()
    print(f"Dropped {n_before - n_after} patients with <4h of data.", flush=True)

    full = full.sort(["patient_id", "ICULOS"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.write_parquet(OUT_PATH, compression="zstd")
    print(f"Wrote {full.height} rows, {n_after} patients -> {OUT_PATH}", flush=True)
    print(full.group_by("hospital").agg(pl.col("patient_id").n_unique()), flush=True)


if __name__ == "__main__":
    main()
