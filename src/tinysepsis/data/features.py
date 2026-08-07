"""Feature engineering: missingness masks, time-since-last-measurement, deltas.

Operates on the long-format (patient_id, ICULOS, ...) Parquet produced by
scripts/ingest.py and returns an enriched long-format frame where every
time-varying clinical column `X` gains three companions:
  X__mask   - 1.0 if X was actually measured at this hour, else 0.0
  X__tslm   - hours since the last real measurement of X (capped at TSLM_CAP)
  X__delta1 - X__ffill(t) - X__ffill(t-1), 0.0 where undefined
plus the forward-filled value itself (X stays as the ffilled value; the raw
missingness is fully captured by X__mask so no information is lost).
"""
from __future__ import annotations

import polars as pl

from tinysepsis.data.schema import NUMERIC_FEATURES

TSLM_CAP = 48.0  # hours; missingness beyond this is treated as "a long time"


def add_missingness_and_dynamics(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["patient_id", "ICULOS"])
    out_cols = []

    for col in NUMERIC_FEATURES:
        mask = pl.col(col).is_not_null().cast(pl.Float32).alias(f"{col}__mask")
        out_cols.append(mask)

    df = df.with_columns(out_cols)

    # Time since last measurement: computed per-patient via a running counter.
    # hours_since = 0 if measured now, else previous_hours_since + step, capped.
    tslm_exprs = []
    ffill_exprs = []
    for col in NUMERIC_FEATURES:
        ffilled = pl.col(col).forward_fill().over("patient_id").alias(f"{col}__ffill")
        ffill_exprs.append(ffilled)
    df = df.with_columns(ffill_exprs)

    for col in NUMERIC_FEATURES:
        # Hour index of most recent observation, forward-filled; NaN if never observed yet.
        obs_hour = (
            pl.when(pl.col(f"{col}__mask") == 1.0)
            .then(pl.col("ICULOS"))
            .otherwise(None)
            .forward_fill()
            .over("patient_id")
        )
        tslm = (pl.col("ICULOS") - obs_hour).fill_null(TSLM_CAP).clip(0, TSLM_CAP)
        tslm_exprs.append(tslm.alias(f"{col}__tslm"))
    df = df.with_columns(tslm_exprs)

    delta_exprs = []
    for col in NUMERIC_FEATURES:
        prev = pl.col(f"{col}__ffill").shift(1).over("patient_id")
        delta = (pl.col(f"{col}__ffill") - prev).fill_null(0.0)
        delta_exprs.append(delta.alias(f"{col}__delta1"))
    df = df.with_columns(delta_exprs)

    # Replace the raw (possibly-null) column with the forward-filled version,
    # then fill any still-missing leading values with a population-level
    # placeholder of 0 after z-scoring is applied downstream (kept as null
    # here; normalize.py handles the final fill so train-set statistics are
    # used consistently).
    df = df.with_columns(
        [pl.col(f"{col}__ffill").alias(col) for col in NUMERIC_FEATURES]
    )
    df = df.drop([f"{col}__ffill" for col in NUMERIC_FEATURES])

    df = df.with_columns(
        (pl.col("ICULOS") - pl.col("ICULOS").min().over("patient_id")).alias("hours_since_admission")
    )

    return df
