"""Early-warning label construction.

The Challenge 2019 `SepsisLabel` column is already shifted 6h before the
Sepsis-3 clinical-suspicion time t_susp (i.e. SepsisLabel(t)=1 starting at
t_susp-6). We reconstruct t_susp per patient and derive our own explicit
4h / 6h / 8h early-warning targets, censoring each patient's timeline at
t_susp so the model is only ever trained/evaluated on data that would have
been available *before* a clinician already recognized sepsis.

For control (never-septic) patients, all labels are 0 and no censoring
applies.
"""
from __future__ import annotations

import polars as pl

CHALLENGE_LABEL_SHIFT_H = 6  # SepsisLabel(t)=1 begins 6h before t_susp by construction
PREDICTION_WINDOWS = (4, 6, 8)


def add_early_warning_labels(df: pl.DataFrame) -> pl.DataFrame:
    onset = (
        df.filter(pl.col("SepsisLabel") == 1)
        .group_by("patient_id")
        .agg(pl.col("ICULOS").min().alias("t_onset"))
    )
    df = df.join(onset, on="patient_id", how="left")
    df = df.with_columns(
        (pl.col("t_onset") + CHALLENGE_LABEL_SHIFT_H).alias("t_susp")
    )

    label_exprs = []
    for w in PREDICTION_WINDOWS:
        lbl = (
            pl.when(pl.col("t_susp").is_not_null() & (pl.col("t_susp") - pl.col("ICULOS") <= w)
                     & (pl.col("t_susp") - pl.col("ICULOS") > 0))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias(f"label_{w}h")
        )
        label_exprs.append(lbl)
    df = df.with_columns(label_exprs)

    # Censor: drop hours at/after t_susp (septic patients are only used for
    # the pre-suspicion window; the challenge's own SepsisLabel for
    # ICULOS>=t_susp is discarded here, not a modeling target).
    df = df.filter(pl.col("t_susp").is_null() | (pl.col("ICULOS") < pl.col("t_susp")))

    return df
