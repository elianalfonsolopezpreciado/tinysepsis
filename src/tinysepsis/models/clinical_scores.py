"""Approximate qSOFA and NEWS2-lite clinical risk scores as zero-training baselines.

The Challenge 2019 dataset lacks Glasgow Coma Scale / mental-status data, so
qSOFA here uses only its 2 available criteria (of 3): SBP <= 100 mmHg and
Resp >= 22 /min (each worth 1 point; score in {0,1,2}, flagged "high risk"
at score >= 2 as in the original qSOFA convention, which is a conservative
approximation since the true 3-criterion score would only be >= this one).

NEWS2-lite similarly uses only the vitals present in this dataset (HR,
Resp, Temp, SBP, O2Sat) with standard NEWS2 breakpoints, omitting the
oxygen-supplementation and consciousness sub-scores that are not recorded.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def qsofa_lite(df: pl.DataFrame) -> pl.Series:
    sbp_pt = (pl.col("SBP") <= 100).cast(pl.Int8)
    resp_pt = (pl.col("Resp") >= 22).cast(pl.Int8)
    return (sbp_pt + resp_pt).alias("qsofa_lite")


def _news2_resp(r):
    return (
        pl.when(r <= 8).then(3)
        .when(r <= 11).then(1)
        .when(r <= 20).then(0)
        .when(r <= 24).then(2)
        .otherwise(3)
    )


def _news2_o2sat(s):
    return (
        pl.when(s <= 91).then(3)
        .when(s <= 93).then(2)
        .when(s <= 95).then(1)
        .otherwise(0)
    )


def _news2_temp(t):
    return (
        pl.when(t <= 35.0).then(3)
        .when(t <= 36.0).then(1)
        .when(t <= 38.0).then(0)
        .when(t <= 39.0).then(1)
        .otherwise(2)
    )


def _news2_sbp(s):
    return (
        pl.when(s <= 90).then(3)
        .when(s <= 100).then(2)
        .when(s <= 110).then(1)
        .when(s <= 219).then(0)
        .otherwise(3)
    )


def _news2_hr(h):
    return (
        pl.when(h <= 40).then(3)
        .when(h <= 50).then(1)
        .when(h <= 90).then(0)
        .when(h <= 110).then(1)
        .when(h <= 130).then(2)
        .otherwise(3)
    )


def news2_lite(df: pl.DataFrame) -> pl.Series:
    score = (
        _news2_resp(pl.col("Resp"))
        + _news2_o2sat(pl.col("O2Sat"))
        + _news2_temp(pl.col("Temp"))
        + _news2_sbp(pl.col("SBP"))
        + _news2_hr(pl.col("HR"))
    )
    return score.alias("news2_lite")
