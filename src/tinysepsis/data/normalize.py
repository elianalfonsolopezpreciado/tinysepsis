"""Compute and apply z-score normalization statistics from the TRAIN split only.

Avoids test/validation leakage: stats are fit once on split=='train' and
reused (as a saved JSON) everywhere else, including external_test (hospital B).
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from tinysepsis.data.schema import NUMERIC_FEATURES

STATIC_NUMERIC = ["Age", "HospAdmTime"]


def fit_stats(df: pl.DataFrame) -> dict:
    train = df.filter(pl.col("split") == "train")
    stats = {}
    cols = (
        NUMERIC_FEATURES
        + [f"{c}__delta1" for c in NUMERIC_FEATURES]
        + [f"{c}__tslm" for c in NUMERIC_FEATURES]
        + STATIC_NUMERIC
    )
    agg = train.select(
        [pl.col(c).mean().alias(f"{c}__mean") for c in cols]
        + [pl.col(c).std().alias(f"{c}__std") for c in cols]
    ).to_dicts()[0]
    for c in cols:
        mean = agg[f"{c}__mean"]
        std = agg[f"{c}__std"]
        if mean is None:
            mean = 0.0
        if not std or std < 1e-6:
            std = 1.0
        stats[c] = {"mean": float(mean), "std": float(std)}
    return stats


def save_stats(stats: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))


def load_stats(path: Path) -> dict:
    return json.loads(path.read_text())


Z_CLIP = 10.0  # guards against data-entry outliers (e.g. FiO2 recorded as
# a raw percentage vs. fraction) blowing up fp16 inference; a z-score
# beyond +/-10 SD carries no additional clinical information anyway.


def apply_stats(df: pl.DataFrame, stats: dict) -> pl.DataFrame:
    cols = (
        NUMERIC_FEATURES
        + [f"{c}__delta1" for c in NUMERIC_FEATURES]
        + [f"{c}__tslm" for c in NUMERIC_FEATURES]
        + STATIC_NUMERIC
    )
    exprs = []
    for c in cols:
        s = stats[c]
        exprs.append(
            ((pl.col(c).fill_null(s["mean"]) - s["mean"]) / s["std"]).clip(-Z_CLIP, Z_CLIP).alias(f"{c}__z")
        )
    return df.with_columns(exprs)
