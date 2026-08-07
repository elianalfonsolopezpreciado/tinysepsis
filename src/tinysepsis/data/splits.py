"""Patient-level, stratified train/val/test splits.

Hospital A -> internal train/val/test (no leakage: split is by patient_id,
never by row). Hospital B is held out in its entirety as an external,
different-institution validation set (a real distribution shift, unlike a
random split), the strongest form of external validation available without
credentialed MIMIC-IV/eICU access.
"""
from __future__ import annotations

import numpy as np
import polars as pl

SEED = 42


def assign_splits(df: pl.DataFrame) -> pl.DataFrame:
    patient_meta = (
        df.group_by("patient_id")
        .agg(
            pl.col("hospital").first(),
            (pl.col("t_onset").first().is_not_null()).alias("ever_septic"),
        )
    )

    rng = np.random.default_rng(SEED)
    assignments = {}

    for hospital in ["A", "B"]:
        for stratum in [True, False]:
            pids = (
                patient_meta.filter(
                    (pl.col("hospital") == hospital) & (pl.col("ever_septic") == stratum)
                )["patient_id"]
                .to_numpy()
            )
            pids = pids.copy()
            rng.shuffle(pids)

            if hospital == "B":
                for pid in pids:
                    assignments[pid] = "external_test"
            else:
                n = len(pids)
                n_train = int(n * 0.70)
                n_val = int(n * 0.15)
                for pid in pids[:n_train]:
                    assignments[pid] = "train"
                for pid in pids[n_train:n_train + n_val]:
                    assignments[pid] = "val"
                for pid in pids[n_train + n_val:]:
                    assignments[pid] = "test"

    split_df = pl.DataFrame(
        {"patient_id": list(assignments.keys()), "split": list(assignments.values())}
    )
    return df.join(split_df, on="patient_id", how="left")
