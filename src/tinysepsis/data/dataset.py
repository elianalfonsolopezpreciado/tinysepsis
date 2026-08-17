"""PyTorch Dataset for TinySepsis: per-patient arrays, sliding observation windows.

Builds each patient's full (T, F) feature matrix once at init (cheap: avg
stay ~38h) and returns left-padded SEQ_LEN-hour windows ending at hour t on
__getitem__, avoiding an exploded per-window Parquet on disk.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from tinysepsis.data.schema import NUMERIC_FEATURES

STATIC_COLS = ["Age__z", "Gender"]


def feature_cols_for_ablation(ablation: str = "full") -> list[str]:
    """
    full           - value + missing-mask + time-since-last-measurement + delta (default)
    no_missingness - value + delta only (drops mask & tslm channels, tests their contribution)
    no_dynamics    - value + mask only (drops the delta / short-term-trend channel)
    """
    value = [f"{c}__z" for c in NUMERIC_FEATURES]
    mask = [f"{c}__mask" for c in NUMERIC_FEATURES]
    tslm = [f"{c}__tslm__z" for c in NUMERIC_FEATURES]
    delta = [f"{c}__delta1__z" for c in NUMERIC_FEATURES]
    if ablation == "full":
        return value + mask + tslm + delta
    if ablation == "no_missingness":
        return value + delta
    if ablation == "no_dynamics":
        return value + mask
    raise ValueError(f"unknown ablation: {ablation}")


class TinySepsisDataset(Dataset):
    def __init__(
        self,
        parquet_path: Path,
        split: str,
        seq_len: int = 24,
        label_col: str = "label_6h",
        max_patients: int | None = None,
        ablation: str = "full",
        extra_label_cols: list[str] | None = None,
    ):
        """extra_label_cols: additional label columns (e.g. label_4h,
        label_8h alongside the primary label_col=label_6h) for multi-horizon
        auxiliary-task training. Backward compatible: when None (default),
        behavior and __getitem__'s output keys are unchanged."""
        self.seq_len = seq_len
        self.label_col = label_col
        self.extra_label_cols = extra_label_cols or []
        FEATURE_COLS = feature_cols_for_ablation(ablation)
        self.feature_cols = FEATURE_COLS

        lf = pl.scan_parquet(parquet_path).filter(pl.col("split") == split)
        df = lf.select(
            ["patient_id", "ICULOS", "hours_since_admission", label_col]
            + self.extra_label_cols
            + FEATURE_COLS
            + STATIC_COLS
        ).collect()
        self.parquet_path = str(parquet_path)
        self.split = split

        patient_ids = df["patient_id"].unique(maintain_order=True).to_list()
        if max_patients is not None:
            patient_ids = patient_ids[:max_patients]
            df = df.filter(pl.col("patient_id").is_in(patient_ids))

        self.patients: dict[str, dict[str, np.ndarray]] = {}
        self.index: list[tuple[str, int]] = []  # (patient_id, row_idx_within_patient)

        for pid, pdf in df.group_by("patient_id", maintain_order=True):
            pid = pid[0] if isinstance(pid, tuple) else pid
            pdf = pdf.sort("ICULOS")
            feats = pdf.select(FEATURE_COLS).to_numpy().astype(np.float32)
            static = pdf.select(STATIC_COLS).to_numpy().astype(np.float32)
            labels = pdf[label_col].to_numpy().astype(np.float32)
            iculos = pdf["ICULOS"].to_numpy().astype(np.int32)
            rec = {"feats": feats, "static": static, "labels": labels, "iculos": iculos}
            if self.extra_label_cols:
                rec["extra_labels"] = pdf.select(self.extra_label_cols).to_numpy().astype(np.float32)
            self.patients[pid] = rec
            for row_idx in range(feats.shape[0]):
                self.index.append((pid, row_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        pid, row_idx = self.index[i]
        rec = self.patients[pid]
        feats = rec["feats"]
        n_feat = feats.shape[1]

        start = row_idx - self.seq_len + 1
        seq = np.zeros((self.seq_len, n_feat), dtype=np.float32)
        pad_mask = np.zeros((self.seq_len,), dtype=np.float32)  # 1 = real, 0 = padding

        real_start = max(0, start)
        n_real = row_idx - real_start + 1
        seq[self.seq_len - n_real:] = feats[real_start:row_idx + 1]
        pad_mask[self.seq_len - n_real:] = 1.0

        static = rec["static"][row_idx]
        label = rec["labels"][row_idx]

        out = {
            "seq": torch.from_numpy(seq),
            "pad_mask": torch.from_numpy(pad_mask),
            "static": torch.from_numpy(static),
            "label": torch.tensor(label, dtype=torch.float32),
            "patient_id": pid,
        }
        if self.extra_label_cols:
            out["extra_labels"] = torch.from_numpy(rec["extra_labels"][row_idx])
        return out
