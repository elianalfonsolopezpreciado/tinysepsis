import polars as pl
import pytest

from tinysepsis.data.dataset import TinySepsisDataset, feature_cols_for_ablation
from tinysepsis.data.schema import NUMERIC_FEATURES


def _write_toy_parquet(path):
    n_feat = len(NUMERIC_FEATURES)
    rows = []
    for pid, split, n_hours in [("p1", "train", 3), ("p2", "train", 30)]:
        for h in range(1, n_hours + 1):
            row = {"patient_id": pid, "ICULOS": h, "hours_since_admission": h - 1,
                   "split": split, "label_6h": 1 if (pid == "p2" and h > 20) else 0,
                   "Age__z": 0.1, "Gender": 1.0}
            for c in NUMERIC_FEATURES:
                row[f"{c}__z"] = 0.0
                row[f"{c}__mask"] = 1.0
                row[f"{c}__tslm__z"] = 0.0
                row[f"{c}__delta1__z"] = 0.0
            rows.append(row)
    pl.DataFrame(rows).write_parquet(path)


def test_short_stay_gets_left_padded(tmp_path):
    path = tmp_path / "toy.parquet"
    _write_toy_parquet(path)
    ds = TinySepsisDataset(path, "train", seq_len=24, label_col="label_6h")

    # patient p1 has only 3 hours; the window at its last hour should be
    # mostly padding
    idx = [i for i, (pid, row_idx) in enumerate(ds.index) if pid == "p1" and row_idx == 2][0]
    item = ds[idx]
    assert item["pad_mask"].sum().item() == 3
    assert item["seq"].shape == (24, len(feature_cols_for_ablation("full")))


def test_long_stay_window_never_exceeds_seq_len(tmp_path):
    path = tmp_path / "toy.parquet"
    _write_toy_parquet(path)
    ds = TinySepsisDataset(path, "train", seq_len=24, label_col="label_6h")
    idx = [i for i, (pid, row_idx) in enumerate(ds.index) if pid == "p2" and row_idx == 29][0]
    item = ds[idx]
    assert item["pad_mask"].sum().item() == 24  # fully real window, no padding


def test_label_matches_source_row(tmp_path):
    path = tmp_path / "toy.parquet"
    _write_toy_parquet(path)
    ds = TinySepsisDataset(path, "train", seq_len=24, label_col="label_6h")
    idx = [i for i, (pid, row_idx) in enumerate(ds.index) if pid == "p2" and row_idx == 25][0]  # hour 26
    item = ds[idx]
    assert item["label"].item() == 1.0


def test_ablation_no_missingness_drops_mask_and_tslm_channels():
    full = feature_cols_for_ablation("full")
    reduced = feature_cols_for_ablation("no_missingness")
    assert len(reduced) == len(NUMERIC_FEATURES) * 2  # value + delta only
    assert len(full) == len(NUMERIC_FEATURES) * 4


def test_unknown_ablation_raises():
    with pytest.raises(ValueError):
        feature_cols_for_ablation("not_a_real_mode")
