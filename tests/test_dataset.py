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
                   "label_4h": 1 if (pid == "p2" and h > 22) else 0,
                   "label_8h": 1 if (pid == "p2" and h > 18) else 0,
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


def test_extra_label_cols_returns_multihorizon_labels(tmp_path):
    """Multi-horizon training (scripts/train_model_multihorizon.py) needs
    label_4h/label_8h alongside the primary label_6h; extra_label_cols must
    not disturb the existing "label" key's shape or the default (None)
    behavior other callers rely on."""
    path = tmp_path / "toy.parquet"
    _write_toy_parquet(path)
    ds = TinySepsisDataset(path, "train", seq_len=24, label_col="label_6h",
                            extra_label_cols=["label_4h", "label_8h"])
    idx = [i for i, (pid, row_idx) in enumerate(ds.index) if pid == "p2" and row_idx == 25][0]  # hour 26
    item = ds[idx]
    assert item["label"].item() == 1.0  # label_6h: h=26 > 20
    assert item["extra_labels"].shape == (2,)
    assert item["extra_labels"][0].item() == 1.0  # label_4h: h=26 > 22
    assert item["extra_labels"][1].item() == 1.0  # label_8h: h=26 > 18


def test_default_dataset_has_no_extra_labels_key(tmp_path):
    path = tmp_path / "toy.parquet"
    _write_toy_parquet(path)
    ds = TinySepsisDataset(path, "train", seq_len=24, label_col="label_6h")
    item = ds[0]
    assert "extra_labels" not in item
