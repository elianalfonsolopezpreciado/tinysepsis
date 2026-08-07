import polars as pl

from tinysepsis.data.labels import add_early_warning_labels


def test_control_patient_gets_all_zero_labels():
    df = pl.DataFrame({
        "patient_id": ["ctrl"] * 5,
        "ICULOS": [1, 2, 3, 4, 5],
        "SepsisLabel": [0, 0, 0, 0, 0],
    })
    out = add_early_warning_labels(df)
    assert out.height == 5
    assert out["label_4h"].sum() == 0
    assert out["label_6h"].sum() == 0
    assert out["label_8h"].sum() == 0


def test_septic_patient_window_labels_and_censoring():
    # SepsisLabel(t)=1 starts at hour 10 (challenge convention: t_onset),
    # so t_susp = t_onset + 6 = 16.
    hours = list(range(1, 21))
    sepsis_label = [0] * 9 + [1] * 11  # 1 from hour 10 onward
    df = pl.DataFrame({
        "patient_id": ["case"] * len(hours),
        "ICULOS": hours,
        "SepsisLabel": sepsis_label,
    })
    out = add_early_warning_labels(df).sort("ICULOS")

    # censored at t_susp=16: only hours < 16 remain
    assert out["ICULOS"].max() == 15

    row = {r["ICULOS"]: r for r in out.to_dicts()}
    # 6h window: hours 10..15 (t_susp - t in (0,6])
    for h in range(10, 16):
        assert row[h]["label_6h"] == 1, h
    for h in range(1, 10):
        assert row[h]["label_6h"] == 0, h

    # 4h window: hours 12..15
    for h in range(12, 16):
        assert row[h]["label_4h"] == 1, h
    assert row[11]["label_4h"] == 0


def test_no_leakage_no_hours_at_or_after_t_susp():
    hours = list(range(1, 25))
    sepsis_label = [0] * 5 + [1] * 19
    df = pl.DataFrame({
        "patient_id": ["case2"] * len(hours),
        "ICULOS": hours,
        "SepsisLabel": sepsis_label,
    })
    out = add_early_warning_labels(df)
    t_susp = out["t_susp"][0]
    assert (out["ICULOS"] < t_susp).all()
