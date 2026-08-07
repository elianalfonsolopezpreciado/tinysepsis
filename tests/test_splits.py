import polars as pl

from tinysepsis.data.splits import assign_splits


def _toy_patients(hospital, n, septic_frac):
    rows = []
    n_septic = int(n * septic_frac)
    for i in range(n):
        pid = f"{hospital}_{i}"
        t_onset = 10 if i < n_septic else None
        for h in range(1, 6):
            rows.append({"patient_id": pid, "hospital": hospital, "ICULOS": h, "t_onset": t_onset})
    return rows


def test_hospital_b_entirely_external_test():
    rows = _toy_patients("A", 40, 0.2) + _toy_patients("B", 20, 0.2)
    df = pl.DataFrame(rows)
    out = assign_splits(df)
    b_splits = out.filter(pl.col("hospital") == "B")["split"].unique().to_list()
    assert b_splits == ["external_test"]


def test_no_patient_appears_in_two_splits():
    rows = _toy_patients("A", 100, 0.15)
    df = pl.DataFrame(rows)
    out = assign_splits(df)
    per_patient_splits = out.group_by("patient_id").agg(pl.col("split").n_unique().alias("n"))
    assert (per_patient_splits["n"] == 1).all()


def test_hospital_a_splits_into_train_val_test_only():
    rows = _toy_patients("A", 200, 0.15)
    df = pl.DataFrame(rows)
    out = assign_splits(df)
    a_splits = set(out.filter(pl.col("hospital") == "A")["split"].unique().to_list())
    assert a_splits <= {"train", "val", "test"}
    assert "train" in a_splits and "val" in a_splits and "test" in a_splits


def test_train_is_majority_of_hospital_a():
    rows = _toy_patients("A", 200, 0.15)
    df = pl.DataFrame(rows)
    out = assign_splits(df)
    patient_split = out.group_by("patient_id").agg(pl.col("split").first())
    counts = patient_split["split"].value_counts()
    counts_d = dict(zip(counts["split"].to_list(), counts["count"].to_list()))
    assert counts_d["train"] > counts_d["val"]
    assert counts_d["train"] > counts_d["test"]
