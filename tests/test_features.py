import numpy as np
import polars as pl

from tinysepsis.data.features import add_missingness_and_dynamics, TSLM_CAP


def _toy_frame():
    return pl.DataFrame({
        "patient_id": ["p1"] * 5,
        "ICULOS": [1, 2, 3, 4, 5],
        "HR": [80.0, None, None, 90.0, None],
        "O2Sat": [None, None, None, None, None],
    }).with_columns([
        pl.lit(None, dtype=pl.Float64).alias(c)
        for c in ["Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3", "FiO2",
                  "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride",
                  "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium",
                  "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
                  "PTT", "WBC", "Fibrinogen", "Platelets"]
    ])


def test_mask_reflects_observed_values():
    df = add_missingness_and_dynamics(_toy_frame())
    mask = df["HR__mask"].to_list()
    assert mask == [1.0, 0.0, 0.0, 1.0, 0.0]


def test_forward_fill_carries_last_value():
    df = add_missingness_and_dynamics(_toy_frame())
    hr = df["HR"].to_list()
    assert hr == [80.0, 80.0, 80.0, 90.0, 90.0]


def test_time_since_last_measurement_increments_and_resets():
    df = add_missingness_and_dynamics(_toy_frame())
    tslm = df["HR__tslm"].to_list()
    assert tslm == [0.0, 1.0, 2.0, 0.0, 1.0]


def test_never_observed_feature_caps_at_tslm_cap():
    df = add_missingness_and_dynamics(_toy_frame())
    tslm = df["O2Sat__tslm"].to_list()
    assert all(t == TSLM_CAP for t in tslm)


def test_delta1_is_zero_at_first_observation_and_correct_after():
    df = add_missingness_and_dynamics(_toy_frame())
    delta = df["HR__delta1"].to_list()
    assert delta[0] == 0.0
    assert delta[3] == 10.0  # 90 - 80
