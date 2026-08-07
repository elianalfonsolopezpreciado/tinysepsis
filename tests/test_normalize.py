import numpy as np
import polars as pl

from tinysepsis.data.normalize import fit_stats, apply_stats, Z_CLIP


def _toy_enriched():
    n = 100
    return pl.DataFrame({
        "split": ["train"] * 80 + ["test"] * 20,
        "HR": list(range(100)),
        "HR__delta1": [0.0] * 100,
        "HR__tslm": [0.0] * 100,
        "Age": [50.0] * 100,
        "HospAdmTime": [-1.0] * 100,
        **{f"{c}": [0.0] * 100 for c in [
            "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3",
            "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium",
            "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
            "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI",
            "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
        ]},
        **{f"{c}__delta1": [0.0] * 100 for c in [
            "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3",
            "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium",
            "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
            "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI",
            "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
        ]},
        **{f"{c}__tslm": [0.0] * 100 for c in [
            "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3",
            "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium",
            "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
            "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI",
            "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
        ]},
    })


def test_fit_stats_uses_only_train_split():
    df = _toy_enriched()
    stats = fit_stats(df)
    # train HR = 0..79 -> mean 39.5
    assert abs(stats["HR"]["mean"] - 39.5) < 1e-6


def test_apply_stats_zscores_train_to_zero_mean():
    df = _toy_enriched()
    stats = fit_stats(df)
    out = apply_stats(df, stats)
    train_z = out.filter(pl.col("split") == "train")["HR__z"]
    assert abs(train_z.mean()) < 1e-6


def test_constant_column_gets_std_floor_no_divide_by_zero():
    df = _toy_enriched()
    stats = fit_stats(df)
    assert stats["Age"]["std"] == 1.0  # constant Age=50 -> std floored to 1.0, no NaN/inf
    out = apply_stats(df, stats)
    assert out["Age__z"].null_count() == 0


def test_extreme_outlier_is_clipped_not_left_unbounded():
    df = _toy_enriched()
    stats = fit_stats(df)
    # simulate a data-entry error far outside the training distribution
    df = df.with_columns(pl.when(pl.arange(0, df.height) == 0).then(1e9).otherwise(pl.col("HR")).alias("HR"))
    out = apply_stats(df, stats)
    z = out["HR__z"].to_numpy()
    assert z.max() <= Z_CLIP
    assert z.min() >= -Z_CLIP
    assert np.isfinite(z).all()
