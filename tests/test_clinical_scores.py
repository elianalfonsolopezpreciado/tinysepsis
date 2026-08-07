import polars as pl

from tinysepsis.models.clinical_scores import qsofa_lite, news2_lite


def test_qsofa_lite_flags_hypotension_and_tachypnea():
    df = pl.DataFrame({"SBP": [80.0, 130.0], "Resp": [25.0, 14.0]})
    out = df.with_columns(qsofa_lite(df))
    assert out["qsofa_lite"].to_list() == [2, 0]


def test_news2_lite_normal_vitals_score_zero():
    df = pl.DataFrame({
        "Resp": [16.0], "O2Sat": [98.0], "Temp": [37.0], "SBP": [120.0], "HR": [75.0],
    })
    out = df.with_columns(news2_lite(df))
    assert out["news2_lite"][0] == 0


def test_news2_lite_critical_vitals_score_high():
    df = pl.DataFrame({
        "Resp": [30.0], "O2Sat": [88.0], "Temp": [34.0], "SBP": [85.0], "HR": [135.0],
    })
    out = df.with_columns(news2_lite(df))
    assert out["news2_lite"][0] >= 10
