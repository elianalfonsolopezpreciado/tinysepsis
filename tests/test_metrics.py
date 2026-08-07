import numpy as np

from tinysepsis.eval.metrics import (
    auroc, auprc, brier, expected_calibration_error, sensitivity_at_specificity,
    specificity_at_sensitivity, alarms_per_1000_patient_hours, decision_curve_net_benefit,
    lead_time_hours,
)


def test_auroc_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auroc(y, p) == 1.0


def test_auroc_random_is_near_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=5000)
    p = rng.random(5000)
    assert 0.45 < auroc(y, p) < 0.55


def test_brier_perfect_predictions_is_zero():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.0, 1.0, 1.0, 0.0])
    assert brier(y, p) == 0.0


def test_ece_perfectly_calibrated_is_near_zero():
    rng = np.random.default_rng(1)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert expected_calibration_error(y, p, n_bins=10) < 0.03


def test_sensitivity_at_specificity_bounds():
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.linspace(0, 0.5, 50), np.linspace(0.5, 1, 50)])
    sens, thr = sensitivity_at_specificity(y, p, 0.9)
    assert 0.0 <= sens <= 1.0


def test_alarms_per_1000_patient_hours():
    y_bin = np.array([1, 0, 1, 0, 1])
    rate = alarms_per_1000_patient_hours(y_bin, n_patient_hours=1000)
    assert rate == 3.0


def test_decision_curve_treat_none_is_always_zero():
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0.2, 0.8, 0.3, 0.6, 0.9])
    dc = decision_curve_net_benefit(y, p, np.array([0.1, 0.2, 0.3]))
    assert np.all(dc["net_benefit_none"] == 0.0)


def test_lead_time_none_when_never_alarmed():
    assert lead_time_hours([], t_susp=20) is None


def test_lead_time_sustained_alarm():
    # alarmed at hours 10,11,...,19 continuously up to t_susp=20
    hours = list(range(10, 20))
    lt = lead_time_hours(hours, t_susp=20, sustained=True)
    assert lt == 10.0
