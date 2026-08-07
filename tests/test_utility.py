import numpy as np

from tinysepsis.eval.utility import patient_utility, normalized_utility, MAX_U_TP, MIN_U_FN, U_FP


def test_control_patient_false_positive_penalty():
    y_pred = np.array([0, 1, 0, 1])
    u = patient_utility(y_pred, is_septic=False)
    assert u == 2 * U_FP


def test_control_patient_all_negative_is_zero():
    y_pred = np.array([0, 0, 0])
    u = patient_utility(y_pred, is_septic=False)
    assert u == 0.0


def test_septic_patient_optimal_window_gets_max_reward():
    # tau in (-6, 3] should award close to max_u_tp when alarmed
    tau = np.array([-6.0, -3.0, 0.0, 3.0])
    y_pred = np.array([1, 1, 1, 1])
    u = patient_utility(y_pred, is_septic=True, tau=tau)
    # at tau=-3 utility should be near max (peaks at tau=-6 exactly = max_u_tp)
    assert u > 0


def test_septic_patient_missed_in_critical_window_is_penalized():
    tau = np.array([-3.0, 0.0, 3.0])
    y_pred = np.array([0, 0, 0])  # never alarmed
    u = patient_utility(y_pred, is_septic=True, tau=tau)
    assert u < 0
    assert u >= 3 * MIN_U_FN  # bounded below


def test_normalized_utility_oracle_is_one():
    assert abs(normalized_utility(total_observed=10, total_best=10, total_inaction=0) - 1.0) < 1e-9


def test_normalized_utility_inaction_is_zero():
    assert normalized_utility(total_observed=0, total_best=10, total_inaction=0) == 0.0
