import numpy as np

from tinysepsis.eval.conformal import calibrate_threshold, empirical_fpr, empirical_tpr


def test_conformal_threshold_controls_false_alarm_rate():
    rng = np.random.default_rng(0)
    n = 5000
    labels = rng.integers(0, 2, size=n)
    probs = np.where(labels == 1, rng.beta(5, 2, n), rng.beta(2, 5, n))

    alpha = 0.10
    tau = calibrate_threshold(probs, labels, alpha)

    # re-check on a *fresh* draw from the same distribution (exchangeable
    # calibration/test split), FPR should be close to (at or below, up to
    # finite-sample slack) the target alpha
    labels2 = rng.integers(0, 2, size=n)
    probs2 = np.where(labels2 == 1, rng.beta(5, 2, n), rng.beta(2, 5, n))
    fpr = empirical_fpr(probs2, labels2, tau)
    assert fpr < alpha + 0.03  # small slack for finite-sample variation


def test_higher_alpha_gives_lower_threshold():
    rng = np.random.default_rng(1)
    n = 3000
    labels = rng.integers(0, 2, size=n)
    probs = np.where(labels == 1, rng.beta(5, 2, n), rng.beta(2, 5, n))
    tau_strict = calibrate_threshold(probs, labels, alpha=0.05)
    tau_loose = calibrate_threshold(probs, labels, alpha=0.20)
    assert tau_loose <= tau_strict


def test_empirical_tpr_and_fpr_are_fractions():
    labels = np.array([0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.6, 0.4, 0.7, 0.9])
    fpr = empirical_fpr(probs, labels, tau=0.5)
    tpr = empirical_tpr(probs, labels, tau=0.5)
    assert fpr == 0.5  # one of two negatives (0.6) exceeds tau
    assert abs(tpr - 2 / 3) < 1e-9  # two of three positives (0.7, 0.9) exceed tau
