import numpy as np

from tinysepsis.eval.calibration import TemperatureScaler, IsotonicCalibrator
from tinysepsis.eval.metrics import expected_calibration_error


def test_temperature_scaling_improves_overconfident_ece():
    rng = np.random.default_rng(0)
    n = 20000
    true_p = rng.random(n)
    y = (rng.random(n) < true_p).astype(np.float32)
    # simulate an overconfident model: logits scaled by 3x
    true_logit = np.log(true_p / (1 - true_p))
    overconfident_logit = true_logit * 3.0
    overconfident_prob = 1 / (1 + np.exp(-overconfident_logit))

    ece_before = expected_calibration_error(y, overconfident_prob)

    ts = TemperatureScaler().fit(overconfident_logit, y)
    calibrated_prob = ts.transform(overconfident_logit)
    ece_after = expected_calibration_error(y, calibrated_prob)

    assert ts.T > 1.0  # should learn to "cool down" overconfidence
    assert ece_after < ece_before


def test_isotonic_calibrator_monotonic_output():
    rng = np.random.default_rng(2)
    probs = np.sort(rng.random(500))
    labels = (rng.random(500) < probs).astype(int)
    iso = IsotonicCalibrator().fit(probs, labels)
    out = iso.transform(np.array([0.1, 0.5, 0.9]))
    assert out[0] <= out[1] <= out[2]
