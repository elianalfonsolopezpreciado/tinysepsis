"""Conformal risk control for a false-alarm-rate budget.

Implements split-conformal threshold selection in the style of Angelopoulos
et al. (2022), "Conformal Risk Control": given a calibration set disjoint
from both training and the reported test set, choose the smallest alarm
threshold tau such that the empirical false-positive rate on the
calibration set does not exceed a target budget alpha, with a finite-sample
correction so the guarantee holds (approximately, marginally) at deployment
time on exchangeable data. We additionally report empirical coverage on the
external (different-hospital) test set as a robustness check, since the
i.i.d./exchangeability assumption is explicitly violated there by design.
"""
from __future__ import annotations

import numpy as np


def calibrate_threshold(cal_probs: np.ndarray, cal_labels: np.ndarray, alpha: float) -> float:
    """Smallest threshold tau in [0,1] such that FPR(tau) <= alpha on the
    calibration set, using the conformal finite-sample correction
    ceil((n+1)(1-alpha))/n applied to the negative-class score distribution.
    """
    neg_probs = np.sort(cal_probs[cal_labels == 0])
    n = len(neg_probs)
    if n == 0:
        return 1.0
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    tau = np.quantile(neg_probs, q_level, method="higher")
    return float(tau)


def empirical_fpr(probs: np.ndarray, labels: np.ndarray, tau: float) -> float:
    neg = labels == 0
    if neg.sum() == 0:
        return 0.0
    return float((probs[neg] >= tau).mean())


def empirical_tpr(probs: np.ndarray, labels: np.ndarray, tau: float) -> float:
    pos = labels == 1
    if pos.sum() == 0:
        return 0.0
    return float((probs[pos] >= tau).mean())
