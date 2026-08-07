"""Clinical utility score, adapted from the official PhysioNet/CinC Challenge 2019
scoring function (Reyna et al. 2020).

Constants and piecewise formula verified verbatim against the official
evaluation script:
https://github.com/physionetchallenges/evaluation-2019/blob/master/evaluate_sepsis_score.py

Adaptation note (documented explicitly, see paper Sec. Methods): the
official script scores predictions over a patient's *entire* ICU stay,
including hours after clinical sepsis suspicion. TinySepsis is designed as
a pre-suspicion early-warning system and is not evaluated after t_susp (see
tinysepsis.data.labels for the censoring rationale), so here the same
formula is applied only over the pre-suspicion decision window we actually
model. This is a deliberate scope narrowing, not a re-derivation of the
formula itself, and is not directly comparable to official leaderboard
scores.
"""
from __future__ import annotations

import numpy as np

DT_EARLY = -12
DT_OPTIMAL = -6
DT_LATE = 3
MAX_U_TP = 1.0
MIN_U_FN = -2.0
U_FP = -0.05
U_TN = 0.0


def _u_tp(tau: np.ndarray) -> np.ndarray:
    m1 = MAX_U_TP / (DT_OPTIMAL - DT_EARLY)
    b1 = -m1 * DT_EARLY
    m2 = -MAX_U_TP / (DT_LATE - DT_OPTIMAL)
    b2 = -m2 * DT_LATE
    u = np.zeros_like(tau, dtype=np.float64)
    seg1 = (tau > DT_EARLY) & (tau <= DT_OPTIMAL)
    seg2 = (tau > DT_OPTIMAL) & (tau <= DT_LATE)
    u[seg1] = m1 * tau[seg1] + b1
    u[seg2] = m2 * tau[seg2] + b2
    return u


def _u_fn(tau: np.ndarray) -> np.ndarray:
    m3 = MIN_U_FN / (DT_LATE - DT_OPTIMAL)
    b3 = -m3 * DT_OPTIMAL
    u = np.zeros_like(tau, dtype=np.float64)
    seg = (tau > DT_OPTIMAL) & (tau <= DT_LATE)
    u[seg] = m3 * tau[seg] + b3
    u[tau > DT_LATE] = MIN_U_FN
    return u


def patient_utility(
    y_pred_binary: np.ndarray,
    is_septic: bool,
    tau: np.ndarray | None = None,
) -> float:
    """Sum of per-hour utility for one patient's decision window.

    y_pred_binary: (T,) array of 0/1 alarm decisions for this patient's
        modeled hours (already restricted to the pre-suspicion window).
    is_septic: whether this patient ever developed sepsis.
    tau: (T,) hours relative to t_susp (negative = before suspicion),
        required if is_septic.
    """
    y_pred_binary = np.asarray(y_pred_binary).astype(bool)
    if not is_septic:
        return float(np.where(y_pred_binary, U_FP, U_TN).sum())

    tau = np.asarray(tau, dtype=np.float64)
    tp_u = _u_tp(tau)
    fn_u = _u_fn(tau)
    per_hour = np.where(y_pred_binary, tp_u, fn_u)
    return float(per_hour.sum())


def normalized_utility(total_observed: float, total_best: float, total_inaction: float) -> float:
    """Challenge-style normalized utility in [~0, 1]: 1.0 = perfect oracle,
    0.0 = the "always predict 0" (inaction) policy."""
    denom = total_best - total_inaction
    if abs(denom) < 1e-9:
        return 0.0
    return (total_observed - total_inaction) / denom
