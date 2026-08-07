"""Discrimination, calibration, and clinical-operational metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve


def auroc(y_true, y_prob) -> float:
    return float(roc_auc_score(y_true, y_prob))


def auprc(y_true, y_prob) -> float:
    return float(average_precision_score(y_true, y_prob))


def brier(y_true, y_prob) -> float:
    return float(brier_score_loss(y_true, y_prob))


def sensitivity_at_specificity(y_true, y_prob, target_specificity: float) -> tuple[float, float]:
    fpr, tpr, thresh = roc_curve(y_true, y_prob)
    spec = 1 - fpr
    idx = np.argmin(np.abs(spec - target_specificity))
    return float(tpr[idx]), float(thresh[idx])


def specificity_at_sensitivity(y_true, y_prob, target_sensitivity: float) -> tuple[float, float]:
    fpr, tpr, thresh = roc_curve(y_true, y_prob)
    idx = np.argmin(np.abs(tpr - target_sensitivity))
    return float(1 - fpr[idx]), float(thresh[idx])


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def calibration_curve(y_true, y_prob, n_bins: int = 15):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, obs_freq, counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_centers.append(y_prob[mask].mean())
        obs_freq.append(y_true[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(bin_centers), np.array(obs_freq), np.array(counts)


def alarms_per_1000_patient_hours(y_pred_binary, n_patient_hours: int) -> float:
    n_alarms = int(np.asarray(y_pred_binary).sum())
    return 1000.0 * n_alarms / max(n_patient_hours, 1)


def decision_curve_net_benefit(y_true, y_prob, thresholds: np.ndarray) -> dict:
    """Vickers & Elkin (2006) standard net-benefit decision-curve analysis."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true)
    prevalence = y_true.mean()
    nb_model, nb_all, nb_none = [], [], []
    for pt in thresholds:
        pred_pos = y_prob >= pt
        tp = (pred_pos & (y_true == 1)).sum()
        fp = (pred_pos & (y_true == 0)).sum()
        w = pt / (1 - pt) if pt < 1 else np.inf
        nb_model.append(tp / n - fp / n * w)
        nb_all.append(prevalence - (1 - prevalence) * w)
        nb_none.append(0.0)
    return {
        "thresholds": thresholds,
        "net_benefit_model": np.array(nb_model),
        "net_benefit_all": np.array(nb_all),
        "net_benefit_none": np.array(nb_none),
    }


def lead_time_hours(patient_alarm_hours: list[int], t_susp: int, sustained: bool = True) -> float | None:
    """Hours between t_susp and the first alarm (or first *sustained* alarm
    that never turns off before t_susp, if `sustained`). Returns None if the
    patient was never alarmed."""
    if not patient_alarm_hours:
        return None
    t_susp = int(t_susp)
    hrs = sorted(int(h) for h in patient_alarm_hours)
    if not sustained:
        return float(t_susp - hrs[0])
    # find first alarm hour h such that all subsequent modeled hours up to
    # t_susp are also alarmed
    hrs_set = set(hrs)
    for h in hrs:
        if all((hh in hrs_set) for hh in range(h, t_susp)):
            return float(t_susp - h)
    return float(t_susp - hrs[0])
