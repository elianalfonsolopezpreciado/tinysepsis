"""Post-hoc probability calibration: temperature scaling and isotonic regression.

Both are fit exclusively on the validation split and then frozen (applied
as-is to test / external_test), following standard practice to avoid any
calibration-set leakage into the numbers being reported.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression


class TemperatureScaler:
    def __init__(self):
        self.T = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, max_iter: int = 200) -> "TemperatureScaler":
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        log_T = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)
        criterion = torch.nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            T = torch.exp(log_T)
            loss = criterion(logits_t / T, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.T = float(torch.exp(log_T).item())
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-logits / self.T))


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        self.model.fit(probs, labels)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return self.model.predict(probs)
