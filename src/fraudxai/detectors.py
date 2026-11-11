"""Proposed XAI-guided drift detector (objective O4): ADWIN on windowed TreeSHAP attributions.

Extends Shap-ADWIN (Zimmermann & Boussard, 2022) to a fraud stream with delayed labels: the
signal needs NO labels. A tree model (fitted on labelled warm-up data) is explained with
TreeSHAP on every `sample_every`-th incoming transaction. For each explained transaction the
signal is the relative L1 distance between its |SHAP| vector and the reference mean |SHAP|
vector from the warm-up window. ADWIN raises an alarm when the mean of this signal changes.
"""
from __future__ import annotations

import numpy as np
import shap
from river import drift


def positive_class_shap(explainer, X: np.ndarray) -> np.ndarray:
    sv = explainer.shap_values(X, check_additivity=False)
    sv = np.asarray(sv)
    if sv.ndim == 3:            # (n, features, classes) in recent SHAP versions
        return sv[:, :, 1]
    return sv


class ShapADWIN:
    def __init__(self, tree_model, X_reference: np.ndarray, delta: float = 0.002,
                 sample_every: int = 5, chunk: int = 2000):
        self.explainer = shap.TreeExplainer(tree_model)
        ref = np.abs(positive_class_shap(self.explainer, X_reference))
        self.mu_ref = ref.mean(axis=0)
        self.scale = self.mu_ref.sum() + 1e-12
        self.delta, self.sample_every, self.chunk = delta, sample_every, chunk
        self.adwin = drift.ADWIN(delta=delta)

    def signal(self, X: np.ndarray) -> np.ndarray:
        s = np.abs(positive_class_shap(self.explainer, X))
        return np.abs(s - self.mu_ref).sum(axis=1) / self.scale

    def run(self, X: np.ndarray) -> tuple[list[int], np.ndarray, np.ndarray]:
        """Replay X in order; returns alarm positions, sampled positions and signal values."""
        idx = np.arange(0, len(X), self.sample_every)
        sig = np.concatenate([self.signal(X[idx[i:i + self.chunk]])
                              for i in range(0, len(idx), self.chunk)]) if len(idx) else np.array([])
        alarms = []
        for pos, v in zip(idx, sig):
            self.adwin.update(float(v))
            if self.adwin.drift_detected:
                alarms.append(int(pos))
        return alarms, idx, sig


def detection_delay(alarms, t0: int, horizon: int | None = None):
    """First alarm at or after t0 (in transactions) and number of alarms before t0."""
    after = [a for a in alarms if a >= t0 and (horizon is None or a < t0 + horizon)]
    return (after[0] - t0 if after else None), sum(a < t0 for a in alarms)
