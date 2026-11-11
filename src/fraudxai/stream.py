"""Time-ordered stream replay with delayed labels and prequential (test-then-train) evaluation
(report Sections 3.22-3.24, objectives O1, O2, O5).

For every transaction, in time order:
  1. release every label whose transaction time + delay <= current time; the model learns from
     it and the error-rate detectors (ADWIN, DDM) see its 0/1 error;
  2. the model scores the current transaction (test) and the transaction joins the queue
     of labels not yet known (train later).
With delay = 0 this is ordinary prequential evaluation. Cross-validation and random splits are
never used, because they break the time order (UREC2).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import config


@dataclass
class StreamResult:
    scores: np.ndarray
    alarms: dict = field(default_factory=dict)      # detector name -> list of stream positions
    labels_used: int = 0


def prequential(model, X_rows: list[dict], y: np.ndarray, times: np.ndarray,
                delay_s: float = 0.0, detectors: dict | None = None,
                threshold: float = config.THRESHOLD) -> StreamResult:
    detectors = detectors or {}
    alarms = {k: [] for k in detectors}
    pending: deque = deque()
    scores = np.zeros(len(X_rows))
    used = 0
    for i, (x, t) in enumerate(zip(X_rows, times)):
        while pending and pending[0][2] + delay_s <= t:
            xj, yj, _, pj = pending.popleft()
            model.learn_one(xj, int(yj))
            used += 1
            err = int((pj >= threshold) != yj)
            for name, det in detectors.items():
                det.update(err)
                if det.drift_detected:
                    alarms[name].append(i)
        p = model.predict_proba_one(x).get(1, 0.0)
        scores[i] = p
        pending.append((x, y[i], t, p))
    return StreamResult(scores=scores, alarms=alarms, labels_used=used)


def to_rows(df, features) -> list[dict]:
    return df[features].to_dict("records")
