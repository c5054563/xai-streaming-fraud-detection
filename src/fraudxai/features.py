"""Feature extraction (report Section 3.21).

The data has no card or customer ID, so per-card aggregates are impossible. Features are:
hour of day from Time, log Amount, and stream-window features that use ONLY past transactions
(no look-ahead): transaction count in the previous 60 s, mean Amount of the previous 100
transactions and the ratio of the current Amount to that mean. Online standardisation is done
inside the River pipeline (preprocessing.StandardScaler) for the linear model.

`add_features` (batch, vectorised) and `OnlineFeatures` (one transaction at a time, used by the
Streamlit demo) must give identical values; tests/test_features.py checks this.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from . import config

WIN_SECONDS = 60
WIN_N = 100
ENGINEERED = ["hour", "log_amount", "tx_count_60s", "amt_mean_100", "amt_ratio"]
MODEL_FEATURES = config.V_COLS + ENGINEERED


def hour_of_day(time_s):
    return (np.asarray(time_s) // 3600).astype(int) % 24


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised feature builder. Requires df sorted by Time."""
    out = df.copy()
    t = out["Time"].to_numpy()
    amt = out["Amount"].to_numpy()
    out["hour"] = hour_of_day(t)
    out["log_amount"] = np.log1p(amt)
    idx = np.arange(len(t))
    # previous transactions with time > t - 60 (strictly earlier rows only)
    out["tx_count_60s"] = idx - np.searchsorted(t, t - WIN_SECONDS, side="right")
    prev_mean = pd.Series(amt).shift(1).rolling(WIN_N, min_periods=1).mean()
    out["amt_mean_100"] = prev_mean.fillna(0.0).to_numpy()
    out["amt_ratio"] = amt / (out["amt_mean_100"].to_numpy() + 1.0)
    return out


class OnlineFeatures:
    """Same features computed one transaction at a time with bounded memory."""

    def __init__(self, win_seconds: int = WIN_SECONDS, win_n: int = WIN_N):
        self.win_seconds = win_seconds
        self.times: deque = deque()
        self.amounts: deque = deque(maxlen=win_n)

    def transform_one(self, x: dict) -> dict:
        t, a = float(x["Time"]), float(x["Amount"])
        while self.times and self.times[0] <= t - self.win_seconds:
            self.times.popleft()
        mean_prev = float(np.mean(self.amounts)) if self.amounts else 0.0
        feats = {k: x[k] for k in config.V_COLS}
        feats.update({
            "hour": int(t // 3600) % 24,
            "log_amount": float(np.log1p(a)),
            "tx_count_60s": len(self.times),
            "amt_mean_100": mean_prev,
            "amt_ratio": a / (mean_prev + 1.0),
        })
        self.times.append(t)
        self.amounts.append(a)
        return feats
