"""Unit/integration tests for metrics, drift injection and the delayed-label stream (TC-U10..U14)."""
import numpy as np
import pandas as pd

from fraudxai import drift, metrics, stream


def test_g_mean_and_precision_at_k():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.6, 0.7, 0.2])
    assert np.isclose(metrics.g_mean(y, s), np.sqrt(0.5 * 0.5))
    assert metrics.precision_at_k(y, s, k=1, block=4) == 1.0
    assert metrics.precision_at_k(y, s, k=2, block=2) == 0.5


def test_pr_auc_perfect_ranking():
    assert metrics.pr_auc([0, 0, 1], [0.1, 0.2, 0.9]) == 1.0


def test_abrupt_drift_changes_only_after_t0(tiny):
    out, info = drift.inject(tiny, "abrupt", t0=3, seed=1, n_pairs=2)
    cols = [c for p in info["pairs"] for c in p]
    assert out.iloc[:3][cols].equals(tiny.iloc[:3][cols])
    a, b = info["pairs"][0]
    assert np.allclose(out.iloc[3:][a], tiny.iloc[3:][b])


def test_drift_is_repeatable_per_seed(tiny):
    a, _ = drift.inject(tiny, "gradual", t0=1, seed=7, width=2)
    b, _ = drift.inject(tiny, "gradual", t0=1, seed=7, width=2)
    assert a.equals(b)


class _Counter:
    """Fake model that records how many labels it has received."""
    def __init__(self):
        self.n = 0
    def learn_one(self, x, y):
        self.n += 1
    def predict_proba_one(self, x):
        return {1: float(self.n)}


def test_delayed_labels_are_not_used_before_delay():
    times = np.array([0, 10, 20, 30, 40], dtype=float)
    rows = [{"a": 0.0}] * 5
    y = np.zeros(5)
    res0 = stream.prequential(_Counter(), rows, y, times, delay_s=0)
    res = stream.prequential(_Counter(), rows, y, times, delay_s=25)
    assert list(res0.scores) == [0, 1, 2, 3, 4]          # test-then-train
    assert list(res.scores) == [0, 0, 0, 1, 2]           # label of t=0 only known at t>=25
