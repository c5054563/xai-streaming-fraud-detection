"""Unit test for cost-sensitive learning wrapper (TC-U15)."""
from fraudxai import models


class _Spy:
    def __init__(self):
        self.weights = []
    def learn_one(self, x, y, w=1.0):
        self.weights.append(w)
    def predict_proba_one(self, x):
        return {1: 0.0}


def test_fraud_labels_get_extra_weight():
    spy = _Spy()
    m = models.ClassWeighted(spy, fraud_weight=50)
    m.learn_one({"a": 1.0}, 0); m.learn_one({"a": 1.0}, 1)
    assert spy.weights == [1.0, 50]
