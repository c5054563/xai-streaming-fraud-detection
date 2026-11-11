"""Detector comparison on the drifted evaluation stream (objective O4; preview for Chapter 4).

One frozen batch Random Forest (fitted on the labelled warm-up slice) is monitored by three
detectors: ADWIN and DDM on its 0/1 errors, which only arrive after the label delay, and the
label-free SHAP-ADWIN on its TreeSHAP attributions.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import config, detectors, drift, metrics, models

_D: dict = {}


def _init(warm, evalp, feats, t0, delays, horizon, shap_delta, sample_every):
    _D.update(warm=warm, evalp=evalp, feats=feats, t0=t0, delays=delays, horizon=horizon,
              shap_delta=shap_delta, sample_every=sample_every)


def label_arrival_positions(times: np.ndarray, delay_s: float) -> np.ndarray:
    """Stream position at which each transaction's label becomes known (t_j + delay <= t_i)."""
    return np.searchsorted(times, times + delay_s, side="left")


def error_detector_alarms(errors, arrival, name):
    det = models.make_detector(name)
    alarms = []
    for e, pos in zip(errors, arrival):
        det.update(int(e))
        if det.drift_detected:
            alarms.append(int(pos))
    return alarms


def run_case(job):
    kind, seed = job
    warm, evalp, feats, t0 = _D["warm"], _D["evalp"], _D["feats"], _D["t0"]
    rf = models.make_batch_rf(seed=seed).fit(warm[feats].to_numpy(), warm.Class.to_numpy())
    drifted, _ = drift.inject(evalp, kind, t0=t0, seed=seed)
    X, y, t = drifted[feats].to_numpy(), drifted.Class.to_numpy(), drifted.Time.to_numpy()
    p = rf.predict_proba(X)[:, 1]
    err = (p >= config.THRESHOLD).astype(int) != y
    out = []
    ref = warm[feats].to_numpy()[np.random.default_rng(seed).choice(len(warm), 2000, replace=False)]
    sd = detectors.ShapADWIN(rf, ref, delta=_D["shap_delta"], sample_every=_D["sample_every"])
    shap_alarms, _, _ = sd.run(X)
    pre = metrics.summarise(y[:t0], p[:t0]); post = metrics.summarise(y[t0:], p[t0:])
    for dname, dsec in _D["delays"].items():
        arr = label_arrival_positions(t, dsec)
        alarms = {"ADWIN": error_detector_alarms(err, arr, "ADWIN"),
                  "DDM": error_detector_alarms(err, arr, "DDM"),
                  "SHAP-ADWIN": shap_alarms}
        for det, al in alarms.items():
            d, fa = detectors.detection_delay(al, t0, _D["horizon"])
            out.append({"scenario": kind, "seed": seed, "delay": dname, "detector": det,
                        "detected": d is not None, "detection_delay_tx": d,
                        "false_alarms_before_t0": fa, "alarms_total": len(al),
                        "rf_pr_auc_pre": pre["pr_auc"], "rf_pr_auc_post": post["pr_auc"]})
    return out


def run_all(warm, evalp, feats, t0, delays, kinds=("none", "abrupt", "gradual", "adversarial"),
            seeds=config.SEEDS, horizon=50_000, shap_delta=0.002, sample_every=5, workers=5):
    jobs = [(k, s) for k in kinds for s in seeds]
    with ProcessPoolExecutor(workers, initializer=_init,
                             initargs=(warm, evalp, feats, t0, delays, horizon, shap_delta, sample_every)) as ex:
        return [r for rows in ex.map(run_case, jobs) for r in rows]
