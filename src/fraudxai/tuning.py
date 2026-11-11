"""Prequential hyperparameter search on the early warm-up slice (report Section 3.24).

Only the first WARMUP_FRACTION of the time-ordered stream is used, so later data (where drift is
injected and final results are measured) is never seen while tuning. Each configuration is run
with every seed and scored with PR-AUC, G-mean and Precision@k (mean +- SD).
"""
from __future__ import annotations

import itertools
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config, metrics, models, stream

GRID = {
    "HT": {"grace_period": [50, 200], "leaf_prediction": ["mc", "nba"], "fraud_weight": [1, 50, 200]},
    "ARF": {"n_models": [5, 10, 20], "grace_period": [50, 200]},
    "LR": {"lr": [0.005, 0.01, 0.05], "fraud_weight": [1, 50]},
}
DETERMINISTIC = {"HT", "LR"}   # no random component: one run equals every seed


def expand(grid: dict) -> list[tuple[str, dict]]:
    out = []
    for name, space in grid.items():
        keys = list(space)
        for vals in itertools.product(*(space[k] for k in keys)):
            out.append((name, dict(zip(keys, vals))))
    return out


_DATA: dict = {}


def _init(X, feats, y, t):
    """Each worker receives compact arrays and builds its own row dicts (keeps pickling small)."""
    _DATA.update(rows=[dict(zip(feats, r)) for r in X.tolist()], y=y, t=t)


def _run(job):
    name, params, seed, delay = job
    m = models.make_online(name, seed=seed, **params)
    t0 = time.time()
    res = stream.prequential(m, _DATA["rows"], _DATA["y"], _DATA["t"], delay_s=delay)
    out = {"model": name, **{f"p_{k}": v for k, v in params.items()}, "params": str(params),
           "seed": seed, "delay_s": delay, "seconds": time.time() - t0,
           **metrics.summarise(_DATA["y"], res.scores)}
    return out


def search(X, feats, y, t, grid=GRID, seeds=config.SEEDS, delay_s=0.0, workers=6) -> pd.DataFrame:
    jobs = []
    for name, params in expand(grid):
        for s in (seeds[:1] if name in DETERMINISTIC else seeds):
            jobs.append((name, params, s, delay_s))
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(X, feats, y, t)) as ex:
        res = list(ex.map(_run, jobs))
    df = pd.DataFrame(res)
    # deterministic models: copy the single run to every seed so mean +- SD is over 5 seeds (SD = 0)
    extra = []
    for _, r in df[df.model.isin(DETERMINISTIC)].iterrows():
        for s in seeds[1:]:
            rr = r.copy(); rr["seed"] = s; extra.append(rr)
    if extra:
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "params"])
    s = g[["pr_auc", "g_mean", "precision_at_k", "seconds"]].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s["n_seeds"] = g.size()
    return s.reset_index().sort_values(["model", "pr_auc_mean"], ascending=[True, False])


def evaluate(X, feats, y, t, configs: list[tuple[str, dict]], seeds=config.SEEDS,
             delays=(0.0,), workers=5) -> pd.DataFrame:
    """Run fixed (tuned) configurations on a stream for every seed and label delay."""
    jobs = [(n, p, s, d) for n, p in configs for d in delays
            for s in (seeds[:1] if n in DETERMINISTIC else seeds)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(X, feats, y, t)) as ex:
        return pd.DataFrame(list(ex.map(_run, jobs)))
