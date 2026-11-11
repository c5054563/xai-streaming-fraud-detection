"""Programmatic drift injection into the real, time-ordered stream (report Section 3.22, O2/O4).

Public data rarely has labelled natural drift, so three controlled scenarios are injected
(drift taxonomy of Gama et al., 2014):
  * abrupt      - from position t0, pairs of strong fraud-indicator features are swapped for every
                  transaction (the feature->label relation changes at once: real concept drift);
  * gradual     - the same swap, applied with a probability that rises linearly from 0 to 1 over
                  `width` transactions after t0;
  * adversarial - from t0, fraud transactions copy the values of their strongest indicator
                  features from randomly chosen legitimate transactions (camouflage).
All randomness is controlled by `seed`, so each seed gives a different but repeatable drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOP_FEATURES = ["V17", "V14", "V12", "V10", "V16", "V3", "V7", "V11"]  # set from EDA (|corr| with Class)


def _pairs(rng, features, n_pairs):
    chosen = rng.choice(features, size=2 * n_pairs, replace=False)
    return [(chosen[2 * i], chosen[2 * i + 1]) for i in range(n_pairs)]


def inject(df: pd.DataFrame, kind: str, t0: int, seed: int = 0, width: int = 20_000,
           n_pairs: int = 3, features=TOP_FEATURES) -> tuple[pd.DataFrame, dict]:
    """Return a drifted copy of df and a description of what was changed."""
    if kind == "none":
        return df.copy(), {"kind": "none"}
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    pos = np.arange(n)
    info = {"kind": kind, "t0": int(t0), "seed": seed}
    if kind in ("abrupt", "gradual"):
        pairs = _pairs(rng, list(features), n_pairs)
        if kind == "abrupt":
            mask = pos >= t0
        else:
            p = np.clip((pos - t0) / width, 0, 1)
            mask = rng.random(n) < p
            info["width"] = width
        for a, b in pairs:
            va, vb = out.loc[mask, a].to_numpy().copy(), out.loc[mask, b].to_numpy().copy()
            out.loc[mask, a], out.loc[mask, b] = vb, va
        info["pairs"] = [list(map(str, p)) for p in pairs]
        info["n_changed"] = int(mask.sum())
    elif kind == "adversarial":
        cols = list(features[:4])
        fraud_after = np.where((pos >= t0) & (out["Class"].to_numpy() == 1))[0]
        legit = np.where(out["Class"].to_numpy() == 0)[0]
        donors = rng.choice(legit, size=len(fraud_after), replace=True)
        out.iloc[fraud_after, [out.columns.get_loc(c) for c in cols]] = \
            out.iloc[donors, [out.columns.get_loc(c) for c in cols]].to_numpy()
        info["camouflaged_features"] = cols
        info["n_changed"] = int(len(fraud_after))
    else:
        raise ValueError(f"unknown drift kind: {kind}")
    return out, info
