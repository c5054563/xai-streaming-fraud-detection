"""Imbalance-aware metrics (UREC2: PR-AUC, G-mean, Precision@k; accuracy is NOT used)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from . import config


def pr_auc(y, scores) -> float:
    """Area under the precision-recall curve (average precision)."""
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, scores))


def g_mean(y, scores, threshold: float = config.THRESHOLD) -> float:
    """sqrt(TPR * TNR) at a fixed decision threshold."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    pos, neg = y == 1, y == 0
    tpr = pred[pos].mean() if pos.any() else 0.0
    tnr = (1 - pred[neg]).mean() if neg.any() else 0.0
    return float(np.sqrt(tpr * tnr))


def precision_at_k(y, scores, k: int = config.PK_K, block: int = config.PK_BLOCK) -> float:
    """Alert-budget precision: in each block of `block` consecutive transactions the k
    highest-scored ones are alerted; returns the mean precision over full blocks."""
    y, s = np.asarray(y), np.asarray(scores)
    vals = []
    for start in range(0, len(y) - block + 1, block):
        yb, sb = y[start:start + block], s[start:start + block]
        top = np.argsort(-sb, kind="stable")[:k]
        vals.append(yb[top].mean())
    return float(np.mean(vals)) if vals else float("nan")


def summarise(y, scores) -> dict:
    return {"pr_auc": pr_auc(y, scores), "g_mean": g_mean(y, scores),
            "precision_at_k": precision_at_k(y, scores)}
