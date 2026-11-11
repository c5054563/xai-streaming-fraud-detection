"""Model and detector factories (report Section 3.23, objectives O1, O3, O4)."""
from __future__ import annotations

from river import drift, forest, linear_model, preprocessing, tree, optim
from sklearn.ensemble import RandomForestClassifier


def make_online(name: str, seed: int = 0, **params):
    """Online learners from River. Trees need no scaling; the linear model uses online
    standardisation (running mean/variance) as its first step."""
    if name == "HT":
        return tree.HoeffdingTreeClassifier(
            grace_period=params.get("grace_period", 200),
            delta=params.get("delta", 1e-7),
            leaf_prediction=params.get("leaf_prediction", "nba"))
    if name == "ARF":
        return forest.ARFClassifier(
            n_models=params.get("n_models", 10),
            grace_period=params.get("grace_period", 50),
            lambda_value=params.get("lambda_value", 6),
            seed=seed)
    if name == "LR":
        return preprocessing.StandardScaler() | linear_model.LogisticRegression(
            optimizer=optim.SGD(params.get("lr", 0.01)))
    raise ValueError(name)


def make_batch_rf(seed: int = 0, n_estimators: int = 100, max_depth: int = 10):
    """Batch comparator (O3) and the tree model explained by TreeSHAP (O4)."""
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  class_weight="balanced_subsample", n_jobs=2,
                                  random_state=seed)


def make_detector(name: str, **params):
    """Classic error-rate detectors (baselines for O4)."""
    if name == "ADWIN":
        return drift.ADWIN(delta=params.get("delta", 0.002))
    if name == "DDM":
        return drift.binary.DDM()
    raise ValueError(name)
