"""Tune, fit, and score one (config, model) on one chronological fold.

Order of operations (the leakage-critical part):
1. Split the TRAINING period by time into fit (older 80%) and validation (newer 20%).
2. Score every hyperparameter setting on validation, keep the best by ROC-AUC.
3. Refit that setting on the full training period.
4. Score the test year exactly once. Nothing about the test year influences steps 1-3.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from jitcost.data import Fold, time_val_split
from jitcost.features import CONFIGS, FINAL_TREES, TUNE_TREES, build_model, param_grid

SEED = 0
BUDGET = 0.20  # reviewers inspect the riskiest 20% of commits (decision D14)


def budget_metrics(y: np.ndarray, score: np.ndarray, frac: float = BUDGET, seed: int = SEED):
    """Recall and precision when only the top `frac` of commits (by score) are reviewed.

    Ties are broken at random with a fixed seed, so a model that gives every commit the
    same score (the prior) lands near `frac` recall instead of an arbitrary time-ordered slice.
    """
    n = len(y)
    k = int(round(n * frac))
    perm = np.random.default_rng(seed).permutation(n)
    order = perm[np.argsort(-score[perm], kind="stable")]
    hits = y[order[:k]].sum()
    return hits / y.sum(), hits / k


def run_one(
    fold: Fold,
    config: str,
    model: str,
    class_weight=None,
    tune_trees: int = TUNE_TREES,
    final_trees: int = FINAL_TREES,
    seed: int = SEED,
) -> tuple[dict, pd.DataFrame]:
    """Return (one results row, per-commit test scores for the paired bootstrap)."""
    t0 = time.perf_counter()
    cols = CONFIGS[config]
    y_test = fold.test["buggy"].to_numpy()

    if model == "prior":
        clf = DummyClassifier(strategy="prior")
        clf.fit(np.zeros((len(fold.train), 1)), fold.train["buggy"])
        score = clf.predict_proba(np.zeros((len(fold.test), 1)))[:, 1]
        best_params, val_auc = {}, float("nan")
    else:
        fit, val = time_val_split(fold.train)
        best_params, val_auc = None, -np.inf
        for params in param_grid(model):
            est = build_model(model, cols, params, class_weight, seed, tune_trees)
            est.fit(fit[cols], fit["buggy"])
            auc = roc_auc_score(val["buggy"], est.predict_proba(val[cols])[:, 1])
            if auc > val_auc:
                best_params, val_auc = params, auc
        est = build_model(model, cols, best_params, class_weight, seed, final_trees)
        est.fit(fold.train[cols], fold.train["buggy"])
        score = est.predict_proba(fold.test[cols])[:, 1]

    recall_b, precision_b = budget_metrics(y_test, score, seed=seed)
    row = {
        "project": fold.project,
        "test_year": fold.test_year,
        "config": config,
        "model": model,
        "n_features": len(cols),
        "n_train": len(fold.train),
        "n_test": len(fold.test),
        "test_base_rate": y_test.mean(),
        "val_auc": val_auc,
        "test_roc_auc": roc_auc_score(y_test, score),
        "test_pr_auc": average_precision_score(y_test, score),
        "recall_at_20": recall_b,
        "precision_at_20": precision_b,
        "best_params": json.dumps(best_params),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    preds = pd.DataFrame({"commit_id": fold.test["commit_id"].to_numpy(), "y": y_test, "score": score})
    return row, preds