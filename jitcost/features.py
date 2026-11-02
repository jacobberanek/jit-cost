"""Cost tiers, experiment configs, and model pipelines.

This file is the single source of truth for which feature belongs to which
cost tier (decision D5). Tiers follow what a feature needs to be computed:
- T1: commit header + list of changed paths (never reads file contents)
- T2: line-level content diff
- T3: scan of repository history

`fix` is excluded from every main config (D3). It appears only in the S3
demonstration config "T1+fix".
"""
from __future__ import annotations

from functools import partial
from itertools import product

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

T1 = ["hour", "weekday", "nf", "nd", "ns"]
T2 = ["la", "ld", "ent"]
T3 = ["ndev", "age", "nuc", "aexp", "arexp", "asexp"]

CONFIGS: dict[str, list[str]] = {
    "prior": [],  # predicts the training base rate; AUC = 0.5
    "la_only": ["la"],  # Zeng et al. 2021 "LApredict"
    "T1": T1,
    "T1+T2": T1 + T2,
    "T1+T2+T3": T1 + T2 + T3,
    "T3_only": T3,  # diagnostic: is the expensive information useful on its own?
    "T1+fix": T1 + ["fix"],  # sensitivity S3 only (dataset-construction artifact)
}
MAIN_CONFIGS = ["prior", "la_only", "T1", "T1+T2", "T1+T2+T3", "T3_only"]


def models_for(config: str) -> tuple[str, ...]:
    """Which models run on a config. LApredict is logistic regression by definition."""
    if config == "prior":
        return ("prior",)
    if config == "la_only":
        return ("lr",)
    return ("lr", "rf")


# --- Logistic regression preprocessing (D: log1p, cyclic time, standardize) ---

# Count-like, right-skewed features get log1p. `ent` (entropy, 0 to 6.6) is already
# compact, so it passes through; `fix` is 0/1.
LOG_COLS = {"nf", "nd", "ns", "la", "ld", "ndev", "age", "nuc", "aexp", "arexp", "asexp"}
CYCLIC = {"hour": 24, "weekday": 7}  # so that 23:00 sits next to 00:00 and Sunday next to Monday
RAW_COLS = {"ent", "fix"}


def _sincos(X, period):
    angle = 2 * np.pi * np.asarray(X, dtype=float) / period
    return np.hstack([np.sin(angle), np.cos(angle)])


def _sincos_names(transformer, input_features):
    return [f"{name}_{part}" for name in input_features for part in ("sin", "cos")]


def _lr_preprocessor(cols: list[str]) -> ColumnTransformer:
    unknown = set(cols) - LOG_COLS - set(CYCLIC) - RAW_COLS
    if unknown:
        raise ValueError(f"No preprocessing rule for columns: {sorted(unknown)}")
    parts = []
    logged = [c for c in cols if c in LOG_COLS]
    if logged:
        parts.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), logged))
    for c in (c for c in cols if c in CYCLIC):
        parts.append(
            (
                f"cyc_{c}",
                FunctionTransformer(
                    partial(_sincos, period=CYCLIC[c]), feature_names_out=_sincos_names
                ),
                [c],
            )
        )
    raw = [c for c in cols if c in RAW_COLS]
    if raw:
        parts.append(("raw", "passthrough", raw))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


# --- Models and tuning grids (plan 5.5; identical budget for every config) ---

TUNE_TREES = 200  # trees while searching the grid
FINAL_TREES = 500  # trees in the refit that gets scored on the test year

# The plan's grids (C from 0.01, min_samples_leaf up to 20) were extended by one value
# each (0.001, 50) after a validation-only check showed the best setting landing on the
# grid edge, which means the search may not have reached the optimum (decision D14).
LR_C = (0.001, 0.01, 0.1, 1, 10)
RF_GRID = {
    "max_depth": (None, 10, 20),
    "min_samples_leaf": (1, 5, 20, 50),
    "max_features": ("sqrt", 0.5),
}


def param_grid(model: str) -> list[dict]:
    if model == "lr":
        return [{"C": c} for c in LR_C]
    if model == "rf":
        keys = list(RF_GRID)
        return [dict(zip(keys, vals)) for vals in product(*RF_GRID.values())]
    raise ValueError(f"No grid for model {model!r}")


def build_model(
    model: str,
    cols: list[str],
    params: dict,
    class_weight=None,
    seed: int = 0,
    n_trees: int = FINAL_TREES,
):
    """Return an unfitted estimator. Every transform lives inside it, so it only ever
    sees the data passed to `.fit` (no preprocessing leakage)."""
    if model == "lr":
        return Pipeline(
            [
                ("prep", _lr_preprocessor(cols)),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(  # L2-regularized by default
                        C=params["C"],
                        class_weight=class_weight,
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model == "rf":
        # Trees compare thresholds, so monotone transforms (log, scaling) change nothing.
        return RandomForestClassifier(
            n_estimators=n_trees,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=seed,
            **params,
        )
    raise ValueError(f"Unknown model {model!r}")