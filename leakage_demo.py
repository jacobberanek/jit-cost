"""Leakage demo (M5): the same models and features, scored three different ways.

Schemes (all use identical, fixed hyperparameters, so only the way the data is split differs):
- chronological:  train on every year before Y, test on year Y (our real design, D9).
- leave-one-year: train on every OTHER year (past AND future), test on the same year Y.
                  Same test commits as chronological; the only change is that the model has
                  seen the future.
- random 5-fold:  shuffled stratified 5-fold over a project's pooled 2010-2017 commits, the
                  common evaluation that the JIT-DP surveys warn about.

The point is not which number is "right" (the chronological one is), but whether a leaky
split changes the conclusion about what each cost tier is worth.

Run from the repo root:  python leakage_demo.py      -> results/leakage_demo.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from jitcost.data import PROJECTS, TEST_YEARS, filter_study, load, rolling_folds
from jitcost.features import CONFIGS, build_model

SEED = 0
LADDER = ["T1", "T1+T2", "T1+T2+T3"]  # cumulative cost tiers
# Fixed, moderate settings (not tuned): both models see identical settings under every scheme.
FIXED = {
    "lr": {"C": 0.01},
    "rf": {"max_depth": 10, "min_samples_leaf": 20, "max_features": 0.5},
}


def score_split(model: str, config: str, train: pd.DataFrame, test: pd.DataFrame, n_trees: int):
    cols = CONFIGS[config]
    est = build_model(model, cols, FIXED[model], None, SEED, n_trees)
    est.fit(train[cols], train["buggy"])
    p = est.predict_proba(test[cols])[:, 1]
    y = test["buggy"]
    return roc_auc_score(y, p), average_precision_score(y, p), y.mean()


def splits(study: pd.DataFrame, project: str):
    """Yield (scheme, fold label, train, test) for one project."""
    p = study[study["project"] == project].reset_index(drop=True)
    for f in rolling_folds(study, project):
        yield "chronological", str(f.test_year), f.train, f.test
    for y in TEST_YEARS:
        yield "leave-one-year", str(y), p[p["year"] != y], p[p["year"] == y]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(skf.split(p, p["buggy"]), 1):
        yield "random 5-fold", f"fold {k}", p.iloc[tr], p.iloc[te]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results", help="output directory (default: results)")
    ap.add_argument("--trees", type=int, default=500, help="random forest size")
    args = ap.parse_args()

    study = filter_study(load())
    rows = []
    for project in PROJECTS:
        for scheme, fold, train, test in splits(study, project):
            for model in ("lr", "rf"):
                for config in LADDER:
                    roc, pr, base = score_split(model, config, train, test, args.trees)
                    rows.append(
                        dict(
                            scheme=scheme, project=project, fold=fold, model=model, config=config,
                            n_train=len(train), n_test=len(test), test_base_rate=base,
                            roc_auc=roc, pr_auc=pr,
                        )
                    )
        print(f"done: {project}")
    d = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    d.to_csv(out / "leakage_demo.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    order = ["chronological", "leave-one-year", "random 5-fold"]
    # Mean over folds within a project first, then over the 3 projects.
    per_proj = d.groupby(["scheme", "project", "model", "config"])[["roc_auc", "pr_auc", "test_base_rate"]].mean()
    m = per_proj.groupby(["scheme", "model", "config"]).mean().reset_index()

    print("\n=== Mean ROC-AUC by scheme (identical fixed hyperparameters) ===")
    t = m.pivot_table(index=["model", "config"], columns="scheme", values="roc_auc")[order]
    t["leave-one-year minus chrono"] = t["leave-one-year"] - t["chronological"]
    t["random minus chrono"] = t["random 5-fold"] - t["chronological"]
    print(t.round(3).to_string())

    print("\n=== Mean PR-AUC and test base rate by scheme ===")
    pr = m.pivot_table(index=["model", "config"], columns="scheme", values="pr_auc")[order]
    print(pr.round(3).to_string())
    print("mean test base rate:", m.groupby("scheme")["test_base_rate"].mean().reindex(order).round(3).to_dict())

    print("\n=== Does the cost conclusion change? Mean gain from the next tier (ROC-AUC) ===")
    w = m.pivot_table(index=["model", "scheme"], columns="config", values="roc_auc")
    steps = pd.DataFrame(
        {"T1 -> T1+T2": w["T1+T2"] - w["T1"], "T1+T2 -> T1+T2+T3": w["T1+T2+T3"] - w["T1+T2"]}
    )
    print(steps.reindex(order, level="scheme").round(3).to_string())


if __name__ == "__main__":
    main()