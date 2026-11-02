"""Run configs x models x folds and write results/results.csv.

Examples (from the repo root):
    python run_experiments.py                        # main grid: 144 runs
    python run_experiments.py --variant gap6         # S1: drop last 6 months of training
    python run_experiments.py --variant cw_balanced  # S2: class_weight="balanced"
    python run_experiments.py --variant fix_demo     # S3: add `fix` to T1
    python run_experiments.py --project camel --year 2016   # one project/fold only

Safe to stop and restart: finished runs are skipped. Each run also saves its test-year
scores to results/preds/ (the paired bootstrap needs both models' scores on the same commits).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from jitcost.data import PROJECTS, TEST_YEARS, filter_study, load, rolling_folds
from jitcost.experiment import run_one
from jitcost.features import FINAL_TREES, MAIN_CONFIGS, TUNE_TREES, models_for

VARIANTS = {
    "main": dict(gap_months=0, class_weight=None, configs=MAIN_CONFIGS),
    "gap6": dict(gap_months=6, class_weight=None, configs=MAIN_CONFIGS),
    "cw_balanced": dict(gap_months=0, class_weight="balanced", configs=MAIN_CONFIGS),
    "fix_demo": dict(gap_months=0, class_weight=None, configs=["T1+fix"]),
}
KEY = ["variant", "project", "test_year", "config", "model"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=VARIANTS, default="main")
    ap.add_argument("--project", help="substring filter, e.g. camel")
    ap.add_argument("--year", type=int, help="test year filter")
    ap.add_argument("--out", default="results", help="output directory (default: results)")
    ap.add_argument("--tune-trees", type=int, default=TUNE_TREES)
    ap.add_argument("--final-trees", type=int, default=FINAL_TREES)
    args = ap.parse_args()

    v = VARIANTS[args.variant]
    out = Path(args.out)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    results_path = out / "results.csv"
    done = set()
    if results_path.exists():
        prev = pd.read_csv(results_path)
        done = set(map(tuple, prev[KEY].itertuples(index=False, name=None)))

    study = filter_study(load())
    projects = [p for p in PROJECTS if args.project is None or args.project in p]
    years = [y for y in TEST_YEARS if args.year is None or y == args.year]

    jobs = [
        (project, fold_year, config, model)
        for project in projects
        for fold_year in years
        for config in v["configs"]
        for model in models_for(config)
    ]
    for i, (project, year, config, model) in enumerate(jobs, 1):
        key = (args.variant, project, year, config, model)
        tag = f"[{i}/{len(jobs)}] {args.variant} {project} {year} {config} {model}"
        if key in done:
            print(f"{tag}: already done, skipping")
            continue
        fold = next(
            f for f in rolling_folds(study, project, (year,), gap_months=v["gap_months"])
        )
        row, preds = run_one(
            fold, config, model, v["class_weight"], args.tune_trees, args.final_trees
        )
        row = {"variant": args.variant, "class_weight": str(v["class_weight"]), "gap_months": v["gap_months"], **row}
        stem = "__".join(str(k).replace("/", "-").replace("+", "_") for k in key)
        preds.to_csv(out / "preds" / f"{stem}.csv.gz", index=False)  # scores first, then the row
        pd.DataFrame([row]).to_csv(results_path, mode="a", header=not results_path.exists(), index=False)
        print(f"{tag}: ROC-AUC {row['test_roc_auc']:.4f} ({row['seconds']}s)")


if __name__ == "__main__":
    main()