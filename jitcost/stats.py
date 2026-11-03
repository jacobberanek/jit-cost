"""Paired bootstrap and across-fold statistics for the cost-tier comparison (RQ1, RQ2).

For each project-fold, model and tier step (e.g. T1 -> T1+T2) this answers: how much ROC-AUC
does the richer config add, and could that difference be noise?

Method (decision D15):
- Both configs scored the SAME test commits (saved in results/preds/), so each bootstrap
  resample draws commits once and scores BOTH models on that resample. Pairing removes the
  commit-to-commit difficulty that both models share and isolates the difference between them.
- delta = AUC(richer) - AUC(cheaper). 1,000 resamples, 95% percentile interval.
- Across the 12 project-folds: median delta, how many folds clear the pre-registered 0.02
  threshold (D8), how many intervals exclude 0 or sit entirely below 0.02, and a Wilcoxon
  signed-rank test. With n = 12 the test has low power, so effect sizes and intervals come first.

Run from the repo root:
    python -m jitcost.stats                      # main results
    python -m jitcost.stats --variant gap6       # any variant in results.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from jitcost.data import PROJECTS, TEST_YEARS

THRESHOLD = 0.02  # D8: "practically meaningful" ROC-AUC gain, fixed before any result was seen
N_BOOT = 1000
SEED = 0

# (label, cheaper config, richer config, models). delta = AUC(richer) - AUC(cheaper).
STEPS = [
    ("T1 -> T1+T2", "T1", "T1+T2", ("lr", "rf")),
    ("T1+T2 -> T1+T2+T3", "T1+T2", "T1+T2+T3", ("lr", "rf")),
    ("T1 -> LA-only", "T1", "la_only", ("lr",)),
    ("LA-only -> T1+T2", "la_only", "T1+T2", ("lr",)),
]


def pred_path(results_dir: Path, variant: str, project: str, year: int, config: str, model: str) -> Path:
    """Where run_experiments.py saved a run's test scores (the naming rule must match it)."""
    stem = "__".join(
        str(k).replace("/", "-").replace("+", "_") for k in (variant, project, year, config, model)
    )
    return results_dir / "preds" / f"{stem}.csv.gz"


# --- The paired bootstrap (M8). Write your own version, then check it against this one. ---
def paired_bootstrap(
    y: np.ndarray,
    score_cheap: np.ndarray,
    score_rich: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Return (delta, ci_low, ci_high) for AUC(rich) - AUC(cheap) on the same commits."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)  # n commits drawn with replacement, shared by both models
        yb = y[idx]
        if yb.min() == yb.max():  # AUC is undefined without both classes (does not happen here)
            continue
        deltas.append(roc_auc_score(yb, score_rich[idx]) - roc_auc_score(yb, score_cheap[idx]))
    delta = roc_auc_score(y, score_rich) - roc_auc_score(y, score_cheap)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return delta, lo, hi


def per_fold(results_dir: Path, variant: str, n_boot: int) -> pd.DataFrame:
    res = pd.read_csv(results_dir / "results.csv")
    res = res[res["variant"] == variant].set_index(["project", "test_year", "config", "model"])
    rows = []
    for label, cheap, rich, models in STEPS:
        for model in models:
            for project in PROJECTS:
                for year in TEST_YEARS:
                    pa = pd.read_csv(pred_path(results_dir, variant, project, year, cheap, model))
                    pb = pd.read_csv(pred_path(results_dir, variant, project, year, rich, model))
                    if not (pa["commit_id"].to_numpy() == pb["commit_id"].to_numpy()).all():
                        raise ValueError(f"Different test commits for {label} {project} {year} {model}")
                    y = pa["y"].to_numpy()
                    # Integrity check: the saved scores must reproduce the AUC in results.csv.
                    for cfg, p in ((cheap, pa), (rich, pb)):
                        saved = res.loc[(project, year, cfg, model), "test_roc_auc"]
                        if abs(roc_auc_score(y, p["score"]) - saved) > 1e-9:
                            raise ValueError(f"preds and results.csv disagree: {project} {year} {cfg} {model}")
                    delta, lo, hi = paired_bootstrap(y, pa["score"].to_numpy(), pb["score"].to_numpy(), n_boot)
                    rows.append(
                        dict(
                            step=label, model=model, project=project, test_year=year,
                            n_test=len(y), n_buggy=int(y.sum()),
                            auc_cheap=roc_auc_score(y, pa["score"]), auc_rich=roc_auc_score(y, pb["score"]),
                            delta=delta, ci_low=lo, ci_high=hi,
                        )
                    )
        print(f"done: {label}")
    return pd.DataFrame(rows)


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (step, model), g in d.groupby(["step", "model"], sort=False):
        delta = g["delta"].to_numpy()
        rows.append(
            dict(
                step=step, model=model, n_folds=len(g),
                median_delta=np.median(delta), mean_delta=delta.mean(),
                min_delta=delta.min(), max_delta=delta.max(),
                folds_point_ge_0_02=int((g["delta"] >= THRESHOLD).sum()),
                folds_ci_above_0=int((g["ci_low"] > 0).sum()),
                folds_ci_below_0=int((g["ci_high"] < 0).sum()),
                folds_ci_below_0_02=int((g["ci_high"] < THRESHOLD).sum()),
                wilcoxon_p_vs_0=wilcoxon(delta).pvalue,
                wilcoxon_p_below_0_02=wilcoxon(delta - THRESHOLD, alternative="less").pvalue,
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", help="results directory (default: results)")
    ap.add_argument("--variant", default="main")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    out = Path(args.results)

    d = per_fold(out, args.variant, args.n_boot)
    s = summarize(d)
    d.to_csv(out / f"deltas_{args.variant}.csv", index=False)
    s.to_csv(out / f"deltas_summary_{args.variant}.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"\n=== Per fold ({args.variant}): delta = AUC(richer) - AUC(cheaper), 95% CI ===")
    show = d.assign(project=d["project"].str.split("/").str[1])
    show["delta (95% CI)"] = show.apply(lambda r: f"{r.delta:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]", axis=1)
    print(show[["step", "model", "project", "test_year", "n_buggy", "delta (95% CI)"]].to_string(index=False))
    print(f"\n=== Across the 12 project-folds (threshold = {THRESHOLD}) ===")
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()