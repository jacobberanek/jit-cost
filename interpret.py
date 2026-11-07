"""RQ3: why does the line-level diff (T2) pay off while the history scan (T3) does not?

Four analyses, all on the main-variant T1+T2+T3 models, plus the three report figures:
1. RF permutation importance (test fold only): how much does test ROC-AUC drop when one
   feature's values are shuffled? Averaged over the 12 project-folds.
2. LR standardized coefficients: direction and relative size of each feature's effect.
3. Spearman correlation of the 14 features over the study set: does T3 just repeat T1/T2?
4. T3 on its own vs T1+T2, straight from results.csv (no new modelling).

Models are refit exactly as experiment.run_one does (best_params from results.csv, 500 trees,
seed 0, whole training period). Each refit must reproduce the test ROC-AUC saved in
results.csv, the same integrity check stats.py uses, so the figures describe the models
that produced the headline numbers. Nothing is re-tuned.

Run from the repo root (after run_experiments.py and `python -m jitcost.stats`):
    python interpret.py              -> results/*.csv and figures/*.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from jitcost.data import PROJECTS, filter_study, load, rolling_folds
from jitcost.experiment import SEED
from jitcost.features import CONFIGS, FINAL_TREES, T1, T2, T3, build_model
from jitcost.stats import THRESHOLD

FULL = "T1+T2+T3"
LADDER = ["T1", "T1+T2", "T1+T2+T3"]
N_REPEATS = 10
TIER_COLOR = {"T1": "#0072B2", "T2": "#E69F00", "T3": "#009E73"}  # colorblind-safe
TIER_OF = {**{c: "T1" for c in T1}, **{c: "T2" for c in T2}, **{c: "T3" for c in T3}}
STEP_COLOR = {"T1 -> T1+T2": TIER_COLOR["T2"], "T1+T2 -> T1+T2+T3": TIER_COLOR["T3"]}


def base_feature(name: str) -> str:
    """hour_sin -> hour, so cyclic LR columns map back to their tier."""
    return name.rsplit("_", 1)[0] if name.endswith(("_sin", "_cos")) else name


def refit(fold, model: str, res: pd.DataFrame):
    """Refit the tuned T1+T2+T3 model for one fold and check it against results.csv."""
    cols = CONFIGS[FULL]
    row = res.loc[(fold.project, fold.test_year, FULL, model)]
    est = build_model(model, cols, json.loads(row["best_params"]), None, SEED, FINAL_TREES)
    est.fit(fold.train[cols], fold.train["buggy"])
    auc = roc_auc_score(fold.test["buggy"], est.predict_proba(fold.test[cols])[:, 1])
    if abs(auc - row["test_roc_auc"]) > 1e-9:
        raise ValueError(
            f"Refit disagrees with results.csv: {fold.project} {fold.test_year} {model} "
            f"({auc:.6f} vs {row['test_roc_auc']:.6f})"
        )
    return est


def importance_and_coefficients(study: pd.DataFrame, res: pd.DataFrame, repeats: int):
    cols = CONFIGS[FULL]
    imp_rows, coef_rows = [], []
    for project in PROJECTS:
        for fold in rolling_folds(study, project):
            tag = project.split("/")[1]
            rf = refit(fold, "rf", res)
            pi = permutation_importance(
                rf, fold.test[cols], fold.test["buggy"], scoring="roc_auc",
                n_repeats=repeats, random_state=SEED,
            )
            ranks = pd.Series(-pi.importances_mean).rank(method="min").astype(int)
            for c, m, r in zip(cols, pi.importances_mean, ranks):
                imp_rows.append(
                    dict(project=project, test_year=fold.test_year, feature=c, tier=TIER_OF[c],
                         auc_drop=m, rank=int(r))
                )
            lr = refit(fold, "lr", res)
            names = lr[:-1].get_feature_names_out()
            for n, w in zip(names, lr.named_steps["clf"].coef_[0]):
                coef_rows.append(
                    dict(project=project, test_year=fold.test_year, term=n,
                         tier=TIER_OF[base_feature(n)], coef=w)
                )
            print(f"done: {tag} {fold.test_year}")
    return pd.DataFrame(imp_rows), pd.DataFrame(coef_rows)


def summarize_importance(imp: pd.DataFrame) -> pd.DataFrame:
    g = imp.groupby(["feature", "tier"])
    s = g.agg(mean_auc_drop=("auc_drop", "mean"), sd_auc_drop=("auc_drop", "std"),
              mean_rank=("rank", "mean"), folds_top3=("rank", lambda r: int((r <= 3).sum())))
    return s.reset_index().sort_values("mean_auc_drop", ascending=False)


def summarize_coefficients(coef: pd.DataFrame) -> pd.DataFrame:
    g = coef.groupby(["term", "tier"])
    s = g.agg(mean_coef=("coef", "mean"), sd_coef=("coef", "std"),
              folds_positive=("coef", lambda c: int((c > 0).sum())),
              folds_negative=("coef", lambda c: int((c < 0).sum())))
    s = s.reset_index()
    return s.reindex(s["mean_coef"].abs().sort_values(ascending=False).index)


def t3_comparison(res: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per fold and model: does T3 alone beat the cheaper tiers, and does it add to T1+T2?"""
    r = res.reset_index()
    r = r[r["config"].isin(["T1", "T1+T2", "T3_only", FULL]) & r["model"].isin(["lr", "rf"])]
    w = r.pivot_table(index=["project", "test_year", "model"], columns="config", values="test_roc_auc")
    w["T3_only - T1"] = w["T3_only"] - w["T1"]
    w["T3_only - (T1+T2)"] = w["T3_only"] - w["T1+T2"]
    w["(T1+T2+T3) - (T1+T2)"] = w[FULL] - w["T1+T2"]
    w = w.reset_index()
    s = w.groupby("model")[["T1", "T1+T2", "T3_only", FULL]].mean()
    s["folds_T3_only_beats_T1"] = w.groupby("model")["T3_only - T1"].apply(lambda d: int((d > 0).sum()))
    s["folds_T3_only_beats_T1+T2"] = w.groupby("model")["T3_only - (T1+T2)"].apply(lambda d: int((d > 0).sum()))
    return w, s.reset_index()


def fold_labels(df: pd.DataFrame) -> list[str]:
    return [f"{p.split('/')[1]} {y}" for p, y in df[["project", "test_year"]].drop_duplicates().itertuples(index=False)]


# --- Figures ---

def fig_cost_curve(res: pd.DataFrame, path: Path) -> None:
    r = res.reset_index()
    r = r[r["config"].isin(LADDER) & r["model"].isin(["lr", "rf"])]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, model, title in zip(axes, ("lr", "rf"), ("Logistic regression", "Random forest")):
        w = r[r["model"] == model].pivot_table(index=["project", "test_year"], columns="config", values="test_roc_auc")[LADDER]
        for _, row in w.iterrows():
            ax.plot(range(3), row.to_numpy(), color="0.75", lw=1, zorder=1)
        ax.plot(range(3), w.mean().to_numpy(), color="black", lw=2.5, marker="o", zorder=2, label="mean of 12 folds")
        ax.set_xticks(range(3), ["T1\nheader + paths", "T1+T2\n+ line diff", "T1+T2+T3\n+ history scan"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Test ROC-AUC")
    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle("Accuracy by cost tier (grey = one project-fold)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_delta_per_fold(deltas: pd.DataFrame, path: Path) -> None:
    keep = deltas[deltas["step"].isin(STEP_COLOR) & deltas["model"].isin(["lr", "rf"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, model, title in zip(axes, ("lr", "rf"), ("Logistic regression", "Random forest")):
        for offset, (step, color) in zip((-0.15, 0.15), STEP_COLOR.items()):
            d = keep[(keep["model"] == model) & (keep["step"] == step)].reset_index(drop=True)
            x = np.arange(len(d)) + offset
            ax.errorbar(x, d["delta"], yerr=[d["delta"] - d["ci_low"], d["ci_high"] - d["delta"]],
                        fmt="o", ms=4, lw=1, color=color, label=step)
        labels = fold_labels(keep[(keep["model"] == model) & (keep["step"] == "T1 -> T1+T2")])
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
        ax.axhline(THRESHOLD, color="black", ls="--", lw=1, label=f"{THRESHOLD} threshold (D8)")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("ROC-AUC gain (95% paired-bootstrap CI)")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_importance(imp_s: pd.DataFrame, coef_s: pd.DataFrame, path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5))
    s = imp_s.sort_values("mean_auc_drop")
    a.barh(s["feature"], s["mean_auc_drop"], xerr=s["sd_auc_drop"], color=[TIER_COLOR[t] for t in s["tier"]],
           error_kw=dict(lw=1, ecolor="0.3"))
    a.set_xlabel("Test ROC-AUC drop when shuffled (mean, sd over 12 folds)")
    a.set_title("Random forest: permutation importance")
    c = coef_s.sort_values("mean_coef", key=lambda v: v.abs())
    b.barh(c["term"], c["mean_coef"], xerr=c["sd_coef"], color=[TIER_COLOR[t] for t in c["tier"]],
           error_kw=dict(lw=1, ecolor="0.3"))
    b.axvline(0, color="0.5", lw=0.8)
    b.set_xlabel("Standardized coefficient (mean, sd over 12 folds)")
    b.set_title("Logistic regression: coefficients")
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in TIER_COLOR.values()]
    a.legend(handles, TIER_COLOR, loc="lower right", frameon=False)
    for ax in (a, b):
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_spearman(corr: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.index)
    for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick.set_color(TIER_COLOR[TIER_OF[tick.get_text()]])
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Spearman rho", shrink=0.8)
    ax.set_title("Feature correlation, 3 projects, 2010-2017 (label colour = tier)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", help="results directory (default: results)")
    ap.add_argument("--figures", default="figures", help="figure directory (default: figures)")
    ap.add_argument("--repeats", type=int, default=N_REPEATS, help="permutation repeats per feature")
    args = ap.parse_args()
    out, figs = Path(args.results), Path(args.figures)
    figs.mkdir(exist_ok=True)

    res = pd.read_csv(out / "results.csv")
    res = res[res["variant"] == "main"].set_index(["project", "test_year", "config", "model"])
    deltas = pd.read_csv(out / "deltas_main.csv")
    study = filter_study(load())

    imp, coef = importance_and_coefficients(study, res, args.repeats)
    imp_s, coef_s = summarize_importance(imp), summarize_coefficients(coef)
    corr = study[CONFIGS[FULL]].corr(method="spearman")
    t3_folds, t3_s = t3_comparison(res)

    imp.to_csv(out / "importance_rf_folds.csv", index=False)
    imp_s.to_csv(out / "importance_rf_summary.csv", index=False)
    coef.to_csv(out / "lr_coefficients_folds.csv", index=False)
    coef_s.to_csv(out / "lr_coefficients_summary.csv", index=False)
    corr.to_csv(out / "spearman.csv")
    t3_folds.to_csv(out / "t3_comparison_folds.csv", index=False)
    t3_s.to_csv(out / "t3_comparison_summary.csv", index=False)

    fig_cost_curve(res, figs / "f4_cost_curve.png")
    fig_delta_per_fold(deltas, figs / "f5_delta_per_fold.png")
    fig_importance(imp_s, coef_s, figs / "f6_importance.png")
    fig_spearman(corr, figs / "f6b_spearman.png")

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== RF permutation importance (ROC-AUC drop), mean over 12 folds ===")
    print(imp_s.round(4).to_string(index=False))
    print("\n=== LR standardized coefficients, mean over 12 folds ===")
    print(coef_s.round(3).to_string(index=False))
    print("\n=== Strongest feature pairs across tiers (|Spearman rho| >= 0.5) ===")
    pairs = [(a, b, corr.at[a, b]) for i, a in enumerate(corr.index) for b in corr.columns[i + 1:]
             if TIER_OF[a] != TIER_OF[b] and abs(corr.at[a, b]) >= 0.5]
    for a, b, v in sorted(pairs, key=lambda p: -abs(p[2])):
        print(f"{a} ({TIER_OF[a]}) vs {b} ({TIER_OF[b]}): {v:+.2f}")
    print("\n=== T3 alone vs cheaper tiers: mean test ROC-AUC over 12 folds ===")
    print(t3_s.round(3).to_string(index=False))
    print(f"\nWrote CSVs to {out}/ and figures to {figs}/")


if __name__ == "__main__":
    main()
