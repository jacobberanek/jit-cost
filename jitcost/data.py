"""Load ApacheJIT, restrict it to the study scope, and build chronological splits.

Every time-based split in the project lives in this file, so temporal leakage
can only enter in one place. Decisions referenced (see PROGRESS_LOG.md):
D1 own splits from the total file, D2 no Hadoop, D4 window 2010-2017,
D7 Camel/HBase/Cassandra, D9 rolling-origin test years 2014-2017,
D13 folds split on the dataset's `year` column, not on `author_date`.

Two different times exist per commit:
- `author_date` (Unix seconds, UTC): when the change was written.
- `year`: almost always equal to the author year or later (1.2% of rows are
  later, concentrated in Nov-Dec), so it most likely reflects when the change
  landed in the repository. A deployed model scores a commit when it lands,
  so folds split on `year`. `author_date` is used only for hour/weekday and
  for ordering commits inside a training period.

Run `python -m jitcost.data` from the repo root to print a sanity check.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, NamedTuple

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "apachejit_total.csv"

PROJECTS = ("apache/camel", "apache/hbase", "apache/cassandra")  # D7 (excludes Hadoop, D2)
WINDOW = (2010, 2017)  # D4, inclusive
TEST_YEARS = (2014, 2015, 2016, 2017)  # D9


class Fold(NamedTuple):
    project: str
    test_year: int
    train: pd.DataFrame
    test: pd.DataFrame


def load(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the full ApacheJIT file, add time columns, and sort by commit time.

    `author_date` is a Unix timestamp in UTC, so `hour` and `weekday` are UTC
    too (a known caveat: they mix developers' time zones).
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["author_date"], unit="s", utc=True)
    df["hour"] = df["date"].dt.hour
    df["weekday"] = df["date"].dt.day_of_week  # Monday = 0
    df["buggy"] = df["buggy"].astype(int)
    return df.sort_values(["year", "date"], kind="stable").reset_index(drop=True)


def filter_study(
    df: pd.DataFrame,
    projects: tuple[str, ...] = PROJECTS,
    window: tuple[int, int] = WINDOW,
) -> pd.DataFrame:
    """Keep only the study projects and the label-maturity window (by `year`)."""
    available = set(df["project"].unique())
    missing = set(projects) - available
    if missing:
        raise ValueError(
            f"Projects not found: {sorted(missing)}. Available: {sorted(available)}"
        )
    lo, hi = window
    mask = df["project"].isin(projects) & df["year"].between(lo, hi)
    return df.loc[mask].reset_index(drop=True)


def rolling_folds(
    df: pd.DataFrame,
    project: str,
    test_years: tuple[int, ...] = TEST_YEARS,
    gap_months: int = 0,
) -> Iterator[Fold]:
    """Yield one fold per test year: train on every `year` before Y, test on `year` == Y.

    gap_months > 0 also drops training commits authored in the last months
    before Y (sensitivity S1: those labels would not be known yet in deployment).
    """
    p = df[df["project"] == project]
    for y in test_years:
        train = p[p["year"] < y]
        if gap_months:
            cutoff = pd.Timestamp(year=y, month=1, day=1, tz="UTC") - pd.DateOffset(
                months=gap_months
            )
            train = train[train["date"] < cutoff]
        test = p[p["year"] == y]
        yield Fold(project, y, train.reset_index(drop=True), test.reset_index(drop=True))


def time_val_split(
    train: pd.DataFrame, frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a training period into (fit, validation): validation is the latest `frac` by time.

    Used for hyperparameter tuning only. Never tune on the test year.
    """
    train = train.sort_values(["year", "date"], kind="stable")
    cut = len(train) - int(round(len(train) * frac))
    return train.iloc[:cut].reset_index(drop=True), train.iloc[cut:].reset_index(drop=True)


if __name__ == "__main__":
    full = load()
    print(f"Loaded {len(full):,} commits, {full['buggy'].mean():.1%} buggy")

    diff = full["year"] - full["date"].dt.year
    print(
        f"`year` vs author year: {(diff > 0).sum():,} later, {(diff < 0).sum():,} earlier "
        f"(expected about 1,310 later and 4 earlier)"
    )

    study = filter_study(full)
    print(f"Study scope: {len(study):,} commits, {study['buggy'].mean():.1%} buggy\n")

    print(f"{'project':<18}{'year':>6}{'train':>8}{'test':>7}{'test buggy':>12}{'rate':>7}")
    for project in PROJECTS:
        for fold in rolling_folds(study, project):
            nb = fold.test["buggy"].sum()
            print(
                f"{project:<18}{fold.test_year:>6}{len(fold.train):>8}"
                f"{len(fold.test):>7}{nb:>12}{nb / len(fold.test):>7.1%}"
            )