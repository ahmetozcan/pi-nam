"""Validation protocols for the rev1 revision (R2-1, R3-2).

P-rand  -- stratified 10-fold CV (the submitted protocol, kept for continuity)
P-block -- 10 contiguous time blocks with a symmetric embargo around the test block
P-year  -- train on 2023, test on 2024 (genuinely future, unseen year)

Every split also defines an inner validation set (`inner_validation`) that is
carved out of the training portion only; it is used for early stopping and for
choosing the decision threshold, so no test-day label ever influences fitting,
early stopping, threshold selection or hyperparameter tuning.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROTOCOLS = ("rand", "block", "year")


@dataclass(frozen=True)
class Split:
    """One outer train/test split; indices refer to rows of the model frame."""

    name: str
    train: np.ndarray
    test: np.ndarray


def split_random(y: np.ndarray, n_splits: int = 10, seed: int = 42) -> List[Split]:
    """Stratified shuffled K-fold, identical to the submitted manuscript."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [Split(f"fold{i:02d}", tr, te)
            for i, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1)]


def split_blocked(n: int, n_blocks: int = 10, embargo: int = 14) -> List[Split]:
    """Contiguous blocks; training rows within `embargo` days of the test block are purged.

    The embargo equals the PI-NAM window length (14 days), so no training sample's
    input window overlaps a test day and short-range autocorrelation is removed.
    """
    bounds = np.linspace(0, n, n_blocks + 1).astype(int)
    splits = []
    for i in range(n_blocks):
        lo, hi = bounds[i], bounds[i + 1]
        test = np.arange(lo, hi)
        keep = (np.arange(n) < lo - embargo) | (np.arange(n) >= hi + embargo)
        splits.append(Split(f"block{i + 1:02d}", np.flatnonzero(keep), test))
    return splits


def split_year(dates: pd.Series, train_year: int = 2023, test_year: int = 2024) -> List[Split]:
    """Single chronological split: fit on one calendar year, evaluate on the next."""
    years = pd.to_datetime(dates).dt.year.to_numpy()
    return [Split(f"{train_year}->{test_year}", np.flatnonzero(years == train_year),
                  np.flatnonzero(years == test_year))]


def make_splits(protocol: str, y: np.ndarray, dates: pd.Series, seed: int = 42) -> List[Split]:
    if protocol == "rand":
        return split_random(y, seed=seed)
    if protocol == "block":
        return split_blocked(len(y))
    if protocol == "year":
        return split_year(dates)
    raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")


def inner_validation(train: np.ndarray, frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Split training rows into (fit, validation); validation = the latest `frac` in time.

    Rows are in chronological order, so the most recent training days act as a
    pseudo-future validation set for early stopping and threshold selection.
    """
    train = np.sort(train)
    n_val = max(int(len(train) * frac), 10)
    return train[:-n_val], train[-n_val:]
