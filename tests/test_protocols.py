"""Protocol and metric checks (R2-1, R3-2, R3-6)."""

import numpy as np
import pandas as pd

from src.revision.metrics import best_threshold, compute_metrics
from src.revision.protocols import inner_validation, split_blocked, split_random, split_year

DATES = pd.Series(pd.date_range("2023-01-04", "2024-12-31", freq="D"))
N = len(DATES)


def _check_disjoint(splits):
    for s in splits:
        assert len(np.intersect1d(s.train, s.test)) == 0


def test_random_matches_submitted_and_covers_all():
    y = np.random.default_rng(0).integers(0, 2, N)
    splits = split_random(y)
    _check_disjoint(splits)
    assert np.array_equal(np.sort(np.concatenate([s.test for s in splits])), np.arange(N))


def test_blocked_embargo_and_coverage():
    splits = split_blocked(N, n_blocks=10, embargo=14)
    _check_disjoint(splits)
    assert np.array_equal(np.sort(np.concatenate([s.test for s in splits])), np.arange(N))
    for s in splits:
        gap = np.abs(s.train[:, None] - s.test[None, :]).min()
        assert gap > 14       # no training day within 14 days of any test day


def test_year_split_is_chronological():
    (s,) = split_year(DATES)
    assert DATES.iloc[s.train].dt.year.eq(2023).all()
    assert DATES.iloc[s.test].dt.year.eq(2024).all()
    assert s.train.max() < s.test.min()


def test_inner_validation_is_latest_part_of_train():
    train = np.arange(100)
    fit, val = inner_validation(train)
    assert len(val) == 20 and fit.max() < val.min()
    assert len(np.intersect1d(fit, val)) == 0


def test_metrics_confusion_and_far():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([.1, .2, .7, .8, .3, .6, .9, .95])
    m = compute_metrics(y, p, 0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 2, 1, 3)
    assert m["false_authorization_rate"] == 0.5 and m["specificity"] == 0.5
    assert 0.2 <= best_threshold(p, y) < 0.8
