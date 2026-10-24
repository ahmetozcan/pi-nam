"""Leakage checks for decision-time information sets (R2-1, R2-2, R3-1)."""

from pathlib import Path

import numpy as np
import pytest

from src.pinam_search import WINDOW, build_windows
from src.revision.info_sets import load_inputs

ROOT = Path(__file__).resolve().parents[1]


def _index_raw(n: int) -> np.ndarray:
    """Raw matrix whose every cell holds its own row index (day id)."""
    return np.repeat(np.arange(n, dtype=np.float32)[:, None], 3, axis=1)


def test_lag0_window_ends_on_same_day():
    win = build_windows(_index_raw(30), W=WINDOW, lag=0)
    assert np.all(win[:, -1, 0] == np.arange(30))


def test_lag1_window_never_contains_decision_day():
    n, h = 30, 3
    ext = _index_raw(n + h)
    win = build_windows(ext[h:], W=WINDOW, lag=1, history=ext[:h])
    day_ids = np.arange(h, n + h)
    assert np.all(win[:, -1, 0] == day_ids - 1)
    assert np.all(win[:, :, 0] < day_ids[:, None])


def test_lag1_without_history_raises():
    with pytest.raises(ValueError):
        build_windows(_index_raw(10), W=WINDOW, lag=1)


def test_s0_matches_submitted_pipeline():
    s0 = load_inputs("S0", ROOT / "config.yaml")
    ref = build_windows(s0.win[:, -1, :])
    np.testing.assert_array_equal(s0.win, ref)
    assert len(s0.y) == 728 and int(s0.y.sum()) == 451


def test_s1_uses_only_previous_days():
    s0 = load_inputs("S0", ROOT / "config.yaml")
    s1 = load_inputs("S1", ROOT / "config.yaml")
    np.testing.assert_array_equal(s1.y, s0.y)
    # S1 tabular input on day t equals the S0 input of day t-1
    np.testing.assert_array_equal(s1.X[1:], s0.X[:-1])
    # and no S1 window contains the decision day's own values
    assert not np.any(np.all(s1.win == s0.X[:, None, :], axis=2)[:, -1])
