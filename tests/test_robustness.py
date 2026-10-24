"""Stage 7 robustness inputs (src/revision/robustness.py)."""

import numpy as np
import pytest

from src.pinn import IDX, RAW_CHANNELS
from src.revision.protocols import Split
from src.revision.robustness import (GROUPS, IMPUTERS, MISSING_FRACS, drop_group_windows,
                                     fill_group, inference_conditions, random_missing,
                                     subsample_splits, windows)

N = 120


@pytest.fixture(scope="module")
def raw():
    return np.random.default_rng(0).normal(10, 3, (N, len(RAW_CHANNELS))).astype(np.float32)


def test_groups_cover_all_channels_once():
    chans = [c for g in GROUPS.values() for c in g]
    assert sorted(chans) == sorted(RAW_CHANNELS)


def test_fill_group_changes_only_the_group(raw):
    fill = raw[:60].mean(0)
    out = fill_group(raw, "wind", fill)
    idx = [IDX[c] for c in GROUPS["wind"]]
    rest = [i for i in range(len(RAW_CHANNELS)) if i not in idx]
    assert np.allclose(out[:, idx], fill[idx]) and np.array_equal(out[:, rest], raw[:, rest])
    assert not np.shares_memory(out, raw)


@pytest.mark.parametrize("method", IMPUTERS)
def test_random_missing_rate_and_imputation(raw, method):
    fill = raw.mean(0)
    out = random_missing(raw, 0.3, np.random.default_rng(1), method, fill)
    changed = out != raw
    assert np.isfinite(out).all() and 0.2 < changed.mean() < 0.4
    if method == "mean":
        assert np.allclose(out[changed], np.broadcast_to(fill, raw.shape)[changed])
    else:                                  # carried forward: equals the previous day's value
        t, j = np.argwhere(changed[1:])[0]
        assert out[t + 1, j] == out[t, j]


def test_inference_conditions_names_shapes_and_clean(raw):
    tr, te = np.arange(0, 80), np.arange(80, N)
    conds = dict(inference_conditions(raw, tr, te, n_reps=2))
    assert len(conds) == 1 + len(GROUPS) + len(MISSING_FRACS) * 2 * len(IMPUTERS)
    assert np.array_equal(conds["clean"], windows(raw)[te])
    assert all(w.shape == (len(te), 14, len(RAW_CHANNELS)) for w in conds.values())
    # the decision-day window row of a masked group equals the training mean
    j = IDX["humidity"]
    assert np.allclose(conds["mask_humidity"][:, :, j], raw[tr, j].mean())
    # repetitions differ, but are reproducible
    again = dict(inference_conditions(raw, tr, te, n_reps=2))
    assert not np.array_equal(conds["miss20_mean_r0"], conds["miss20_mean_r1"])
    assert np.array_equal(conds["miss20_mean_r1"], again["miss20_mean_r1"])


def test_windows_rebuild_matches_s0_decision_day(raw):
    w = windows(raw)
    assert np.array_equal(w[:, -1, :], raw) and np.array_equal(w[20, 0], raw[7])


def test_drop_group_is_constant_everywhere(raw):
    w = drop_group_windows(raw, "pressure")
    assert np.ptp(w[..., IDX["sealevelpressure"]]) == 0


def test_subsample_splits_stratified_and_keeps_test():
    y = np.r_[np.zeros(60, int), np.ones(40, int)]
    s = [Split("a", np.arange(0, 80), np.arange(80, 100))]
    sub = subsample_splits(s, y, 0.5, seed=42)[0]
    assert len(sub.train) == 40 and np.array_equal(sub.test, s[0].test)
    assert set(sub.train) <= set(s[0].train) and np.all(np.diff(sub.train) > 0)
    assert abs(y[sub.train].mean() - y[s[0].train].mean()) < 0.05
    assert subsample_splits(s, y, 1.0, seed=0) is s
