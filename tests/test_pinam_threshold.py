"""PI-NAM threshold policies (agreed 2026-10-09): the inner-CV policy must use training rows
only, cover every training row exactly once, mirror the baselines' splitter, and leave the
test predictions unchanged."""

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

import src.revision.models as models
from src.revision.models import PINAM_V1, fit_pinam
from src.revision.protocols import Split, inner_validation

N = 120


def _fake_worker(win_ti, y_ti, win_vi, y_vi, win_te, *args, **kwargs):
    """Deterministic stand-in: probability = first feature of the target rows, plus the seed."""
    seed = args[-2]
    return win_te[:, -1, 0] + 1e-3 * seed, win_vi[:, -1, 0], {"seed": seed}


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(models, "_fold_seed_worker", _fake_worker)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, N)
    win = np.zeros((N, 3, 16), dtype=np.float32)
    win[:, -1, 0] = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.1, N), 0, 1)
    splits = [Split("a", np.arange(0, 90), np.arange(90, N)),
              Split("b", np.arange(30, N), np.arange(0, 30))]
    return win, y, splits


def _fit(win, y, splits, policy):
    return fit_pinam(PINAM_V1, win, y, splits, n_seeds=3, n_jobs=1, threshold=policy)


def test_inner_cv_threshold_uses_training_rows_only(setup):
    win, y, splits = setup
    for r, s in zip(_fit(win, y, splits, "inner_cv"), splits):
        rows = r.extra["thr_rows"]
        assert np.array_equal(rows, np.sort(s.train))          # every training row exactly once
        assert not np.intersect1d(rows, s.test).size
        assert np.allclose(r.extra["p_thr"], win[rows, -1, 0] + 1e-3 * 43, atol=1e-6)


def test_inner_cv_matches_baseline_splitter(setup, monkeypatch):
    win, y, splits = setup
    seen = []
    orig = models._pinam_task

    def spy(cfg, w, yy, train, target, seed):
        seen.append((np.sort(train), np.sort(target)))
        return orig(cfg, w, yy, train, target, seed)

    monkeypatch.setattr(models, "_pinam_task", spy)
    _fit(win, y, splits[:1], "inner_cv")
    tr = splits[0].train
    expected = [np.sort(tr[te]) for _, te in StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y[tr])]
    inner_targets = [t for _, t in seen if not np.array_equal(t, splits[0].test)]
    assert all(any(np.array_equal(t, e) for t in inner_targets) for e in expected)


def test_test_predictions_identical_across_policies(setup):
    win, y, splits = setup
    for a, b in zip(_fit(win, y, splits, "val"), _fit(win, y, splits, "inner_cv")):
        assert np.array_equal(a.p_test, b.p_test)
        assert np.array_equal(a.extra["p_seeds"], b.extra["p_seeds"])


def test_val_policy_keeps_submitted_behaviour(setup):
    win, y, splits = setup
    for r, s in zip(_fit(win, y, splits, "val"), splits):
        assert np.array_equal(r.extra["thr_rows"], inner_validation(s.train)[1])


def test_unknown_policy_rejected(setup):
    win, y, splits = setup
    with pytest.raises(ValueError):
        _fit(win, y, splits, "test")
