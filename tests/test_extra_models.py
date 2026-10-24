"""Stage 6 comparison models (src/revision/extra_models.py and the physics MLP)."""

import numpy as np
import pytest
import torch

from src.pinam_search import INTERACTION_PAIRS, MONO_DEC, PhysMLP, _n_factors, train_pinamx
from src.revision.extra_models import (derived_features, ensemble_fd, fit_tabular, raw_features)
from src.revision.models import PINAM_V1, PinamConfig, fit_baseline
from src.revision.protocols import Split

N, W = 160, 14


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    win = rng.normal(10, 3, (N, W, 16)).astype(np.float32)
    win[..., 4] = rng.uniform(0, 100, (N, W))        # humidity-like channel in range
    y = (win[:, -1, 1] + rng.normal(0, 2, N) < 10).astype(int)
    splits = [Split("a", np.arange(0, 120), np.arange(120, N)),
              Split("b", np.arange(40, N), np.arange(0, 40))]
    return win, y, splits


def test_derived_features_shape_and_raw_prefix(data):
    win, _, _ = data
    F = derived_features(win)
    assert F.shape == (N, _n_factors(PINAM_V1.spec) + len(INTERACTION_PAIRS))
    assert np.allclose(F[:, :16], raw_features(win))
    assert np.isfinite(F).all()


def test_original_baseline_refit_matches_grid_fit(data):
    win, y, splits = data
    ref = fit_baseline("XGBoost", {"n_estimators": 50}, win[:, -1, :], y, splits, n_jobs=1)
    new = fit_tabular("XGBoost", {"n_estimators": 50}, win, y, splits, n_jobs=1)
    for a, b in zip(ref, new):
        assert np.allclose(a.p_test, b.p_test) and a.threshold == b.threshold
        assert b.extra["fd"].shape == (len(MONO_DEC), len(b.test))


@pytest.mark.parametrize("name", ["XGBoost-mono", "LightGBM-mono"])
def test_monotone_trees_never_increase(data, name):
    win, y, splits = data
    for r in fit_tabular(name, {"n_estimators": 50}, win, y, splits, n_jobs=1):
        assert r.extra["fd"].max() <= 1e-9


def test_ebm_and_derived_models_run(data):
    win, y, splits = data
    for name in ("EBM", "LightGBM-derived"):
        res = fit_tabular(name, {} if name == "EBM" else {"n_estimators": 30}, win, y, splits[:1],
                          n_jobs=1)
        assert res[0].p_test.shape == (40,) and 0 < res[0].threshold < 1


def test_physics_mlp_trains_and_fd_runs(data):
    win, y, splits = data
    cfg = PinamConfig(spec=PINAM_V1.spec, arch="mlp")
    m = train_pinamx(win[:100], y[:100], win[100:], y[100:], cfg.spec, lam_mono=10.0,
                     hidden=cfg.hidden, dropout=cfg.dropout, depth=cfg.depth, max_epochs=3,
                     arch="mlp")
    assert isinstance(m, PhysMLP) and not len(m.shapes)
    st = {k: v.clone() for k, v in m.state_dict().items()}

    class R:                       # minimal SplitResult stand-in
        test = np.arange(100, N)
        extra = {"states": [st]}
    fd = ensemble_fd(cfg, win, [R()])[0]
    assert fd.shape == (len(MONO_DEC), N - 100) and np.isfinite(fd).all()
    with torch.no_grad():
        assert m(torch.tensor(win[:5])).shape == (5,)
