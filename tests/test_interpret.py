"""Stage 8 interpretation helpers (src/revision/interpret.py)."""

import numpy as np
import torch

from src.pinam_search import _n_factors, train_pinamx
from src.revision.extra_models import derived_features
from src.revision.interpret import (build_model, contributions, factor_group, factor_names,
                                    group_importance, pairwise_tau, shape_curve, term_importance)
from src.revision.models import PINAM_V1

SPEC = PINAM_V1.spec


def _model():
    rng = np.random.default_rng(0)
    win = rng.normal(10, 3, (80, 14, 16)).astype(np.float32)
    win[..., 6] = rng.uniform(20, 90, (80, 14))
    y = (win[:, -1, 10] < 10).astype(int)
    m = train_pinamx(win[:60], y[:60], win[60:], y[60:], SPEC, hidden=8, max_epochs=2, depth=2)
    state = {k: v.clone() for k, v in m.state_dict().items()}
    return build_model(SPEC, 8, 0.1, 2, state), win


def test_names_match_factor_order_and_groups():
    names = factor_names(SPEC)
    assert len(names) == _n_factors(SPEC) + 6
    assert names[:2] == ["tempmax", "tempmin"] and names[-1] == "humidity x cloudcover"
    assert factor_group("windgust_min3") == "wind" and factor_group("air_density") == "temperature"
    assert factor_group("sealevelpressure_min7") == "pressure"
    assert factor_group("temp x humidity") == "interaction"
    assert {factor_group(n) for n in names} <= {"wind", "cloud", "pressure", "humidity",
                                                 "temperature", "precipitation", "interaction"}


def test_contributions_sum_to_logit():
    m, win = _model()
    c = contributions(m, win)
    assert c.shape == (80, len(factor_names(SPEC))) and np.allclose(c.mean(0), 0, atol=1e-5)
    with torch.no_grad():
        raw = m._contributions(torch.tensor(win)).numpy()
        logit = torch.logit(m(torch.tensor(win))).numpy()
    assert np.allclose(raw.sum(1) + float(m.bias.detach()), logit, atol=1e-3)


def test_shape_curve_matches_model_term():
    m, win = _model()
    names = factor_names(SPEC)
    idx = names.index("cloudcover")
    F = derived_features(win, {**SPEC, "interactions": False})
    data = F[:, idx]
    curve = shape_curve(m, idx, data, data)          # evaluated at the observed values
    assert np.allclose(curve, contributions(m, win)[:, idx], atol=1e-4)


def test_group_importance_cancels_opposite_terms():
    c = np.zeros((50, 3))
    c[:, 0] = np.linspace(-1, 1, 50)
    c[:, 1] = -c[:, 0]
    c[:, 2] = np.linspace(-0.5, 0.5, 50)
    g = group_importance(c, ["air_density", "temp", "windspeed"])
    assert g["temperature"] < 1e-9 and g["wind"] > 0.2
    assert np.allclose(term_importance(c)[:2], 0.5102, atol=1e-3)


def test_pairwise_tau():
    a = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]], float)
    t = pairwise_tau(a)
    assert len(t) == 3 and np.allclose(sorted(t), [-1, -1, 1])
