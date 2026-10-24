"""Checks of the significance tests used in stage 4 (src/revision/stats.py)."""

import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from src.revision.stats import delong, holm, mcnemar_exact, paired_bootstrap_diff, wilcoxon_paired


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    good = np.clip(y * 0.6 + rng.normal(0.2, 0.25, n), 0, 1)
    weak = np.clip(y * 0.2 + rng.normal(0.4, 0.25, n), 0, 1)
    return y, good, weak


def test_holm_matches_reference():
    # reference: statsmodels multipletests(method="holm")
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])   # monotone: 0.9 is lifted to 1.0


def test_delong_auc_equals_sklearn_and_detects_difference():
    y, good, weak = _toy()
    r = delong(y, good, weak)
    assert r["auc1"] == pytest.approx(roc_auc_score(y, good))
    assert r["auc2"] == pytest.approx(roc_auc_score(y, weak))
    assert r["p"] < 1e-6


def test_delong_identical_predictions():
    y, good, _ = _toy()
    assert delong(y, good, good)["p"] == 1.0


def test_delong_matches_bootstrap_se():
    y, good, weak = _toy(n=300, seed=3)
    rng = np.random.default_rng(1)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        boots.append(roc_auc_score(y[i], good[i]) - roc_auc_score(y[i], weak[i]))
    r = delong(y, good, weak)
    assert abs(r["diff"]) / abs(r["z"]) == pytest.approx(np.std(boots), rel=0.15)


def test_mcnemar_counts_and_null():
    y = np.array([1, 1, 0, 0, 1, 0])
    a = np.array([1, 1, 0, 0, 0, 0])
    b = np.array([1, 0, 1, 0, 1, 0])
    r = mcnemar_exact(y, a, b)
    assert (r["only1_correct"], r["only2_correct"]) == (2, 1)
    assert mcnemar_exact(y, a, a)["p"] == 1.0


def test_wilcoxon_sign_and_effect():
    a = np.array([0.80, 0.82, 0.85, 0.81, 0.83, 0.84, 0.86, 0.80, 0.82, 0.85])
    r = wilcoxon_paired(a, a - 0.01 * np.arange(1, 11))
    assert r["rank_biserial"] == 1.0 and r["p"] < 0.01 and r["mean_diff"] > 0
    assert wilcoxon_paired(a, a)["p"] == 1.0


def test_paired_bootstrap_contains_point_estimate():
    y, good, weak = _toy()
    thr = np.full(len(y), 0.5)
    f1 = lambda yy, pp, tt: f1_score(yy, (pp >= tt).astype(int))  # noqa: E731
    r = paired_bootstrap_diff(y, good, thr, weak, thr, f1, groups=np.arange(len(y)) // 7, n_boot=300)
    assert r["ci_lo"] <= r["diff"] <= r["ci_hi"] and r["ci_lo"] > 0


def test_circular_mean_sd_wraps_around_north():
    from src.revision.stats import circular_mean_sd
    m, sd = circular_mean_sd(np.array([350.0, 10.0]))
    assert min(abs(m - 0.0), abs(m - 360.0)) < 1e-6      # not the linear mean 180
    assert 9.9 < sd < 10.2                               # sqrt(-2 ln cos 10°) ≈ 10.0°
    m2, sd2 = circular_mean_sd(np.array([90.0, 90.0, 90.0]))
    assert abs(m2 - 90.0) < 1e-6 and sd2 < 1e-6
