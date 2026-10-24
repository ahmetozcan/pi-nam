"""Paired significance tests for the rev1 model comparison (R2-6, R3-6, R3-8).

* CV protocols (rand, block): fold-level paired Wilcoxon signed-rank test with
  rank-biserial effect size; Holm correction across the baselines compared.
* Year protocol (single 2024 test set): DeLong test for correlated ROC AUCs,
  exact McNemar test on the thresholded decisions, and a paired (weekly block)
  bootstrap CI for differences of threshold metrics.
"""

from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy import stats


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Two-sided Wilcoxon signed-rank test of fold scores a vs b.

    Returns the mean difference, the p-value (exact for small n, zero differences
    dropped) and the matched-pairs rank-biserial correlation r in [-1, 1].
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    nz = d[d != 0]
    if len(nz) == 0:
        return {"mean_diff": 0.0, "p": 1.0, "rank_biserial": 0.0, "n": int(len(d))}
    p = stats.wilcoxon(nz, alternative="two-sided").pvalue
    ranks = stats.rankdata(np.abs(nz))
    r = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
    return {"mean_diff": float(d.mean()), "p": float(p), "rank_biserial": float(r), "n": int(len(d))}


def _midrank(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x)


def delong(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Dict[str, float]:
    """DeLong et al. (1988) test for two correlated ROC AUCs on the same cases."""
    y = np.asarray(y).astype(bool)
    pos, neg = y.sum(), (~y).sum()
    aucs, v10, v01 = [], [], []
    for p in (p1, p2):
        p = np.asarray(p, float)
        r_all, r_pos, r_neg = _midrank(p), _midrank(p[y]), _midrank(p[~y])
        tx = r_all[y] - r_pos                      # placements of positives
        ty = r_all[~y] - r_neg                     # placements of negatives
        aucs.append((r_all[y].sum() - pos * (pos + 1) / 2) / (pos * neg))
        v10.append(tx / neg)
        v01.append(1.0 - ty / pos)
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / pos + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / neg
    diff = aucs[0] - aucs[1]
    if var <= 0:
        return {"auc1": aucs[0], "auc2": aucs[1], "diff": diff, "z": 0.0, "p": 1.0}
    z = diff / np.sqrt(var)
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": float(diff), "z": float(z),
            "p": float(2 * stats.norm.sf(abs(z)))}


def mcnemar_exact(y: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> Dict[str, float]:
    """Exact (binomial) McNemar test on the correctness of two decision vectors."""
    c1, c2 = np.asarray(pred1) == y, np.asarray(pred2) == y
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    p = 1.0 if b + c == 0 else stats.binomtest(b, b + c, 0.5).pvalue
    return {"only1_correct": b, "only2_correct": c, "p": float(p)}


def paired_bootstrap_diff(y: np.ndarray, p1: np.ndarray, t1: np.ndarray, p2: np.ndarray,
                          t2: np.ndarray, metric: Callable, groups: Optional[np.ndarray] = None,
                          n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
    """Percentile CI of metric(model1) - metric(model2), resampling the same units for both.

    `metric(y, p, thr)` takes per-row thresholds; `groups` (e.g. ISO weeks) gives a block
    bootstrap that respects day-to-day autocorrelation.
    """
    rng = np.random.default_rng(seed)
    units = np.arange(len(y)) if groups is None else np.unique(groups)
    index = None if groups is None else {g: np.flatnonzero(groups == g) for g in units}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(units, size=len(units), replace=True)
        idx = pick if groups is None else np.concatenate([index[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(metric(y[idx], p1[idx], t1[idx]) - metric(y[idx], p2[idx], t2[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(metric(y, p1, t1) - metric(y, p2, t2)), "ci_lo": float(lo),
            "ci_hi": float(hi)}


def circular_mean_sd(deg: np.ndarray) -> tuple:
    """Circular mean and circular SD (degrees) of directions such as wind direction.

    The circular SD is sqrt(-2 ln R), where R is the mean resultant length (Mardia & Jupp).
    """
    rad = np.deg2rad(np.asarray(deg, dtype=float))
    s, c = np.sin(rad).mean(), np.cos(rad).mean()
    r = min(np.hypot(s, c), 1.0)
    mean = np.rad2deg(np.arctan2(s, c)) % 360.0
    sd = np.rad2deg(np.sqrt(-2.0 * np.log(r))) if r > 0 else float("inf")
    return float(mean), float(sd)
