"""Metric suite for the rev1 revision (R2-6, R3-6, R3-7, R3-8).

All threshold-dependent metrics use a threshold chosen on training-portion
predictions only (`best_threshold`), never on test labels.
"""

from typing import Callable, Dict, Optional

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

THR_GRID = np.arange(0.2, 0.8, 0.005)   # same grid as the submitted manuscript


def best_threshold(p: np.ndarray, y: np.ndarray) -> float:
    """Threshold maximizing F1 on (training-portion) predictions."""
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in THR_GRID]
    return float(THR_GRID[int(np.argmax(scores))])


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-bin ECE."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def compute_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    """Discrimination, threshold, safety and calibration metrics for one prediction set.

    false_authorization_rate = FP / (FP + TN): share of no-flight days the model
    would have authorized -- the safety-critical error for this application.
    """
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)), "prevalence": float(y.mean()), "threshold": float(thr),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p) if both else np.nan,
        "pr_auc": average_precision_score(y, p) if both else np.nan,
        "accuracy": accuracy_score(y, pred),
        "balanced_acc": balanced_accuracy_score(y, pred) if both else np.nan,
        "mcc": matthews_corrcoef(y, pred),
        "precision_fly": precision_score(y, pred, pos_label=1, zero_division=0),
        "recall_fly": recall_score(y, pred, pos_label=1, zero_division=0),
        "precision_nofly": precision_score(y, pred, pos_label=0, zero_division=0),
        "recall_nofly": recall_score(y, pred, pos_label=0, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "false_authorization_rate": fp / (fp + tn) if (fp + tn) else np.nan,
        "missed_flight_rate": fn / (fn + tp) if (fn + tp) else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "brier": brier_score_loss(y, p),
        "ece": expected_calibration_error(y, p),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, thr: np.ndarray,
                 metric: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
                 n_boot: int = 2000, seed: int = 42, alpha: float = 0.05,
                 groups: Optional[np.ndarray] = None) -> tuple:
    """Percentile bootstrap CI over days (or over `groups`, e.g. weeks, if given).

    `thr` is per-row so pooled CV predictions keep their own fold thresholds.
    """
    rng = np.random.default_rng(seed)
    units = np.arange(len(y)) if groups is None else np.unique(groups)
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(units, size=len(units), replace=True)
        idx = pick if groups is None else np.concatenate([np.flatnonzero(groups == g) for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(metric(y[idx], p[idx], thr[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """Mean predicted vs observed frequency per equal-width bin (for calibration plots)."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows
