"""Model fitting under the rev1 protocols: tuned baselines and the PI-NAM deep ensemble.

Both families follow the same leakage-free rule per outer split:
  * hyperparameters come from a tuning run that never saw the test rows,
  * the decision threshold is chosen on training-portion predictions only
    (baselines: stratified 5-fold inner CV predictions; PI-NAM: either the chronological
    inner validation set -- submitted behaviour -- or, with threshold="inner_cv", the same
    stratified 5-fold inner CV as the baselines, each inner fold a full 15-seed ensemble),
  * test rows are predicted exactly once.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.pinam_search import _fold_seed_worker
from src.revision.metrics import best_threshold
from src.revision.protocols import Split, inner_validation

logger = logging.getLogger(__name__)

BASELINES = ("SVM", "XGBoost", "LightGBM", "Random Forest", "CatBoost", "TabPFN")


@dataclass
class SplitResult:
    """Predictions of one model on one outer split."""

    split: str
    test: np.ndarray
    p_test: np.ndarray
    threshold: float
    extra: Dict = field(default_factory=dict)


def make_baseline(name: str, params: Optional[Dict], seed: int):
    """Scaled classifier with the given (tuned) hyperparameters."""
    p = dict(params or {})
    if name == "SVM":
        from sklearn.svm import SVC
        clf = SVC(kernel="rbf", probability=True, random_state=seed, **p)
    elif name == "XGBoost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(**p, random_state=seed, verbosity=0, eval_metric="logloss", n_jobs=2)
    elif name == "LightGBM":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(**p, random_state=seed, verbose=-1, n_jobs=2)
    elif name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(**p, random_state=seed, n_jobs=2)
    elif name == "CatBoost":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(**p, random_seed=seed, verbose=0, thread_count=2)
    elif name == "TabPFN":
        import torch
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu",
                               random_state=seed)
    else:
        raise ValueError(f"unknown baseline {name!r}")
    return make_pipeline(StandardScaler(), clf)


def _baseline_one_split(name, params, X, y, split: Split, seed, inner_folds) -> SplitResult:
    tr, te = split.train, split.test
    inner = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    p_inner = cross_val_predict(make_baseline(name, params, seed), X[tr], y[tr], cv=inner,
                                method="predict_proba")[:, 1]
    thr = best_threshold(p_inner, y[tr])
    model = make_baseline(name, params, seed).fit(X[tr], y[tr])
    # inner predictions are kept so alternative threshold policies can be applied post hoc
    return SplitResult(split.name, te, model.predict_proba(X[te])[:, 1], thr,
                       {"thr_rows": tr, "p_thr": p_inner})


def fit_baseline(name: str, params: Optional[Dict], X: np.ndarray, y: np.ndarray,
                 splits: List[Split], seed: int = 42, inner_folds: int = 5,
                 n_jobs: int = 10) -> List[SplitResult]:
    """Fit one baseline on every outer split (splits run in parallel; TabPFN serially on GPU)."""
    jobs = 1 if name == "TabPFN" else n_jobs
    return Parallel(n_jobs=jobs)(delayed(_baseline_one_split)(name, params, X, y, s, seed, inner_folds)
                                 for s in splits)


@dataclass(frozen=True)
class PinamConfig:
    """PI-NAM structure + training hyperparameters (V1 by default)."""

    spec: Dict
    lam_mono: float = 0.081
    hidden: int = 48
    dropout: float = 0.2
    lr: float = 2e-4
    weight_decay: float = 1e-3
    batch_size: int = 32
    depth: int = 2
    arch: str = "nam"          # "mlp": non-additive physics MLP control (stage 6)


PINAM_V1 = PinamConfig(spec={"physics": True, "stats": ["min"], "scales": [3, 7],
                             "interactions": True, "synoptic": True, "syn_scale": 5})


THRESHOLD_POLICIES = ("val", "inner_cv")


def _pinam_task(cfg: PinamConfig, win, y, train, target, seed):
    """One PI-NAM fit on `train` (chronological early-stopping split) predicting `target`."""
    fit, val = inner_validation(train)
    return delayed(_fold_seed_worker)(win[fit], y[fit], win[val], y[val], win[target], cfg.spec,
                                      cfg.lam_mono, cfg.hidden, cfg.dropout, cfg.lr,
                                      cfg.weight_decay, cfg.batch_size, seed, cfg.depth,
                                      arch=cfg.arch)


def fit_pinam(cfg: PinamConfig, win: np.ndarray, y: np.ndarray, splits: List[Split],
              n_seeds: int = 15, base_seed: int = 42, n_jobs: int = 28,
              keep_states: bool = False, threshold: str = "val",
              inner_folds: int = 5) -> List[SplitResult]:
    """Deep-ensemble PI-NAM on every split; all fits run in one parallel pool.

    threshold="val": threshold from the ensemble's predictions on the chronological inner
    validation set (submitted behaviour). threshold="inner_cv": threshold from out-of-fold
    predictions of a stratified `inner_folds`-fold CV on the training rows (same splitter and
    seed as the baselines), each inner fold an `n_seeds` ensemble of the same configuration.
    Test predictions are identical under both policies.
    """
    if threshold not in THRESHOLD_POLICIES:
        raise ValueError(f"unknown threshold policy {threshold!r}; expected {THRESHOLD_POLICIES}")
    tasks, meta = [], []
    for k, s in enumerate(splits):
        val = inner_validation(s.train)[1]
        for j in range(n_seeds):
            tasks.append(_pinam_task(cfg, win, y, s.train, s.test, base_seed + j))
            meta.append((k, "final", val))
        if threshold == "inner_cv":
            skf = StratifiedKFold(inner_folds, shuffle=True, random_state=base_seed)
            for f, (itr, ite) in enumerate(skf.split(s.train, y[s.train])):
                for j in range(n_seeds):
                    tasks.append(_pinam_task(cfg, win, y, s.train[itr], s.train[ite], base_seed + j))
                    meta.append((k, f, s.train[ite]))
    out = Parallel(n_jobs=n_jobs, backend="loky")(tasks)

    results = []
    for k, s in enumerate(splits):
        final = [r for r, m in zip(out, meta) if m[0] == k and m[1] == "final"]
        p_test = np.mean([r[0] for r in final], axis=0)
        if threshold == "val":
            thr_rows = inner_validation(s.train)[1]
            p_thr = np.mean([r[1] for r in final], axis=0)
        else:
            thr_rows, p_thr = _inner_oof(out, meta, k, inner_folds)
        # threshold-source predictions and per-seed test predictions are kept
        extra = {"thr_rows": thr_rows, "p_thr": p_thr, "p_seeds": np.stack([r[0] for r in final])}
        if keep_states:
            extra["states"] = [r[2] for r in final]
        results.append(SplitResult(s.name, s.test, p_test, best_threshold(p_thr, y[thr_rows]), extra))
    return results


def _inner_oof(out, meta, k: int, inner_folds: int):
    """Out-of-fold ensemble predictions on the training rows of split k (sorted by row)."""
    rows, preds = [], []
    for f in range(inner_folds):
        fold = [(r, m[2]) for r, m in zip(out, meta) if m[0] == k and m[1] == f]
        rows.append(fold[0][1])
        preds.append(np.mean([r[0] for r, _ in fold], axis=0))
    rows, preds = np.concatenate(rows), np.concatenate(preds)
    order = np.argsort(rows)
    return rows[order], preds[order]
