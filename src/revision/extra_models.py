"""Stage 6 comparison models (R2-5, R3-3, R1): architecture vs representation.

Tabular models (one fit per outer split, threshold from the baselines' stratified 5-fold
inner CV, exactly as `models.fit_baseline`):
  XGBoost-mono, LightGBM-mono       raw-16 with monotone_constraints = -1 on MONO_DEC (R3-3)
  XGBoost-derived, LightGBM-derived PI-NAM V1's own factors (raw-16 + physics + temporal +
                                    synoptic) plus the products of its interaction pairs (R2-5)
  EBM                               Explainable Boosting Machine on raw-16, default settings
  <original baselines>              refitted unchanged, only to measure monotonicity

Every tabular fit also records a finite-difference monotonicity check on its test rows:
the change in predicted probability when one MONO_DEC channel of the decision day is raised
by one (all-row) standard deviation, with derived features recomputed from the perturbed
window. The PI-NAM family (incl. the physics MLP, `PinamConfig(arch="mlp")`) goes through
`models.fit_pinam`; its check is computed from the saved states (`ensemble_fd`).
"""

import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.optuna_tuning import _cv_f1, _suggest
from src.pinam_search import ARCHS, INTERACTION_PAIRS, MONO_DEC, PINAMX, _n_factors
from src.pinn import IDX, RAW_CHANNELS
from src.revision.metrics import best_threshold
from src.revision.models import PINAM_V1, BASELINES, PinamConfig, SplitResult, make_baseline
from src.revision.protocols import Split

logger = logging.getLogger(__name__)

MONO_MODELS = ("XGBoost-mono", "LightGBM-mono")
DERIVED_MODELS = ("XGBoost-derived", "LightGBM-derived")
TABULAR_EXTRA = MONO_MODELS + DERIVED_MODELS + ("EBM",)
FeatureFn = Callable[[np.ndarray], np.ndarray]


# ── Feature maps (window -> tabular matrix) ──────────────────────────────────

def raw_features(win: np.ndarray) -> np.ndarray:
    """Decision-day raw-16, i.e. the baselines' X."""
    return win[:, -1, :]


def derived_features(win: np.ndarray, spec: Optional[Dict] = None) -> np.ndarray:
    """PI-NAM's factor matrix for `spec` (V1 by default) plus interaction-pair products."""
    spec = spec or PINAM_V1.spec
    n = _n_factors(spec)
    with torch.no_grad():
        m = PINAMX(torch.zeros(n), torch.ones(n), n, {**spec, "interactions": False})
        F = m._raw_factors(torch.tensor(win, dtype=torch.float32)).numpy()
    parts = [F]
    if spec.get("interactions"):
        cur = win[:, -1, :]
        parts.append(np.stack([cur[:, IDX[a]] * cur[:, IDX[b]] for a, b in INTERACTION_PAIRS], 1))
    return np.nan_to_num(np.concatenate(parts, 1).astype(np.float32), posinf=0.0, neginf=0.0)


def feature_fn(name: str) -> FeatureFn:
    return derived_features if name in DERIVED_MODELS else raw_features


def base_learner(name: str) -> str:
    """Optuna search space / estimator family of an extra model."""
    return name.split("-")[0]


# ── Estimators and tuning ────────────────────────────────────────────────────

def _mono_vector() -> List[int]:
    return [-1 if c in MONO_DEC else 0 for c in RAW_CHANNELS]


def make_extra(name: str, params: Optional[Dict], seed: int):
    """Estimator for an extra or original model (scaling keeps monotone directions)."""
    p = dict(params or {})
    if name in BASELINES:
        return make_baseline(name, p, seed)
    if name == "EBM":
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(random_state=seed, n_jobs=2, **p)
    if name in MONO_MODELS:
        p["monotone_constraints"] = (tuple(_mono_vector()) if name.startswith("XGBoost")
                                     else _mono_vector())
    if name.startswith("XGBoost"):
        from xgboost import XGBClassifier
        clf = XGBClassifier(**p, random_state=seed, verbosity=0, eval_metric="logloss", n_jobs=2)
    elif name.startswith("LightGBM"):
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(**p, random_state=seed, verbose=-1, n_jobs=2)
    else:
        raise ValueError(f"unknown extra model {name!r}")
    return make_pipeline(StandardScaler(), clf)


def tune_extra(name: str, X: np.ndarray, y: np.ndarray, seed: int = 42, n_trials: int = 40) -> Dict:
    """Submitted Optuna search of the base learner (10-fold CV F1), fitted as `name`
    (constraints are active during the search). EBM is not tuned."""
    if name == "EBM":
        return {}
    import optuna

    def objective(trial):
        _, p = _suggest(trial, base_learner(name), seed)
        return _cv_f1(lambda: make_extra(name, p, seed), X, y, seed)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("tuned %s on %d rows: CV-F1=%.4f", name, len(y), study.best_value)
    return study.best_params


# ── Fitting with the monotonicity check ──────────────────────────────────────

def raw_sd(win: np.ndarray) -> np.ndarray:
    """Per-channel SD of the decision-day values over all rows (perturbation size)."""
    return win[:, -1, :].std(0)


def _perturbed(win: np.ndarray, channel: str, sd: np.ndarray) -> np.ndarray:
    w = win.copy()
    w[:, -1, IDX[channel]] += sd[IDX[channel]]
    return w


def fit_with_threshold(name, params, X: np.ndarray, y: np.ndarray, tr: np.ndarray, seed: int = 42,
                       inner_folds: int = 5):
    """Fitted estimator on rows `tr` and its threshold from stratified inner-CV predictions
    (the baselines' procedure). Returns (model, threshold, p_inner)."""
    inner = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    p_inner = cross_val_predict(make_extra(name, params, seed), X[tr], y[tr], cv=inner,
                                method="predict_proba")[:, 1]
    model = make_extra(name, params, seed).fit(X[tr], y[tr])
    return model, best_threshold(p_inner, y[tr]), p_inner


def _tab_one_split(name, params, feat: FeatureFn, win, y, split: Split, seed, inner_folds, sd):
    tr, te = split.train, split.test
    X = feat(win)
    model, thr, p_inner = fit_with_threshold(name, params, X, y, tr, seed, inner_folds)
    p_te = model.predict_proba(X[te])[:, 1]
    fd = np.stack([model.predict_proba(feat(_perturbed(win[te], c, sd)))[:, 1] - p_te
                   for c in MONO_DEC])
    return SplitResult(split.name, te, p_te, thr, {"thr_rows": tr, "p_thr": p_inner, "fd": fd})


def fit_tabular(name: str, params: Optional[Dict], win: np.ndarray, y: np.ndarray,
                splits: List[Split], seed: int = 42, inner_folds: int = 5,
                n_jobs: int = 10) -> List[SplitResult]:
    """Fit a tabular model on every outer split; `extra["fd"]` is (len(MONO_DEC), n_test)."""
    sd = raw_sd(win)
    jobs = 1 if name == "TabPFN" else n_jobs
    return Parallel(n_jobs=jobs)(delayed(_tab_one_split)(name, params, feature_fn(name), win, y, s,
                                                         seed, inner_folds, sd) for s in splits)


def load_ensemble(cfg: PinamConfig, states: List[Dict]) -> Callable[[np.ndarray], np.ndarray]:
    """Mean-probability predictor of a saved PI-NAM-family ensemble (one state per seed)."""
    n = _n_factors(cfg.spec)
    models = []
    for st in states:
        m = ARCHS[cfg.arch](torch.zeros(n), torch.ones(n), n, cfg.spec, cfg.hidden, cfg.dropout,
                            raw_mean=torch.zeros(len(RAW_CHANNELS)),
                            raw_std=torch.ones(len(RAW_CHANNELS)), depth=cfg.depth)
        m.load_state_dict(st)
        models.append(m.eval())

    def pred(w: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(w, dtype=torch.float32)
            return torch.stack([m(x) for m in models]).mean(0).numpy()
    return pred


def ensemble_fd(cfg: PinamConfig, win: np.ndarray, results: List[SplitResult]) -> List[np.ndarray]:
    """Same check for a PI-NAM-family ensemble, from `extra["states"]` of `fit_pinam`."""
    sd = raw_sd(win)
    out = []
    for r in results:
        pred = load_ensemble(cfg, r.extra["states"])
        base = pred(win[r.test])
        out.append(np.stack([pred(_perturbed(win[r.test], c, sd)) - base for c in MONO_DEC]))
    return out
