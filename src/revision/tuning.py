"""Hyperparameter tuning restricted to a given set of rows (nested tuning for P-year).

Baselines reuse the submitted Optuna search (`src.optuna_tuning`, 10-fold F1).
PI-NAM reuses the submitted structure/hyperparameter search space; its objective
is the mean inner-CV F1 with a validation-chosen threshold (leakage-free).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import optuna
from sklearn.metrics import f1_score

from src.optuna_tuning import tune_baseline
from src.revision.models import PinamConfig, fit_pinam
from src.revision.protocols import Split, split_random

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_baselines(X: np.ndarray, y: np.ndarray, rows: np.ndarray, names, n_trials: int,
                   seed: int = 42) -> Dict[str, Dict]:
    """Optuna per baseline on `rows` only (TabPFN is not tuned)."""
    out = {}
    for name in names:
        if name == "TabPFN":
            out[name] = {}
            continue
        params, score = tune_baseline(name, X[rows], y[rows], seed, n_trials=n_trials)
        logger.info("tuned %s on %d rows: inner F1=%.4f", name, len(rows), score)
        out[name] = params
    return out


def _suggest_pinam(trial, fixed_spec: Optional[Dict] = None) -> PinamConfig:
    """Search space of the submitted PI-NAM search; `fixed_spec` freezes the architecture
    (factor groups, windows) so that only the training hyperparameters are searched."""
    if fixed_spec is not None:
        return _suggest_training(trial, dict(fixed_spec))
    stats = [s for s in ["mean", "std", "max", "min", "trend"]
             if trial.suggest_categorical(f"stat_{s}", [True, False])]
    scales = [sc for sc in [3, 7, 14] if trial.suggest_categorical(f"scale_{sc}", [True, False])]
    spec = {"physics": trial.suggest_categorical("physics", [True, False]),
            "stats": stats, "scales": scales,
            "interactions": trial.suggest_categorical("interactions", [True, False]),
            "synoptic": trial.suggest_categorical("synoptic", [True, False]),
            "syn_scale": trial.suggest_categorical("syn_scale", [3, 5, 7])}
    if not stats and not spec["physics"]:
        spec["physics"] = True     # at least one factor group (same guard as submitted search)
    return _suggest_training(trial, spec)


def _suggest_training(trial, spec: Dict) -> PinamConfig:
    return PinamConfig(spec=spec,
                       lam_mono=trial.suggest_float("lam_mono", 0.0, 0.1),
                       hidden=trial.suggest_categorical("hidden", [16, 24, 32, 48, 64]),
                       dropout=trial.suggest_categorical("dropout", [0.1, 0.2, 0.3]),
                       lr=trial.suggest_categorical("lr", [1e-3, 5e-4, 2e-4]),
                       weight_decay=trial.suggest_categorical("weight_decay", [1e-4, 3e-4, 1e-3]),
                       batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
                       depth=trial.suggest_categorical("depth", [2, 3]))


def tune_pinam(win: np.ndarray, y: np.ndarray, rows: np.ndarray, n_trials: int,
               n_seeds: int = 3, seed: int = 42, n_jobs: int = 28,
               fixed_spec: Optional[Dict] = None) -> PinamConfig:
    """PI-NAM search using inner stratified 10-fold CV on `rows` only.

    Without `fixed_spec` the architecture is searched too (structure search); with it the
    architecture is frozen and only the training hyperparameters are tuned.
    """
    rows = np.sort(rows)
    inner = [Split(s.name, rows[s.train], rows[s.test]) for s in split_random(y[rows], seed=seed)]

    def objective(trial):
        cfg = _suggest_pinam(trial, fixed_spec)
        res = fit_pinam(cfg, win, y, inner, n_seeds=n_seeds, base_seed=seed, n_jobs=n_jobs)
        f1s = [f1_score(y[r.test], (r.p_test >= r.threshold).astype(int), zero_division=0)
               for r in res]
        return float(np.mean(f1s))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = _suggest_pinam(optuna.trial.FixedTrial(study.best_params), fixed_spec)
    logger.info("tuned PI-NAM on %d rows: inner F1=%.4f spec=%s", len(rows), study.best_value, best.spec)
    return best


def save_tuning(path: Path, baselines: Dict, pinam: PinamConfig, meta: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "baselines": baselines,
                                "pinam": pinam.__dict__}, indent=2, default=str))


def load_tuning(path: Path):
    d = json.loads(Path(path).read_text())
    return d["baselines"], PinamConfig(**d["pinam"])
