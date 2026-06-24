"""
Fair Optuna hyperparameter search for all baseline classifiers on RAW-16 inputs.

Every tunable model is optimized under the SAME protocol: maximize mean 10-fold
stratified-CV F1 (StandardScaler in-fold). Search spaces and best parameters are
saved to JSON for transparent reporting in the paper.

TabPFN v2 is a zero-shot foundation model and is not hyperparameter-tuned.

Usage: python -m src.optuna_tuning   (or import run_all)
"""

import json
import logging
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

# Human-readable search-space description for the paper table
SEARCH_SPACES = {
    "SVM": {
        "C": "log-uniform [0.1, 100]",
        "gamma": "log-uniform [1e-4, 1.0]",
        "kernel": "{rbf}",
    },
    "XGBoost": {
        "n_estimators": "int [200, 600]",
        "max_depth": "int [3, 8]",
        "learning_rate": "log-uniform [0.01, 0.2]",
        "subsample": "uniform [0.6, 1.0]",
        "colsample_bytree": "uniform [0.6, 1.0]",
        "reg_lambda": "uniform [0, 5]",
    },
    "LightGBM": {
        "n_estimators": "int [200, 600]",
        "num_leaves": "int [15, 63]",
        "max_depth": "int [3, 8]",
        "learning_rate": "log-uniform [0.01, 0.2]",
        "subsample": "uniform [0.6, 1.0]",
        "colsample_bytree": "uniform [0.6, 1.0]",
        "min_child_samples": "int [5, 40]",
        "reg_lambda": "uniform [0, 5]",
    },
    "Random Forest": {
        "n_estimators": "int [200, 600]",
        "max_depth": "int [4, 20]",
        "min_samples_split": "int [2, 10]",
        "min_samples_leaf": "int [1, 8]",
        "max_features": "{sqrt, log2, 0.5}",
    },
    "CatBoost": {
        "iterations": "int [300, 800]",
        "depth": "int [4, 8]",
        "learning_rate": "log-uniform [0.01, 0.2]",
        "l2_leaf_reg": "uniform [1, 10]",
    },
}


def _cv_f1(make_model, X, y, seed, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p = cross_val_predict(make_model(), X, y, cv=skf, method="predict_proba", n_jobs=4)[:, 1]
    return f1_score(y, (p >= 0.5).astype(int), zero_division=0)


def _suggest(trial, name, seed):
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    if name == "SVM":
        p = dict(C=trial.suggest_float("C", 0.1, 100, log=True),
                 gamma=trial.suggest_float("gamma", 1e-4, 1.0, log=True))
        return lambda: make_pipeline(StandardScaler(),
                                     SVC(C=p["C"], gamma=p["gamma"], kernel="rbf",
                                         probability=True, random_state=seed)), p
    if name == "XGBoost":
        p = dict(n_estimators=trial.suggest_int("n_estimators", 200, 600),
                 max_depth=trial.suggest_int("max_depth", 3, 8),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0))
        return lambda: make_pipeline(StandardScaler(),
                                     XGBClassifier(**p, random_state=seed, verbosity=0, eval_metric="logloss")), p
    if name == "LightGBM":
        p = dict(n_estimators=trial.suggest_int("n_estimators", 200, 600),
                 num_leaves=trial.suggest_int("num_leaves", 15, 63),
                 max_depth=trial.suggest_int("max_depth", 3, 8),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                 min_child_samples=trial.suggest_int("min_child_samples", 5, 40),
                 reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0))
        return lambda: make_pipeline(StandardScaler(),
                                     LGBMClassifier(**p, random_state=seed, verbose=-1)), p
    if name == "Random Forest":
        p = dict(n_estimators=trial.suggest_int("n_estimators", 200, 600),
                 max_depth=trial.suggest_int("max_depth", 4, 20),
                 min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                 min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
                 max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]))
        return lambda: make_pipeline(StandardScaler(),
                                     RandomForestClassifier(**p, random_state=seed, n_jobs=4)), p
    if name == "CatBoost":
        p = dict(iterations=trial.suggest_int("iterations", 300, 800),
                 depth=trial.suggest_int("depth", 4, 8),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                 l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0))
        return lambda: make_pipeline(StandardScaler(),
                                     CatBoostClassifier(**p, random_seed=seed, verbose=0)), p
    raise ValueError(name)


def tune_baseline(name, X, y, seed, n_trials=40):
    def objective(trial):
        make_model, _ = _suggest(trial, name, seed)
        return _cv_f1(make_model, X, y, seed)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("%s: best CV-F1=%.4f", name, study.best_value)
    return study.best_params, study.best_value


def run_all(X, y, seed=42, n_trials=40, out_dir="outputs/results"):
    out = Path(out_dir)
    best_all = {}
    for name in ["SVM", "XGBoost", "LightGBM", "Random Forest", "CatBoost"]:
        bp, bv = tune_baseline(name, X, y, seed, n_trials)
        best_all[name] = {"best_params": bp, "best_cv_f1": bv}
        logger.info("  %s best params: %s", name, bp)
    with open(out / "optuna_search_spaces.json", "w") as f:
        json.dump(SEARCH_SPACES, f, indent=2)
    with open(out / "optuna_best_params.json", "w") as f:
        json.dump(best_all, f, indent=2, default=str)
    return best_all
