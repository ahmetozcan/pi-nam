"""
Optuna structure/feature + hyperparameter search for PI-NAM-X.
Objective: maximize mean 10-fold deployable F1 (same fair protocol as baselines).
Saves search space + per-trial records + best spec to outputs/results/pinamx_search.json.
"""
import json, logging, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
ROOT = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT))
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from src.feature_engineering import build_features
from src.pinn import RAW_CHANNELS
from src.pinam_search import run_pinamx_cv_parallel, deployable_f1

SEARCH_SPACE = {
    "physics": "{True, False}",
    "stats": "subset of {mean, std, max, min, trend}",
    "scales": "subset of {3, 7, 14}",
    "interactions": "{True, False}  (6 physically-motivated raw pairs)",
    "synoptic": "{True, False}  (10 multi-day synoptic + composite physics rules)",
    "syn_scale": "{3, 5, 7}  (synoptic lookback window in days)",
    "lam_mono": "uniform [0.0, 0.1]",
    "hidden": "{16, 24, 32, 48, 64}",
    "depth": "{2, 3}  (hidden layers per shape function)",
    "dropout": "{0.1, 0.2, 0.3}",
    "lr": "{1e-3, 5e-4, 2e-4}",
    "batch_size": "{16, 32, 64}",
    "weight_decay": "{1e-4, 3e-4, 1e-3}",
    "search_seeds": 3,
    "objective": "mean 10-fold deployable F1",
}
SEED = 42
N_TRIALS = 60
N_JOBS = 20

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
n_splits = cfg["features"]["test_folds"]
_, df = build_features(str(ROOT / "config.yaml")); df = df.reset_index(drop=True)
y = df["flight"].values
raw = df[RAW_CHANNELS].values.astype(np.float32)

trials_log = []


def objective(trial):
    stats = [s for s in ["mean", "std", "max", "min", "trend"]
             if trial.suggest_categorical(f"stat_{s}", [True, False])]
    scales = [sc for sc in [3, 7, 14] if trial.suggest_categorical(f"scale_{sc}", [True, False])]
    synoptic = trial.suggest_categorical("synoptic", [True, False])
    spec = {"physics": trial.suggest_categorical("physics", [True, False]),
            "stats": stats, "scales": scales,
            "interactions": trial.suggest_categorical("interactions", [True, False]),
            "synoptic": synoptic,
            "syn_scale": trial.suggest_categorical("syn_scale", [3, 5, 7])}
    lam = trial.suggest_float("lam_mono", 0.0, 0.1)
    hidden = trial.suggest_categorical("hidden", [16, 24, 32, 48, 64])
    depth = trial.suggest_categorical("depth", [2, 3])
    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])
    lr = trial.suggest_categorical("lr", [1e-3, 5e-4, 2e-4])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-4, 3e-4, 1e-3])
    if not stats and not spec["physics"]:
        spec["physics"] = True  # guard: need at least one factor group
    rows, oof = run_pinamx_cv_parallel(raw, y, cfg, spec, lam_mono=lam, hidden=hidden,
                                       dropout=dropout, lr=lr, weight_decay=weight_decay,
                                       batch_size=batch_size, depth=depth, n_seeds=3, n_jobs=N_JOBS)
    dep = deployable_f1(oof, y, n_splits, SEED)
    trials_log.append({"spec": spec, "lam_mono": lam, "hidden": hidden, "depth": depth,
                       "dropout": dropout, "lr": lr, "batch_size": batch_size,
                       "weight_decay": weight_decay, "deployable_f1": dep,
                       "std_f1": float(pd.DataFrame(rows).f1.mean())})
    print(f"trial: phys={spec['physics']} stats={stats} scales={scales} "
          f"syn={spec['synoptic']}@{spec['syn_scale']} "
          f"lam={lam:.3f} h={hidden} depth={depth} drop={dropout} lr={lr} bs={batch_size} "
          f"wd={weight_decay} -> deployF1={dep:.4f}", flush=True)
    return dep


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

best = max(trials_log, key=lambda r: r["deployable_f1"])
out = {"search_space": SEARCH_SPACE, "n_trials": N_TRIALS, "trials": trials_log, "best": best}
json.dump(out, open(ROOT / "outputs/results/pinamx_search.json", "w"), indent=2)
print(f"\nBEST deployF1={best['deployable_f1']:.4f}  spec={best['spec']} "
      f"lam={best['lam_mono']:.3f} h={best['hidden']} depth={best.get('depth')} "
      f"drop={best['dropout']} lr={best.get('lr')} bs={best.get('batch_size')} "
      f"wd={best.get('weight_decay')}")
print("Tuned baselines deployable bar: LightGBM 0.874, RF 0.873, SVM 0.872")
