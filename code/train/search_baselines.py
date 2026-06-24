"""
Run the baseline Optuna hyperparameter search from scratch on the raw-16 inputs.
Produces outputs/results/optuna_search_spaces.json + optuna_best_params.json,
which retrain_tuned_baselines.py then uses to refit and save weight bundles.
(TabPFN v2 is a zero-shot foundation model and is not tuned; it is provided as a
frozen out-of-fold array in reproducibility/oof/tabpfn_oof.npy.)
"""
import logging, warnings
from pathlib import Path
import numpy as np, yaml
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
ROOT = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT))
from src.feature_engineering import build_features
from src.pinn import RAW_CHANNELS
from src.optuna_tuning import run_all

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
seed = cfg["features"]["random_seed"]
_, df = build_features(str(ROOT / "config.yaml")); df = df.reset_index(drop=True)
y = df["flight"].values
X = df[RAW_CHANNELS].values.astype(np.float32)
run_all(X, y, seed=seed, n_trials=40, out_dir=str(ROOT / "outputs/results"))
print("Saved optuna_search_spaces.json + optuna_best_params.json")
