"""
Quick all-in test: add ALL synoptic physics rules to the physics-based winner
config and evaluate at 15 seeds for scales {3,5,7}, plus a no-synoptic reference.
Reports deployable F1 / std-F1 / AUC / BAcc / MCC / threshold stability.
"""
import logging, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
ROOT = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT))
from src.feature_engineering import build_features
from src.pinn import RAW_CHANNELS
from src.pinam_search import run_pinamx_cv_parallel, deployable_f1

N_SEEDS = 15
N_JOBS = 20
SEED = 42
# physics-based winner (expanded-search config #2 -> 15-seed 0.8708)
HP = dict(lam_mono=0.081, hidden=48, dropout=0.2, lr=2e-4, weight_decay=1e-3,
          batch_size=32, depth=2)
BASE_SPEC = {"physics": True, "stats": ["min"], "scales": [3, 7], "interactions": False}

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
n_splits = cfg["features"]["test_folds"]
_, df = build_features(str(ROOT / "config.yaml")); df = df.reset_index(drop=True)
y = df["flight"].values
raw = df[RAW_CHANNELS].values.astype(np.float32)


def run(tag, spec):
    rows, oof = run_pinamx_cv_parallel(raw, y, cfg, spec, n_seeds=N_SEEDS, n_jobs=N_JOBS, **HP)
    dep = deployable_f1(oof, y, n_splits, SEED)
    r = pd.DataFrame(rows)
    print(f"{tag:<22} deploy15={dep:.4f}  stdF1={r.f1.mean():.4f}  AUC={r.roc_auc.mean():.4f}  "
          f"BAcc={r.balanced_acc.mean():.4f}  MCC={r.mcc.mean():.4f}  "
          f"thr={r.threshold.mean():.3f}±{r.threshold.std():.3f}", flush=True)
    return dep


print(f"{'config':<22}{'metrics':<10}  (bar: LightGBM 0.874, RF 0.873)")
run("base (no synoptic)", dict(BASE_SPEC))
for s in (3, 5, 7):
    run(f"+synoptic scale={s}", {**BASE_SPEC, "synoptic": True, "syn_scale": s})
