"""
Persist the winning synoptic config (physics base + ALL synoptic rules, scale=5)
at 15 seeds: save per-fold weight bundles, OOF probabilities, and metrics JSON
so the barrier-breaking result is reproducible.
"""
import json, logging, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
ROOT = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT))
from src.feature_engineering import build_features
from src.pinn import RAW_CHANNELS
from src.pinam_search import run_pinamx_cv_parallel, deployable_f1

N_SEEDS, N_JOBS, SEED = 15, 20, 42
HP = dict(lam_mono=0.081, hidden=48, dropout=0.2, lr=2e-4, weight_decay=1e-3,
          batch_size=32, depth=2)
SPEC = {"physics": True, "stats": ["min"], "scales": [3, 7],
        "interactions": False, "synoptic": True, "syn_scale": 5}

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
n_splits = cfg["features"]["test_folds"]
_, df = build_features(str(ROOT / "config.yaml")); df = df.reset_index(drop=True)
y = df["flight"].values
raw = df[RAW_CHANNELS].values.astype(np.float32)

save_dir = ROOT / "reproducibility/folds/pinamx_synoptic5"
rows, oof = run_pinamx_cv_parallel(raw, y, cfg, SPEC, n_seeds=N_SEEDS, n_jobs=N_JOBS,
                                   save_dir=save_dir, **HP)
dep = deployable_f1(oof, y, n_splits, SEED)
r = pd.DataFrame(rows)
rec = {"spec": SPEC, "hp": HP, "n_seeds": N_SEEDS,
       "deploy15": dep, "std_f1": float(r.f1.mean()), "auc": float(r.roc_auc.mean()),
       "bacc": float(r.balanced_acc.mean()), "mcc": float(r.mcc.mean()),
       "thr_mean": float(r.threshold.mean()), "thr_std": float(r.threshold.std()),
       "bar": {"LightGBM": 0.874, "RF": 0.873, "SVM": 0.872},
       "save_dir": str(save_dir)}
json.dump(rec, open(ROOT / "outputs/results/pinamx_synoptic5.json", "w"), indent=2)
np.save(ROOT / "outputs/results/pinamx_synoptic5_oof.npy", oof)
r.to_csv(ROOT / "outputs/results/pinamx_synoptic5_folds.csv", index=False)
print(f"SAVED synoptic5: deploy15={dep:.4f} stdF1={rec['std_f1']:.4f} AUC={rec['auc']:.4f} "
      f"BAcc={rec['bacc']:.4f} MCC={rec['mcc']:.4f} thr={rec['thr_mean']:.3f}±{rec['thr_std']:.3f}")
print(f"bundles -> {save_dir}")
