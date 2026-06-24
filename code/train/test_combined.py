"""
Combined test: synoptic@5 + interactions=True, at 15 seeds.
V1 = winning synoptic@5 config + interactions on (isolate interaction effect).
V2 = synoptic@5 + interactions + richer temporal stats (best-of-both).
Compares to saved synoptic@5 reference (deploy15=0.8748).
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
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
n_splits = cfg["features"]["test_folds"]
_, df = build_features(str(ROOT / "config.yaml")); df = df.reset_index(drop=True)
y = df["flight"].values
raw = df[RAW_CHANNELS].values.astype(np.float32)

VARIANTS = {
    # V1: winning synoptic@5 HP, interactions ON
    "V1 syn5+inter (min)": (
        {"physics": True, "stats": ["min"], "scales": [3, 7],
         "interactions": True, "synoptic": True, "syn_scale": 5},
        dict(lam_mono=0.081, hidden=48, dropout=0.2, lr=2e-4, weight_decay=1e-3,
             batch_size=32, depth=2)),
    # V2: synoptic@5 + interactions + richer temporal (best of both search worlds)
    "V2 syn5+inter (rich)": (
        {"physics": True, "stats": ["mean", "std", "min", "trend"], "scales": [7],
         "interactions": True, "synoptic": True, "syn_scale": 5},
        dict(lam_mono=0.01, hidden=24, dropout=0.2, lr=5e-4, weight_decay=1e-4,
             batch_size=16, depth=2)),
}

print("reference: synoptic@5 (no inter) deploy15=0.8748 AUC=0.9039 MCC=0.633 thr0.380±0.030")
print(f"bar: LightGBM 0.874, RF 0.873\n")
results = []
for tag, (spec, hp) in VARIANTS.items():
    save_dir = ROOT / f"reproducibility/folds/pinamx_{tag.split()[0].lower()}"
    rows, oof = run_pinamx_cv_parallel(raw, y, cfg, spec, n_seeds=N_SEEDS, n_jobs=N_JOBS,
                                       save_dir=save_dir, **hp)
    dep = deployable_f1(oof, y, n_splits, SEED)
    r = pd.DataFrame(rows)
    rec = {"tag": tag, "spec": spec, "hp": hp, "deploy15": dep,
           "std_f1": float(r.f1.mean()), "auc": float(r.roc_auc.mean()),
           "bacc": float(r.balanced_acc.mean()), "mcc": float(r.mcc.mean()),
           "thr_mean": float(r.threshold.mean()), "thr_std": float(r.threshold.std())}
    results.append(rec)
    np.save(ROOT / f"outputs/results/pinamx_{tag.split()[0].lower()}_oof.npy", oof)
    print(f"{tag:<24} deploy15={dep:.4f}  stdF1={rec['std_f1']:.4f}  AUC={rec['auc']:.4f}  "
          f"BAcc={rec['bacc']:.4f}  MCC={rec['mcc']:.4f}  thr={rec['thr_mean']:.3f}±{rec['thr_std']:.3f}",
          flush=True)
json.dump(results, open(ROOT / "outputs/results/pinamx_combined.json", "w"), indent=2)
best = max(results, key=lambda r: r["deploy15"])
print(f"\nBEST combined: {best['tag']} deploy15={best['deploy15']:.4f} "
      f"(vs synoptic@5 0.8748, bar 0.874)")
