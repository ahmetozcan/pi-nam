"""
Final 15-seed deep-ensemble evaluation of the top-3 PINAMX configs from the
Optuna search. Picks the best by deployable F1, saves its per-fold weight
bundles + OOF, and reports standard/deployable F1 + AUC/BAcc/MCC.
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

N_SEEDS = 15
N_JOBS = 20
SEED = 42

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
n_splits = cfg["features"]["test_folds"]
_, df = build_features(str(ROOT / "config.yaml")); df = df.reset_index(drop=True)
y = df["flight"].values
raw = df[RAW_CHANNELS].values.astype(np.float32)

d = json.load(open(ROOT / "outputs/results/pinamx_search.json"))
ts = sorted(d["trials"], key=lambda r: -r["deployable_f1"])
seen, uniq = [], []
for t in ts:
    key = (t["spec"]["physics"], tuple(t["spec"]["stats"]), tuple(t["spec"]["scales"]),
           t["spec"]["interactions"], t["spec"].get("synoptic", False),
           t["spec"].get("syn_scale", 0), round(t["lam_mono"], 3), t["hidden"],
           t.get("depth", 2), t["dropout"], t["lr"], t["batch_size"], t["weight_decay"])
    if key in seen:
        continue
    seen.append(key); uniq.append(t)
    if len(uniq) == 3:
        break

results = []
best = {"deploy15": -1}
for i, t in enumerate(uniq, 1):
    sp = t["spec"]
    print(f"\n=== Config #{i} (dep3={t['deployable_f1']:.4f}) {sp} "
          f"lam={t['lam_mono']:.3f} h={t['hidden']} drop={t['dropout']} "
          f"lr={t['lr']} bs={t['batch_size']} wd={t['weight_decay']} ===", flush=True)
    save_dir = ROOT / f"reproducibility/folds/pinamx_cfg{i}"
    rows, oof = run_pinamx_cv_parallel(
        raw, y, cfg, sp, lam_mono=t["lam_mono"], hidden=t["hidden"], dropout=t["dropout"],
        lr=t["lr"], weight_decay=t["weight_decay"], batch_size=t["batch_size"],
        depth=t.get("depth", 2), n_seeds=N_SEEDS, n_jobs=N_JOBS, save_dir=save_dir)
    dep15 = deployable_f1(oof, y, n_splits, SEED)
    rdf = pd.DataFrame(rows)
    rec = {"config": i, "spec": sp, "lam_mono": t["lam_mono"], "hidden": t["hidden"],
           "depth": t.get("depth", 2),
           "dropout": t["dropout"], "lr": t["lr"], "batch_size": t["batch_size"],
           "weight_decay": t["weight_decay"], "deploy3": t["deployable_f1"],
           "deploy15": dep15, "std_f1": float(rdf.f1.mean()),
           "auc": float(rdf.roc_auc.mean()), "bacc": float(rdf.balanced_acc.mean()),
           "mcc": float(rdf.mcc.mean()), "thr_mean": float(rdf.threshold.mean()),
           "thr_std": float(rdf.threshold.std()), "save_dir": str(save_dir)}
    results.append(rec)
    np.save(ROOT / f"outputs/results/pinamx_cfg{i}_oof.npy", oof)
    print(f"  -> deploy15={dep15:.4f}  stdF1={rec['std_f1']:.4f}  AUC={rec['auc']:.4f} "
          f"BAcc={rec['bacc']:.4f} MCC={rec['mcc']:.4f}  thr={rec['thr_mean']:.3f}±{rec['thr_std']:.3f}",
          flush=True)
    if dep15 > best["deploy15"]:
        best = rec

out = {"n_seeds": N_SEEDS, "configs": results, "best_config": best["config"],
       "best_deploy15": best["deploy15"],
       "bar": {"LightGBM": 0.874, "RF": 0.873, "SVM": 0.872}}
json.dump(out, open(ROOT / "outputs/results/pinamx_final.json", "w"), indent=2)
np.save(ROOT / "outputs/results/pinamx_best_oof.npy",
        np.load(ROOT / f"outputs/results/pinamx_cfg{best['config']}_oof.npy"))
print(f"\n############ BEST = config #{best['config']}  deploy15={best['deploy15']:.4f} "
      f"(bar LightGBM=0.874) ############")
print(f"std_f1={best['std_f1']:.4f} AUC={best['auc']:.4f} BAcc={best['bacc']:.4f} "
      f"MCC={best['mcc']:.4f} thr={best['thr_mean']:.3f}±{best['thr_std']:.3f}")
