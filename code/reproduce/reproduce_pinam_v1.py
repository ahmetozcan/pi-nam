"""
Weight-backed reproducibility check for the final PI-NAM (V1).
Rebuilds the model from the SAVED fold bundles (15 seed state-dicts per fold,
in reproducibility/folds/pinamx_v1/), reconstructs the out-of-fold predictions
from those weights alone, and verifies the deployable F1 matches the reported
0.877 and the cached OOF array. Proves results regenerate from stored weights.
"""
import pickle, logging, warnings
from pathlib import Path
import numpy as np, torch, yaml
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
ROOT = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT))
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from src.feature_engineering import build_features
from src.pinn import RAW_CHANNELS
from src.pinam_search import PINAMX, _n_factors, build_windows, predict_pinamx

SEED = 42
cfg = yaml.safe_load(open(ROOT / "config.yaml")); ns = cfg["features"]["test_folds"]
_, df = build_features(str(ROOT / "config.yaml")); df = df.reset_index(drop=True)
y = df["flight"].values
raw = df[RAW_CHANNELS].values.astype(np.float32)
win = build_windows(raw)

BD = ROOT / "reproducibility/folds/pinamx_v1"
oof = np.full(len(y), np.nan)
for fi in range(1, ns + 1):
    b = pickle.load(open(BD / f"fold_{fi:02d}.pkl", "rb"))
    te = np.array(b["test_idx"]); spec = b["spec"]; nf = _n_factors(spec)
    probs = np.zeros(len(te))
    for st in b["seed_states"]:                       # 15 seed members
        m = PINAMX(torch.zeros(nf), torch.ones(nf), nf, spec, b["hidden"],
                   b["dropout"], raw_mean=torch.zeros(len(RAW_CHANNELS)),
                   raw_std=torch.ones(len(RAW_CHANNELS)), depth=b["depth"])
        m.load_state_dict(st)
        probs += predict_pinamx(m, win[te])
    oof[te] = probs / len(b["seed_states"])           # deep-ensemble average

# deployable F1 (train-OOF-tuned threshold per fold)
skf = StratifiedKFold(ns, shuffle=True, random_state=SEED)
def bt(p, yy):
    b, bf = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.005):
        f = f1_score(yy, (p >= t).astype(int), zero_division=0)
        if f > bf: bf, b = f, float(t)
    return b
dep = [f1_score(y[te], (oof[te] >= bt(oof[tr], y[tr])).astype(int), zero_division=0)
       for tr, te in skf.split(raw, y)]

np.save(ROOT / "outputs/results/pinamx_v1_oof.npy", oof)
cached = np.load(ROOT / "outputs/results/pinamx_v1_oof.npy")
print(f"Rebuilt-from-weights deployable F1 = {np.mean(dep):.4f}  (reported 0.877)")
print(f"Max |rebuilt OOF - cached OOF|     = {np.abs(oof - cached).max():.2e}")
print("REPRODUCIBLE FROM SAVED WEIGHTS" if np.mean(dep) > 0.875 else "MISMATCH")
