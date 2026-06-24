"""
Regenerate the TabPFN v2 out-of-fold (OOF) predictions FROM SCRATCH.

This reproduces, with the exact configuration used in the paper, the array that
is also shipped frozen at reproducibility/oof/tabpfn_oof.npy. TabPFN is a
zero-shot foundation model (no hyperparameter tuning); for each fold it is fitted
on the standardized training split and queried on the held-out test split.

Requirements: `pip install tabpfn==7.0.1` and PyTorch. A CUDA GPU is strongly
recommended (CPU inference is ~40 min for 10 folds). NOTE: exact probabilities
may differ by ~1e-5 across GPU models / driver / library versions due to
floating-point non-determinism, which can shift borderline predictions and move
F1 by ~0.001-0.002. For a bit-exact reference use the shipped frozen OOF.
"""
import logging, warnings
from pathlib import Path
import numpy as np, yaml
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
ROOT = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT))
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from tabpfn import TabPFNClassifier
from src.feature_engineering import build_features
from src.pinn import RAW_CHANNELS

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
ns = cfg["features"]["test_folds"]; seed = cfg["features"]["random_seed"]
n_est = cfg["models"]["tabpfn"]["n_estimators"]          # = 32 (paper config)
_, df = build_features(str(ROOT / "config.yaml")); df = df.reset_index(drop=True)
y = df["flight"].values
X = df[RAW_CHANNELS].values.astype(np.float32)

skf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=seed)
oof = np.full(len(y), np.nan)
for fi, (tr, te) in enumerate(skf.split(X, y), 1):
    sc = StandardScaler().fit(X[tr])
    clf = TabPFNClassifier(n_estimators=n_est, random_state=0)  # device="auto"
    clf.fit(sc.transform(X[tr]), y[tr])
    oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    print(f"  fold {fi:2d}/{ns} done", flush=True)

out = ROOT / "reproducibility/oof/tabpfn_oof.npy"
np.save(out, oof)
# quick deployable check (single global F1-optimal threshold as a sanity print)
def bt(p, yy):
    b, bf = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.005):
        f = f1_score(yy, (p >= t).astype(int), zero_division=0)
        if f > bf: bf, b = f, float(t)
    return b
thr = bt(oof, y)
print(f"\nRegenerated TabPFN OOF -> {out}")
print(f"AUC={roc_auc_score(y,oof):.4f}  F1@{thr:.2f}={f1_score(y,(oof>=thr).astype(int)):.4f} "
      f"(paper TabPFN: AUC 0.908, deployable F1 0.846)")
