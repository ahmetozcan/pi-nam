"""
Wilcoxon signed-rank test: PI-NAM (V1) vs each tuned baseline on per-fold
deployable F1 (train-OOF-tuned threshold applied to test fold), shared 10-fold
StratifiedKFold(seed=42). Reconstructs baseline OOF from saved weight bundles.
Writes outputs/results/wilcoxon_pinam.json + a LaTeX table.
"""
import json, pickle, logging, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
ROOT = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT))
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon
from src.feature_engineering import build_features
from src.pinn import RAW_CHANNELS

SEED = 42
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
n_splits = cfg["features"]["test_folds"]
_, df = build_features(str(ROOT / "config.yaml")); df = df.reset_index(drop=True)
y = df["flight"].values
X = df[RAW_CHANNELS].values.astype(np.float32)
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y))

BASE = {"SVM": "svm", "XGBoost": "xgboost", "LightGBM": "lightgbm",
        "Random Forest": "randomforest", "CatBoost": "catboost", "TabPFN": "tabpfn"}
FB = ROOT / "reproducibility/folds/base_models"


def _load_bundle(path):
    return pickle.load(open(path, "rb"))


def baseline_oof(sub):
    if sub == "tabpfn":
        return np.load(ROOT / "reproducibility/oof/tabpfn_oof.npy")
    """Reconstruct per-sample OOF probability from saved fold bundles."""
    oof = np.full(len(y), np.nan)
    for fi in range(1, n_splits + 1):
        b = _load_bundle(FB / sub / f"fold_{fi:02d}.pkl")
        te = np.array(b["test_idx"]); m, sc = b["model"], b["scaler"]
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return oof


def best_thr(p, yy):
    b, bf = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.005):
        f = f1_score(yy, (p >= t).astype(int), zero_division=0)
        if f > bf:
            bf, b = f, float(t)
    return b


def deployable_per_fold(oof):
    """10 per-fold deployable F1: tune thr on train-OOF, apply to test."""
    out = []
    for tr, te in folds:
        thr = best_thr(oof[tr], y[tr])
        out.append(f1_score(y[te], (oof[te] >= thr).astype(int), zero_division=0))
    return np.array(out)


pinam_oof = np.load(ROOT / "outputs/results/pinamx_v1_oof.npy")
pinam_fold = deployable_per_fold(pinam_oof)
print(f"PI-NAM (V1) deployable F1: mean={pinam_fold.mean():.4f}  per-fold={np.round(pinam_fold,3)}\n")

rows = []
for name, sub in BASE.items():
    try:
        b_oof = baseline_oof(sub)
        b_fold = deployable_per_fold(b_oof)
    except Exception as e:
        print(f"{name}: SKIP ({e})"); continue
    stat, p = wilcoxon(pinam_fold, b_fold, zero_method="wilcox", alternative="two-sided")
    diff = pinam_fold.mean() - b_fold.mean()
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
    rows.append({"model": name, "base_mean": float(b_fold.mean()),
                 "pinam_mean": float(pinam_fold.mean()), "diff": float(diff),
                 "wilcoxon_p": float(p), "sig": sig})
    print(f"{name:<14} base={b_fold.mean():.4f}  PI-NAM={pinam_fold.mean():.4f}  "
          f"Δ={diff:+.4f}  p={p:.4f}  {sig}")

out = {"metric": "per-fold deployable F1", "n_folds": n_splits, "seed": SEED,
       "pinam_per_fold": pinam_fold.tolist(), "comparisons": rows}
json.dump(out, open(ROOT / "outputs/results/wilcoxon_pinam.json", "w"), indent=2)

# LaTeX table
lines = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Baseline & Baseline F1 & PI-NAM F1 & $\Delta$ & Wilcoxon $p$ \\", r"\midrule"]
for r in rows:
    lines.append(f"{r['model']} & {r['base_mean']:.3f} & {r['pinam_mean']:.3f} & "
                 f"{r['diff']:+.3f} & {r['wilcoxon_p']:.3f}{'' if r['sig']=='ns' else '$^{'+r['sig']+'}$'} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
Path(ROOT / "outputs/results/tab_wilcoxon_pinam.tex").write_text("\n".join(lines))
print("\nSaved: outputs/results/wilcoxon_pinam.json + tab_wilcoxon_pinam.tex")
