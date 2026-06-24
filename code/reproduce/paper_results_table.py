"""
Authoritative paper results table: ALL models on raw-16, shared 10-fold split,
under the deployable-threshold protocol (threshold tuned on train-OOF, applied
to test fold). Reports per-fold deployable F1, ROC-AUC, BAcc, MCC (mean+/-std)
and Wilcoxon signed-rank p-values of PI-NAM (V1) vs each baseline.
Writes outputs/results/tab_main_deployable.tex + tab_wilcoxon_pinam.tex + JSON.
"""
import json, pickle, logging, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
ROOT = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT))
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, roc_auc_score, balanced_accuracy_score,
                             matthews_corrcoef)
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
FB = ROOT / "reproducibility/folds/base_models"
BASE = {"SVM": "svm", "XGBoost": "xgboost", "LightGBM": "lightgbm",
        "Random Forest": "randomforest", "CatBoost": "catboost", "TabPFN": "tabpfn"}


def _load(path):
    return pickle.load(open(path, "rb"))


def baseline_oof(sub):
    if sub == "tabpfn":
        return np.load(ROOT / "reproducibility/oof/tabpfn_oof.npy")
    oof = np.full(len(y), np.nan)
    for fi in range(1, n_splits + 1):
        b = _load(FB / sub / f"fold_{fi:02d}.pkl")
        te = np.array(b["test_idx"])
        oof[te] = b["model"].predict_proba(b["scaler"].transform(X[te]))[:, 1]
    return oof


def best_thr(p, yy):
    b, bf = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.005):
        f = f1_score(yy, (p >= t).astype(int), zero_division=0)
        if f > bf:
            bf, b = f, float(t)
    return b


def per_fold_metrics(oof):
    """Per-fold deployable F1/BAcc/MCC (train-tuned threshold) + AUC."""
    f1s, baccs, mccs, aucs, thrs = [], [], [], [], []
    for tr, te in folds:
        thr = best_thr(oof[tr], y[tr]); pred = (oof[te] >= thr).astype(int)
        f1s.append(f1_score(y[te], pred, zero_division=0))
        baccs.append(balanced_accuracy_score(y[te], pred))
        mccs.append(matthews_corrcoef(y[te], pred))
        aucs.append(roc_auc_score(y[te], oof[te])); thrs.append(thr)
    return (np.array(f1s), np.array(aucs), np.array(baccs), np.array(mccs), np.array(thrs))


# PI-NAM (V1)
pin_oof = np.load(ROOT / "outputs/results/pinamx_v1_oof.npy")
pin = per_fold_metrics(pin_oof)

rows = {}
for name, sub in BASE.items():
    try:
        rows[name] = per_fold_metrics(baseline_oof(sub))
    except Exception as e:
        print(f"skip {name}: {e}")
rows["PI-NAM (proposed)"] = pin

# ---- main metrics table ----
order = ["SVM", "XGBoost", "LightGBM", "Random Forest", "CatBoost", "TabPFN",
         "PI-NAM (proposed)"]
best_f1 = max(rows[m][0].mean() for m in order)
lines = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Model & Deployable F1 & ROC-AUC & BAcc & MCC \\", r"\midrule"]
for m in order:
    f1, auc, bacc, mcc, thr = rows[m]
    nm = f"\\textbf{{{m}}}" if m.startswith("PI-NAM") else m
    f1s = f"{f1.mean():.3f} $\\pm$ {f1.std():.3f}"
    if abs(f1.mean() - best_f1) < 1e-9:
        f1s = f"\\textbf{{{f1.mean():.3f}}} $\\pm$ {f1.std():.3f}"
    if m.startswith("PI-NAM"):
        lines.append(r"\midrule")
    lines.append(f"{nm} & {f1s} & {auc.mean():.3f} $\\pm$ {auc.std():.3f} & "
                 f"{bacc.mean():.3f} & {mcc.mean():.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(ROOT / "outputs/results/tab_main_deployable.tex").write_text("\n".join(lines))

# ---- Wilcoxon table ----
wl = [r"\begin{tabular}{lcccc}", r"\toprule",
      r"Baseline & Baseline F1 & PI-NAM F1 & $\Delta$ & Wilcoxon $p$ \\", r"\midrule"]
wj = []
for m in ["SVM", "XGBoost", "LightGBM", "Random Forest", "CatBoost", "TabPFN"]:
    bf = rows[m][0]; stat, p = wilcoxon(pin[0], bf, alternative="two-sided")
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    tag = "" if sig == "" else f"$^{{{sig}}}$"
    wl.append(f"{m} & {bf.mean():.3f} & {pin[0].mean():.3f} & "
              f"{pin[0].mean()-bf.mean():+.3f} & {p:.3f}{tag} \\\\")
    wj.append({"model": m, "base_f1": float(bf.mean()), "pinam_f1": float(pin[0].mean()),
               "delta": float(pin[0].mean()-bf.mean()), "p": float(p)})
wl += [r"\bottomrule", r"\end{tabular}"]
(ROOT / "outputs/results/tab_wilcoxon_pinam.tex").write_text("\n".join(wl))

json.dump({"metric": "per-fold deployable F1/AUC/BAcc/MCC",
           "models": {m: {"deploy_f1": float(rows[m][0].mean()),
                          "deploy_f1_std": float(rows[m][0].std()),
                          "auc": float(rows[m][1].mean()),
                          "bacc": float(rows[m][2].mean()),
                          "mcc": float(rows[m][3].mean()),
                          "thr_mean": float(rows[m][4].mean()),
                          "thr_std": float(rows[m][4].std())} for m in order},
           "wilcoxon": wj},
          open(ROOT / "outputs/results/paper_results.json", "w"), indent=2)

print(f"{'Model':<20}{'deplF1':>8}{'AUC':>8}{'BAcc':>7}{'MCC':>7}{'thr±std':>14}")
for m in order:
    f1, auc, bacc, mcc, thr = rows[m]
    print(f"{m:<20}{f1.mean():>8.4f}{auc.mean():>8.4f}{bacc.mean():>7.3f}"
          f"{mcc.mean():>7.3f}{thr.mean():>8.3f}±{thr.std():.3f}")
print("\nWilcoxon vs PI-NAM:")
for w in wj:
    print(f"  {w['model']:<14} Δ={w['delta']:+.4f} p={w['p']:.4f}")
print("\nSaved tab_main_deployable.tex, tab_wilcoxon_pinam.tex, paper_results.json")
