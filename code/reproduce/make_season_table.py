"""
Reproduce the season-stratified performance table (paper Table 3) for the final
PI-NAM (V1) from its out-of-fold predictions. Writes outputs/results/tab_seasonal.tex.
"""
import logging, warnings
from pathlib import Path
import numpy as np, yaml
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
ROOT = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT))
from sklearn.metrics import (f1_score, roc_auc_score, balanced_accuracy_score,
                             matthews_corrcoef)
from src.feature_engineering import build_features
from src.pinn import RAW_CHANNELS  # noqa: F401 (ensures env parity)

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
_, df = build_features(str(ROOT / "config.yaml")); df = df.reset_index(drop=True)
y = df["flight"].values
season = df["season"].values
oof = np.load(ROOT / "outputs/results/pinamx_v1_oof.npy")

def bt(p, yy):
    b, bf = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.005):
        f = f1_score(yy, (p >= t).astype(int), zero_division=0)
        if f > bf: bf, b = f, float(t)
    return b
thr = bt(oof, y); pred = (oof >= thr).astype(int)
NAME = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}

lines = [r"\begin{tabular}{lrcccc}", r"\toprule",
         r"Season (FR\%) & $N$ & F1 & ROC-AUC & BAcc & MCC \\", r"\midrule"]
for s in [1, 2, 3, 4]:
    m = season == s; fr = 100 * y[m].mean()
    lines.append(f"{NAME[s]} ({fr:.1f}\\%) & {m.sum()} & {f1_score(y[m],pred[m],zero_division=0):.3f} & "
                 f"{roc_auc_score(y[m],oof[m]):.3f} & {balanced_accuracy_score(y[m],pred[m]):.3f} & "
                 f"{matthews_corrcoef(y[m],pred[m]):.3f} \\\\")
lines.append(r"\midrule")
lines.append(f"\\textbf{{Overall ({100*y.mean():.1f}\\%)}} & \\textbf{{{len(y)}}} & "
             f"$\\mathbf{{{f1_score(y,pred,zero_division=0):.3f}}}$ & $\\mathbf{{{roc_auc_score(y,oof):.3f}}}$ & "
             f"$\\mathbf{{{balanced_accuracy_score(y,pred):.3f}}}$ & $\\mathbf{{{matthews_corrcoef(y,pred):.3f}}}$ \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(ROOT / "outputs/results/tab_seasonal.tex").write_text("\n".join(lines))
print(f"Saved tab_seasonal.tex (pooled threshold {thr:.3f}); overall F1="
      f"{f1_score(y,pred,zero_division=0):.3f} AUC={roc_auc_score(y,oof):.3f}")
