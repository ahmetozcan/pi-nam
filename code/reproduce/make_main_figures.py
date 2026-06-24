"""
Regenerate the three main comparison figures with the final PI-NAM (V1) and
all Optuna-tuned baselines on raw-16 under the deployable protocol:
  figure-01-main-comparison : per-fold deployable F1 + ROC-AUC boxplots
  figure-02-mcc-boxplot      : per-fold deployable MCC boxplots
  figure-03-auc-trajectory   : fold-level ROC-AUC trajectory
Per-fold values reconstructed from saved weight bundles + PI-NAM V1 OOF.
"""
import pickle, logging, warnings
from pathlib import Path
import numpy as np, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
ROOT = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT))
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from src.feature_engineering import build_features
from src.pinn import RAW_CHANNELS

SEED = 42
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
ns = cfg["features"]["test_folds"]
_, df = build_features(str(ROOT / "config.yaml")); df = df.reset_index(drop=True)
y = df["flight"].values
X = df[RAW_CHANNELS].values.astype(np.float32)
folds = list(StratifiedKFold(ns, shuffle=True, random_state=SEED).split(X, y))
FB = ROOT / "reproducibility/folds/base_models"
BASE = {"SVM": "svm", "XGBoost": "xgboost", "LightGBM": "lightgbm",
        "Random Forest": "randomforest", "CatBoost": "catboost", "TabPFN": "tabpfn"}


def _load(p):
    return pickle.load(open(p, "rb"))


def base_oof(sub):
    if sub == "tabpfn":
        return np.load(ROOT / "reproducibility/oof/tabpfn_oof.npy")
    oof = np.full(len(y), np.nan)
    for fi in range(1, ns + 1):
        b = _load(FB / sub / f"fold_{fi:02d}.pkl"); te = np.array(b["test_idx"])
        oof[te] = b["model"].predict_proba(b["scaler"].transform(X[te]))[:, 1]
    return oof


def bt(p, yy):
    b, bf = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.005):
        f = f1_score(yy, (p >= t).astype(int), zero_division=0)
        if f > bf:
            bf, b = f, float(t)
    return b


def per_fold(oof):
    f1s, aucs, mccs = [], [], []
    for tr, te in folds:
        thr = bt(oof[tr], y[tr]); pred = (oof[te] >= thr).astype(int)
        f1s.append(f1_score(y[te], pred, zero_division=0))
        aucs.append(roc_auc_score(y[te], oof[te]))
        mccs.append(matthews_corrcoef(y[te], pred))
    return np.array(f1s), np.array(aucs), np.array(mccs)


names = list(BASE) + ["PI-NAM"]
oofs = {n: base_oof(s) for n, s in BASE.items()}
oofs["PI-NAM"] = np.load(ROOT / "outputs/results/pinamx_v1_oof.npy")
M = {n: per_fold(oofs[n]) for n in names}

# Okabe-Ito colorblind-safe; PI-NAM highlighted red
COL = {"SVM": "#999999", "XGBoost": "#E69F00", "LightGBM": "#56B4E9",
       "Random Forest": "#009E73", "CatBoost": "#0072B2", "TabPFN": "#CC79A7",
       "PI-NAM": "#D55E00"}
outdirs = [ROOT / "outputs/figures", ROOT / "outputs/figures"]


def save(fig, stem):
    for d in outdirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(d / f"{stem}.png", bbox_inches="tight", dpi=150)


def box(ax, vals, title, ylab):
    bp = ax.boxplot(vals, patch_artist=True, widths=0.6, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=5))
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(COL[n]); patch.set_alpha(0.85)
        patch.set_edgecolor("black"); patch.set_linewidth(1.4 if n == "PI-NAM" else 0.7)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylab, fontsize=9); ax.grid(axis="y", alpha=0.3)


# Figure 1: F1 + AUC
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
box(a1, [M[n][0] for n in names], "Deployable F1-score", "F1")
box(a2, [M[n][1] for n in names], "ROC-AUC", "AUC")
fig.suptitle("Per-fold deployable F1 and ROC-AUC across 10 folds (raw-16, Optuna-tuned)",
             fontsize=11.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96]); save(fig, "figure-01-main-comparison"); plt.close(fig)

# Figure 2: MCC
fig, ax = plt.subplots(figsize=(8.5, 4.8))
box(ax, [M[n][2] for n in names], "Matthews Correlation Coefficient (deployable threshold)", "MCC")
fig.tight_layout(); save(fig, "figure-02-mcc-boxplot"); plt.close(fig)

# Figure 3: AUC trajectory
fig, ax = plt.subplots(figsize=(9.5, 5.0))
x = np.arange(1, ns + 1)
for n in names:
    lw = 2.6 if n == "PI-NAM" else 1.3
    z = 5 if n == "PI-NAM" else 2
    ax.plot(x, M[n][1], marker="o", ms=4, lw=lw, color=COL[n], label=n, zorder=z)
ax.set_xlabel("Fold", fontsize=10); ax.set_ylabel("ROC-AUC", fontsize=10)
ax.set_xticks(x); ax.grid(alpha=0.3)
ax.set_title("Fold-level ROC-AUC trajectory", fontsize=11, fontweight="bold")
ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.30))
fig.tight_layout(); save(fig, "figure-03-auc-trajectory"); plt.close(fig)

print("Regenerated figure-01/02/03 with V1 + raw-16 deployable baselines.")
for n in names:
    print(f"  {n:<14} F1={M[n][0].mean():.3f}  AUC={M[n][1].mean():.3f}  MCC={M[n][2].mean():.3f}")
