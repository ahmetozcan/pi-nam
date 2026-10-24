"""Stage 10 result figures (S0, final PI-NAM with lambda = 10), drawn at print size.

  fig_main_comparison   ROC-AUC and balanced accuracy of PI-NAM and the main-table models
                        under random CV, blocked CV (fold values, mean and 95% t-interval)
                        and year-ahead 2023 -> 2024 (95% weekly-block bootstrap interval)
  fig_robustness        (a) one variable group masked at prediction time, (b) random missing
                        cells, (c) model refitted without a group -- blocked CV, change in
                        ROC-AUC; (d) learning curve (share of the training data)

Called by make_all_figures.py; inputs: outputs/rev1/{grid,extra,robustness/summary}.
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ablation import _folds, _load  # noqa: E402
from src.revision import plot_style as ps  # noqa: E402
from src.revision.info_sets import load_inputs  # noqa: E402

OUT = ROOT / "outputs/rev1"
RB = OUT / "robustness/summary"
MAIN_MODELS = ["PI-NAM", "PI-NAM-fixed", "EBM", "XGBoost", "LightGBM", "CatBoost",
               "Random Forest", "SVM", "TabPFN"]
_SLUG = {"PI-NAM": "grid/S0_{p}_pinam_thrcv_lam10", "PI-NAM-fixed": "grid/S0_{p}_pinamfixed_thrcv_lam10",
         "EBM": "extra/S0_{p}_ebm", "XGBoost": "grid/S0_{p}_xgboost", "LightGBM": "grid/S0_{p}_lightgbm",
         "CatBoost": "grid/S0_{p}_catboost", "Random Forest": "grid/S0_{p}_randomforest",
         "SVM": "grid/S0_{p}_svm", "TabPFN": "grid/S0_{p}_tabpfn"}
PROTOCOLS = [("rand", "Random CV"), ("block", "Blocked CV"), ("year", "Year-ahead (2023 → 2024)")]
METRICS = [("roc_auc", "AUC"), ("balanced_acc", "Balanced accuracy")]
DISPLAY = {"TabPFN": "TabPFN v2"}  # names as in Table tab:main
GROUPS = [("wind", "Wind"), ("cloud", "Cloud & visibility"), ("temperature", "Temperature"),
          ("humidity", "Humidity"), ("pressure", "Pressure"), ("precipitation", "Precipitation")]


def _path(model: str, proto: str) -> Path:
    return OUT / (_SLUG[model].format(p=proto) + ".npz")


def _t_ci(x: np.ndarray) -> float:
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def _metric_fn(m: str) -> Callable:
    if m == "roc_auc":
        return lambda y, p, t: roc_auc_score(y, p)
    return lambda y, p, t: balanced_accuracy_score(y, (p >= t).astype(int))


def _block_boot_ci(d: Dict, weeks: np.ndarray, fn: Callable, n_boot: int = 2000,
                   seed: int = 42) -> List[float]:
    """95% percentile interval, resampling ISO weeks of the test year (as in the tests)."""
    rng = np.random.default_rng(seed)
    w = weeks[d["test"]]
    units = np.unique(w)
    index = {u: np.flatnonzero(w == u) for u in units}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([index[u] for u in rng.choice(units, len(units))])
        if len(np.unique(d["y"][idx])) == 2:
            vals.append(fn(d["y"][idx], d["p"][idx], d["thr"][idx]))
    return list(np.percentile(vals, [2.5, 97.5]))


def _weeks() -> np.ndarray:
    iso = load_inputs("S0", ROOT / "config.yaml").dates.reset_index(drop=True).dt.isocalendar()
    return (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()


def main_comparison(out_dir: Path) -> None:
    weeks = _weeks()
    fig, axes = plt.subplots(2, 3, figsize=(ps.TEXT_WIDTH, 5.4), sharey=True)
    ypos = {m: i for i, m in enumerate(MAIN_MODELS[::-1])}
    for c, (proto, title) in enumerate(PROTOCOLS):
        models = [m for m in MAIN_MODELS if _path(m, proto).exists()]
        data = {m: _load(_path(m, proto)) for m in models}
        folds = {m: _folds(d) for m, d in data.items()} if proto != "year" else {}
        for r, (metric, label) in enumerate(METRICS):
            ax = axes[r, c]
            for m in models:
                y0, col = ypos[m], ps.MODEL_COLORS[m]
                if proto == "year":
                    d = data[m]
                    est = _metric_fn(metric)(d["y"], d["p"], d["thr"])
                    lo, hi = _block_boot_ci(d, weeks, _metric_fn(metric))
                else:
                    v = folds[m][metric].to_numpy()
                    est, h = v.mean(), _t_ci(v)
                    lo, hi = est - h, est + h
                    jit = np.linspace(-0.18, 0.18, len(v))
                    ax.scatter(v, y0 + jit, s=6, color=col, alpha=0.35, lw=0, zorder=2)
                ax.plot([lo, hi], [y0, y0], color=col, lw=1.6, zorder=3)
                ax.plot(est, y0, "o", ms=5.5 if m.startswith("PI-NAM") else 4.5,
                        color=col, mec="black", mew=0.5, zorder=4)
                if m == "PI-NAM":
                    ax.axvline(est, color=col, lw=0.8, ls="--", zorder=1)
            ax.grid(axis="x")
            ax.set_xlabel(label)
            if r == 0:
                ax.set_title(f"({'abc'[c]}) {title}", fontsize=ps.FONT)
    for ax in axes[:, 0]:
        ax.set_yticks(list(ypos.values()), [DISPLAY.get(m, m) for m in ypos])
    for r in range(2):
        lo = min(a.get_xlim()[0] for a in axes[r])
        hi = max(a.get_xlim()[1] for a in axes[r])
        for a in axes[r]:
            a.set_xlim(lo, hi)
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    ps.save(fig, out_dir, "fig_main_comparison")


# ── robustness ───────────────────────────────────────────────────────────────

def _group_panel(ax, df: pd.DataFrame, key: Callable[[str], str], title: str) -> None:
    for i, (g, label) in enumerate(GROUPS[::-1]):
        sub = df[df["key"] == key(g)]
        others = sub[sub.model != "PI-NAM"]
        ax.plot([others.d_roc_auc.min(), others.d_roc_auc.max()], [i, i], color="#BBBBBB", lw=4,
                solid_capstyle="round", zorder=1)
        ax.scatter(others.d_roc_auc, np.full(len(others), i), s=10, color="#666666", zorder=2)
        pin = sub[sub.model == "PI-NAM"].d_roc_auc
        ax.plot(pin, np.full(len(pin), i), "o", ms=6, color=ps.PINAM, mec="black", mew=0.5, zorder=3)
    ax.set_yticks(range(len(GROUPS)), [lab for _, lab in GROUPS[::-1]])
    ax.axvline(0, color="black", lw=0.7)
    ax.grid(axis="x")
    ax.set_xlabel("Change in AUC")
    ax.set_title(title, fontsize=ps.FONT)


def robustness(out_dir: Path) -> None:
    inf = pd.read_csv(RB / "inference_summary.csv")
    inf = inf[inf.protocol == "block"].rename(columns={"family": "key"})
    drop = pd.read_csv(RB / "drop_summary.csv")
    drop = drop[drop.protocol == "block"].rename(columns={"dropped": "key"})
    curve = pd.read_csv(RB / "curve_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(ps.TEXT_WIDTH, 5.6))
    _group_panel(axes[0, 0], inf, lambda g: f"mask_{g}", "(a) Group masked at prediction")
    _group_panel(axes[1, 0], drop, lambda g: g, "(c) Model refitted without group")
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_xlim(min(axes[0, 0].get_xlim()[0], axes[1, 0].get_xlim()[0]), 0.01)

    ax = axes[0, 1]
    fracs = [10, 20, 30]
    for imp, ls, label in (("mean", "-", "mean imputation"), ("ffill", "--", "previous day")):
        fam = [f"miss{f}_{imp}" for f in fracs]
        oth = inf[(inf.model != "PI-NAM") & inf.key.isin(fam)].groupby("key").d_roc_auc
        ax.fill_between(fracs, oth.min().reindex(fam), oth.max().reindex(fam), color="#BBBBBB",
                        alpha=0.5 if imp == "mean" else 0.3, lw=0)
        pin = inf[(inf.model == "PI-NAM")].set_index("key").d_roc_auc.reindex(fam)
        ax.plot(fracs, pin, ls, marker="o", ms=4, color=ps.PINAM, label=f"PI-NAM, {label}")
    ax.set_xticks(fracs, [f"{f}%" for f in fracs])
    ax.set_xlabel("Missing cells")
    ax.set_ylabel("Change in AUC")
    ax.axhline(0, color="black", lw=0.7)
    ax.grid(axis="y")
    ax.legend(loc="lower left", fontsize=ps.FONT - 1)
    ax.set_title("(b) Random missing cells", fontsize=ps.FONT)

    ax = axes[1, 1]
    for proto, ls, label in (("year", "-", "2023 → 2024"), ("block", "--", "blocked CV")):
        sub = curve[curve.protocol == proto]
        oth = sub[sub.model != "PI-NAM"].groupby("frac").roc_auc
        x = 100 * oth.min().index.to_numpy()
        ax.fill_between(x, oth.min(), oth.max(), color="#BBBBBB", alpha=0.45 if proto == "year" else 0.25,
                        lw=0)
        pin = sub[sub.model == "PI-NAM"].sort_values("frac")
        ax.plot(100 * pin.frac, pin.roc_auc, ls, marker="o", ms=4, color=ps.PINAM,
                label=f"PI-NAM, {label}")
    ax.set_xticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of training data")
    ax.set_ylabel("AUC")
    ax.grid(axis="y")
    ax.legend(loc="lower right", fontsize=ps.FONT - 1)
    ax.set_title("(d) Learning curve", fontsize=ps.FONT)
    n_other = inf[inf.key == "clean"].model.nunique() - 1
    handles = [Line2D([], [], marker="o", ls="", ms=6, color=ps.PINAM, mec="black", mew=0.5),
               Line2D([], [], marker="o", ls="", ms=4, color="#666666"),
               Patch(color="#BBBBBB")]
    fig.legend(handles, ["PI-NAM", f"other models ({n_other})", "range of the other models"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(h_pad=1.2, w_pad=1.0, rect=(0, 0.04, 1, 1))
    ps.save(fig, out_dir, "fig_robustness")


def make(out_dir: Path) -> None:
    ps.apply()
    main_comparison(out_dir)
    robustness(out_dir)


if __name__ == "__main__":
    make(OUT / "figures")
