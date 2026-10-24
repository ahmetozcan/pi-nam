"""Stage 4 analysis of the rev1 grid (R2-6, R3-6, R3-7, R3-8, R2-1, R3-2).

Reads outputs/rev1/grid/*.npz (+ aux/) and writes, under outputs/rev1/analysis/:
  tests_cv.csv          PI-NAM (and PI-NAM-fixed, if run) vs each other model, fold-level
                        Wilcoxon + Holm per reference (rand, block)
  tests_year.csv        2023->2024: DeLong (AUC), exact McNemar, weekly-block bootstrap CIs
                        of F1 / balanced accuracy / false-authorization-rate differences
  calibration.csv       Brier, ECE, mean predicted probability vs prevalence
  threshold_shift.csv   chosen vs test-optimal threshold and predicted-positive rate (year)
  seasonal.csv          per-season metrics with season prevalence
  safety_operating.csv  thresholds set on threshold-source predictions to cap the
                        false-authorization rate, evaluated on test
  reliability_year.png  reliability diagrams, year protocol
Thresholds are never chosen on test labels; the test-optimal threshold in
threshold_shift.csv is a diagnostic only.

With --pinam-threshold inner_cv the PI-NAM rows use the *_thrcv grid files (threshold by
the baselines' 5-fold inner CV) and outputs go to outputs/rev1/analysis_thrcv/.
With --pinam-lam X the PI-NAM rows use the *_lam<X> files (run_grid.py --lam-mono X) and
the output folder gets the same suffix.

Run from the repository root:  python3 scripts/rev1/analyze_grid.py [--pinam-threshold inner_cv]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.revision.info_sets import load_inputs  # noqa: E402
from src.revision.metrics import THR_GRID, compute_metrics, reliability_curve  # noqa: E402
from src.revision.stats import (delong, holm, mcnemar_exact, paired_bootstrap_diff,  # noqa: E402
                                wilcoxon_paired)

logger = logging.getLogger("analyze_grid")
GRID = ROOT / "outputs/rev1/grid"
OUT = ROOT / "outputs/rev1/analysis"
BASELINES = {"XGBoost": "xgboost", "LightGBM": "lightgbm", "Random Forest": "randomforest",
             "CatBoost": "catboost", "SVM": "svm", "TabPFN": "tabpfn"}
PINAMS = {"PI-NAM": "pinam", "PI-NAM-fixed": "pinamfixed"}
CV_METRICS = ["f1", "roc_auc", "balanced_acc", "mcc", "false_authorization_rate", "brier"]
FAR_TARGETS = (0.10, 0.05)
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def load(info: str, proto: str, slug: str) -> Dict[str, np.ndarray]:
    """Grid predictions sorted by test row (so different models align row by row)."""
    d = np.load(GRID / f"{info}_{proto}_{slug}.npz")
    o = np.argsort(d["test"])
    r = {k: d[k][o] for k in ("test", "y", "p", "thr", "split")}
    a = np.load(GRID / "aux" / f"{info}_{proto}_{slug}.npz")
    r |= {k: a[k] for k in ("thr_y", "thr_p", "thr_split")}
    return r


def available(info: str, proto: str) -> Dict[str, str]:
    models = {**PINAMS, **BASELINES}
    return {m: s for m, s in models.items() if (GRID / f"{info}_{proto}_{s}.npz").exists()}


def row_metrics(y, p, thr) -> Dict[str, float]:
    """Threshold metrics with a per-row threshold (pooled CV predictions keep fold thresholds)."""
    pred = (p >= thr).astype(int)
    neg = y == 0
    out = {"n": len(y), "prevalence": y.mean(), "pred_pos_rate": pred.mean(),
           "f1": f1_score(y, pred, zero_division=0),
           "balanced_acc": balanced_accuracy_score(y, pred) if 0 < y.mean() < 1 else np.nan,
           "mcc": matthews_corrcoef(y, pred),
           "false_authorization_rate": pred[neg].mean() if neg.any() else np.nan,
           "recall_fly": pred[~neg].mean() if (~neg).any() else np.nan}
    out["roc_auc"] = roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan
    return out


def _f1(y, p, t):
    return f1_score(y, (p >= t).astype(int), zero_division=0)


def _bacc(y, p, t):
    return balanced_accuracy_score(y, (p >= t).astype(int))


def _far(y, p, t):
    return float(((p >= t) & (y == 0)).sum() / max((y == 0).sum(), 1))


# ---------------------------------------------------------------- significance tests
def tests_cv() -> pd.DataFrame:
    rows = []
    for info in ("S0", "S1"):
        for proto in ("rand", "block"):
            models = available(info, proto)
            folds = {}
            for m, s in models.items():
                d = load(info, proto, s)
                folds[m] = pd.DataFrame([compute_metrics(d["y"][d["split"] == k], d["p"][d["split"] == k],
                                                         float(d["thr"][d["split"] == k][0]))
                                         for k in np.unique(d["split"])])
            for ref in [m for m in PINAMS if m in folds]:
                others = [b for b in BASELINES if b in folds] + [o for o in PINAMS
                                                                 if o in folds and o != ref]
                for metric in CV_METRICS:
                    block = []
                    for b in others:
                        r = wilcoxon_paired(folds[ref][metric], folds[b][metric])
                        block.append({"info_set": info, "protocol": proto, "metric": metric,
                                      "reference": ref, "baseline": b,
                                      "pinam": folds[ref][metric].mean(),
                                      "base": folds[b][metric].mean(), **r})
                    for r, ph in zip(block, holm([r["p"] for r in block])):
                        r["p_holm"] = ph
                    rows += block
    return pd.DataFrame(rows)


def tests_year(weeks: np.ndarray) -> pd.DataFrame:
    rows = []
    for info in ("S0", "S1"):
        data = {m: load(info, "year", s) for m, s in available(info, "year").items()}
        y = data["PI-NAM"]["y"]
        g = weeks[data["PI-NAM"]["test"]]
        for pm in PINAMS:
            block = []
            others = list(BASELINES) + [o for o in PINAMS if o != pm]
            for b in others:
                a, c = data[pm], data[b]
                r = {"info_set": info, "pinam": pm, "other": b}
                dl = delong(y, a["p"], c["p"])
                r |= {"auc_pinam": dl["auc1"], "auc_other": dl["auc2"], "auc_diff": dl["diff"],
                      "p_delong": dl["p"]}
                mc = mcnemar_exact(y, a["p"] >= a["thr"], c["p"] >= c["thr"])
                r |= {"mcnemar_only_pinam": mc["only1_correct"], "mcnemar_only_other": mc["only2_correct"],
                      "p_mcnemar": mc["p"]}
                for name, fn in (("f1", _f1), ("bacc", _bacc), ("far", _far)):
                    bs = paired_bootstrap_diff(y, a["p"], a["thr"], c["p"], c["thr"], fn, groups=g)
                    r |= {f"{name}_diff": bs["diff"], f"{name}_ci_lo": bs["ci_lo"], f"{name}_ci_hi": bs["ci_hi"]}
                block.append(r)
            for col in ("p_delong", "p_mcnemar"):
                for r, ph in zip(block, holm([r[col] for r in block])):
                    r[col + "_holm"] = ph
            rows += block
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- calibration and thresholds
def calibration_and_shift() -> tuple:
    cal, shift = [], []
    for info in ("S0", "S1"):
        for proto in ("rand", "block", "year"):
            for m, s in available(info, proto).items():
                d = load(info, proto, s)
                cm = compute_metrics(d["y"], d["p"], 0.5)
                cal.append({"info_set": info, "protocol": proto, "model": m, "brier": cm["brier"],
                            "ece": cm["ece"], "mean_p": d["p"].mean(), "prevalence": d["y"].mean()})
                if proto != "year":
                    continue
                f1s = [_f1(d["y"], d["p"], t) for t in THR_GRID]
                shift.append({"info_set": info, "model": m, "thr_chosen": d["thr"][0],
                              "thr_test_optimal": THR_GRID[int(np.argmax(f1s))],
                              "f1_chosen": _f1(d["y"], d["p"], d["thr"]), "f1_test_optimal": max(f1s),
                              "pred_pos_rate": (d["p"] >= d["thr"]).mean(),
                              "prevalence_test": d["y"].mean(),
                              "prevalence_thr_source": d["thr_y"].mean(),
                              "n_thr_source": len(d["thr_y"])})
    return pd.DataFrame(cal), pd.DataFrame(shift)


def plot_reliability(path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, info in zip(axes, ("S0", "S1")):
        for m, s in available(info, "year").items():
            d = load(info, "year", s)
            pts = np.array(reliability_curve(d["y"], d["p"]))
            style = dict(lw=2.2) if m.startswith("PI-NAM") else dict(lw=1, alpha=0.7)
            ax.plot(pts[:, 0], pts[:, 1], marker="o", ms=3, label=m, **style)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(title=f"{info}, train 2023 / test 2024", xlabel="mean predicted probability")
    axes[0].set_ylabel("observed flight frequency")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- seasons and safety
def seasonal(dates: pd.Series) -> pd.DataFrame:
    season = dates.dt.month.map(SEASONS).to_numpy()
    rows = []
    for info in ("S0", "S1"):
        for proto in ("rand", "block", "year"):
            for m, s in available(info, proto).items():
                d = load(info, proto, s)
                for se in ("DJF", "MAM", "JJA", "SON"):
                    k = season[d["test"]] == se
                    if k.any():
                        rows.append({"info_set": info, "protocol": proto, "model": m, "season": se,
                                     **row_metrics(d["y"][k], d["p"][k], d["thr"][k])})
    return pd.DataFrame(rows)


def _far_threshold(p_neg: np.ndarray, target: float) -> float:
    """Smallest threshold whose false-authorization rate on the source negatives is <= target."""
    if len(p_neg) == 0:
        return 0.5
    return float(np.quantile(p_neg, 1 - target, method="higher") + 1e-9)


def safety_operating() -> pd.DataFrame:
    rows = []
    for info in ("S0", "S1"):
        for proto in ("rand", "block", "year"):
            for m, s in available(info, proto).items():
                d = load(info, proto, s)
                for target in FAR_TARGETS:
                    thr = np.empty(len(d["y"]))
                    for k in np.unique(d["split"]):
                        src = (d["thr_split"] == k) & (d["thr_y"] == 0)
                        thr[d["split"] == k] = _far_threshold(d["thr_p"][src], target)
                    r = row_metrics(d["y"], d["p"], thr)
                    rows.append({"info_set": info, "protocol": proto, "model": m, "far_target": target,
                                 "far_test": r["false_authorization_rate"], "recall_fly": r["recall_fly"],
                                 "f1": r["f1"], "balanced_acc": r["balanced_acc"],
                                 "mean_thr": thr.mean()})
    return pd.DataFrame(rows)


def main() -> None:
    global OUT, PINAMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinam-threshold", default="val", choices=["val", "inner_cv"])
    ap.add_argument("--pinam-lam", type=float, default=None)
    args = ap.parse_args()
    if args.pinam_threshold == "inner_cv":
        PINAMS = {m: s + "_thrcv" for m, s in PINAMS.items()}
        OUT = OUT.with_name("analysis_thrcv")
    if args.pinam_lam is not None:
        PINAMS = {m: s + f"_lam{args.pinam_lam:g}" for m, s in PINAMS.items()}
        OUT = OUT.with_name(OUT.name + f"_lam{args.pinam_lam:g}")
    OUT.mkdir(parents=True, exist_ok=True)
    dates = load_inputs("S0", ROOT / "config.yaml").dates.reset_index(drop=True)
    iso = dates.dt.isocalendar()
    weeks = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()
    outputs: List[tuple] = [("tests_cv.csv", tests_cv()), ("tests_year.csv", tests_year(weeks))]
    cal, shift = calibration_and_shift()
    outputs += [("calibration.csv", cal), ("threshold_shift.csv", shift),
                ("seasonal.csv", seasonal(dates)), ("safety_operating.csv", safety_operating())]
    for name, df in outputs:
        df.to_csv(OUT / name, index=False, float_format="%.4f")
        logger.info("wrote %s (%d rows)", name, len(df))
    plot_reliability(OUT / "reliability_year.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("src.feature_engineering", "src.revision.info_sets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    main()
