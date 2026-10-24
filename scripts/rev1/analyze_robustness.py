"""Summarise stage 7 (outputs/rev1/robustness) -> outputs/rev1/robustness/summary/.

  inference_summary.csv  per protocol / model / condition: fold-mean metrics with the clean
                         thresholds and the change from clean (random-missing conditions
                         averaged over repetitions)
  inference_tests.csv    block: per condition, fold-wise AUC/BAcc change of PI-NAM vs each
                         model (Wilcoxon + Holm across models)
  drop_summary.csv       re-fitted without a variable group, change from the model's full fit
  curve_summary.csv      learning curve (mean over repetitions; 100 % = the full fit)
  reverse_metrics.csv    2024 -> 2023; reverse_tests.csv: DeLong + weekly-block bootstrap
                         vs PI-NAM (as for the 2023 -> 2024 grid)

Run from the repository root:  python3 scripts/rev1/analyze_robustness.py [--smoke]
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ablation import METRICS, _folds, _load  # noqa: E402
from analyze_extra import _tests_year  # noqa: E402
from run_grid import _slug  # noqa: E402
from run_robustness import PINAM_GRID  # noqa: E402
from src.revision.info_sets import load_inputs  # noqa: E402
from src.revision.metrics import compute_metrics  # noqa: E402
from src.revision.stats import holm, wilcoxon_paired  # noqa: E402

logger = logging.getLogger("analyze_robustness")
RB = ROOT / "outputs/rev1/robustness"
DELTA = ["roc_auc", "balanced_acc", "f1", "false_authorization_rate"]
EXTRA_MODELS = {"EBM", "XGBoost-mono"}


def full_fit(proto: str, model: str) -> Path:
    """The clean, full-data result a re-fitted variant is compared with."""
    if model == "PI-NAM":
        return ROOT / "outputs/rev1/grid" / f"S0_{proto}_{PINAM_GRID[proto]}.npz"
    sub = "extra" if model in EXTRA_MODELS else "grid"
    return ROOT / f"outputs/rev1/{sub}" / f"S0_{proto}_{_slug(model)}.npz"


# ── inference ────────────────────────────────────────────────────────────────

def _family(cond: str) -> str:
    return re.sub(r"_r\d+$", "", cond)


def inference_folds(tail: str) -> pd.DataFrame:
    rows = []
    for f in sorted((RB / "inference").glob(f"S0_*{tail}.npz")):
        if f.stem.endswith("_smoke") != bool(tail):
            continue
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        for k, cond in enumerate(d["conditions"]):
            for s in np.unique(d["split"]):
                m = d["split"] == s
                rows.append({"protocol": meta["protocol"], "model": meta["model"], "condition": str(cond),
                             "family": _family(str(cond)), "split": s,
                             **compute_metrics(d["y"][m], d["p"][k][m], float(d["thr"][m][0]))})
    return pd.DataFrame(rows)


def inference_summary(folds: pd.DataFrame) -> pd.DataFrame:
    fam = folds.groupby(["protocol", "model", "family", "split"])[METRICS].mean().reset_index()
    clean = fam[fam.family == "clean"].set_index(["protocol", "model", "split"])[DELTA]
    d = fam.join(clean, on=["protocol", "model", "split"], rsuffix="_clean")
    for m in DELTA:
        d[f"d_{m}"] = d[m] - d[f"{m}_clean"]
    keep = ["protocol", "model", "family"] + METRICS + [f"d_{m}" for m in DELTA]
    return d[keep].groupby(["protocol", "model", "family"], sort=False).mean().reset_index(), d


def inference_tests(per_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    blk = per_fold[(per_fold.protocol == "block") & (per_fold.family != "clean")]
    for fam, g in blk.groupby("family"):
        for m in ("d_roc_auc", "d_balanced_acc"):
            piv = g.pivot(index="split", columns="model", values=m)
            if "PI-NAM" not in piv:
                continue
            block = [{"family": fam, "metric": m, "other": o,
                      "pinam_mean": piv["PI-NAM"].mean(), "other_mean": piv[o].mean(),
                      **wilcoxon_paired(piv["PI-NAM"].to_numpy(), piv[o].to_numpy())}
                     for o in piv.columns if o != "PI-NAM"]
            for r, ph in zip(block, holm([r["p"] for r in block])):
                r["p_holm"] = ph
            rows += block
    return pd.DataFrame(rows)


# ── re-fitting experiments ───────────────────────────────────────────────────

def _mean_metrics(path: Path) -> Dict[str, float]:
    return _folds(_load(path))[METRICS].mean().to_dict()


def refit_summary(sub: str, key: str, tail: str) -> pd.DataFrame:
    rows = []
    for f in sorted((RB / sub).glob(f"S0_*{tail}.npz")):
        if f.stem.endswith("_smoke") != bool(tail):
            continue
        meta = _load(f)["meta"]
        ref = full_fit(meta["protocol"], meta["model"])
        r = {"protocol": meta["protocol"], "model": meta["model"], key: meta[key]}
        if sub == "curve":
            r |= {"rep": meta["rep"], "n_train_mean": float(np.mean(meta["n_train"]))}
        r |= _mean_metrics(f)
        if ref.exists():
            full = _mean_metrics(ref)
            r |= {f"d_{m}": r[m] - full[m] for m in DELTA}
        rows.append(r)
    df = pd.DataFrame(rows)
    if sub == "curve" and len(df):
        df = df.groupby(["protocol", "model", "frac"]).mean(numeric_only=True).reset_index()
        full = [{"protocol": p, "model": m, "frac": 1.0, **_mean_metrics(full_fit(p, m))}
                for p, m in df[["protocol", "model"]].drop_duplicates().itertuples(index=False)
                if full_fit(p, m).exists()]
        df = pd.concat([df, pd.DataFrame(full)], ignore_index=True).sort_values(
            ["protocol", "model", "frac"])
    return df


def reverse(tail: str):
    files = {f.stem: f for f in sorted((RB / "reverse").glob(f"S0_yearrev_*{tail}.npz"))
             if f.stem.endswith("_smoke") == bool(tail)}
    data = {_load(f)["meta"]["model"]: _load(f) for f in files.values()}
    if not data:
        return pd.DataFrame(), pd.DataFrame()
    metrics = pd.DataFrame([{"model": m, **_folds(d)[METRICS].mean().to_dict()} for m, d in data.items()])
    tests = pd.DataFrame()
    if "PI-NAM" in data:
        iso = load_inputs("S0", ROOT / "config.yaml").dates.reset_index(drop=True).dt.isocalendar()
        weeks = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()
        tests = pd.DataFrame(_tests_year(data, weeks, "S0")).assign(protocol="yearrev")
    return metrics, tests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tail = "_smoke" if args.smoke else ""
    out = RB / ("summary_smoke" if args.smoke else "summary")
    out.mkdir(parents=True, exist_ok=True)
    tables = {}
    folds = inference_folds(tail)
    if len(folds):
        tables["inference_summary.csv"], per_fold = inference_summary(folds)
        tables["inference_tests.csv"] = inference_tests(per_fold)
    tables["drop_summary.csv"] = refit_summary("drop", "dropped", tail)
    tables["curve_summary.csv"] = refit_summary("curve", "frac", tail)
    tables["reverse_metrics.csv"], tables["reverse_tests.csv"] = reverse(tail)
    for name, df in tables.items():
        df.to_csv(out / name, index=False, float_format="%.4f")
        logger.info("wrote %s (%d rows)", name, len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("src.feature_engineering", "src.revision.info_sets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    main()
