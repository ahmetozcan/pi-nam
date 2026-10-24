"""Summarise stage 6 (outputs/rev1/extra) against the final PI-NAM (lambda = 10, inner-CV
threshold) -> outputs/rev1/extra/summary/.

  extra_metrics.csv       fold-mean metrics of every model (extra, fd-only originals, PI-NAM)
  extra_tests.csv         PI-NAM vs each model -- rand/block: fold Wilcoxon + Holm (AUC, BAcc,
                          MCC, F1); year: DeLong + weekly-block bootstrap CIs (F1, BAcc, FAR)
  extra_monotonicity.csv  share of test days where +1 SD in a MONO_DEC channel raises the
                          predicted probability by more than 0.01 (PI-NAM rows from
                          analysis_thrcv_lam10/monotonicity_grid.csv, same definition)

Run from the repository root:  python3 scripts/rev1/analyze_extra.py [--smoke]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ablation import METRICS, TESTED, _folds, _load  # noqa: E402
from src.pinam_search import MONO_DEC  # noqa: E402
from src.revision.info_sets import load_inputs  # noqa: E402
from src.revision.stats import delong, holm, paired_bootstrap_diff, wilcoxon_paired  # noqa: E402

logger = logging.getLogger("analyze_extra")
EXTRA = ROOT / "outputs/rev1/extra"
GRID = ROOT / "outputs/rev1/grid"
FD_THRESHOLD = 0.01
PINAM_SUFFIX = "_thrcv_lam10"


def _f1(y, p, t):
    return f1_score(y, (p >= t).astype(int), zero_division=0)


def _bacc(y, p, t):
    return balanced_accuracy_score(y, (p >= t).astype(int))


def _far(y, p, t):
    neg = y == 0
    return float(((p >= t) & neg).sum() / max(neg.sum(), 1))


def _files(info: str, proto: str, smoke: bool):
    tail = "_smoke" if smoke else ""
    files = {f.stem.removeprefix(f"{info}_{proto}_").removesuffix(tail): f
             for f in sorted(EXTRA.glob(f"{info}_{proto}_*{tail}.npz"))
             if f.stem.endswith("_smoke") == smoke}
    refs = {"PI-NAM": GRID / f"{info}_{proto}_pinam{PINAM_SUFFIX}.npz"}
    if proto == "year":
        refs["PI-NAM-fixed"] = GRID / f"{info}_{proto}_pinamfixed{PINAM_SUFFIX}.npz"
    return files, refs


def _tests_cv(data, folds, proto, info):
    rows = []
    others = [v for v in data if not v.startswith("PI-NAM")]
    for m in TESTED:
        block = [{"info_set": info, "protocol": proto, "pinam": "PI-NAM", "other": v, "metric": m,
                  **wilcoxon_paired(folds["PI-NAM"][m], folds[v][m])} for v in others]
        for r, ph in zip(block, holm([r["p"] for r in block])):
            r["p_holm"] = ph
        rows += block
    return rows


def _tests_year(data, weeks, info):
    rows = []
    for ref in [r for r in ("PI-NAM", "PI-NAM-fixed") if r in data]:
        a = data[ref]
        g = weeks[a["test"]]
        block = []
        for v in [v for v in data if not v.startswith("PI-NAM")]:
            b = data[v]
            assert np.array_equal(a["test"], b["test"])
            r = {"info_set": info, "protocol": "year", "pinam": ref, "other": v}
            dl = delong(a["y"], a["p"], b["p"])
            r |= {"auc_diff": dl["diff"], "p_delong": dl["p"]}
            for name, fn in (("f1", _f1), ("bacc", _bacc), ("far", _far)):
                bs = paired_bootstrap_diff(a["y"], a["p"], a["thr"], b["p"], b["thr"], fn, groups=g)
                r |= {f"{name}_diff": bs["diff"], f"{name}_ci_lo": bs["ci_lo"], f"{name}_ci_hi": bs["ci_hi"]}
            block.append(r)
        for r, ph in zip(block, holm([r["p_delong"] for r in block])):
            r["p_delong_holm"] = ph
        rows += block
    return rows


def _fd_rates(path: Path) -> dict:
    fd = np.load(path.parent / "fd" / path.name)["fd"]
    return dict(zip(MONO_DEC, (fd > FD_THRESHOLD).mean(1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--info-set", default="S0")
    args = ap.parse_args()
    info = args.info_set
    out = EXTRA / ("summary_smoke" if args.smoke else "summary")
    out.mkdir(parents=True, exist_ok=True)
    iso = load_inputs(info, ROOT / "config.yaml").dates.reset_index(drop=True).dt.isocalendar()
    weeks = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()
    mono_ref = pd.read_csv(ROOT / f"outputs/rev1/analysis{PINAM_SUFFIX}/monotonicity_grid.csv"
                           ).set_index("file")
    metrics, tests, mono = [], [], []
    for proto in ("rand", "block", "year"):
        files, refs = _files(info, proto, args.smoke)
        if not files:
            continue
        data = {v: _load(f) for v, f in (refs | files).items() if f.exists()}
        folds = {v: _folds(d) for v, d in data.items()}
        metrics += [{"info_set": info, "protocol": proto, "model": v, **folds[v][METRICS].mean().to_dict()}
                    for v in data]
        tests += _tests_year(data, weeks, info) if proto == "year" else _tests_cv(data, folds, proto, info)
        mono += [{"info_set": info, "protocol": proto, "model": v, **_fd_rates(f)} for v, f in files.items()]
        mono += [{"info_set": info, "protocol": proto, "model": v,
                  **mono_ref.loc[f.stem, MONO_DEC].to_dict()} for v, f in refs.items()
                 if f.stem in mono_ref.index]
    for name, rows in (("extra_metrics.csv", metrics), ("extra_tests.csv", tests),
                       ("extra_monotonicity.csv", mono)):
        pd.DataFrame(rows).to_csv(out / name, index=False, float_format="%.4f")
        logger.info("wrote %s (%d rows)", name, len(rows))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("src.feature_engineering", "src.revision.info_sets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    main()
