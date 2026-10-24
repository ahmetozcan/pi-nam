"""Summarise the stage 5 ablation (outputs/rev1/ablation) -> outputs/rev1/ablation/summary/.

  ablation_metrics.csv     fold-mean metrics per variant (block: 10 folds; year: one split)
  ablation_tests.csv       each variant vs its reference (V1, or the final S_lam10 for the
                           _l10 variants, the cumulative chain and the lambda grid) -- block: fold Wilcoxon + Holm (AUC, BAcc, MCC, F1);
                           year: DeLong (AUC) + weekly-block bootstrap CI of the BAcc difference
  ablation_monotonicity.csv  share of test days where +1 SD in a constrained channel raises
                           the ensemble probability by more than 0.01 (same check as stage 9)

Run from the repository root:  python3 scripts/rev1/analyze_ablation.py [--smoke]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.pinam_search import MONO_DEC, PINAMX, _n_factors  # noqa: E402
from src.pinn import IDX, RAW_CHANNELS  # noqa: E402
from src.revision.info_sets import load_inputs  # noqa: E402
from src.revision.metrics import compute_metrics  # noqa: E402
from src.revision.stats import delong, holm, paired_bootstrap_diff, wilcoxon_paired  # noqa: E402

logger = logging.getLogger("analyze_ablation")
ABL = ROOT / "outputs/rev1/ablation"
METRICS = ["roc_auc", "balanced_acc", "mcc", "f1", "false_authorization_rate", "brier"]
TESTED = ["roc_auc", "balanced_acc", "mcc", "f1"]
FD_THRESHOLD = 0.01


def _load(path: Path) -> Dict:
    d = np.load(path)
    o = np.argsort(d["test"])
    r = {k: d[k][o] for k in ("test", "y", "p", "thr", "split")}
    r["meta"] = json.loads(str(d["meta"]))
    return r


def _folds(d: Dict) -> pd.DataFrame:
    return pd.DataFrame([compute_metrics(d["y"][d["split"] == k], d["p"][d["split"] == k],
                                         float(d["thr"][d["split"] == k][0]))
                         for k in np.unique(d["split"])])


def _bacc(y, p, t):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y, (p >= t).astype(int))


def metrics_and_tests(files: Dict[str, Path], proto: str, weeks: np.ndarray):
    data = {v: _load(f) for v, f in files.items()}
    folds = {v: _folds(d) for v, d in data.items()}
    rows = [{"protocol": proto, "variant": v, **folds[v][METRICS].mean().to_dict()} for v in data]
    tests: List[Dict] = []
    for ref_name, members in REFERENCES.items():
        if ref_name in data:
            tests += _tests_vs(data, folds, proto, weeks, ref_name,
                               [v for v in data if v != ref_name and members(v)])
    return rows, tests


# Each variant is tested against the model it was derived from: the submitted V1
# (lambda = 0.081) and the final model S_lam10 (lambda = 10, agreed 2026-10-10).
REFERENCES = {
    "V1": lambda v: not v.endswith("_l10"),
    "S_lam10": lambda v: v.endswith("_l10") or v.startswith(("C", "S_lam")) or v == "V1",
}


def _tests_vs(data, folds, proto, weeks, ref_name, others) -> List[Dict]:
    ref, tests = data[ref_name], []
    if proto == "year":
        g = weeks[ref["test"]]
        for v in others:
            dl = delong(ref["y"], data[v]["p"], ref["p"])
            bs = paired_bootstrap_diff(ref["y"], data[v]["p"], data[v]["thr"], ref["p"], ref["thr"],
                                       _bacc, groups=g)
            tests.append({"protocol": proto, "reference": ref_name, "variant": v,
                          "auc_diff": dl["diff"], "p_delong": dl["p"], "bacc_diff": bs["diff"],
                          "bacc_ci_lo": bs["ci_lo"], "bacc_ci_hi": bs["ci_hi"]})
        for r, ph in zip(tests, holm([r["p_delong"] for r in tests])):
            r["p_delong_holm"] = ph
        return tests
    for m in TESTED:
        block = [{"protocol": proto, "reference": ref_name, "variant": v, "metric": m,
                  **wilcoxon_paired(folds[v][m], folds[ref_name][m])} for v in others]
        for r, ph in zip(block, holm([r["p"] for r in block])):
            r["p_holm"] = ph
        tests += block
    return tests


def _ensemble(params: Dict, states: List[Dict]) -> List[PINAMX]:
    nf = _n_factors(params["spec"])
    models = []
    for st in states:
        m = PINAMX(torch.zeros(nf), torch.ones(nf), nf, params["spec"], params["hidden"],
                   params["dropout"], raw_mean=torch.zeros(len(RAW_CHANNELS)),
                   raw_std=torch.ones(len(RAW_CHANNELS)), depth=params["depth"])
        m.load_state_dict(st)
        models.append(m.eval())
    return models


def monotonicity(path: Path, win: np.ndarray) -> Dict[str, float]:
    """FD check (+1 SD on the decision day) on every split's test days with that split's ensemble."""
    params = json.loads(str(np.load(path)["meta"]))["params"]
    bundle = torch.load(path.parent / "states" / f"{path.stem}.pt", weights_only=False)
    sd = win[:, -1, :].std(0)
    fd = {c: [] for c in MONO_DEC}
    with torch.no_grad():
        for split in bundle.values():
            models = _ensemble(params, split["states"])
            x = torch.tensor(win[split["test"]])
            base = torch.stack([m(x) for m in models]).mean(0)
            for c in MONO_DEC:
                xp = x.clone()
                xp[:, -1, IDX[c]] += float(sd[IDX[c]])
                fd[c].append((torch.stack([m(xp) for m in models]).mean(0) - base).numpy())
    return {c: float((np.concatenate(v) > FD_THRESHOLD).mean()) for c, v in fd.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--info-set", default="S0")
    args = ap.parse_args()
    out = ABL / ("summary_smoke" if args.smoke else "summary")
    out.mkdir(parents=True, exist_ok=True)
    inp = load_inputs(args.info_set, ROOT / "config.yaml")
    iso = inp.dates.reset_index(drop=True).dt.isocalendar()
    weeks = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()
    rows, tests, mono = [], [], []
    for proto in ("year", "block"):
        files = {}
        for f in sorted(ABL.glob(f"{args.info_set}_{proto}_*.npz")):
            if f.stem.endswith("_smoke") != args.smoke:
                continue
            files[f.stem.removeprefix(f"{args.info_set}_{proto}_").removesuffix("_smoke")] = f
        if not files:
            continue
        r, t = metrics_and_tests(files, proto, weeks)
        rows += r
        tests += t
        for v, f in files.items():
            if (f.parent / "states" / f"{f.stem}.pt").exists():
                mono.append({"protocol": proto, "variant": v, **monotonicity(f, inp.win)})
    for name, df in (("ablation_metrics.csv", rows), ("ablation_tests.csv", tests),
                     ("ablation_monotonicity.csv", mono)):
        if not df:  # e.g. no saved model states: keep the existing summary
            logger.info("skipped %s (nothing to write)", name)
            continue
        pd.DataFrame(df).to_csv(out / name, index=False, float_format="%.4f")
        logger.info("wrote %s (%d rows)", name, len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("src.feature_engineering", "src.revision.info_sets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    main()
