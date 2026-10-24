"""Summarize rev1 grid predictions into metric tables.

For CV protocols (rand, block) metrics are reported as the mean over outer folds
(fold-level, as in the manuscript) and on the pooled out-of-fold predictions
(pooled; used for calibration, confusion matrices and bootstrap CIs).
For the year protocol there is a single test set (2024).

Run from the repository root:  python3 scripts/rev1/summarize_grid.py [--smoke]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.revision.metrics import compute_metrics  # noqa: E402

SHOW = ["f1", "roc_auc", "pr_auc", "balanced_acc", "mcc", "specificity",
        "false_authorization_rate", "brier", "ece", "threshold"]


def summarize(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    y, p, thr, split = d["y"], d["p"], d["thr"], d["split"]
    # threshold-free metrics on raw probabilities; threshold metrics with each row's fold threshold
    pooled = compute_metrics(y, p, thr=0.5) | _thresholded(y, p, thr)
    folds = [compute_metrics(y[split == s], p[split == s], float(thr[split == s][0]))
             for s in np.unique(split)]
    fold_mean = pd.DataFrame(folds).mean(numeric_only=True).to_dict()
    row = {"info_set": meta["info_set"], "protocol": meta["protocol"], "model": meta["model"],
           "n_splits": len(folds)}
    row |= {f"{k}": fold_mean[k] for k in SHOW}
    row |= {f"pooled_{k}": pooled[k] for k in ("f1", "roc_auc", "brier", "ece", "mcc",
                                               "false_authorization_rate")}
    return row


def _thresholded(y, p, thr) -> dict:
    """Pooled threshold metrics where each row keeps its own fold's threshold."""
    shift = p - thr + 0.5            # row-wise threshold -> common 0.5 cut, ranking kept within fold
    m = compute_metrics(y, np.clip(shift, 0, 1), thr=0.5)
    keep = ("f1", "accuracy", "balanced_acc", "mcc", "specificity", "false_authorization_rate",
            "missed_flight_rate", "tn", "fp", "fn", "tp")
    return {k: m[k] for k in keep}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=Path, default=ROOT / "outputs/rev1/grid")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    files = sorted(f for f in args.grid.glob("*.npz") if f.stem.endswith("_smoke") == args.smoke)
    tab = pd.DataFrame([summarize(f) for f in files])
    if tab.empty:
        print("no grid results found"); return
    tab = tab.sort_values(["info_set", "protocol", "f1"], ascending=[True, True, False])
    out = args.grid.parent / f"grid_summary{'_smoke' if args.smoke else ''}.csv"
    tab.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(tab.round(4).to_string(index=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
