"""R3-6: class-specific metrics and confusion matrices -> outputs/rev1/class_metrics/.

For every main-table model and protocol (random CV, blocked CV, 2023 -> 2024) the held-out
predictions are thresholded with each split's own threshold (inner CV on the training part)
and the confusion matrices of the splits are summed (728 days for CV, 2024 for year-ahead).

  class_metrics.csv      S0 and S1: TN, FP, FN, TP, precision / recall per class,
                         specificity, false-authorization rate (FP / (FP + TN)),
                         missed-flight rate (FN / (FN + TP))
  tab_class_metrics.tex  S0 table for the manuscript (or supplementary material)

Run from the repository root:  python3 scripts/rev1/class_metrics.py
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ablation import _load  # noqa: E402
from fig_results import MAIN_MODELS, _SLUG  # noqa: E402

OUT = ROOT / "outputs/rev1/class_metrics"
PROTOCOLS = {"rand": "Random 10-fold CV", "block": "Blocked CV",
             "year": "Train 2023, test 2024"}


def pooled_counts(d: Dict) -> Dict[str, int]:
    """Sum of per-split confusion matrices, each at that split's threshold."""
    total = np.zeros(4, dtype=int)
    for s in np.unique(d["split"]):
        m = d["split"] == s
        pred = (d["p"][m] >= d["thr"][m]).astype(int)
        total += confusion_matrix(d["y"][m], pred, labels=[0, 1]).ravel()
    return dict(zip(("tn", "fp", "fn", "tp"), (int(v) for v in total)))


def class_rates(c: Dict[str, int]) -> Dict[str, float]:
    tn, fp, fn, tp = c["tn"], c["fp"], c["fn"], c["tp"]
    return {"precision_fly": tp / (tp + fp), "recall_fly": tp / (tp + fn),
            "precision_nofly": tn / (tn + fn), "specificity": tn / (tn + fp),
            "false_authorization_rate": fp / (fp + tn), "missed_flight_rate": fn / (fn + tp)}


def table() -> pd.DataFrame:
    rows = []
    for info in ("S0", "S1"):
        for proto in PROTOCOLS:
            for model in MAIN_MODELS:
                path = ROOT / "outputs/rev1" / (_SLUG[model].replace("S0_", f"{info}_")
                                                .format(p=proto) + ".npz")
                if not path.exists():
                    continue
                c = pooled_counts(_load(path))
                rows.append({"info_set": info, "protocol": proto, "model": model, **c,
                             **class_rates(c)})
    return pd.DataFrame(rows)


def write_latex(df: pd.DataFrame, path: Path) -> None:
    s0 = df[df.info_set == "S0"]
    lines = [r"\begin{table}[!ht]", r"\centering\small",
             r"\caption{Class-specific results with same-day information (S0). Confusion "
             r"matrices are summed over the held-out splits, each thresholded at its own "
             r"inner-CV threshold. Fly = green flag. FAR: false-authorization rate "
             r"FP/(FP+TN); MFR: missed-flight rate FN/(FN+TP).}",
             r"\label{tab:class_metrics}",
             r"\begin{tabular}{lrrrrcccccc}", r"\toprule",
             r" & \multicolumn{4}{c}{Confusion matrix} & \multicolumn{2}{c}{Fly} & "
             r"\multicolumn{2}{c}{No fly} & & \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r"Model & TN & FP & FN & TP & Prec. & Rec. & Prec. & Spec. & FAR & MFR \\",
             r"\midrule"]
    for proto, title in PROTOCOLS.items():
        sub = s0[s0.protocol == proto]
        n0, n1 = sub.iloc[0][["tn", "fp"]].sum(), sub.iloc[0][["fn", "tp"]].sum()
        lines.append(rf"\multicolumn{{11}}{{l}}{{\textit{{{title}}} "
                     rf"({n0} no-flight, {n1} flight days)}} \\")
        for r in sub.itertuples():
            lines.append(f"{r.model} & {r.tn} & {r.fp} & {r.fn} & {r.tp} & "
                         f"{r.precision_fly:.3f} & {r.recall_fly:.3f} & {r.precision_nofly:.3f} & "
                         f"{r.specificity:.3f} & {r.false_authorization_rate:.3f} & "
                         f"{r.missed_flight_rate:.3f} \\\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = table()
    df.to_csv(OUT / "class_metrics.csv", index=False, float_format="%.4f")
    write_latex(df, OUT / "tab_class_metrics.tex")
    print(df[df.info_set == "S0"].drop(columns="info_set").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
