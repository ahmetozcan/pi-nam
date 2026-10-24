"""Stage 5 figure: cumulative ablation of PI-NAM (S0) -> outputs/rev1/fig_ablation_steps.{pdf,png}.

Panel (a): ROC-AUC and balanced accuracy along the cumulative chain raw-16 NAM -> +physics
-> +minima -> +synoptic -> +interactions -> +monotonicity (final, lambda = 10); blocked CV
fold means with 95% t-intervals, year-ahead (2023->2024) as single points.
Panel (b): share of held-out days where +1 SD wind speed / precipitation / wind gust / cloud
cover raises the flight probability by more than 0.01 (from ablation_monotonicity.csv).

Run from the repository root after analyze_ablation.py:  python3 scripts/rev1/plot_ablation.py
(make_all_figures.py calls make() with outputs/rev1/figures.)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ablation import ABL, _folds, _load  # noqa: E402
from src.revision import plot_style as ps  # noqa: E402

STEPS = [("C1_raw", "Raw variables"), ("C2_phys", "+ physics"), ("C3_temp", "+ minima"),
         ("C4_syn", "+ synoptic"), ("C5_inter", "+ interactions"), ("S_lam10", "+ monotonicity")]
CHANNELS = [("windspeed", "wind speed"), ("windgust", "wind gust"),
            ("cloudcover", "cloud cover"), ("precip", "precipitation")]
METRICS = [("roc_auc", "AUC", ps.PINAM), ("balanced_acc", "Balanced accuracy", ps.OKABE_ITO["vermillion"])]


def _ci(x: np.ndarray) -> float:
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def make(out_dir: Path) -> None:
    ps.apply()
    x = np.arange(len(STEPS))
    labels = [s for _, s in STEPS]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(ps.TEXT_WIDTH, 5.6), sharex=True)
    for k, (m, label, color) in enumerate(METRICS):
        block = [_folds(_load(ABL / f"S0_block_{v}.npz"))[m].to_numpy() for v, _ in STEPS]
        year = [_folds(_load(ABL / f"S0_year_{v}.npz"))[m].iloc[0] for v, _ in STEPS]
        off = (k - 0.5) * 0.2 - 0.05          # block and year points side by side
        ax.errorbar(x + off, [b.mean() for b in block], yerr=[_ci(b) for b in block], fmt="o-",
                    color=color, ms=4, capsize=2.5, label=f"{label}, blocked CV")
        ax.plot(x + off + 0.1, year, "s--", color=color, ms=4, mfc="white", label=f"{label}, year-ahead")
    ax.set_ylabel("Score")
    ax.set_title("(a) Discrimination along the cumulative ablation")
    ax.grid(axis="y")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_ylim(0.6, 0.95)

    mono = pd.read_csv(ABL / "summary" / "ablation_monotonicity.csv")
    for proto, style in (("block", "o-"), ("year", "s--")):
        sub = mono[mono.protocol == proto].set_index("variant")
        for (c, label), color in zip(CHANNELS, (ps.PINAM, ps.OKABE_ITO["green"],
                                                ps.OKABE_ITO["purple"], ps.OKABE_ITO["vermillion"])):
            bx.plot(x, [100 * sub.loc[v, c] for v, _ in STEPS], style, color=color, ms=4,
                    mfc="white" if proto == "year" else color,
                    label=f"{label}, {'blocked CV' if proto == 'block' else 'year-ahead'}")
    bx.set_xticks(x, labels, rotation=20, ha="right")
    bx.set_ylabel("Violating days (%)")
    bx.set_title("(b) Monotonicity violations (+1 SD raises flight probability)")
    bx.grid(axis="y")
    bx.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout(h_pad=1.0)
    ps.save(fig, out_dir, "fig_ablation_steps")


if __name__ == "__main__":
    make(ROOT / "outputs/rev1")
