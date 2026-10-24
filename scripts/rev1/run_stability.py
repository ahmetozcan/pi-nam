"""Stage 8: stability of the final PI-NAM's interpretation (R3-9, R2-4) -> outputs/rev1/stability/.

Members: S0 blocked CV, 10 folds x 15 seeds (grid PI-NAM, lambda = 10); the year-ahead
PI-NAM-fixed ensemble (15 seeds, trained on 2023) as an independent check. Every term is
evaluated on all 728 rows (same population for every member) and centred.

  term_importance.csv   importance of each term (fold ensembles): mean, SD, CV, mean rank,
                        seed-level CV, year-ahead value
  group_importance.csv  importance of each source-variable group (sum of its terms)
  rank_stability.csv    Kendall tau of importance rankings: fold vs fold, seed vs seed
                        within a fold, blocked-CV mean vs year-ahead
  concurvity.csv        factor pairs with |r| >= 0.9: stability of each importance vs the
                        importance of their summed contribution
  factor_correlation.csv
  fig_shapes_stability.{pdf,png}  shape functions, mean of 150 members with 95 % band,
                        x axis cut to the 1st-99th percentile of observed values
  fig_factor_correlation.{pdf,png}

Run from the repository root:  python3 scripts/rev1/run_stability.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.revision.extra_models import derived_features  # noqa: E402
from src.revision.info_sets import load_inputs  # noqa: E402
from src.revision.interpret import (build_model, contributions, factor_group,  # noqa: E402
                                    factor_names, group_importance, pairwise_tau, shape_curve,
                                    term_importance)
from src.revision.models import PinamConfig  # noqa: E402
from src.revision import plot_style as ps  # noqa: E402

logger = logging.getLogger("run_stability")
GRID = ROOT / "outputs/rev1/grid"
OUT = ROOT / "outputs/rev1/stability"
SOURCES = {"block": "S0_block_pinam_thrcv_lam10", "year": "S0_year_pinamfixed_thrcv_lam10"}
CONCURVITY_R = 0.9
PANELS = [("Cloud cover", "cloudcover", "%"), ("Wind gust", "windgust", "km/h"),
          ("Wind speed", "windspeed", "km/h"), ("Visibility", "visibility", "km"),
          ("Relative humidity", "humidity", "%"), ("Air density", "air_density", "kg m$^{-3}$"),
          ("Pressure tendency", "pressure_tendency", "hPa"),
          ("Calm–clear index", "calm_clear", "index"), ("Diurnal range", "diurnal_range", "°C")]


def load_members(stem: str):
    cfg = PinamConfig(**json.loads(str(np.load(GRID / f"{stem}.npz")["meta"]))["params"])
    saved = torch.load(GRID / "states" / f"{stem}.pt", weights_only=False)
    members = {k: [build_model(cfg.spec, cfg.hidden, cfg.dropout, cfg.depth, st)
                   for st in v["states"]] for k, v in saved.items()}
    return cfg, members


def importance_tables(names, contrib: Dict[str, List[np.ndarray]], year: List[np.ndarray]):
    folds = list(contrib)
    ens = {k: np.mean(v, axis=0) for k, v in contrib.items()}          # fold ensembles
    imp = np.stack([term_importance(ens[k]) for k in folds])            # (10, n_terms)
    seed_cv = np.mean([(lambda a: a.std(0, ddof=1) / a.mean(0))(
        np.stack([term_importance(c) for c in contrib[k]])) for k in folds], axis=0)
    year_imp = term_importance(np.mean(year, axis=0))
    ranks = (-imp).argsort(1).argsort(1) + 1
    terms = pd.DataFrame({"term": names, "group": [factor_group(n) for n in names],
                          "importance": imp.mean(0), "sd_folds": imp.std(0, ddof=1),
                          "cv_folds": imp.std(0, ddof=1) / imp.mean(0),
                          "rank_mean": ranks.mean(0), "rank_sd": ranks.std(0, ddof=1),
                          "cv_seeds": seed_cv, "importance_year": year_imp}
                         ).sort_values("importance", ascending=False)
    g = pd.DataFrame([group_importance(ens[k], names) for k in folds])
    gy = group_importance(np.mean(year, axis=0), names)
    groups = pd.DataFrame({"group": g.columns, "importance": g.mean().values,
                           "sd_folds": g.std(ddof=1).values,
                           "cv_folds": (g.std(ddof=1) / g.mean()).values,
                           "share": (g.mean() / g.mean().sum()).values,
                           "importance_year": [gy[c] for c in g.columns]}
                          ).sort_values("importance", ascending=False)
    seed_tau = np.concatenate([pairwise_tau(np.stack([term_importance(c) for c in contrib[k]]))
                               for k in folds])
    stab = [{"level": "term", "comparison": "fold vs fold", **_tau(pairwise_tau(imp))},
            {"level": "term", "comparison": "seed vs seed (within fold)", **_tau(seed_tau)},
            {"level": "term", "comparison": "blocked CV vs year-ahead",
             **_tau(pairwise_tau(np.stack([imp.mean(0), year_imp])))},
            {"level": "group", "comparison": "fold vs fold", **_tau(pairwise_tau(g.to_numpy()))},
            {"level": "group", "comparison": "blocked CV vs year-ahead",
             **_tau(pairwise_tau(np.stack([g.mean().to_numpy(), [gy[c] for c in g.columns]])))}]
    return terms, groups, pd.DataFrame(stab), ens


def _tau(t: np.ndarray) -> Dict[str, float]:
    return {"tau_mean": float(np.mean(t)), "tau_min": float(np.min(t)),
            "tau_max": float(np.max(t)), "n_pairs": int(len(t))}


def concurvity(F: pd.DataFrame, names, ens: Dict[str, np.ndarray]) -> pd.DataFrame:
    r = F.corr()
    rows = []
    cols = list(F.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if abs(r.loc[a, b]) < CONCURVITY_R:
                continue
            ia, ib = names.index(a), names.index(b)
            ea = np.array([np.abs(e[:, ia]).mean() for e in ens.values()])
            eb = np.array([np.abs(e[:, ib]).mean() for e in ens.values()])
            es = np.array([np.abs(e[:, ia] + e[:, ib]).mean() for e in ens.values()])
            rows.append({"a": a, "b": b, "r": r.loc[a, b], "cv_a": ea.std(ddof=1) / ea.mean(),
                         "cv_b": eb.std(ddof=1) / eb.mean(), "cv_sum": es.std(ddof=1) / es.mean(),
                         "corr_imp_across_folds": np.corrcoef(ea, eb)[0, 1]})
    return pd.DataFrame(rows).sort_values("r", key=np.abs, ascending=False)


def plot_shapes(F: pd.DataFrame, names, members, year_members, out_dir: Path) -> None:
    ps.apply()
    fig, axes = plt.subplots(3, 3, figsize=(ps.TEXT_WIDTH, 6.6))
    for k, (ax, (label, fname, unit)) in enumerate(zip(axes.ravel(), PANELS)):
        idx, data = names.index(fname), F[fname].to_numpy()
        lo, hi = np.percentile(data, [1, 99])
        x = np.linspace(lo, hi, 120)
        cs = np.stack([shape_curve(m, idx, x, data) for ms in members.values() for m in ms])
        yc = np.stack([shape_curve(m, idx, x, data) for m in year_members]).mean(0)
        ax.fill_between(x, *np.percentile(cs, [2.5, 97.5], axis=0), color=ps.PINAM, alpha=0.2,
                        lw=0, label="95% band (150 members)")
        ax.plot(x, cs.mean(0), color=ps.PINAM, lw=1.8, label="Blocked CV mean")
        ax.plot(x, yc, color=ps.OKABE_ITO["vermillion"], lw=1.3, ls="--",
                label="Year-ahead (2023 fit)")
        inside = data[(data >= lo) & (data <= hi)]
        ax.plot(inside, np.full(len(inside), ax.get_ylim()[0]), "|", color="#555555", alpha=0.25,
                ms=5)
        ax.axhline(0, color="#999999", lw=0.6, ls=":")
        ax.set_title(label, fontsize=ps.FONT, loc="center")
        ax.set_xlabel(unit, fontsize=ps.FONT)
        if k % 3 == 0:
            ax.set_ylabel("log-odds")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.6, w_pad=0.5)
    ps.save(fig, out_dir, "fig_shapes_stability")


def plot_correlation(F: pd.DataFrame, out_dir: Path) -> None:
    """Supplementary: 48 x 48 factor correlations (tick labels are necessarily small)."""
    ps.apply()
    r = F.corr()
    fig, ax = plt.subplots(figsize=(ps.TEXT_WIDTH, 6.0))
    im = ax.imshow(r.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(r)), r.columns, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(r)), r.columns, fontsize=5.5)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    ps.save(fig, out_dir, "fig_factor_correlation")


def _setup():
    inp = load_inputs("S0", ROOT / "config.yaml")
    cfg, members = load_members(SOURCES["block"])
    _, year = load_members(SOURCES["year"])
    names = factor_names(cfg.spec)
    n1d = len(names) - (6 if cfg.spec.get("interactions") else 0)
    F = pd.DataFrame(derived_features(inp.win, {**cfg.spec, "interactions": False})[:, :n1d],
                     columns=names[:n1d])
    return inp, members, year[next(iter(year))], names, F


def make_figures(out_dir: Path) -> None:
    """Figures only (used by make_all_figures.py)."""
    _, members, year_members, names, F = _setup()
    plot_shapes(F, names, members, year_members, out_dir)
    plot_correlation(F, out_dir)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inp, members, year_members, names, F = _setup()
    contrib = {k: [contributions(m, inp.win) for m in ms] for k, ms in members.items()}
    year_c = [contributions(m, inp.win) for m in year_members]
    terms, groups, stab, ens = importance_tables(names, contrib, year_c)
    tables = {"term_importance.csv": terms, "group_importance.csv": groups,
              "rank_stability.csv": stab, "concurvity.csv": concurvity(F, names, ens),
              "factor_correlation.csv": F.corr()}
    for name, df in tables.items():
        df.to_csv(OUT / name, index=name == "factor_correlation.csv", float_format="%.4f")
        logger.info("wrote %s (%d rows)", name, len(df))
    plot_shapes(F, names, members, year_members, OUT)
    plot_correlation(F, OUT)
    logger.info("wrote figures to %s", OUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("src.feature_engineering", "src.revision.info_sets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    main()
