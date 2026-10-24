"""Manuscript tables for Section 5 (final PI-NAM: lambda = 10, inner-CV threshold).

  tab_main_results     ROC-AUC, balanced accuracy and MCC of every main-table model in the six
                       settings (S0/S1 x random CV, blocked CV, year-ahead 2023 -> 2024), with
                       significance against PI-NAM
  tests_main.csv       the tests behind the marks: one Holm family per setting and metric
                       containing every comparison of PI-NAM in the table
  tab_monotonicity     share of held-out days with a monotonicity violation (S0)

Tests: random/blocked CV -- two-sided Wilcoxon signed-rank on the ten fold values, Holm;
year-ahead -- ROC-AUC with DeLong's test (Holm), balanced accuracy and MCC with 95% percentile
intervals of 2,000 paired bootstrap resamples of ISO weeks (significant if the interval excludes
zero). Under S0 random/blocked CV PI-NAM-fixed is the same model as PI-NAM and is not tested.

Outputs: outputs/rev1/tables/. Run from the repository root:  python3 scripts/rev1/make_tables.py
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ablation import _folds, _load  # noqa: E402
from src.revision.info_sets import load_inputs  # noqa: E402
from src.revision.metrics import compute_metrics  # noqa: E402
from src.revision.stats import delong, holm, paired_bootstrap_diff, wilcoxon_paired  # noqa: E402

logger = logging.getLogger("make_tables")
OUT = ROOT / "outputs/rev1"
TAB = OUT / "tables"
ALPHA = 0.05

MODELS = ["PI-NAM", "PI-NAM-fixed", "EBM", "XGBoost", "LightGBM", "CatBoost", "Random Forest",
          "SVM", "TabPFN"]
LABEL = {"TabPFN": "TabPFN~v2"}
SLUG = {"PI-NAM": "grid/{i}_{p}_pinam_thrcv_lam10", "PI-NAM-fixed": "grid/{i}_{p}_pinamfixed_thrcv_lam10",
        "EBM": "extra/{i}_{p}_ebm", "XGBoost": "grid/{i}_{p}_xgboost", "LightGBM": "grid/{i}_{p}_lightgbm",
        "CatBoost": "grid/{i}_{p}_catboost", "Random Forest": "grid/{i}_{p}_randomforest",
        "SVM": "grid/{i}_{p}_svm", "TabPFN": "grid/{i}_{p}_tabpfn"}
INFOS = [("S0", "Same-day information (S0)"), ("S1", "Previous-day information (S1)")]
PROTOCOLS = [("rand", "Random CV"), ("block", "Blocked CV"), ("year", "Year-ahead")]
METRICS = ["roc_auc", "balanced_acc", "mcc"]
HEAD = {"roc_auc": "AUC", "balanced_acc": "BAcc", "mcc": "MCC"}
_ROW_FN = {"balanced_acc": lambda y, p, t: balanced_accuracy_score(y, (p >= t).astype(int)),
           "mcc": lambda y, p, t: matthews_corrcoef(y, (p >= t).astype(int))}


def _path(model: str, info: str, proto: str) -> Path:
    return OUT / (SLUG[model].format(i=info, p=proto) + ".npz")


def _weeks(info: str) -> np.ndarray:
    iso = load_inputs(info, ROOT / "config.yaml").dates.reset_index(drop=True).dt.isocalendar()
    return (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()


def _same_as_pinam(model: str, info: str, proto: str) -> bool:
    """Under S0 random/blocked CV, PI-NAM uses the reference configuration (Table pinam_configs)."""
    return model == "PI-NAM-fixed" and info == "S0" and proto != "year"


def _setting(info: str, proto: str, weeks: np.ndarray):
    """Metric values and tests of every available model in one setting."""
    data = {m: _load(_path(m, info, proto)) for m in MODELS
            if not _same_as_pinam(m, info, proto) and _path(m, info, proto).exists()}
    others = [m for m in data if m != "PI-NAM"]
    ref = data["PI-NAM"]
    values: Dict[str, Dict[str, float]] = {}
    tests: List[Dict] = []
    if proto == "year":
        for m, d in data.items():
            values[m] = {k: compute_metrics(d["y"], d["p"], float(d["thr"][0]))[k] for k in METRICS}
        g = weeks[ref["test"]]
        auc = []
        for m in others:
            d = data[m]
            assert np.array_equal(ref["test"], d["test"])
            dl = delong(ref["y"], ref["p"], d["p"])
            auc.append({"info_set": info, "protocol": proto, "model": m, "metric": "roc_auc",
                        "diff": dl["diff"], "p": dl["p"]})
            for k, fn in _ROW_FN.items():
                bs = paired_bootstrap_diff(ref["y"], ref["p"], ref["thr"], d["p"], d["thr"], fn, groups=g)
                tests.append({"info_set": info, "protocol": proto, "model": m, "metric": k,
                              "diff": bs["diff"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                              "significant": bool(bs["ci_lo"] > 0 or bs["ci_hi"] < 0)})
        for r, ph in zip(auc, holm([r["p"] for r in auc])):
            r |= {"p_holm": ph, "significant": bool(ph < ALPHA)}
        tests += auc
    else:
        folds = {m: _folds(d) for m, d in data.items()}
        for m in data:
            values[m] = {k: float(folds[m][k].mean()) for k in METRICS}
        for k in METRICS:
            block = [{"info_set": info, "protocol": proto, "model": m, "metric": k,
                      **wilcoxon_paired(folds["PI-NAM"][k], folds[m][k])} for m in others]
            for r, ph in zip(block, holm([r["p"] for r in block])):
                r |= {"diff": r["mean_diff"], "p_holm": ph, "significant": bool(ph < ALPHA)}
            tests += block
    return values, tests


def _cell(v: float, sig: bool, better: bool) -> str:
    s = f"{v:.3f}"
    if sig:
        s += r"$^{+}$" if better else r"$^{-}$"
    return s


def main_results() -> pd.DataFrame:
    rows, all_tests = [], []
    body = []
    for info, info_label in INFOS:
        weeks = _weeks(info)
        res = {p: _setting(info, p, weeks) for p, _ in PROTOCOLS}
        for p, (vals, tests) in res.items():
            all_tests += tests
            rows += [{"info_set": info, "protocol": p, "model": m, **v} for m, v in vals.items()]
        body.append(r"\midrule" if body else "")
        body.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{info_label}}}}} \\")
        for m in MODELS:
            cells = []
            for p, _ in PROTOCOLS:
                vals, tests = res[p]
                if _same_as_pinam(m, info, p):
                    cells.append(r"\multicolumn{3}{c}{as PI-NAM}")
                    continue
                if m not in vals:
                    cells.append(r"\multicolumn{3}{c}{--}")
                    continue
                for k in METRICS:
                    t = next((t for t in tests if t["model"] == m and t["metric"] == k), None)
                    sig = bool(t and t["significant"])
                    cells.append(_cell(vals[m][k], sig, vals[m][k] > vals["PI-NAM"][k]))
            body.append(f"{LABEL.get(m, m)} & " + " & ".join(cells) + r" \\")
    head = [r"\begin{tabular}{l" + "ccc" * 3 + "}", r"\toprule",
            " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{t}}}" for _, t in PROTOCOLS) + r" \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
            "Model & " + " & ".join(HEAD[k] for _ in PROTOCOLS for k in METRICS) + r" \\", r"\midrule"]
    tex = "\n".join(head + [b for b in body if b] + [r"\bottomrule", r"\end{tabular}"]) + "\n"
    (TAB / "tab_main_results.tex").write_text(tex)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "main_results.csv", index=False, float_format="%.4f")
    pd.DataFrame(all_tests).to_csv(TAB / "tests_main.csv", index=False, float_format="%.4f")
    logger.info("wrote tab_main_results.tex, main_results.csv, tests_main.csv")
    return df


MONO = [("PI-NAM", "PI-NAM"), ("PI-NAM-fixed", "PI-NAM-fixed"), ("ebm", "EBM"),
        ("xgboostmono", "XGBoost-mono"), ("lightgbmmono", "LightGBM-mono"),
        ("xgboost_fd", "XGBoost"), ("lightgbm_fd", "LightGBM"), ("catboost_fd", "CatBoost"),
        ("randomforest_fd", "Random Forest"), ("svm_fd", "SVM"), ("tabpfn_fd", "TabPFN~v2")]
CHANNELS = ["windgust", "windspeed", "cloudcover", "precip", "snow"]


def monotonicity() -> pd.DataFrame:
    """Largest violation share over the five channels, and the channel it occurs in (S0)."""
    e = pd.read_csv(OUT / "extra/summary/extra_monotonicity.csv")
    e = e[e.info_set == "S0"]
    short = {"windgust": "gust", "windspeed": "wind", "cloudcover": "cloud", "precip": "precip",
             "snow": "snow"}
    lines, rows = [], []
    for key, label in MONO:
        cells = []
        for p, _ in PROTOCOLS:
            r = e[(e.model == key) & (e.protocol == p)]
            if r.empty:
                cells.append(r"\multicolumn{1}{c}{as PI-NAM}" if key == "PI-NAM-fixed" else "--")
                continue
            vals = r[CHANNELS].iloc[0]
            mx = float(vals.max())
            rows.append({"model": label, "protocol": p, "max_share": mx, "channel": vals.idxmax(),
                         **vals.to_dict()})
            cells.append("0.0" if mx == 0 else f"{100 * mx:.1f} ({short[vals.idxmax()]})")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    tex = "\n".join([r"\begin{tabular}{lccc}", r"\toprule",
                     "Model & " + " & ".join(t for _, t in PROTOCOLS) + r" \\", r"\midrule",
                     *lines, r"\bottomrule", r"\end{tabular}"]) + "\n"
    (TAB / "tab_monotonicity.tex").write_text(tex)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "monotonicity_main.csv", index=False, float_format="%.4f")
    logger.info("wrote tab_monotonicity.tex")
    return df


ABL_GROUPS = [
    ("Cumulative construction ($\\lambda = 0$ until the last step)",
     [("C1_raw", "Raw variables only"), ("C2_phys", "+ physics factors"),
      ("C3_temp", "+ minima over 3 and 7 days"), ("C4_syn", "+ synoptic factors"),
      ("C5_inter", "+ interactions"), ("S_lam10", "+ monotonicity penalty, $\\lambda = 10$ (reference)")]),
    ("Reference without one group",
     [("L_phys_l10", "without physics factors"), ("L_temp_l10", "without minima"),
      ("L_syn_l10", "without synoptic factors"), ("L_inter_l10", "without interactions")]),
    ("Synoptic window ($s = 5$ in the reference)",
     [("S_w3_l10", "$s = 3$"), ("S_w7_l10", "$s = 7$")]),
    ("Monotonicity weight ($\\lambda = 10$ in the reference)",
     [("S_lam0.03", "$\\lambda = 0.03$"), ("V1", "$\\lambda = 0.081$ (submitted model)"),
      ("S_lam0.2", "$\\lambda = 0.2$"), ("S_lam1", "$\\lambda = 1$")]),
    ("Single network instead of the 15-member ensemble",
     [(f"E_seed{s}_l10", f"seed {s}") for s in range(42, 47)]),
]


def _abl_mark(t: pd.DataFrame, proto: str, v: str, metric: str) -> bool:
    """Significance of variant v against the reference S_lam10 (analyze_ablation.py tests)."""
    t = t[(t.protocol == proto) & (t.reference == "S_lam10") & (t.variant == v)]
    if t.empty:
        return False
    if proto == "year":
        if metric == "roc_auc":
            return bool(t.p_delong_holm.iloc[0] < ALPHA)
        if metric == "balanced_acc":
            return bool(t.bacc_ci_lo.iloc[0] > 0 or t.bacc_ci_hi.iloc[0] < 0)
        return False
    r = t[t.metric == metric]
    return bool(not r.empty and r.p_holm.iloc[0] < ALPHA)


def ablation() -> None:
    m = pd.read_csv(OUT / "ablation/summary/ablation_metrics.csv").set_index(["protocol", "variant"])
    t = pd.read_csv(OUT / "ablation/summary/ablation_tests.csv")
    lines = []
    for title, rows in ABL_GROUPS:
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{title}}}}} \\")
        for v, label in rows:
            cells = []
            for p in ("block", "year"):
                ref = m.loc[(p, "S_lam10")]
                for k in METRICS:
                    val = m.loc[(p, v), k]
                    cells.append(_cell(val, v != "S_lam10" and _abl_mark(t, p, v, k), val > ref[k]))
            lines.append(rf"\quad {label} & " + " & ".join(cells) + r" \\")
    tex = "\n".join([r"\begin{tabular}{lcccccc}", r"\toprule",
                     r" & \multicolumn{3}{c}{Blocked CV} & \multicolumn{3}{c}{Year-ahead} \\",
                     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
                     "Variant & " + " & ".join(HEAD[k] for _ in range(2) for k in METRICS) + r" \\",
                     r"\midrule", *lines, r"\bottomrule", r"\end{tabular}"]) + "\n"
    (TAB / "tab_s_ablation.tex").write_text(tex)
    logger.info("wrote tab_s_ablation.tex")


EXTRA = [("PI-NAM", "PI-NAM"), ("xgboostmono", "XGBoost-mono"), ("lightgbmmono", "LightGBM-mono"),
         ("xgboostderived", "XGBoost-derived"), ("lightgbmderived", "LightGBM-derived"),
         ("mlpphys", "MLP-phys ($\\lambda = 0$)"), ("mlpphysmono", "MLP-phys-mono ($\\lambda = 10$)")]


def extra_baselines() -> None:
    """Architecture or features? Extra models of stage 6 against PI-NAM (S0)."""
    m = pd.read_csv(OUT / "extra/summary/extra_metrics.csv")
    m = m[m.info_set == "S0"].set_index(["protocol", "model"])
    t = pd.read_csv(OUT / "extra/summary/extra_tests.csv")
    t = t[(t.info_set == "S0") & (t.pinam == "PI-NAM")]
    year_vals = {key: _year_metrics(key) for key, _ in EXTRA}
    lines = []
    for key, label in EXTRA:
        cells = []
        for p, _ in PROTOCOLS:
            for k in METRICS:
                if p == "year":
                    val, ref = year_vals[key][k], year_vals["PI-NAM"][k]
                    r = t[(t.protocol == "year") & (t.other == key)]
                    sig = bool(not r.empty and (
                        (k == "roc_auc" and r.p_delong_holm.iloc[0] < ALPHA)
                        or (k == "balanced_acc" and (r.bacc_ci_lo.iloc[0] > 0 or r.bacc_ci_hi.iloc[0] < 0))))
                else:
                    val, ref = m.loc[(p, key), k], m.loc[(p, "PI-NAM"), k]
                    r = t[(t.protocol == p) & (t.other == key) & (t.metric == k)]
                    sig = bool(not r.empty and r.p_holm.iloc[0] < ALPHA)
                cells.append(_cell(val, key != "PI-NAM" and sig, val > ref))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    tex = "\n".join([r"\begin{tabular}{l" + "ccc" * 3 + "}", r"\toprule",
                     " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{t_}}}" for _, t_ in PROTOCOLS) + r" \\",
                     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
                     "Model & " + " & ".join(HEAD[k] for _ in PROTOCOLS for k in METRICS) + r" \\",
                     r"\midrule", *lines, r"\bottomrule", r"\end{tabular}"]) + "\n"
    (TAB / "tab_s_extra.tex").write_text(tex)
    logger.info("wrote tab_s_extra.tex")


def _year_metrics(key: str) -> Dict[str, float]:
    """Pooled year-ahead metrics (the year protocol has one test set, not ten folds)."""
    path = _path("PI-NAM", "S0", "year") if key == "PI-NAM" else OUT / f"extra/S0_year_{key}.npz"
    d = _load(path)
    return {k: compute_metrics(d["y"], d["p"], float(d["thr"][0]))[k] for k in METRICS}


def calibration_safety() -> None:
    """Brier score and ECE (pooled predictions, as analyze_grid.calibration_and_shift), and the
    safety-oriented operating point (threshold capping the training FAR at 10%), S0."""
    cal = pd.read_csv(OUT / "analysis_thrcv_lam10/calibration.csv")
    cal = cal[cal.info_set == "S0"].set_index(["protocol", "model"])
    saf = pd.read_csv(OUT / "analysis_thrcv_lam10/safety_operating.csv")
    saf = saf[(saf.info_set == "S0") & (saf.far_target == 0.10)].set_index(["protocol", "model"])
    lines = []
    for m in MODELS:
        cells = []
        for p, _ in PROTOCOLS:
            if _same_as_pinam(m, "S0", p):
                cells.append(r"\multicolumn{4}{c}{as PI-NAM}")
                continue
            if (p, m) in cal.index:
                brier, ece = cal.loc[(p, m), "brier"], cal.loc[(p, m), "ece"]
            else:
                d = _load(_path(m, "S0", p))
                cm = compute_metrics(d["y"], d["p"], 0.5)
                brier, ece = cm["brier"], cm["ece"]
            far, rec = ((saf.loc[(p, m), "far_test"], saf.loc[(p, m), "recall_fly"])
                        if (p, m) in saf.index else _safety_point(m, p))
            cells += [f"{brier:.3f}", f"{ece:.3f}",
                      "--" if far is None else f"{far:.3f}", "--" if rec is None else f"{rec:.3f}"]
        lines.append(f"{LABEL.get(m, m)} & " + " & ".join(cells) + r" \\")
    sub = ["Brier", "ECE", "FAR", "Recall"]
    tex = "\n".join([r"\begin{tabular}{l" + "cccc" * 3 + "}", r"\toprule",
                     " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{t}}}" for _, t in PROTOCOLS) + r" \\",
                     r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
                     "Model & " + " & ".join(s for _ in PROTOCOLS for s in sub) + r" \\",
                     r"\midrule", *lines, r"\bottomrule", r"\end{tabular}"]) + "\n"
    (TAB / "tab_s_calibration.tex").write_text(tex)
    logger.info("wrote tab_s_calibration.tex")


SEASONS = [("DJF", "Winter"), ("MAM", "Spring"), ("JJA", "Summer"), ("SON", "Autumn")]


def seasonal() -> None:
    """Balanced accuracy and FAR by meteorological season (pooled held-out days, S0)."""
    s = pd.read_csv(OUT / "analysis_thrcv_lam10/seasonal.csv")
    s = s[s.info_set == "S0"]
    lines = []
    for p, title in PROTOCOLS[1:]:
        ps_ = s[s.protocol == p]
        info = "days and share authorized: " + ", ".join(
            f"{int(ps_[ps_.season == k].n.iloc[0])}/{100 * ps_[ps_.season == k].prevalence.iloc[0]:.0f}\\%"
            for k, _ in SEASONS)
        lines.append(rf"\multicolumn{{9}}{{l}}{{\textit{{{title}}} ({info})}} \\")
        for m in MODELS:
            r = ps_[ps_.model == m].set_index("season")
            if r.empty and _path(m, "S0", p).exists():
                r = _seasonal_rows(m, p)
            if r.empty:
                continue
            cells = [f"{r.loc[k, c]:.2f}" for c in ("balanced_acc", "false_authorization_rate")
                     for k, _ in SEASONS]
            lines.append(f"{LABEL.get(m, m)} & " + " & ".join(cells) + r" \\")
    head = " & ".join(n for _, n in SEASONS)
    tex = "\n".join([r"\begin{tabular}{lcccccccc}", r"\toprule",
                     r" & \multicolumn{4}{c}{Balanced accuracy} & \multicolumn{4}{c}{FAR} \\",
                     r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
                     f"Model & {head} & {head}" + r" \\", r"\midrule", *lines,
                     r"\bottomrule", r"\end{tabular}"]) + "\n"
    (TAB / "tab_s_seasonal.tex").write_text(tex)
    logger.info("wrote tab_s_seasonal.tex")


def _seasonal_rows(model: str, proto: str) -> pd.DataFrame:
    """Per-season metrics from a prediction file outside the grid (used for EBM), as analyze_grid.seasonal."""
    from analyze_grid import SEASONS as MONTH_SEASON, row_metrics
    d = _load(_path(model, "S0", proto))
    dates = load_inputs("S0", ROOT / "config.yaml").dates.reset_index(drop=True)
    season = dates.dt.month.map(MONTH_SEASON).to_numpy()[d["test"]]
    return pd.DataFrame({se: row_metrics(d["y"][season == se], d["p"][season == se], d["thr"][season == se])
                         for se, _ in SEASONS}).T


def _safety_point(model: str, proto: str, target: float = 0.10) -> tuple:
    """FAR cap on the training predictions, as analyze_grid.safety_operating (used for EBM)."""
    from analyze_grid import _far_threshold, row_metrics
    path = _path(model, "S0", proto)
    d = _load(path)
    a = np.load(path.parent / "aux" / path.name)
    thr = np.empty(len(d["y"]))
    for k in np.unique(d["split"]):
        src = (a["thr_split"] == k) & (a["thr_y"] == 0)
        thr[d["split"] == k] = _far_threshold(a["thr_p"][src], target)
    r = row_metrics(d["y"], d["p"], thr)
    return r["false_authorization_rate"], r["recall_fly"]


def class_metrics() -> None:
    """Table S2: tabular of class_metrics/tab_class_metrics.tex, without its table wrapper."""
    src = (OUT / "class_metrics/tab_class_metrics.tex").read_text().splitlines()
    i = next(k for k, s in enumerate(src) if s.startswith("\\begin{tabular}"))
    j = next(k for k, s in enumerate(src) if s.startswith("\\end{tabular}"))
    body = [s.replace("TabPFN &", f"{LABEL['TabPFN']} &") for s in src[i:j + 1]]
    (TAB / "tab_s_class_metrics.tex").write_text("\n".join(body) + "\n")


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    class_metrics()
    main_results()
    monotonicity()
    ablation()
    extra_baselines()
    calibration_safety()
    seasonal()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("src.feature_engineering", "src.revision.info_sets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    main()
