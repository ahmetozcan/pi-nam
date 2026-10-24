"""Physics documentation and audit for the rev1 response (R2-7, R2-12, R3-4, R3-9).

1. Definition table of every derived factor (formula, inputs and units, constants,
   output unit, meaning, reference)          -> physics_definitions.{md,tex}
2. Redundancy: max |Pearson r| of each derived factor with the 16 raw channels
   on the model frame                          -> physics_redundancy.csv
3. Monotonicity audit of the submitted PI-NAM V1 ensemble (saved weights):
   share of test days with a positive gradient / positive finite-difference
   response for the five constrained channels  -> physics_monotonicity.csv

Run from the repository root:  python3 scripts/rev1/physics_audit.py
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.pinam_search import (MONO_DEC, PINAMX, SYNOPTIC_NAMES, _n_factors,  # noqa: E402
                              synoptic_physics, temporal_block)
from src.pinn import IDX, RAW_CHANNELS, physics_features  # noqa: E402
from src.revision.info_sets import load_inputs  # noqa: E402

logger = logging.getLogger("physics_audit")
OUT = ROOT / "outputs/rev1"
BUNDLES = ROOT / "reproducibility/folds/pinamx_v1"
FD_THRESHOLD = 0.01        # a probability increase above 1 point counts as a material violation

# name, formula, inputs (units), constants, output unit, meaning, reference
INSTANT = [
    ("air_density", r"$\rho = 100P/(R_d T_v)$, $T_v = T_K/[1-(e/P)(1-\varepsilon)]$",
     "T (°C), P_sl (hPa), RH (%)", r"$R_d=287.05$ J kg$^{-1}$K$^{-1}$, $\varepsilon=0.622$",
     "kg m$^{-3}$", "moist-air density", "Wallace & Hobbs (2006)"),
    ("density_altitude", r"$h_\rho = 44330\,[1-(\rho/\rho_0)^{1/4.256}]$",
     r"$\rho$", r"$\rho_0=1.225$ kg m$^{-3}$ (ISA)", "m",
     "ISA altitude with the same density (relative index)", "U.S. Standard Atmosphere (1976)"),
    ("lift_capacity", r"$L = \rho\,[1 - T_K/(T_K+\Delta T)]$", r"$\rho$, T (°C)",
     r"$\Delta T = 100$ K (assumed envelope excess)", "kg m$^{-3}$",
     "gross lift per m$^3$ of envelope", "FAA (2024) Balloon Flying Handbook"),
    ("dewpoint_depression", r"$T - T_d \approx (100-RH)/5$", "RH (%)", "—", "K",
     "proximity to saturation", "Lawrence (2005)"),
    ("gust_factor", r"$G = v_{gust}/(v + 10^{-3})$", "windgust, windspeed (km/h)", "—", "—",
     "gustiness relative to sustained wind", "Harper et al. (2010, WMO)"),
    ("specific_humidity", r"$q = \varepsilon e/(P - 0.378e)$, $e = (RH/100)\,e_s(T)$, "
     r"$e_s = 6.112\exp[17.67T/(T+243.5)]$",
     "T (°C), P_sl (hPa), RH (%)", r"$\varepsilon=0.622$; $e$, $e_s$ in hPa", "kg kg$^{-1}$",
     "water-vapour content", "Bolton (1980)"),
    ("wind_dynamic_pressure", r"$q_w = \frac{1}{2}\rho v^2$", r"$\rho$, windspeed (km/h)", "—",
     r"kg m$^{-3}$ (km/h)$^2$ = Pa/12.96", "wind loading on the envelope (index)", "Wallace & Hobbs (2006)"),
    ("clear_sky_fraction", r"$c = (100 - \mathrm{cloud})/100$", "cloudcover (%)", "—", "—",
     "fraction of sky free of cloud", "—"),
]
SYNOPTIC = [  # computed on the last s = 5 days (syn_scale of PI-NAM V1)
    ("pressure_tendency", r"$P_t - P_{t-s+1}$", "P_sl (hPa)", "s = 5", "hPa", "barometric tendency over 4 days", "Wallace & Hobbs (2006)"),
    ("pressure_std", r"$\mathrm{sd}(P_{t-s+1..t})$", "P_sl (hPa)", "s = 5", "hPa", "synoptic variability", "—"),
    ("frontal_passage", r"$\mathrm{ReLU}(-\Delta P)\,\mathrm{ReLU}(\Delta\mathrm{cloud})$", "P_sl, cloudcover", "s = 5", "hPa·%", "falling pressure, rising cloud (heuristic)", "—"),
    ("clearness_trend", r"$c_t - \overline{c}_{t-s+1..t}$", "cloudcover (%)", "s = 5", "—", "clearing or clouding up", "—"),
    ("dewdepr_trend", r"$(T-T_d)_t - (T-T_d)_{t-s+1}$", "RH (%)", "s = 5", "K", "drying or moistening", "Lawrence (2005)"),
    ("winddir_steadiness", r"$\|\overline{(\cos\theta,\sin\theta)}\|$", "winddir (°)", "s = 5", "0–1", "mean resultant length", "Mardia & Jupp (1999)"),
    ("recent_storminess", r"$\max v_{gust,t-s+1..t}$", "windgust (km/h)", "s = 5", "km/h", "peak recent gust", "—"),
    ("calm_clear", r"$c\,/[(T-T_d+1)(v+1)]$", "cloudcover, RH, windspeed", "—", "index", "calm, clear, near-saturated air", "Gultepe et al. (2007)"),
    ("diurnal_range", r"$T_{max}-T_{min}$", "tempmax, tempmin (°C)", "—", "K", "stability and clear-sky proxy", "Stull (1988)"),
    ("thermal_turbulence", r"$T_{max}\,c$", "tempmax (°C), cloudcover", "—", "°C", "daytime convection proxy (heuristic)", "—"),
]


READABLE = {
    "air_density": "Air density $\\rho$", "density_altitude": "Density altitude $h_\\rho$",
    "lift_capacity": "Lift capacity $L$", "dewpoint_depression": "Dew-point depression",
    "gust_factor": "Gust factor $G$", "specific_humidity": "Specific humidity $q$",
    "wind_dynamic_pressure": "Wind dynamic pressure $q_w$", "clear_sky_fraction": "Clear-sky fraction $c$",
    "pressure_tendency": "Pressure tendency $\\Delta P$", "pressure_std": "Pressure variability",
    "frontal_passage": "Frontal passage", "clearness_trend": "Clearness trend",
    "dewdepr_trend": "Dew-point depression trend", "winddir_steadiness": "Wind-direction steadiness",
    "recent_storminess": "Recent storminess", "calm_clear": "Calm--clear index",
    "diurnal_range": "Diurnal range", "thermal_turbulence": "Thermal turbulence",
}
SHORT_UNIT = {"wind_dynamic_pressure": "index ($\\propto$ Pa)"}


def _tex_text(s: str) -> str:
    """Escape the plain-text parts of a cell (math in $...$ never contains these symbols)."""
    return (s.replace("%", "\\%").replace("P_sl", "$P_{sl}$").replace("·", "$\\cdot$")
             .replace("°C", "$^\\circ$C").replace("°", "$^\\circ$").replace("–", "--"))


def write_definitions() -> None:
    cols = ["factor", "formula", "inputs (units)", "constants", "unit", "meaning", "reference"]
    rows = [("instantaneous",) + r for r in INSTANT] + [("synoptic (s=5)",) + r for r in SYNOPTIC]
    df = pd.DataFrame(rows, columns=["group"] + cols)
    md = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    (OUT / "physics_definitions.md").write_text("\n".join(md) + "\n")
    # Manuscript table (R2-11, R2-12, R3-4): five columns at 16 cm text width; symbols, input
    # units and references are given in the text, the caption stays short.
    tex = ["\\begin{table}[!ht]", "\\centering\\footnotesize\\linespread{1}\\selectfont",
           "\\setlength{\\tabcolsep}{3pt}",
           "\\caption{Derived factors of PI-NAM. Inputs in the units of Table~\\ref{tab:features}; "
           "$\\Delta$cloud is defined like $\\Delta P$.}",
           "\\label{tab:physics}",
           "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{2.7cm}>{\\raggedright\\arraybackslash}p{5.0cm}"
           ">{\\raggedright\\arraybackslash}p{2.9cm}>{\\raggedright\\arraybackslash}p{1.3cm}"
           ">{\\raggedright\\arraybackslash}p{2.9cm}}", "\\toprule",
           "Factor & Definition & Constants & Unit & Meaning \\\\", "\\midrule"]
    for g, label in (("instantaneous", "Instantaneous"), ("synoptic (s=5)", "Synoptic ($s=5$)")):
        tex.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{label}}}}} \\\\")
        for r in df[df.group == g].itertuples(index=False):
            const = "" if r[4] == "s = 5" else _tex_text(r[4])
            unit = SHORT_UNIT.get(r[1], _tex_text(r[5]))
            cells = [READABLE[r[1]], r[2], const, unit, _tex_text(r[6])]
            tex.append(" & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT / "physics_definitions.tex").write_text("\n".join(tex) + "\n")


def redundancy(win: np.ndarray) -> pd.DataFrame:
    """Max |r| of each derived factor of PI-NAM V1 with the 16 raw channels (S0 frame)."""
    w = torch.tensor(win)
    cur = w[:, -1, :]
    tb, tnames = temporal_block(w, ["min"], [3, 7])
    blocks = [(physics_features(cur), [r[0] for r in INSTANT]),
              (synoptic_physics(w, 5), SYNOPTIC_NAMES), (tb, tnames)]
    raw = cur.numpy()
    rows = []
    for F, names in blocks:
        for j, n in enumerate(names):
            f = F[:, j].numpy()
            r = [abs(np.corrcoef(f, raw[:, k])[0, 1]) if raw[:, k].std() > 0 else 0.0
                 for k in range(raw.shape[1])]
            k = int(np.nanargmax(r))
            rows.append({"factor": n, "closest_raw": RAW_CHANNELS[k], "abs_r": round(float(r[k]), 3)})
    return pd.DataFrame(rows).sort_values("abs_r", ascending=False)


def _ensemble(bundle) -> list:
    nf = _n_factors(bundle["spec"])
    models = []
    for st in bundle["seed_states"]:
        m = PINAMX(torch.zeros(nf), torch.ones(nf), nf, bundle["spec"], bundle["hidden"],
                   bundle["dropout"], raw_mean=torch.zeros(len(RAW_CHANNELS)),
                   raw_std=torch.ones(len(RAW_CHANNELS)), depth=bundle["depth"])
        m.load_state_dict(st)
        models.append(m.eval())
    return models


def monotonicity(win: np.ndarray) -> pd.DataFrame:
    """Gradient and +1 SD finite-difference checks on the out-of-fold V1 ensembles."""
    sd = win[:, -1, :].std(0)
    grad_pos = {c: [] for c in MONO_DEC}
    fd = {c: [] for c in MONO_DEC}
    for path in sorted(BUNDLES.glob("fold_*.pkl")):
        b = pickle.load(open(path, "rb"))
        te = np.asarray(b["test_idx"])
        models = _ensemble(b)
        x = torch.tensor(win[te]).requires_grad_(True)
        p = torch.stack([m(x) for m in models]).mean(0)
        g = torch.autograd.grad(p.sum(), x)[0][:, -1, :].numpy()
        with torch.no_grad():
            base = p.detach().numpy()
            for c in MONO_DEC:
                grad_pos[c].append(g[:, IDX[c]] > 0)
                xp = torch.tensor(win[te]).clone()
                xp[:, -1, IDX[c]] += float(sd[IDX[c]])
                fd[c].append(torch.stack([m(xp) for m in models]).mean(0).numpy() - base)
    rows = []
    for c in MONO_DEC:
        gp, d = np.concatenate(grad_pos[c]), np.concatenate(fd[c])
        rows.append({"channel": c, "share_grad_positive": round(float(gp.mean()), 4),
                     "share_fd_increase": round(float((d > 0).mean()), 4),
                     f"share_fd_increase_gt_{FD_THRESHOLD}": round(float((d > FD_THRESHOLD).mean()), 4),
                     "max_fd_increase": round(float(d.max()), 4),
                     "median_fd_change": round(float(np.median(d)), 4)})
    return pd.DataFrame(rows)


def main() -> None:
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    write_definitions()
    win = load_inputs("S0", ROOT / "config.yaml").win
    red = redundancy(win)
    red.to_csv(OUT / "physics_redundancy.csv", index=False)
    logger.info("redundancy (|r| >= 0.9):\n%s", red[red.abs_r >= 0.9].to_string(index=False))
    mono = monotonicity(win)
    mono.to_csv(OUT / "physics_monotonicity.csv", index=False)
    logger.info("monotonicity of submitted V1 ensemble:\n%s", mono.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("src.feature_engineering", "src.revision.info_sets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    main()
