"""Stage 1 data audit for the rev1 revision (R2-8, R3-10, Editor E1).

Produces, under outputs/rev1/:
  * data_audit.json             -- record counts, dropped rows, value ranges
  * tab_descriptive.tex         -- descriptive statistics per class (16 variables)
  * fig_variable_distributions.pdf/.png -- per-class distributions of all 16 variables

Run from the repository root:  python3 scripts/rev1/data_audit.py
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.feature_engineering import build_features  # noqa: E402
from src.pinn import RAW_CHANNELS  # noqa: E402
from src.revision import plot_style as ps  # noqa: E402
from src.revision.stats import circular_mean_sd  # noqa: E402

logger = logging.getLogger(__name__)
OUT = ROOT / "outputs" / "rev1"

CIRCULAR = {"winddir"}       # directions: circular mean and circular SD, not linear moments

# Units and daily aggregation as defined by Visual Crossing (metric unit group).
VAR_META = {
    "tempmax": ("°C", "Daily maximum temperature"),
    "tempmin": ("°C", "Daily minimum temperature"),
    "temp": ("°C", "Daily mean temperature"),
    "feelslikemax": ("°C", "Daily maximum apparent temperature"),
    "feelslikemin": ("°C", "Daily minimum apparent temperature"),
    "feelslike": ("°C", "Daily mean apparent temperature"),
    "humidity": ("%", "Daily mean relative humidity"),
    "precip": ("mm", "Daily liquid-equivalent precipitation total"),
    "snow": ("cm", "Daily snowfall"),
    "snowdepth": ("cm", "Snow depth on the ground"),
    "windgust": ("km/h", "Daily maximum hourly wind gust"),
    "windspeed": ("km/h", "Daily maximum hourly sustained wind speed"),
    "winddir": ("°", "Daily mean wind direction"),
    "sealevelpressure": ("hPa", "Daily mean sea-level pressure"),
    "cloudcover": ("%", "Daily mean cloud cover"),
    "visibility": ("km", "Daily mean horizontal visibility"),
}



def audit_counts(full: pd.DataFrame, model: pd.DataFrame) -> dict:
    """Record counts, class balance and the rows dropped during warm-up."""
    dropped = full.loc[~full["datetime"].isin(model["datetime"]), ["datetime", "flight"]]
    meteo_cols = [c for c in pd.read_excel(ROOT / "data/datasetCappadociaBalloonFlight.xlsx",
                                           nrows=1).columns if c not in ("datetime", "flight")]
    x = model[RAW_CHANNELS]
    return {
        "raw_records": int(len(full)),
        "raw_class_counts": {str(k): int(v) for k, v in full["flight"].value_counts().items()},
        "usable_records": int(len(model)),
        "usable_class_counts": {str(k): int(v) for k, v in model["flight"].value_counts().items()},
        "usable_period": [str(model["datetime"].min().date()), str(model["datetime"].max().date())],
        "dropped_rows": [{"date": str(d.date()), "flight": int(f)}
                         for d, f in zip(dropped["datetime"], dropped["flight"])],
        "n_meteorological_columns_in_xlsx": len(meteo_cols),
        "n_model_input_channels": len(RAW_CHANNELS),
        "missing_cells": int(x.isna().sum().sum()),
        "nonfinite_cells": int((~np.isfinite(x.to_numpy())).sum()),
        "windgust_range": [float(x["windgust"].min()), float(x["windgust"].max())],
        "temp_range": [float(x["temp"].min()), float(x["temp"].max())],
        "temp_mean_std": [float(x["temp"].mean()), float(x["temp"].std())],
        "tempmin_tempmax_range": [float(x["tempmin"].min()), float(x["tempmax"].max())],
        "flight_days_with_gust_gt_50": int(((model["flight"] == 1) & (x["windgust"] > 50)).sum()),
        "flight_days_max_gust": float(x.loc[model["flight"] == 1, "windgust"].max()),
    }


def descriptive_table(model: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SD overall and per class, plus min/max, for each variable."""
    rows = []
    for c in RAW_CHANNELS:
        v, fly, nofly = model[c], model.loc[model.flight == 1, c], model.loc[model.flight == 0, c]
        if c in CIRCULAR:
            (m, sd), (fm, fsd), (nm, nsd) = (circular_mean_sd(x) for x in (v, fly, nofly))
        else:
            m, sd, fm, fsd, nm, nsd = v.mean(), v.std(), fly.mean(), fly.std(), nofly.mean(), nofly.std()
        rows.append({"variable": c, "unit": VAR_META[c][0],
                     "mean": m, "sd": sd, "min": v.min(), "max": v.max(),
                     "fly_mean": fm, "fly_sd": fsd, "nofly_mean": nm, "nofly_sd": nsd})
    return pd.DataFrame(rows)


def write_latex(tab: pd.DataFrame, n_fly: int, n_nofly: int, path: Path) -> None:
    lines = [
        r"\begin{table}[htbp]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Descriptive statistics of the 16 meteorological variables "
        rf"($N = {n_fly + n_nofly}$). Values are mean $\pm$ SD; for wind direction, circular mean "
        r"$\pm$ circular SD.}",
        r"\label{tab:descriptive}",
        r"\begin{tabular}{llcccc}", r"\toprule",
        r"Variable & Unit & All & Authorized & Not authorized & Range \\",
        rf" & & ($N={n_fly + n_nofly}$) & ($n={n_fly}$) & ($n={n_nofly}$) & \\",
        r"\midrule",
    ]
    for r in tab.itertuples():
        unit = r.unit.replace("°", r"$^\circ$").replace("%", r"\%")
        lines.append(
            rf"\texttt{{{r.variable}}} & {unit} & {r.mean:.1f} $\pm$ {r.sd:.1f} & "
            rf"{r.fly_mean:.1f} $\pm$ {r.fly_sd:.1f} & {r.nofly_mean:.1f} $\pm$ {r.nofly_sd:.1f} & "
            rf"[${r.min:.1f}$, ${r.max:.1f}$] \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def plot_distributions(model: pd.DataFrame, out_dir: Path) -> None:
    """4x4 grid of per-class violin plots, drawn at print size (text width)."""
    ps.apply()
    fig, axes = plt.subplots(4, 4, figsize=(ps.TEXT_WIDTH, 7.4))
    for ax, c in zip(axes.ravel(), RAW_CHANNELS):
        data = [model.loc[model.flight == k, c].to_numpy() for k in (0, 1)]
        parts = ax.violinplot(data, positions=[0, 1], showmedians=True, widths=0.8)
        for body, k in zip(parts["bodies"], (0, 1)):
            body.set_facecolor(ps.CLASS_COLORS[k]); body.set_edgecolor("black"); body.set_alpha(0.7)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[key].set_color("black"); parts[key].set_linewidth(0.8)
        ax.set_xticks([])
        ax.set_title(f"{c} ({VAR_META[c][0]})", fontsize=ps.FONT, loc="center")
        ax.grid(axis="y")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=ps.CLASS_COLORS[k], edgecolor="black", alpha=0.7)
               for k in (0, 1)]
    fig.legend(handles, ["Not authorized (red or yellow flag)", "Authorized (green flag)"],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(h_pad=0.6, w_pad=0.4, rect=(0, 0, 1, 0.965))
    ps.save(fig, out_dir, "fig_variable_distributions")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("src.feature_engineering").setLevel(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    full, model = build_features(str(ROOT / "config.yaml"))
    model = model.reset_index(drop=True)

    counts = audit_counts(full, model)
    (OUT / "data_audit.json").write_text(json.dumps(counts, indent=2))
    tab = descriptive_table(model)
    tab.to_csv(OUT / "tab_descriptive.csv", index=False)
    vc = model["flight"].value_counts()
    write_latex(tab, int(vc[1]), int(vc[0]), OUT / "tab_descriptive.tex")
    plot_distributions(model, OUT)
    logger.info(json.dumps(counts, indent=2))
    logger.info(tab.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
