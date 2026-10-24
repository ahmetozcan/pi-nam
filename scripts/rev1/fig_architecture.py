"""PI-NAM (V1, final) architecture diagram, redrawn at print size for the revision (R2-9).

Coordinates are inches on a TEXT_WIDTH-wide canvas, so every font size is the printed size
(>= 8 pt). Changes from the submitted figure: no title (the caption carries it), the
schematic shape directions follow the learned shapes (stage 8: wind gust, wind energy and the
3-day minimum gust decrease, diurnal range and the calm-clear index increase), lambda = 10,
threshold from inner cross-validation. All text is dark on light fills: white text on
filled boxes is flagged as hidden text by similarity checkers (Turnitin).

Called by make_all_figures.py; standalone: python3 scripts/rev1/fig_architecture.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.revision import plot_style as ps  # noqa: E402

C = {"input": "#2b3a4a", "raw": "#6f6f6f", "phys": ps.OKABE_ITO["blue"],
     "syn": ps.OKABE_ITO["green"], "temp": ps.OKABE_ITO["purple"],
     "inter": ps.OKABE_ITO["orange"], "sum": "#4d4d4d", "out": ps.OKABE_ITO["vermillion"],
     "note": "#9aa5ad"}
W, H = ps.TEXT_WIDTH, 4.32
ROW_H, ROW_Y = 0.64, [3.36, 2.64, 1.92, 1.20, 0.48 + 0.0]
IN_X, IN_W = 0.0, 0.74
IBUS_X = 0.84
GX, GW = 0.94, 2.02
SX, SW = 3.12, 1.52
BUS_X = 4.76
SUM_X, SUM_W = 4.88, 0.62
OUT_X, OUT_W = 5.66, 0.64
MID_Y = ROW_Y[2] + ROW_H / 2
TXT, SMALL = 8.5, 8.0

GROUPS = [
    ("raw", "Raw inputs", 16, "temperature, humidity, wind,\ncloud, visibility, pressure, …"),
    ("phys", "Instantaneous physics", 8, "air density, density altitude,\nlift capacity, gustiness, …"),
    ("syn", "Synoptic physics", 10, "5-day: pressure tendency,\nfrontal passage, calm–clear, …"),
    ("temp", "Temporal minima", 14, "3- and 7-day minima of\n7 safety-critical channels"),
    ("inter", "Pairwise interactions", 6, "2-D shapes, e.g. cloud × gust,\ntemperature × humidity"),
]
SHAPES = [("dec", "wind gust ↓"), ("dec", "wind energy ↓"), ("inc", "diurnal range ↑"),
          ("dec", "3-day min gust ↓"), ("heat", "cloud × gust")]


def _box(ax, x, y, w, h, color, fill_alpha=0.10, lw=1.2):
    style = "round,pad=0.0,rounding_size=0.06"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, ec="none", fc=color,
                                alpha=fill_alpha, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, ec=color, fc="none", lw=lw,
                                zorder=3))


def _arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                 lw=lw, color=color, zorder=4, shrinkA=0, shrinkB=0))


def _mini(ax, x0, y0, w, h, kind, color):
    if kind == "heat":
        g = np.linspace(-2, 2, 40)
        xx, yy = np.meshgrid(g, g)
        zz = np.exp(-((xx - 0.5) ** 2 + (yy + 0.4) ** 2) / 1.6) \
            - 0.9 * np.exp(-((xx + 1.0) ** 2 + (yy - 1.0) ** 2) / 1.8)
        ax.imshow(zz, extent=(x0, x0 + w, y0, y0 + h), origin="lower", cmap="RdBu_r",
                  alpha=0.85, zorder=5, aspect="auto")
        return
    t = np.linspace(-2.2, 2.2, 60)
    c = np.tanh(1.6 * t) * (-1 if kind == "dec" else 1)
    ax.plot([x0, x0 + w], [y0 + h / 2] * 2, color="#d0d0d0", lw=0.6, zorder=5)
    ax.plot(x0 + (t + 2.2) / 4.4 * w, y0 + h / 2 + c * (h / 2 - 0.02), color=color, lw=1.5,
            zorder=6)


def draw() -> plt.Figure:
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    for xc, text in ((IN_X + IN_W / 2, "INPUT"), (GX + GW / 2, "FACTORS (48 + 6)"),
                     (SX + SW / 2, "SHAPE FUNCTIONS"), (SUM_X + SUM_W / 2, "SUM"),
                     (OUT_X + OUT_W / 2, "OUTPUT")):
        ax.text(xc, H - 0.12, text, ha="center", va="center", fontsize=SMALL, color="#555555",
                fontweight="bold")

    iy, ih = MID_Y - 0.62, 1.24
    _box(ax, IN_X, iy, IN_W, ih, C["input"], fill_alpha=0.12)
    ax.text(IN_X + IN_W / 2, MID_Y, "16 raw\nweather\nchannels,\n14-day\nwindow",
            ha="center", va="center", fontsize=SMALL, color="#222222", zorder=5)
    ax.plot([IN_X + IN_W, IBUS_X], [MID_Y] * 2, color="#555555", lw=1.3, zorder=4)
    ax.plot([IBUS_X] * 2, [ROW_Y[-1] + ROW_H / 2, ROW_Y[0] + ROW_H / 2], color="#555555",
            lw=1.3, zorder=4)

    for (key, title, n, desc), (kind, lab), y0 in zip(GROUPS, SHAPES, ROW_Y):
        col, yc = C[key], y0 + ROW_H / 2
        _box(ax, GX, y0, GW, ROW_H, col)
        ax.text(GX + 0.07, y0 + ROW_H - 0.13, title, ha="left", va="center", fontsize=TXT,
                color=col, fontweight="bold", zorder=5)
        ax.text(GX + GW - 0.07, y0 + ROW_H - 0.13, f"({n})", ha="right", va="center",
                fontsize=TXT, color=col, fontweight="bold", zorder=5)
        ax.text(GX + 0.07, y0 + 0.2, desc, ha="left", va="center", fontsize=SMALL,
                color="#222222", zorder=5, linespacing=1.1)
        _arrow(ax, IBUS_X, yc, GX, yc, col)
        _box(ax, SX, y0, SW, ROW_H, "#b0b0b0", fill_alpha=0.05, lw=0.8)
        _mini(ax, SX + 0.07, y0 + 0.12, 0.34, ROW_H - 0.24, kind, col)
        ax.text(SX + 0.47, yc, lab, ha="left", va="center", fontsize=SMALL, color="#333333",
                zorder=6)
        _arrow(ax, GX + GW, yc, SX, yc, col)
        ax.plot([SX + SW, BUS_X], [yc] * 2, color="#555555", lw=1.0, zorder=4)

    ax.plot([BUS_X] * 2, [ROW_Y[-1] + ROW_H / 2, ROW_Y[0] + ROW_H / 2], color="#555555",
            lw=1.3, zorder=4)
    _arrow(ax, BUS_X, MID_Y, SUM_X, MID_Y, lw=1.3)
    _box(ax, SUM_X, iy, SUM_W, ih, C["sum"], fill_alpha=0.12)
    ax.text(SUM_X + SUM_W / 2, MID_Y, "$\\sum_i f_i$\n$+\\,b$\n\nlog-odds", ha="center",
            va="center", fontsize=TXT, color="#222222", zorder=5)
    _arrow(ax, SUM_X + SUM_W, MID_Y, OUT_X, MID_Y, lw=1.3)
    ax.text((SUM_X + SUM_W + OUT_X) / 2, MID_Y + 0.14, "$\\sigma$", ha="center", va="center",
            fontsize=TXT, color="#333333")
    _box(ax, OUT_X, iy, OUT_W, ih, C["out"], fill_alpha=0.12)
    ax.text(OUT_X + OUT_W / 2, MID_Y, "flight\nprob.\n$\\hat{p}$", ha="center", va="center",
            fontsize=TXT, color="#222222", zorder=5)
    ax.text((SUM_X + OUT_X + OUT_W) / 2, iy - 0.28, "fly if $\\hat{p} \\geq t^{*}$",
            ha="center", va="center", fontsize=SMALL, color="#333333")

    ny, nh = 0.0, 0.40
    _box(ax, 0.0, ny, W, nh, C["note"], fill_alpha=0.12, lw=0.8)
    ax.text(0.07, ny + nh / 2,
            "Monotonicity penalty ($\\lambda = 10$): output non-increasing in wind speed, gust, "
            "cloud, precipitation, snow\n15-seed ensemble; hyperparameters by Optuna; "
            "$t^{*}$ from inner CV on the training data",
            ha="left", va="center", fontsize=SMALL, color="#222222", zorder=5,
            linespacing=1.2)
    return fig


def make(out_dir: Path) -> None:
    ps.apply()
    fig = draw()
    out_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": "standard"}):   # fixed canvas keeps the 16 cm width
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"fig_architecture.{ext}",
                        **({"dpi": ps.PNG_DPI} if ext == "png" else {}))
    plt.close(fig)


if __name__ == "__main__":
    make(ROOT / "outputs/rev1/figures")
