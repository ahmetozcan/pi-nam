"""
Detailed architecture diagram of the final PI-NAM (V1) model:
raw-16 + instantaneous physics (8) + synoptic physics (10, 5-day) +
temporal min-aggregates (scales 3,7) + pairwise interactions (6),
each through its own shape function, summed to log-odds -> sigmoid suitability.
Annotated with monotonicity priors, 15-seed deep ensemble, train-tuned threshold.
Outputs PDF + PNG to outputs/figures/ and outputs/figures/.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(13.5, 8.2))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 8.2); ax.axis("off")

C = {"input": "#34495e", "phys": "#2980b9", "syn": "#16a085", "temp": "#8e44ad",
     "inter": "#d35400", "shape": "#7f8c8d", "out": "#c0392b", "bias": "#95a5a6"}


def box(x, y, w, h, text, color, fc=None, fs=8.5, tc="white", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.06",
                 ec=color, fc=fc or color, lw=lw, alpha=1.0 if fc is None else 0.18))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc if fc is None else "black", wrap=True, zorder=5)


def arrow(x1, y1, x2, y2, color="#555555", lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=11, lw=lw, color=color, zorder=1))


# ---- Input ----
box(0.2, 3.6, 1.7, 1.0, "Raw weather\n16 channels\n(14-day causal\nwindow)", C["input"], fs=8)

# ---- Feature/rule groups (column 2) ----
gx, gw = 2.5, 3.0
groups = [
    (6.55, C["input"], "Raw inputs (16)",
     "temp, humidity, windgust, windspeed, cloudcover,\nvisibility, sea-level pressure, ... (identity factors)"),
    (5.25, C["phys"], "Instantaneous physics (8)",
     "air density, density altitude, lift capacity, dewpoint\ndepression, gustiness, spec. humidity, wind energy, clearness"),
    (3.75, C["syn"], "Synoptic physics (10)  —  5-day window",
     "pressure tendency $\\Delta P$, pressure std, frontal passage,\nclearness trend, dewdepr trend, winddir steadiness,\nrecent storminess, fog risk, diurnal range, thermal turbulence"),
    (2.35, C["temp"], "Temporal min-aggregates",
     "min over scales {3, 7} days\non 7 key channels"),
    (1.15, C["inter"], "Pairwise interactions (6)",
     "cloud×visibility, cloud×gust, gust×wind,\ntemp×humidity, pressure×gust, humidity×cloud  (2-D shapes)"),
]
for yc, col, title, desc in groups:
    box(gx, yc, gw, 1.05, "", col, fc=col)
    ax.text(gx + 0.12, yc + 0.82, title, ha="left", va="center", fontsize=8.6,
            color="black", fontweight="bold", zorder=6)
    ax.text(gx + 0.12, yc + 0.34, desc, ha="left", va="center", fontsize=7.0,
            color="#222222", zorder=6)
    arrow(1.9, 4.1, gx, yc + 0.52, color=col, lw=1.0)

# ---- Shape functions bank (column 3) ----
sx = 6.05
box(sx, 1.15, 2.0, 6.45, "", C["shape"], fc=C["shape"])
ax.text(sx + 1.0, 7.3, "Shape functions", ha="center", fontweight="bold", fontsize=9)
ax.text(sx + 1.0, 6.95, "per factor: $f_i(x_i)$", ha="center", fontsize=8, style="italic")
# mini-curves drawn directly in ax data coordinates (inside the shape box)
import numpy as np
t = np.linspace(-2, 2, 60)
cx0, cw, ch = sx + 0.18, 0.62, 0.34   # curve x-start, width, half-height
for yy, lab, shp in [
        (6.45, "cloud ↓", -1), (5.55, "wind ↓", -1), (4.65, "visibility ↑", 1),
        (3.75, "fog risk ↓", -1), (2.85, "$\\Delta P$ ↑", 1), (1.85, "interaction", 0)]:
    if shp == -1:
        c = -(1 / (1 + np.exp(-2.5 * t)))
    elif shp == 1:
        c = 1 / (1 + np.exp(-2.5 * t))
    else:
        c = 0.6 * np.tanh(2 * t) * np.cos(1.5 * t)
    xs = cx0 + (t + 2) / 4 * cw
    ys = yy + c * ch
    ax.plot(xs, ys, color=C["out"], lw=1.4, zorder=7)
    ax.plot([cx0, cx0 + cw], [yy - ch - 0.04, yy - ch - 0.04], color="#cccccc", lw=0.5, zorder=6)
    ax.text(sx + 1.35, yy, lab, ha="left", va="center", fontsize=6.8, color="#222", zorder=7)

for yc, col, *_ in groups:
    arrow(gx + gw, yc + 0.52, sx, 4.3, color=col, lw=0.8)

# ---- Sum + bias ----
box(8.5, 3.9, 1.6, 0.95, "$\\sum_i f_i(x_i)$\n$+\\ b$\n(log-odds)", C["bias"], fs=9)
arrow(sx + 2.0, 4.35, 8.5, 4.37)

# ---- Sigmoid / output ----
box(10.6, 3.9, 1.7, 0.95, "$\\sigma(\\cdot)$\nflight\nsuitability", C["out"], fs=9)
arrow(10.1, 4.37, 10.6, 4.37)

# ---- Annotations ----
box(8.5, 6.3, 3.8, 1.1, "", "#bdc3c7", fc="#ecf0f1", tc="black")
ax.text(8.62, 7.15, "Physics priors & training", ha="left", fontweight="bold", fontsize=8.4)
ax.text(8.62, 6.55,
        "• Monotonicity: suitability non-increasing in\n   wind, gust, cloud, precip, snow (gradient penalty)\n"
        "• 15-seed deep ensemble  • train-OOF-tuned threshold",
        ha="left", va="center", fontsize=7.0)

box(10.45, 2.05, 2.85, 1.45, "", "#bdc3c7", fc="#fdf2e9", tc="black")
ax.text(10.57, 3.28, "Final model (V1)", ha="left", fontweight="bold", fontsize=8.4)
ax.text(10.57, 2.62,
        "deploy F1 = 0.877\nAUC = 0.903,  MCC = 0.643\nthreshold 0.385 ± 0.038\n"
        "hidden=48, depth=2",
        ha="left", va="center", fontsize=7.2)

ax.text(6.75, 7.95, "PI-NAM: Physics-Informed Neural Additive Model (final architecture)",
        ha="center", fontweight="bold", fontsize=11.5)

outdir1 = ROOT / "outputs/figures"; outdir2 = ROOT / "outputs/figures"
outdir1.mkdir(parents=True, exist_ok=True); outdir2.mkdir(parents=True, exist_ok=True)
for d in (outdir1, outdir2):
    fig.savefig(d / "figure-pinam-v1-architecture.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(d / "figure-pinam-v1-architecture.png", bbox_inches="tight", dpi=160)
print(f"Saved figure-pinam-v1-architecture.{{pdf,png}} to {outdir1} and {outdir2}")
