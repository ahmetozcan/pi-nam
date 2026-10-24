"""Shared figure style for the revised manuscript (R2-9: readable fonts at print size).

Figures are drawn at their final printed size (text width 16 cm = A4 with 2.5 cm margins)
and included with width=\\linewidth, so the point sizes below are the printed sizes.
Colours: Okabe-Ito palette (colour-blind safe); PI-NAM is always the same blue.
Lettering: the journal asks for Helvetica or Arial at 8-12 pt; Arimo (metric-compatible with
Arial, Apache-licensed) is used when installed, DejaVu Sans only for missing glyphs (arrows).
"""

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ARIMO_DIRS = (Path("/usr/share/fonts/truetype/croscore"), Path.home() / ".fonts")

TEXT_WIDTH = 6.3            # inches (16 cm)
HALF_WIDTH = 3.1
FONT = 9.5                  # tick labels, legends
LABEL = 10.0                # axis labels, panel titles
PNG_DPI = 600

OKABE_ITO = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
             "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000",
             "grey": "#7F7F7F"}
PINAM = OKABE_ITO["blue"]
PINAM_FIXED = OKABE_ITO["sky"]
MODEL_COLORS: Dict[str, str] = {
    "PI-NAM": PINAM, "PI-NAM-fixed": PINAM_FIXED, "EBM": OKABE_ITO["green"],
    "XGBoost": OKABE_ITO["orange"], "XGBoost-mono": "#B07800", "LightGBM": OKABE_ITO["yellow"],
    "Random Forest": OKABE_ITO["purple"], "CatBoost": OKABE_ITO["vermillion"],
    "SVM": OKABE_ITO["grey"], "TabPFN": OKABE_ITO["black"],
}
CLASS_COLORS = {0: OKABE_ITO["vermillion"], 1: OKABE_ITO["blue"]}       # no flight / flight


def _font_family() -> List[str]:
    """Register Arimo explicitly (matplotlib does not always index it) and return the stack."""
    files = [f for d in ARIMO_DIRS if d.is_dir() for f in d.glob("Arimo-*.ttf")]
    for f in files:
        font_manager.fontManager.addfont(str(f))
    return (["Arimo"] if files else []) + ["DejaVu Sans"]


def apply() -> None:
    """Set matplotlib rcParams for every manuscript figure."""
    family = _font_family()
    plt.rcParams.update({
        "font.family": family, "mathtext.fontset": "custom", "mathtext.rm": family[0],
        "mathtext.it": f"{family[0]}:italic", "mathtext.bf": f"{family[0]}:bold",
        "font.size": FONT,
        "axes.titlesize": LABEL, "axes.labelsize": LABEL, "axes.titlelocation": "left",
        "xtick.labelsize": FONT, "ytick.labelsize": FONT, "legend.fontsize": FONT,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.linewidth": 1.5, "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })


def save(fig: plt.Figure, out_dir: Path, stem: str, exts: Iterable[str] = ("pdf", "png")) -> None:
    """Write vector PDF and 600-dpi PNG, then close the figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in exts:
        fig.savefig(out_dir / f"{stem}.{ext}", **({"dpi": PNG_DPI} if ext == "png" else {}))
    plt.close(fig)
