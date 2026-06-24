"""
Learned per-factor shape functions f_i(x_i) of the final PI-NAM (V1), averaged
over the 150 ensemble members (10 folds x 15 seeds). Each curve = log-odds
contribution to flight suitability vs the factor value (in real units), centered
for additive identifiability, with +/-1 std band across ensemble members.
Outputs a 3x3 panel PDF+PNG + a text summary of monotonic directions.
"""
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
ROOT = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT))
from src.pinn import RAW_CHANNELS
from src.pinam import PHYS_NAMES
from src.pinam_search import (PINAMX, _n_factors, TEMP_CHANNELS, SYNOPTIC_NAMES)

SPEC = {"physics": True, "stats": ["min"], "scales": [3, 7],
        "interactions": True, "synoptic": True, "syn_scale": 5}
HIDDEN, DROPOUT, DEPTH = 48, 0.2, 2
NF = _n_factors(SPEC)

# factor names in _raw_factors concatenation order
names = list(RAW_CHANNELS) + list(PHYS_NAMES)
for sc in SPEC["scales"]:
    names += [f"{c}_min{sc}" for c in TEMP_CHANNELS]
names += list(SYNOPTIC_NAMES)
assert len(names) == NF, (len(names), NF)
NAME2I = {n: i for i, n in enumerate(names)}

# curated, physically interpretable subset (label, factor name, unit)
PANELS = [
    ("Cloud cover", "cloudcover", "%"),
    ("Wind gust", "windgust", "km/h"),
    ("Visibility", "visibility", "km"),
    ("Relative humidity", "humidity", "%"),
    ("Air density", "air_density", "kg/m$^3$"),
    ("Dewpoint depression", "dewpoint_depression", "$^\\circ$C"),
    ("Pressure tendency $\\Delta P$ (5-day)", "pressure_tendency", "hPa"),
    ("Fog risk (synoptic)", "fog_risk", "index"),
    ("Diurnal range", "diurnal_range", "$^\\circ$C"),
]

# load all ensemble members
states = []
for fi in range(1, 11):
    b = pickle.load(open(ROOT / f"reproducibility/folds/pinamx_v1/fold_{fi:02d}.pkl", "rb"))
    states.extend(b["seed_states"])
print(f"loaded {len(states)} ensemble members")

dummy = torch.zeros(NF)
model = PINAMX(dummy.clone(), torch.ones(NF), NF, SPEC, HIDDEN, DROPOUT,
               raw_mean=torch.zeros(len(RAW_CHANNELS)), raw_std=torch.ones(len(RAW_CHANNELS)),
               depth=DEPTH)

Z = np.linspace(-2.5, 2.5, 80)
zt = torch.tensor(Z, dtype=torch.float32).unsqueeze(1)


def curves_for(idx):
    """Return raw-unit x grid + centered contribution curves (n_models, 80)."""
    cs, xms, xss = [], [], []
    for st in states:
        model.load_state_dict(st)
        model.eval()
        fm = float(st["feat_mean"][idx]); fs = float(st["feat_std"][idx])
        with torch.no_grad():
            c = model.shapes[idx](zt).squeeze(1).numpy()
        cs.append(c - c.mean())          # center for identifiability
        xms.append(fm); xss.append(fs)
    x = np.mean(xms) + Z * np.mean(xss)  # representative raw-unit axis
    return x, np.array(cs)


fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.0))
plt.rcParams.update({"font.size": 9})
summary = []
for ax, (label, fname, unit) in zip(axes.ravel(), PANELS):
    idx = NAME2I[fname]
    x, cs = curves_for(idx)
    mean, std = cs.mean(0), cs.std(0)
    ax.plot(x, mean, color="#c0392b", lw=2.0, zorder=3)
    ax.fill_between(x, mean - std, mean + std, color="#c0392b", alpha=0.15, zorder=2)
    ax.axhline(0, color="#999999", lw=0.6, ls="--")
    ax.set_title(label, fontsize=9.5, fontweight="bold")
    ax.set_xlabel(f"{label.split(' (')[0]} ({unit})", fontsize=8)
    ax.set_ylabel("log-odds contribution", fontsize=8)
    ax.tick_params(labelsize=7.5)
    slope = mean[-1] - mean[0]
    summary.append(f"{label}: {'↓ decreasing' if slope < 0 else '↑ increasing'} "
                   f"(net {slope:+.2f} log-odds across range)")

fig.suptitle("PI-NAM (V1): learned per-factor shape functions  $f_i(x_i)\\!\\to$ flight-suitability log-odds",
             fontsize=12.5, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
for d in (ROOT / "outputs/figures", ROOT / "outputs/figures"):
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "figure-pinam-v1-shapes.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(d / "figure-pinam-v1-shapes.png", bbox_inches="tight", dpi=160)
(ROOT / "outputs/results/pinam_v1_shape_summary.txt").write_text("\n".join(summary))
print("Saved figure-pinam-v1-shapes.{pdf,png}\n")
print("\n".join(summary))
