"""Stage 8: stability of PI-NAM's interpretation across folds and seeds (R3-9, R2-4).

The additive model is  logit p = bias + sum_i f_i(x_i) + sum_pairs g_ij(x_i, x_j).  For a
saved ensemble member we evaluate every term on a common set of rows; contributions are
centred so that each term has mean zero over those rows (additive identifiability).

  importance of a term   mean |centred contribution| over the rows
  source group           every factor is assigned to the raw variable group it is
                         computed from (FACTOR_GROUP); composite factors take their
                         dominant input, interaction surfaces form their own group.
                         Group importance uses the *sum* of its terms' contributions, so
                         contributions shared by collinear factors (concurvity) cancel.
"""

from typing import Dict, List, Sequence

import numpy as np
import torch
from scipy.stats import kendalltau

from src.pinam import PHYS_NAMES
from src.pinam_search import (INTERACTION_PAIRS, PINAMX, SYNOPTIC_NAMES, TEMP_CHANNELS,
                              _n_factors)
from src.pinn import RAW_CHANNELS
from src.revision.robustness import GROUPS

_RAW_GROUP = {c: g for g, cs in GROUPS.items() for c in cs}
_COMPOSITE_GROUP = {
    # physics layer (inputs: T, P, RH unless stated) -> dominant input (|r| >= 0.97, stage 9)
    "air_density": "temperature", "density_altitude": "temperature", "lift_capacity": "temperature",
    "dewpoint_depression": "humidity", "specific_humidity": "humidity",
    "gustiness": "wind", "wind_energy": "wind", "cloud_clearness": "cloud",
    # synoptic layer
    "pressure_tendency": "pressure", "pressure_std": "pressure", "frontal_passage": "pressure",
    "clearness_trend": "cloud", "dewdepr_trend": "humidity", "winddir_steadiness": "wind",
    "recent_storminess": "wind", "calm_clear": "cloud", "diurnal_range": "temperature",
    "thermal_turbulence": "temperature",
}


def factor_names(spec: Dict) -> List[str]:
    """Names of the 1-D factors in `PINAMX._raw_factors` order (+ interaction surfaces)."""
    names = list(RAW_CHANNELS)
    if spec["physics"]:
        names += PHYS_NAMES
    for sc in spec["scales"]:
        for st in ("mean", "std", "max", "min"):
            if st in spec["stats"]:
                names += [f"{c}_{st}{sc}" for c in TEMP_CHANNELS]
    if "trend" in spec["stats"]:
        names += [f"{c}_trend" for c in TEMP_CHANNELS]
    if spec.get("synoptic"):
        names += SYNOPTIC_NAMES
    assert len(names) == _n_factors(spec)
    if spec.get("interactions"):
        names += [f"{a} x {b}" for a, b in INTERACTION_PAIRS]
    return names


def factor_group(name: str) -> str:
    if " x " in name:
        return "interaction"
    if name in _RAW_GROUP:
        return _RAW_GROUP[name]
    if name in _COMPOSITE_GROUP:
        return _COMPOSITE_GROUP[name]
    return _RAW_GROUP[name.rsplit("_", 1)[0]]          # temporal statistic of a channel


def build_model(spec: Dict, hidden: int, dropout: float, depth: int, state: Dict) -> PINAMX:
    n = _n_factors(spec)
    m = PINAMX(torch.zeros(n), torch.ones(n), n, spec, hidden, dropout,
               raw_mean=torch.zeros(len(RAW_CHANNELS)), raw_std=torch.ones(len(RAW_CHANNELS)),
               depth=depth)
    m.load_state_dict(state)
    return m.eval()


def contributions(model: PINAMX, win: np.ndarray) -> np.ndarray:
    """Centred per-term contributions (n_rows, n_terms) in log-odds."""
    with torch.no_grad():
        c = model._contributions(torch.tensor(win, dtype=torch.float32)).numpy()
    return c - c.mean(0, keepdims=True)


def term_importance(c: np.ndarray) -> np.ndarray:
    return np.abs(c).mean(0)


def group_importance(c: np.ndarray, names: Sequence[str]) -> Dict[str, float]:
    groups = [factor_group(n) for n in names]
    out = {}
    for g in dict.fromkeys(groups):
        s = c[:, [i for i, x in enumerate(groups) if x == g]].sum(1)
        out[g] = float(np.abs(s - s.mean()).mean())
    return out


def pairwise_tau(rankings: np.ndarray) -> np.ndarray:
    """Kendall tau between every pair of rows of an (n_runs, n_items) importance matrix."""
    k = len(rankings)
    return np.array([kendalltau(rankings[i], rankings[j])[0]
                     for i in range(k) for j in range(i + 1, k)])


def shape_curve(model: PINAMX, idx: int, x: np.ndarray, x_data: np.ndarray) -> np.ndarray:
    """f_idx on the raw-unit grid `x`, centred by its mean over the observed values `x_data`."""
    fm, fs = float(model.feat_mean[idx]), float(model.feat_std[idx])
    net = model.shapes[idx]
    with torch.no_grad():
        def f(v):
            z = torch.tensor(np.clip((v - fm) / fs, -10, 10), dtype=torch.float32)[:, None]
            return net(z).squeeze(1).numpy()
        return f(x) - f(x_data).mean()
