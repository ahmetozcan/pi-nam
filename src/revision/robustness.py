"""Stage 7 robustness inputs (R1: missing weather variables, record length).

All corruptions act on the raw daily series (decision-day rows, 16 channels) and the model
windows are rebuilt from it, so a corrupted day is corrupted consistently in every window
that contains it. Models are always trained on clean inputs unless stated otherwise.

  mask      one variable group replaced by its training-split mean on every day (the
            sensor/feed for that group is unavailable at prediction time)
  missing   a random share of (day, channel) cells removed and imputed, either with the
            training-split mean or by carrying the last observed value forward
  drop      the group is constant on all rows, for training and test (model re-fitted
            without that information)
  subsample stratified random share of each outer training set (learning curve)

S0 only: the decision-day rows of the windows are the raw series itself.
"""

from typing import Dict, Iterator, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from src.pinam_search import WINDOW, build_windows
from src.pinn import IDX
from src.revision.info_sets import Inputs
from src.revision.protocols import Split

GROUPS: Dict[str, List[str]] = {
    "wind": ["windgust", "windspeed", "winddir"],
    "cloud": ["cloudcover", "visibility"],
    "pressure": ["sealevelpressure"],
    "humidity": ["humidity"],
    "temperature": ["tempmax", "tempmin", "temp", "feelslikemax", "feelslikemin", "feelslike"],
    "precipitation": ["precip", "snow", "snowdepth"],
}
MISSING_FRACS = (0.1, 0.2, 0.3)
IMPUTERS = ("mean", "ffill")


def raw_series(inp: Inputs) -> np.ndarray:
    """Raw daily series of an S0 input set (row t = decision day t)."""
    if inp.info_set != "S0":
        raise ValueError("robustness inputs are defined for S0 only")
    return inp.X.copy()


def windows(raw: np.ndarray) -> np.ndarray:
    return build_windows(raw.astype(np.float32), WINDOW)


def group_index(group: str) -> List[int]:
    return [IDX[c] for c in GROUPS[group]]


def fill_group(raw: np.ndarray, group: str, fill: np.ndarray) -> np.ndarray:
    """Copy of `raw` with the group's channels set to `fill` (per-channel values) on every day."""
    out = raw.copy()
    idx = group_index(group)
    out[:, idx] = fill[idx]
    return out


def _ffill(a: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Carry the last observed value forward per column; leading gaps take `fallback`."""
    out = a.copy()
    for j in range(a.shape[1]):
        last = fallback[j]
        for t in range(a.shape[0]):
            if np.isnan(out[t, j]):
                out[t, j] = last
            else:
                last = out[t, j]
    return out


def random_missing(raw: np.ndarray, frac: float, rng: np.random.Generator, method: str,
                   fill: np.ndarray) -> np.ndarray:
    """Remove a random `frac` of all (day, channel) cells and impute them."""
    if method not in IMPUTERS:
        raise ValueError(f"unknown imputer {method!r}; expected {IMPUTERS}")
    a = raw.astype(np.float64).copy()
    a[rng.random(a.shape) < frac] = np.nan
    if method == "mean":
        return np.where(np.isnan(a), fill[None, :], a).astype(np.float32)
    return _ffill(a, fill).astype(np.float32)


def inference_conditions(raw: np.ndarray, train: np.ndarray, test: np.ndarray, n_reps: int = 5,
                         seed: int = 42) -> Iterator[Tuple[str, np.ndarray]]:
    """(name, test-row windows) for the clean input and every prediction-time corruption.

    The fill value is the mean of the training rows of this split, so no test information
    enters the imputation; random masks depend only on (seed, rep) and are shared across
    models and splits.
    """
    fill = raw[train].mean(0)
    yield "clean", windows(raw)[test]
    for g in GROUPS:
        yield f"mask_{g}", windows(fill_group(raw, g, fill))[test]
    for frac in MISSING_FRACS:
        for k in range(n_reps):
            for method in IMPUTERS:
                rng = np.random.default_rng([seed, k, int(round(100 * frac))])
                yield (f"miss{int(round(100 * frac))}_{method}_r{k}",
                       windows(random_missing(raw, frac, rng, method, fill))[test])


def drop_group_windows(raw: np.ndarray, group: str) -> np.ndarray:
    """Windows in which `group` carries no information (constant = all-row mean of the
    inputs, labels are not involved), used for training and test alike."""
    return windows(fill_group(raw, group, raw.mean(0)))


def subsample_splits(splits: List[Split], y: np.ndarray, frac: float, seed: int) -> List[Split]:
    """Stratified random `frac` of every outer training set; test rows unchanged."""
    if frac >= 1.0:
        return splits
    out = []
    for s in splits:
        tr, _ = train_test_split(s.train, train_size=frac, stratify=y[s.train], random_state=seed)
        out.append(Split(s.name, np.sort(tr), s.test))
    return out
