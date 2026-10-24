"""Decision-time information sets for the rev1 revision (R2-2, R3-1).

S0 -- same-day daily values are used as a proxy for the day-ahead forecast on
      which the flight authorization is based (perfect-forecast upper bound).
S1 -- only observations up to day t-1 are used; nothing from the decision day
      enters the model (observation-only lower bound).

Every model (PI-NAM and baselines) receives inputs through `load_inputs`, so
both information sets are applied identically across the comparison.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import build_features
from src.pinam_search import WINDOW, build_windows
from src.pinn import RAW_CHANNELS

logger = logging.getLogger(__name__)

INFO_SET_LAG = {"S0": 0, "S1": 1}


@dataclass(frozen=True)
class Inputs:
    """Model inputs for one information set (row order = decision days)."""

    info_set: str
    dates: pd.Series      # decision day t for each row
    y: np.ndarray         # flight authorization on day t
    win: np.ndarray       # (N, W, 16) PI-NAM windows
    X: np.ndarray         # (N, 16) tabular baseline inputs = last window row


def load_inputs(info_set: str, config_path: str | Path) -> Inputs:
    """Build PI-NAM windows and baseline features for the given information set.

    S0 reproduces the submitted pipeline exactly (X = day-t raw-16).
    S1 shifts everything one day back (X = day t-1 raw-16).
    """
    if info_set not in INFO_SET_LAG:
        raise ValueError(f"unknown info_set {info_set!r}; expected one of {list(INFO_SET_LAG)}")
    lag = INFO_SET_LAG[info_set]
    full, model = build_features(str(config_path))
    model = model.reset_index(drop=True)
    raw = model[RAW_CHANNELS].to_numpy(dtype=np.float32)
    history = None
    if lag > 0:
        first = full.index[full["datetime"] == model["datetime"].iloc[0]][0]
        history = full[RAW_CHANNELS].iloc[:first].to_numpy(dtype=np.float32)
    win = build_windows(raw, WINDOW, lag=lag, history=history)
    logger.info("info_set=%s lag=%d N=%d", info_set, lag, len(model))
    return Inputs(info_set=info_set, dates=model["datetime"], y=model["flight"].to_numpy(),
                  win=win, X=win[:, -1, :].copy())
