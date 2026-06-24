"""
Feature engineering for meteorological balloon flight prediction.

Adds temporal, seasonal, rolling, and lag features to the raw dataset.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} records from {path}")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["datetime"].dt.month
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(int)

    # Cyclical encoding to capture periodicity
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Season (1=Winter, 2=Spring, 3=Summer, 4=Autumn)
    df["season"] = df["month"].map(
        {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}
    )
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Previous day flight status — key signal for operational continuity."""
    df = df.copy()
    df["prev_flight_1d"] = df["flight"].shift(1)
    df["prev_flight_2d"] = df["flight"].shift(2)
    df["prev_flight_3d"] = df["flight"].shift(3)

    # Consecutive flight days streak
    streak = []
    count = 0
    for val in df["flight"]:
        count = count + 1 if val == 1 else 0
        streak.append(count)
    df["flight_streak"] = streak
    df["flight_streak"] = df["flight_streak"].shift(1)  # only past info
    return df


def add_rolling_features(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Multi-resolution rolling statistics (1d, 3d, 7d, 14d windows).

    All features use shift(1) to prevent target leakage.
    Rationale: Cappadocia's volatile climate means 7d windows alone miss sudden
    changes; 1d and 3d windows capture rapid meteorological shifts.
    """
    df = df.copy()
    met_cols = [
        "windgust", "windspeed", "temp", "humidity",
        "cloudcover", "visibility", "sealevelpressure",
    ]

    # Multi-resolution windows: 1d, 3d, primary window, 14d
    windows = sorted(set([1, 3, window, 14]))
    for col in met_cols:
        for w in windows:
            shifted = df[col].shift(1)
            df[f"{col}_roll{w}m"]   = shifted.rolling(w, min_periods=1).mean()
            df[f"{col}_roll{w}std"] = shifted.rolling(w, min_periods=1).std().fillna(0)
            df[f"{col}_roll{w}max"] = shifted.rolling(w, min_periods=1).max()

    # Cappadocia-specific: acute wind gust change vs 1d rolling mean
    df["windgust_change_1d"] = (
        df["windgust"] - df["windgust"].shift(1).rolling(1, min_periods=1).mean()
    )

    # Wind trend: difference between today and 3-day rolling mean
    df["windgust_trend"] = df["windgust"] - df["windgust"].shift(1).rolling(3, min_periods=1).mean()
    df["temp_trend"] = df["temp"] - df["temp"].shift(1).rolling(3, min_periods=1).mean()
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-informed interaction features."""
    df = df.copy()
    # High wind + low visibility = dangerous combination
    df["wind_vis_ratio"] = df["windgust"] / (df["visibility"] + 1e-3)
    # Thermal comfort index
    df["thermal_index"] = df["temp"] - 0.4 * (df["humidity"] - 85)
    # Pressure gradient proxy
    df["pressure_change"] = df["sealevelpressure"] - df["sealevelpressure"].shift(1)
    df["pressure_change"] = df["pressure_change"].fillna(0)
    return df


def build_features(config_path: str = "config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full feature engineering pipeline.

    Returns:
        df_full: DataFrame with all features including datetime and flight label
        df_model: Feature matrix + target (NaN rows from rolling/lag dropped)
    """
    cfg = load_config(config_path)
    df = load_raw_data(cfg["data"]["path"])

    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df, window=cfg["features"]["window_size"])
    df = add_interaction_features(df)

    # Drop rows where lag/rolling features are NaN (first window_size rows)
    df_model = df.dropna().reset_index(drop=True)

    n_dropped = len(df) - len(df_model)
    logger.info(
        f"Feature engineering complete: {len(df_model)} usable rows "
        f"({n_dropped} dropped due to lag/rolling warm-up)"
    )

    # Save enriched dataset
    out_path = Path(cfg["paths"]["results"]) / "dataset_engineered.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_model.to_csv(out_path, index=False)
    logger.info(f"Saved engineered dataset → {out_path}")

    return df, df_model


def get_feature_columns(df: pd.DataFrame, target: str = "flight") -> list:
    """Return model feature columns (exclude datetime, target)."""
    exclude = {target, "datetime"}
    return [c for c in df.columns if c not in exclude]


def get_season_label(season_id: int) -> str:
    return {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}.get(season_id, "Unknown")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    import os
    os.chdir(Path(__file__).parent.parent)
    df_full, df_model = build_features()
    print(f"\nTotal features: {len(get_feature_columns(df_model))}")
    print(df_model.head(3))
