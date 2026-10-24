"""Physics layer against reference values (R2-12, R3-4).

Documents the units the formulas assume: temperature in degC, pressure in hPa
(sea-level-reduced, as supplied by Visual Crossing), relative humidity in %,
wind in km/h, cloud cover in %.
"""

import numpy as np
import pytest
import torch

from src.pinam_search import synoptic_physics
from src.pinn import IDX, RAW_CHANNELS, physics_features

NAMES = ["air_density", "density_altitude", "lift_capacity", "dewpoint_depression",
         "gust_factor", "specific_humidity", "wind_dynamic_pressure", "clear_sky_fraction"]


def _raw(**kw) -> torch.Tensor:
    row = dict(temp=15.0, tempmax=15.0, tempmin=15.0, humidity=0.0, sealevelpressure=1013.25,
               windspeed=10.0, windgust=10.0, cloudcover=0.0, winddir=0.0)
    row.update(kw)
    x = torch.zeros(1, len(RAW_CHANNELS))
    for k, v in row.items():
        x[0, IDX[k]] = v
    return x


def _phys(**kw) -> dict:
    return dict(zip(NAMES, physics_features(_raw(**kw))[0].tolist()))


def test_isa_sea_level_density_and_altitude():
    f = _phys()                                    # 15 degC, 1013.25 hPa, dry air
    assert f["air_density"] == pytest.approx(1.2250, abs=1e-4)
    assert f["density_altitude"] == pytest.approx(0.0, abs=1.0)


def test_density_altitude_increases_with_temperature():
    assert _phys(temp=35.0)["density_altitude"] > _phys(temp=25.0)["density_altitude"] > 0


def test_tetens_saturation_vapour_pressure_via_specific_humidity():
    # Bolton (1980) form: e_s(20 degC) = 23.37 hPa; at RH 50 % and 1013.25 hPa q = 7.20 g/kg
    es20 = 6.112 * np.exp(17.67 * 20 / (20 + 243.5))
    assert es20 == pytest.approx(23.37, abs=0.02)
    assert _phys(temp=20.0, humidity=50.0)["specific_humidity"] == pytest.approx(7.20e-3, abs=2e-5)


def test_lift_assumes_envelope_100K_above_ambient():
    f = _phys()
    assert f["lift_capacity"] == pytest.approx(1.2250 * 100 / (288.15 + 100), abs=1e-4)


def test_dewpoint_depression_is_linear_in_rh():
    # Lawrence (2005) approximation: T - Td ~= (100 - RH) / 5, independent of T
    assert _phys(humidity=60.0)["dewpoint_depression"] == pytest.approx(8.0, abs=1e-5)
    assert _phys(temp=-5.0, humidity=60.0)["dewpoint_depression"] == pytest.approx(8.0, abs=1e-5)


def test_wind_dynamic_pressure_uses_kmh():
    # 0.5 rho v^2 with v in km/h equals 3.6^2 = 12.96 times the SI value in Pa
    f = _phys(windspeed=36.0)
    assert f["wind_dynamic_pressure"] / 12.96 == pytest.approx(0.5 * 1.2250 * 10.0 ** 2, rel=1e-4)


def test_gust_factor_and_clear_sky():
    f = _phys(windspeed=20.0, windgust=30.0, cloudcover=25.0)
    assert f["gust_factor"] == pytest.approx(1.5, rel=1e-4)
    assert f["clear_sky_fraction"] == pytest.approx(0.75, abs=1e-6)


def test_synoptic_tendency_spans_scale_minus_one_days():
    # window of 5 days with pressure rising 1 hPa/day: tendency = P_t - P_{t-4} = 4 hPa
    win = _raw().repeat(5, 1).unsqueeze(0).clone()
    win[0, :, IDX["sealevelpressure"]] = torch.arange(1010.0, 1015.0)
    syn = synoptic_physics(win, scale=5)[0]
    assert syn[0].item() == pytest.approx(4.0)
