| group | factor | formula | inputs (units) | constants | unit | meaning | reference |
|---|---|---|---|---|---|---|---|
| instantaneous | air_density | $\rho = 100P/(R_d T_v)$, $T_v = T_K/[1-(e/P)(1-\varepsilon)]$ | T (°C), P_sl (hPa), RH (%) | $R_d=287.05$ J kg$^{-1}$K$^{-1}$, $\varepsilon=0.622$ | kg m$^{-3}$ | moist-air density | Wallace & Hobbs (2006) |
| instantaneous | density_altitude | $h_\rho = 44330\,[1-(\rho/\rho_0)^{1/4.256}]$ | $\rho$ | $\rho_0=1.225$ kg m$^{-3}$ (ISA) | m | ISA altitude with the same density (relative index) | U.S. Standard Atmosphere (1976) |
| instantaneous | lift_capacity | $L = \rho\,[1 - T_K/(T_K+\Delta T)]$ | $\rho$, T (°C) | $\Delta T = 100$ K (assumed envelope excess) | kg m$^{-3}$ | gross lift per m$^3$ of envelope | FAA (2024) Balloon Flying Handbook |
| instantaneous | dewpoint_depression | $T - T_d \approx (100-RH)/5$ | RH (%) | — | K | proximity to saturation | Lawrence (2005) |
| instantaneous | gust_factor | $G = v_{gust}/(v + 10^{-3})$ | windgust, windspeed (km/h) | — | — | gustiness relative to sustained wind | Harper et al. (2010, WMO) |
| instantaneous | specific_humidity | $q = \varepsilon e/(P - 0.378e)$, $e = (RH/100)\,e_s(T)$, $e_s = 6.112\exp[17.67T/(T+243.5)]$ | T (°C), P_sl (hPa), RH (%) | $\varepsilon=0.622$; $e$, $e_s$ in hPa | kg kg$^{-1}$ | water-vapour content | Bolton (1980) |
| instantaneous | wind_dynamic_pressure | $q_w = \frac{1}{2}\rho v^2$ | $\rho$, windspeed (km/h) | — | kg m$^{-3}$ (km/h)$^2$ = Pa/12.96 | wind loading on the envelope (index) | Wallace & Hobbs (2006) |
| instantaneous | clear_sky_fraction | $c = (100 - \mathrm{cloud})/100$ | cloudcover (%) | — | — | fraction of sky free of cloud | — |
| synoptic (s=5) | pressure_tendency | $P_t - P_{t-s+1}$ | P_sl (hPa) | s = 5 | hPa | barometric tendency over 4 days | Wallace & Hobbs (2006) |
| synoptic (s=5) | pressure_std | $\mathrm{sd}(P_{t-s+1..t})$ | P_sl (hPa) | s = 5 | hPa | synoptic variability | — |
| synoptic (s=5) | frontal_passage | $\mathrm{ReLU}(-\Delta P)\,\mathrm{ReLU}(\Delta\mathrm{cloud})$ | P_sl, cloudcover | s = 5 | hPa·% | falling pressure, rising cloud (heuristic) | — |
| synoptic (s=5) | clearness_trend | $c_t - \overline{c}_{t-s+1..t}$ | cloudcover (%) | s = 5 | — | clearing or clouding up | — |
| synoptic (s=5) | dewdepr_trend | $(T-T_d)_t - (T-T_d)_{t-s+1}$ | RH (%) | s = 5 | K | drying or moistening | Lawrence (2005) |
| synoptic (s=5) | winddir_steadiness | $\|\overline{(\cos\theta,\sin\theta)}\|$ | winddir (°) | s = 5 | 0–1 | mean resultant length | Mardia & Jupp (1999) |
| synoptic (s=5) | recent_storminess | $\max v_{gust,t-s+1..t}$ | windgust (km/h) | s = 5 | km/h | peak recent gust | — |
| synoptic (s=5) | calm_clear | $c\,/[(T-T_d+1)(v+1)]$ | cloudcover, RH, windspeed | — | index | calm, clear, near-saturated air | Gultepe et al. (2007) |
| synoptic (s=5) | diurnal_range | $T_{max}-T_{min}$ | tempmax, tempmin (°C) | — | K | stability and clear-sky proxy | Stull (1988) |
| synoptic (s=5) | thermal_turbulence | $T_{max}\,c$ | tempmax (°C), cloudcover | — | °C | daytime convection proxy (heuristic) | — |
