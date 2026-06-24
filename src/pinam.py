"""
Shape-function building blocks for PI-NAM.

Only the components used by the final model (`src/pinam_search.py`) are kept:
per-factor 1-D shape networks, 2-D interaction shape networks, and the physics
feature-name list. (Earlier exploratory model variants are not part of the
released proposed model.)
"""
import torch.nn as nn

# Physics feature names (order matches src.pinn.physics_features output)
PHYS_NAMES = ["air_density", "density_altitude", "lift_capacity",
              "dewpoint_depression", "gustiness", "specific_humidity",
              "wind_energy", "cloud_clearness"]


class ShapeNet(nn.Module):
    """1-D shape function f_i(x_i) -> scalar log-odds contribution."""
    def __init__(self, hidden=24, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):           # x: (N,1)
        return self.net(x)


class Shape2D(nn.Module):
    """2-D interaction shape function f_ij(x_i, x_j) -> log-odds contribution."""
    def __init__(self, hidden=16, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):           # x: (N,2)
        return self.net(x)
