"""
PINN: Physics-Informed Neural Network for balloon-flight meteorological suitability.

Two physics-informed mechanisms (the genuinely novel part):
  1. Differentiable physics layer — computes air density, lift capacity,
     density altitude, dewpoint depression from RAW weather inputs via
     atmospheric formulas (ideal gas law, Tetens), inside the network.
  2. Monotonicity regularization — soft physical priors enforced in the loss:
     flyability must NOT increase with wind gust, wind speed, cloud cover,
     or precipitation. Penalizes positive input-gradients (ReLU(∂p/∂x_neg)).

Operates on RAW weather channels only (no hand-engineered rolling features),
testing whether physics priors let a neural net match/beat tree models at N=728.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
torch.set_num_threads(4)

# Raw channel order (must match the feature matrix columns passed in)
RAW_CHANNELS = [
    "tempmax", "tempmin", "temp", "feelslikemax", "feelslikemin", "feelslike",
    "humidity", "precip", "snow", "snowdepth", "windgust", "windspeed",
    "winddir", "sealevelpressure", "cloudcover", "visibility",
]
IDX = {c: i for i, c in enumerate(RAW_CHANNELS)}

# Channels where flyability must be MONOTONICALLY NON-INCREASING (physical prior)
MONO_DECREASING = ["windgust", "windspeed", "cloudcover", "precip", "snow"]


N_PHYS = 8


def physics_features(raw: torch.Tensor) -> torch.Tensor:
    """Differentiable atmospheric physics features from raw weather (N, 16).

    Returns (N, 8): [air_density, density_altitude, lift_capacity,
                     dewpoint_depression, gustiness, specific_humidity,
                     wind_energy, cloud_clearness].
    """
    T   = raw[:, IDX["temp"]]
    P   = raw[:, IDX["sealevelpressure"]]
    RH  = raw[:, IDX["humidity"]] / 100.0
    T_K = T + 273.15

    es  = 6.112 * torch.exp(17.67 * T / (T + 243.5))
    e   = RH * es
    Tv  = T_K / (1 - (e / P) * (1 - 0.622))
    rho = (P * 100.0) / (287.05 * Tv)
    dens_alt = 44330 * (1 - torch.clamp(rho / 1.225, min=1e-3) ** (1 / 4.256))
    lift = rho - rho * T_K / (T_K + 100)
    dewpoint = T - ((100 - raw[:, IDX["humidity"]]) / 5.0)
    dp_depr  = T - dewpoint
    gust = raw[:, IDX["windgust"]] / (raw[:, IDX["windspeed"]] + 1e-3)
    spec_hum = 0.622 * e / (P - 0.378 * e)
    wind_energy = 0.5 * rho * raw[:, IDX["windspeed"]] ** 2     # dynamic pressure
    clearness = (100.0 - raw[:, IDX["cloudcover"]]) / 100.0      # clear-sky fraction

    return torch.stack([rho, dens_alt, lift, dp_depr, gust,
                        spec_hum, wind_energy, clearness], dim=1)


class PINN(nn.Module):
    """Physics-informed net: raw + differentiable physics → MLP → sigmoid.

    Args:
        n_raw: number of raw input channels (16).
        raw_mean/raw_std: standardization stats (tensors) for raw inputs.
        phys_mean/phys_std: standardization stats for physics features.
        hidden: hidden width.
        dropout: dropout prob.
    """

    def __init__(self, n_raw, raw_mean, raw_std, phys_mean, phys_std,
                 hidden=64, dropout=0.4):
        super().__init__()
        self.register_buffer("raw_mean", raw_mean)
        self.register_buffer("raw_std", raw_std)
        self.register_buffer("phys_mean", phys_mean)
        self.register_buffer("phys_std", phys_std)
        self.net = nn.Sequential(
            nn.Linear(n_raw + N_PHYS, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1), nn.Sigmoid(),
        )

    def forward(self, raw):
        phys = physics_features(raw)
        rs = (raw - self.raw_mean) / self.raw_std
        ps = (phys - self.phys_mean) / self.phys_std
        return self.net(torch.cat([rs, ps], dim=1)).squeeze(1)


def monotonicity_penalty(model: PINN, raw_batch: torch.Tensor) -> torch.Tensor:
    """Penalize positive ∂p/∂x for channels that should decrease flyability."""
    raw_batch = raw_batch.clone().requires_grad_(True)
    p = model(raw_batch)
    grads = torch.autograd.grad(p.sum(), raw_batch, create_graph=True)[0]
    pen = 0.0
    for ch in MONO_DECREASING:
        g = grads[:, IDX[ch]]
        pen = pen + torch.relu(g).mean()   # positive gradient = violation
    return pen


def train_pinn(
    raw_tr, y_tr, raw_val, y_val,
    lam_mono=0.1, lr=1e-3, weight_decay=1e-3,
    max_epochs=400, patience=40, batch_size=32, seed=42,
) -> PINN:
    torch.manual_seed(seed)
    rm = raw_tr.mean(0); rs = raw_tr.std(0) + 1e-6
    with torch.no_grad():
        ph = physics_features(torch.tensor(raw_tr))
    pm = ph.mean(0); ps = ph.std(0) + 1e-6

    model = PINN(raw_tr.shape[1],
                 torch.tensor(rm, dtype=torch.float32), torch.tensor(rs, dtype=torch.float32),
                 pm.float(), ps.float())
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCELoss()

    Xtr = torch.tensor(raw_tr, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.float32)
    Xva = torch.tensor(raw_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.float32)

    best, best_state, ni = 1e9, None, 0
    N = len(y_tr)
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(N)
        for b in range(0, N, batch_size):
            idx = perm[b:b + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            loss = crit(model(xb), yb) + lam_mono * monotonicity_penalty(model, xb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xva), yva).item()
        if vl < best - 1e-5:
            best, best_state, ni = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            ni += 1
            if ni >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_pinn(model: PINN, raw: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.tensor(raw, dtype=torch.float32)).numpy()


def _best_threshold(p: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    bt, bf = 0.5, 0.0
    for t in np.arange(0.20, 0.80, 0.01):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > bf:
            bf, bt = f, float(t)
    return bt


def run_pinn_cv(
    raw: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    lam_mono: float = 0.1,
    n_seeds: int = 5,
    tune_threshold: bool = True,
) -> Tuple[List[dict], np.ndarray]:
    """10-fold CV for the strengthened PINN (seed ensemble + threshold tuning).

    Args:
        raw: Raw weather matrix (N, 16).
        y: Binary labels.
        cfg: Config dict (features.test_folds, features.random_seed).
        lam_mono: Monotonicity penalty weight.
        n_seeds: Number of seed-ensemble members (predictions averaged).
        tune_threshold: If True, tune decision threshold on inner validation.

    Returns:
        (per_fold_metrics, oof_probs)
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                  f1_score, matthews_corrcoef, roc_auc_score)

    n_splits = cfg["features"]["test_folds"]
    base_seed = cfg["features"]["random_seed"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed)

    oof = np.zeros(len(y))
    rows: List[dict] = []

    for fold_idx, (tr, te) in enumerate(skf.split(raw, y), start=1):
        nval = max(int(len(tr) * 0.2), 10)
        vi, ti = tr[:nval], tr[nval:]

        # Seed ensemble: average probabilities over n_seeds models
        te_probs = np.zeros(len(te))
        val_probs = np.zeros(len(vi))
        for s in range(n_seeds):
            m = train_pinn(raw[ti], y[ti], raw[vi], y[vi],
                           lam_mono=lam_mono, seed=base_seed + s)
            te_probs += predict_pinn(m, raw[te])
            val_probs += predict_pinn(m, raw[vi])
        te_probs /= n_seeds
        val_probs /= n_seeds
        oof[te] = te_probs

        thr = _best_threshold(val_probs, y[vi]) if tune_threshold else 0.5
        pred = (te_probs >= thr).astype(int)
        rows.append({
            "fold": fold_idx,
            "accuracy": accuracy_score(y[te], pred),
            "f1": f1_score(y[te], pred, zero_division=0),
            "roc_auc": roc_auc_score(y[te], te_probs),
            "balanced_acc": balanced_accuracy_score(y[te], pred),
            "mcc": matthews_corrcoef(y[te], pred),
            "threshold": thr,
        })
        logger.info("PINN Fold %d/%d: F1=%.4f AUC=%.4f thr=%.2f",
                    fold_idx, n_splits, rows[-1]["f1"], rows[-1]["roc_auc"], thr)

    return rows, oof
