"""
PI-NAM: Physics-Informed Neural Additive Model for balloon flight suitability.

Realizes the per-factor suitability idea: each physical factor learns its OWN
shape function f_i(x_i) → a log-odds suitability contribution. The final
prediction is sigmoid(Σ_i f_i(x_i) + bias), making every factor's effect
directly interpretable as a 1-D suitability curve.

Physics-informed elements:
  - Physics-derived factors (air density, lift, gustiness, clearness) get
    their own shape functions alongside raw channels.
  - Monotonicity priors: shape functions for wind/cloud/precip/snow are
    penalized for positive slope (more of these never increases flyability).

This is a NAM (Agarwal et al., NeurIPS 2021) with physics features +
monotonicity — fully interpretable, strong additive inductive bias for small N.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.pinn import RAW_CHANNELS, IDX, physics_features, N_PHYS

logger = logging.getLogger(__name__)
torch.set_num_threads(4)

# Physics feature names (order matches physics_features output)
PHYS_NAMES = ["air_density", "density_altitude", "lift_capacity",
              "dewpoint_depression", "gustiness", "specific_humidity",
              "wind_energy", "cloud_clearness"]

# Factors whose suitability contribution must be monotonically NON-INCREASING
MONO_DEC_RAW = ["windgust", "windspeed", "cloudcover", "precip", "snow"]
MONO_DEC_PHYS = ["gustiness", "wind_energy"]          # higher => less flyable
MONO_INC_PHYS = ["cloud_clearness", "lift_capacity"]  # higher => more flyable


class ShapeNet(nn.Module):
    """1-D shape function f_i(x_i) → scalar log-odds contribution."""
    def __init__(self, hidden=24, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):           # x: (N,1)
        return self.net(x)          # (N,1)


class PINAM(nn.Module):
    """Physics-Informed Neural Additive Model.

    Inputs are RAW weather channels; physics features are computed internally.
    Each factor (raw + physics) has its own ShapeNet. Output is the sigmoid of
    the sum of contributions.
    """

    def __init__(self, raw_mean, raw_std, phys_mean, phys_std, hidden=24, dropout=0.1):
        super().__init__()
        self.register_buffer("raw_mean", raw_mean)
        self.register_buffer("raw_std", raw_std)
        self.register_buffer("phys_mean", phys_mean)
        self.register_buffer("phys_std", phys_std)
        self.n_raw = len(RAW_CHANNELS)
        self.shapes = nn.ModuleList(
            [ShapeNet(hidden, dropout) for _ in range(self.n_raw + N_PHYS)]
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def _contributions(self, raw):
        phys = physics_features(raw)
        rs = (raw - self.raw_mean) / self.raw_std
        ps = (phys - self.phys_mean) / self.phys_std
        feats = torch.cat([rs, ps], dim=1)          # (N, n_raw+N_PHYS)
        contribs = [self.shapes[i](feats[:, i:i + 1]) for i in range(feats.shape[1])]
        return torch.cat(contribs, dim=1)            # (N, n_factors) log-odds

    def forward(self, raw):
        c = self._contributions(raw)
        return torch.sigmoid(c.sum(dim=1) + self.bias).squeeze(-1)

    @torch.no_grad()
    def factor_contributions(self, raw):
        """Per-factor log-odds contributions (interpretability)."""
        return self._contributions(raw).cpu().numpy()


def _mono_penalty(model: PINAM, raw_batch: torch.Tensor) -> torch.Tensor:
    """Penalize wrong-sign slopes of shape functions (physical priors)."""
    raw_batch = raw_batch.clone().requires_grad_(True)
    c = model._contributions(raw_batch)               # (N, n_factors)
    total = c.sum()
    grads = torch.autograd.grad(total, raw_batch, create_graph=True)[0]  # ∂Σc/∂raw
    pen = 0.0
    for ch in MONO_DEC_RAW:
        pen = pen + torch.relu(grads[:, IDX[ch]]).mean()
    return pen


def train_pinam(raw_tr, y_tr, raw_val, y_val,
                lam_mono=0.05, lr=1e-3, weight_decay=1e-4,
                max_epochs=400, patience=40, batch_size=32, seed=42) -> PINAM:
    torch.manual_seed(seed)
    rm = torch.tensor(raw_tr.mean(0), dtype=torch.float32)
    rs = torch.tensor(raw_tr.std(0) + 1e-6, dtype=torch.float32)
    with torch.no_grad():
        ph = physics_features(torch.tensor(raw_tr))
    pm, ps = ph.mean(0).float(), (ph.std(0) + 1e-6).float()

    model = PINAM(rm, rs, pm, ps)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCELoss()
    Xtr = torch.tensor(raw_tr, dtype=torch.float32); ytr = torch.tensor(y_tr, dtype=torch.float32)
    Xva = torch.tensor(raw_val, dtype=torch.float32); yva = torch.tensor(y_val, dtype=torch.float32)

    best, bs, ni = 1e9, None, 0
    N = len(y_tr)
    for ep in range(max_epochs):
        model.train(); perm = torch.randperm(N)
        for b in range(0, N, batch_size):
            idx = perm[b:b + batch_size]
            loss = crit(model(Xtr[idx]), ytr[idx]) + lam_mono * _mono_penalty(model, Xtr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xva), yva).item()
        if vl < best - 1e-5:
            best, bs, ni = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            ni += 1
            if ni >= patience:
                break
    if bs:
        model.load_state_dict(bs)
    return model


@torch.no_grad()
def predict_pinam(model: PINAM, raw: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.tensor(raw, dtype=torch.float32)).numpy()


# ── NA²M: PI-NAM + physically-meaningful pairwise interactions ────────────────

# Physically-motivated interaction pairs (raw channel names)
INTERACTION_PAIRS = [
    ("cloudcover", "visibility"),    # overcast + poor visibility
    ("cloudcover", "windgust"),      # convective turbulence
    ("windgust", "windspeed"),       # gust factor
    ("windspeed", "winddir"),        # directional wind load
    ("temp", "humidity"),            # density / fog tendency
    ("humidity", "cloudcover"),      # moisture → cloud
    ("sealevelpressure", "windgust"),# pressure-system wind
    ("temp", "sealevelpressure"),    # air density
    ("visibility", "humidity"),      # fog
    ("cloudcover", "precip"),        # active weather
]


class Shape2D(nn.Module):
    """2-D interaction shape function f_ij(x_i, x_j) → log-odds contribution."""
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


# Physics-feature interaction pairs (indices into physics_features output)
PHYS_INTERACTION_PAIRS = [
    ("lift_capacity", "cloud_clearness"),     # buoyancy × clear sky
    ("air_density", "wind_energy"),           # density × wind load
    ("dewpoint_depression", "cloud_clearness"),  # fog risk × clarity
    ("gustiness", "wind_energy"),             # turbulence
    ("density_altitude", "lift_capacity"),    # altitude vs lift
    ("specific_humidity", "cloud_clearness"), # moisture × sky
]


class NA2M(PINAM):
    """PI-NAM + pairwise interaction terms (NA²M).

    Args:
        use_phys_inter: if True, add physics-feature interaction shape nets.
        inter_hidden: width of interaction shape nets.
    """

    def __init__(self, raw_mean, raw_std, phys_mean, phys_std,
                 hidden=24, dropout=0.1, inter_hidden=16, inter_dropout=0.2,
                 use_phys_inter=False):
        super().__init__(raw_mean, raw_std, phys_mean, phys_std, hidden, dropout)
        self.pairs = [(IDX[a], IDX[b]) for a, b in INTERACTION_PAIRS]
        self.inters = nn.ModuleList(
            [Shape2D(inter_hidden, inter_dropout) for _ in self.pairs]
        )
        self.use_phys_inter = use_phys_inter
        if use_phys_inter:
            self.phys_pairs = [(PHYS_NAMES.index(a), PHYS_NAMES.index(b))
                               for a, b in PHYS_INTERACTION_PAIRS]
            self.phys_inters = nn.ModuleList(
                [Shape2D(inter_hidden, inter_dropout) for _ in self.phys_pairs]
            )

    def _contributions(self, raw):
        main = super()._contributions(raw)               # (N, n_factors)
        rs = (raw - self.raw_mean) / self.raw_std
        inter_c = []
        for (i, j), net in zip(self.pairs, self.inters):
            inter_c.append(net(torch.stack([rs[:, i], rs[:, j]], dim=1)))
        if self.use_phys_inter:
            phys = physics_features(raw)
            ps = (phys - self.phys_mean) / self.phys_std
            for (i, j), net in zip(self.phys_pairs, self.phys_inters):
                inter_c.append(net(torch.stack([ps[:, i], ps[:, j]], dim=1)))
        inter = torch.cat(inter_c, dim=1)
        return torch.cat([main, inter], dim=1)


def train_na2m(raw_tr, y_tr, raw_val, y_val,
               lam_mono=0.05, lr=1e-3, weight_decay=3e-4,
               max_epochs=400, patience=40, batch_size=32, seed=42,
               inter_hidden=16, use_phys_inter=False) -> NA2M:
    torch.manual_seed(seed)
    rm = torch.tensor(raw_tr.mean(0), dtype=torch.float32)
    rs = torch.tensor(raw_tr.std(0) + 1e-6, dtype=torch.float32)
    with torch.no_grad():
        ph = physics_features(torch.tensor(raw_tr))
    pm, ps = ph.mean(0).float(), (ph.std(0) + 1e-6).float()

    model = NA2M(rm, rs, pm, ps, inter_hidden=inter_hidden, use_phys_inter=use_phys_inter)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCELoss()
    Xtr = torch.tensor(raw_tr, dtype=torch.float32); ytr = torch.tensor(y_tr, dtype=torch.float32)
    Xva = torch.tensor(raw_val, dtype=torch.float32); yva = torch.tensor(y_val, dtype=torch.float32)
    best, bs, ni = 1e9, None, 0
    N = len(y_tr)
    for ep in range(max_epochs):
        model.train(); perm = torch.randperm(N)
        for b in range(0, N, batch_size):
            idx = perm[b:b + batch_size]
            loss = crit(model(Xtr[idx]), ytr[idx]) + lam_mono * _mono_penalty(model, Xtr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xva), yva).item()
        if vl < best - 1e-5:
            best, bs, ni = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            ni += 1
            if ni >= patience:
                break
    if bs:
        model.load_state_dict(bs)
    return model


def run_na2m_cv(raw, y, cfg, lam_mono=0.05, n_seeds=5, tune_threshold=True,
                inter_hidden=16, use_phys_inter=False):
    """10-fold CV: seed-ensemble NA²M with inner-val threshold tuning."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                  f1_score, matthews_corrcoef, roc_auc_score)
    n_splits = cfg["features"]["test_folds"]; base_seed = cfg["features"]["random_seed"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed)

    def best_t(p, yy):
        bt, bf = 0.5, 0.0
        for t in np.arange(0.20, 0.80, 0.01):
            f = f1_score(yy, (p >= t).astype(int), zero_division=0)
            if f > bf:
                bf, bt = f, float(t)
        return bt

    oof = np.zeros(len(y)); rows = []
    for fi, (tr, te) in enumerate(skf.split(raw, y), start=1):
        nval = max(int(len(tr) * 0.2), 10); vi, ti = tr[:nval], tr[nval:]
        tep = np.zeros(len(te)); vap = np.zeros(len(vi))
        for s in range(n_seeds):
            m = train_na2m(raw[ti], y[ti], raw[vi], y[vi], lam_mono=lam_mono,
                           seed=base_seed + s, inter_hidden=inter_hidden,
                           use_phys_inter=use_phys_inter)
            tep += predict_pinam(m, raw[te]); vap += predict_pinam(m, raw[vi])
        tep /= n_seeds; vap /= n_seeds; oof[te] = tep
        thr = best_t(vap, y[vi]) if tune_threshold else 0.5
        pred = (tep >= thr).astype(int)
        rows.append({"fold": fi, "f1": f1_score(y[te], pred, zero_division=0),
                     "roc_auc": roc_auc_score(y[te], tep),
                     "balanced_acc": balanced_accuracy_score(y[te], pred),
                     "mcc": matthews_corrcoef(y[te], pred), "threshold": thr})
        logger.info("NA2M Fold %d/%d: F1=%.4f AUC=%.4f thr=%.2f",
                    fi, n_splits, rows[-1]["f1"], rows[-1]["roc_auc"], thr)
    return rows, oof


# ── Official PI-NAM entry point (proposed model) ──────────────────────────────
# PI-NAM = physics-informed neural additive model with monotonicity priors and
# physics-feature pairwise interactions, deep-ensembled over seeds. This is the
# configuration reported in the paper.

PINAM_CONFIG = dict(lam_mono=0.03, n_seeds=10, inter_hidden=16, use_phys_inter=True)


def run_pinam_cv(
    raw: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    save_dir: Optional["Path"] = None,
    **overrides,
) -> Tuple[List[dict], np.ndarray, List[dict]]:
    """10-fold CV for the proposed PI-NAM model (paper configuration).

    Args:
        raw: Raw weather matrix (N, 16) in RAW_CHANNELS order.
        y: Binary labels (N,).
        cfg: Config dict (features.test_folds, features.random_seed).
        save_dir: If set, save per-fold reproducibility bundle (seed weights,
                  standardization stats, threshold, indices).
        **overrides: Override any PINAM_CONFIG entry.

    Returns:
        (per_fold_metrics, oof_probs, threshold_log)
    """
    import pickle
    from pathlib import Path
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                  f1_score, matthews_corrcoef, roc_auc_score)

    conf = {**PINAM_CONFIG, **overrides}
    lam_mono, n_seeds = conf["lam_mono"], conf["n_seeds"]
    inter_hidden, use_phys_inter = conf["inter_hidden"], conf["use_phys_inter"]

    n_splits = cfg["features"]["test_folds"]
    base_seed = cfg["features"]["random_seed"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed)

    out_dir: Optional[Path] = Path(save_dir) if save_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    def best_t(p, yy):
        bt, bf = 0.5, 0.0
        for t in np.arange(0.20, 0.80, 0.01):
            f = f1_score(yy, (p >= t).astype(int), zero_division=0)
            if f > bf:
                bf, bt = f, float(t)
        return bt

    oof = np.zeros(len(y)); rows = []; thr_log = []
    for fi, (tr, te) in enumerate(skf.split(raw, y), start=1):
        nval = max(int(len(tr) * 0.2), 10); vi, ti = tr[:nval], tr[nval:]
        tep = np.zeros(len(te)); vap = np.zeros(len(vi))
        seed_states = []
        for s in range(n_seeds):
            m = train_na2m(raw[ti], y[ti], raw[vi], y[vi], lam_mono=lam_mono,
                           seed=base_seed + s, inter_hidden=inter_hidden,
                           use_phys_inter=use_phys_inter)
            tep += predict_pinam(m, raw[te]); vap += predict_pinam(m, raw[vi])
            seed_states.append({k: v.cpu() for k, v in m.state_dict().items()})
        tep /= n_seeds; vap /= n_seeds; oof[te] = tep
        thr = best_t(vap, y[vi])
        pred = (tep >= thr).astype(int)
        rows.append({"fold": fi, "f1": f1_score(y[te], pred, zero_division=0),
                     "roc_auc": roc_auc_score(y[te], tep),
                     "accuracy": accuracy_score(y[te], pred),
                     "balanced_acc": balanced_accuracy_score(y[te], pred),
                     "mcc": matthews_corrcoef(y[te], pred), "threshold": thr})
        thr_log.append({"fold": fi, "threshold": thr})
        logger.info("PI-NAM Fold %d/%d: F1=%.4f AUC=%.4f BAcc=%.4f thr=%.2f",
                    fi, n_splits, rows[-1]["f1"], rows[-1]["roc_auc"],
                    rows[-1]["balanced_acc"], thr)

        if out_dir is not None:
            with open(out_dir / f"fold_{fi:02d}.pkl", "wb") as fh:
                pickle.dump({
                    "fold_idx": fi,
                    "train_idx": tr.tolist(), "test_idx": te.tolist(),
                    "seed_states": seed_states, "n_seeds": n_seeds,
                    "inter_hidden": inter_hidden, "use_phys_inter": use_phys_inter,
                    "threshold": thr, "config": conf,
                }, fh, protocol=4)

    return rows, oof, thr_log


@torch.no_grad()
def predict_from_pinam_bundle(bundle: dict, raw: np.ndarray):
    """Reproduce PI-NAM predictions from a saved fold bundle (no retraining).

    Args:
        bundle: dict from fold_XX.pkl (seed_states, inter_hidden, use_phys_inter, threshold).
        raw: Raw weather matrix (N, 16).

    Returns:
        (y_prob, y_pred) using the saved per-fold threshold.
    """
    n_raw = len(RAW_CHANNELS)
    dummy_r = torch.zeros(n_raw); dummy_r1 = torch.ones(n_raw)
    dummy_p = torch.zeros(N_PHYS); dummy_p1 = torch.ones(N_PHYS)
    probs = np.zeros(len(raw))
    for state in bundle["seed_states"]:
        m = NA2M(dummy_r, dummy_r1, dummy_p, dummy_p1,
                 inter_hidden=bundle["inter_hidden"],
                 use_phys_inter=bundle["use_phys_inter"])
        m.load_state_dict(state)
        probs += predict_pinam(m, raw)
    probs /= len(bundle["seed_states"])
    return probs, (probs >= bundle["threshold"]).astype(int)
