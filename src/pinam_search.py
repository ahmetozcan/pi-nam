"""
PI-NAM-X: structure/feature search for the interpretable additive model.

Rather than fixing PI-NAM's factor set by hand, we define a domain-grounded
candidate feature space and let Optuna SELECT which feature families and
hyperparameters yield the best deployable F1 — the same fair optimization the
baselines received. Candidate factors:

  * raw-16 channels                         (always included)
  * 8 differentiable physics features        (toggle)
  * multi-scale temporal "rules" on 7 key channels:
       stats {mean, std, trend} x scales {3, 7, 14}   (subset selected)
  * pairwise physics/raw interactions        (toggle)

Each selected factor keeps its own interpretable shape function; monotonicity
priors constrain wind/cloud/precip. The searched space + best spec are saved
for transparent reporting.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.pinn import RAW_CHANNELS, IDX, physics_features, N_PHYS
from src.pinam import ShapeNet, Shape2D, PHYS_NAMES

# Physically-motivated interaction pairs on current-day raw channels
INTERACTION_PAIRS = [
    ("cloudcover", "visibility"), ("cloudcover", "windgust"),
    ("windgust", "windspeed"), ("temp", "humidity"),
    ("sealevelpressure", "windgust"), ("humidity", "cloudcover"),
]

logger = logging.getLogger(__name__)
torch.set_num_threads(4)

WINDOW = 14
TEMP_CHANNELS = ["windgust", "windspeed", "cloudcover", "visibility",
                 "humidity", "temp", "sealevelpressure"]
TKI = [RAW_CHANNELS.index(c) for c in TEMP_CHANNELS]
MONO_DEC = ["windgust", "windspeed", "cloudcover", "precip", "snow"]


# ── Feature construction (the searchable candidate space) ────────────────────

def build_windows(raw, W=WINDOW):
    N = len(raw); win = np.zeros((N, W, raw.shape[1]), dtype=np.float32)
    for t in range(N):
        lo = max(0, t - W + 1); s = raw[lo:t + 1]
        if len(s) < W:
            s = np.vstack([np.repeat(s[:1], W - len(s), 0), s])
        win[t] = s
    return win


def temporal_block(win, stats, scales):
    """Differentiable temporal aggregates for selected stats/scales on key channels.
    Stats: mean, std, max, min (per scale) + trend (scale-free).
    win: (B,W,16) tensor.  Returns (B, n) and feature names."""
    wk = win[:, :, TKI]
    feats, names = [], []
    for sc in scales:
        seg = wk[:, -sc:, :]
        if "mean" in stats:
            feats.append(seg.mean(1)); names += [f"{c}_mean{sc}" for c in TEMP_CHANNELS]
        if "std" in stats:
            feats.append(torch.sqrt(seg.var(1, unbiased=False) + 1e-6)); names += [f"{c}_std{sc}" for c in TEMP_CHANNELS]
        if "max" in stats:
            feats.append(seg.amax(1)); names += [f"{c}_max{sc}" for c in TEMP_CHANNELS]
        if "min" in stats:
            feats.append(seg.amin(1)); names += [f"{c}_min{sc}" for c in TEMP_CHANNELS]
    if "trend" in stats:
        feats.append(wk[:, -1, :] - wk[:, -3:, :].mean(1)); names += [f"{c}_trend" for c in TEMP_CHANNELS]
    if feats:
        return torch.cat(feats, dim=1), names
    return None, []


N_SYNOPTIC = 10
SYNOPTIC_NAMES = ["pressure_tendency", "pressure_std", "frontal_passage",
                  "clearness_trend", "dewdepr_trend", "winddir_steadiness",
                  "recent_storminess", "fog_risk", "diurnal_range", "thermal_turbulence"]


def synoptic_physics(win, scale=3):
    """Differentiable multi-day synoptic + composite physics from window (B,W,16).

    scale: lookback (days) for tendencies/aggregates. Returns (B, 10):
      [pressure_tendency, pressure_std, frontal_passage, clearness_trend,
       dewdepr_trend, winddir_steadiness, recent_storminess, fog_risk,
       diurnal_range, thermal_turbulence].
    """
    sc = min(scale, win.shape[1])
    cur = win[:, -1, :]
    past = win[:, -sc, :]
    seg = win[:, -sc:, :]
    P, P0 = cur[:, IDX["sealevelpressure"]], past[:, IDX["sealevelpressure"]]
    Pseg = seg[:, :, IDX["sealevelpressure"]]
    cloud, cloud0 = cur[:, IDX["cloudcover"]], past[:, IDX["cloudcover"]]
    clr = (100.0 - cloud) / 100.0
    clr_seg = (100.0 - seg[:, :, IDX["cloudcover"]]) / 100.0
    RH, RH0 = cur[:, IDX["humidity"]], past[:, IDX["humidity"]]
    dpdepr, dpdepr0 = (100.0 - RH) / 5.0, (100.0 - RH0) / 5.0   # T - dewpoint
    ws = cur[:, IDX["windspeed"]]
    p_tend = P - P0                                              # barometric tendency
    p_std = torch.sqrt(Pseg.var(1, unbiased=False) + 1e-6)       # synoptic variability
    frontal = torch.relu(-(p_tend)) * torch.relu(cloud - cloud0)  # falling P + clouds up
    clr_trend = clr - clr_seg.mean(1)
    dpdepr_trend = dpdepr - dpdepr0
    rad = seg[:, :, IDX["winddir"]] * (np.pi / 180.0)
    steady = torch.sqrt(torch.cos(rad).mean(1) ** 2 + torch.sin(rad).mean(1) ** 2 + 1e-9)
    storm = seg[:, :, IDX["windgust"]].amax(1)                   # peak recent gust
    fog = clr * (1.0 / (dpdepr + 1.0)) * (1.0 / (ws + 1.0))      # radiative fog risk
    diurnal = cur[:, IDX["tempmax"]] - cur[:, IDX["tempmin"]]    # stability/clarity proxy
    thermal = cur[:, IDX["tempmax"]] * clr                       # convective turbulence
    return torch.stack([p_tend, p_std, frontal, clr_trend, dpdepr_trend, steady,
                        storm, fog, diurnal, thermal], dim=1)


class DeepShapeNet(nn.Module):
    """1-D shape function with configurable depth (number of hidden layers)."""
    def __init__(self, hidden=24, dropout=0.1, depth=2):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.ReLU()]
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Dropout(dropout), nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PINAMX(nn.Module):
    """Flexible additive model; factor set determined by `spec`."""

    def __init__(self, feat_mean, feat_std, n_factors, spec, hidden=24, dropout=0.1,
                 raw_mean=None, raw_std=None, depth=2):
        super().__init__()
        self.register_buffer("feat_mean", feat_mean)
        self.register_buffer("feat_std", feat_std)
        self.spec = spec
        self.shapes = nn.ModuleList([DeepShapeNet(hidden, dropout, depth) for _ in range(n_factors)])
        self.use_inter = bool(spec.get("interactions", False))
        if self.use_inter:
            self.pairs = [(IDX[a], IDX[b]) for a, b in INTERACTION_PAIRS]
            self.inters = nn.ModuleList([Shape2D(16, 0.2) for _ in self.pairs])
            self.register_buffer("raw_mean", raw_mean if raw_mean is not None else torch.zeros(len(RAW_CHANNELS)))
            self.register_buffer("raw_std", raw_std if raw_std is not None else torch.ones(len(RAW_CHANNELS)))
        self.bias = nn.Parameter(torch.zeros(1))

    def _raw_factors(self, win):
        cur = win[:, -1, :]                       # raw-16 (always)
        parts = [cur]
        if self.spec["physics"]:
            parts.append(physics_features(cur))
        if self.spec["stats"]:
            tb, _ = temporal_block(win, self.spec["stats"], self.spec["scales"])
            if tb is not None:
                parts.append(tb)
        if self.spec.get("synoptic"):
            parts.append(synoptic_physics(win, self.spec.get("syn_scale", 3)))
        return torch.cat(parts, dim=1)

    def _contributions(self, win):
        f = (self._raw_factors(win) - self.feat_mean) / self.feat_std
        f = torch.nan_to_num(f, 0.0, 10.0, -10.0).clamp(-10, 10)
        contribs = [self.shapes[i](f[:, i:i + 1]) for i in range(len(self.shapes))]
        if self.use_inter:
            rs = (win[:, -1, :] - self.raw_mean) / self.raw_std
            rs = torch.nan_to_num(rs, 0.0, 10.0, -10.0).clamp(-10, 10)
            for (i, j), net in zip(self.pairs, self.inters):
                contribs.append(net(torch.stack([rs[:, i], rs[:, j]], dim=1)))
        return torch.cat(contribs, dim=1)

    def forward(self, win):
        return torch.sigmoid(self._contributions(win).sum(1) + self.bias).squeeze(-1)


def _n_factors(spec):
    """Number of additive 1-D factors (raw + physics + per-scale temporal stats + trend)."""
    n = len(RAW_CHANNELS)
    if spec["physics"]:
        n += N_PHYS
    per_scale = sum(s in spec["stats"] for s in ("mean", "std", "max", "min"))
    nstat_ch = per_scale * len(spec["scales"]) * len(TEMP_CHANNELS)
    nstat_ch += len(TEMP_CHANNELS) if "trend" in spec["stats"] else 0
    n += nstat_ch
    if spec.get("synoptic"):
        n += N_SYNOPTIC
    return n


def _feat_stats(win_tr, spec):
    with torch.no_grad():
        m = PINAMX(torch.zeros(_n_factors(spec)), torch.ones(_n_factors(spec)), _n_factors(spec), spec)
        F = m._raw_factors(torch.tensor(win_tr, dtype=torch.float32)).numpy()
    return F.mean(0).astype(np.float32), (F.std(0) + 1e-6).astype(np.float32)


def _mono_penalty(model, win):
    win = win.clone().requires_grad_(True)
    p = model(win)
    g = torch.autograd.grad(p.sum(), win, create_graph=True)[0][:, -1, :]
    return sum(torch.relu(g[:, IDX[c]]).mean() for c in MONO_DEC)


def train_pinamx(win_tr, y_tr, win_val, y_val, spec, lam_mono=0.03, hidden=24,
                 dropout=0.1, lr=5e-4, weight_decay=3e-4, max_epochs=400,
                 patience=40, batch_size=32, seed=42, depth=2):
    torch.manual_seed(seed)
    fm, fs = _feat_stats(win_tr, spec)
    cur = win_tr[:, -1, :]
    rm = torch.tensor(cur.mean(0), dtype=torch.float32)
    rsd = torch.tensor(cur.std(0) + 1e-6, dtype=torch.float32)
    model = PINAMX(torch.tensor(fm), torch.tensor(fs), _n_factors(spec), spec, hidden, dropout,
                   raw_mean=rm, raw_std=rsd, depth=depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCELoss()
    Xtr = torch.tensor(win_tr, dtype=torch.float32); ytr = torch.tensor(y_tr, dtype=torch.float32)
    Xva = torch.tensor(win_val, dtype=torch.float32); yva = torch.tensor(y_val, dtype=torch.float32)
    best, bs, ni = 1e9, None, 0
    N = len(y_tr)
    for ep in range(max_epochs):
        model.train(); perm = torch.randperm(N)
        for b in range(0, N, batch_size):
            idx = perm[b:b + batch_size]
            pred = model(Xtr[idx]).clamp(1e-6, 1 - 1e-6)
            loss = crit(pred, ytr[idx]) + lam_mono * _mono_penalty(model, Xtr[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xva).clamp(1e-6, 1 - 1e-6), yva).item()
        if not np.isfinite(vl):
            break
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
def predict_pinamx(model, win):
    model.eval()
    return model(torch.tensor(win, dtype=torch.float32)).numpy()


def run_pinamx_cv(raw, y, cfg, spec, lam_mono=0.03, hidden=24, dropout=0.1,
                  n_seeds=5, save_dir: Optional[Path] = None):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                  matthews_corrcoef, roc_auc_score)
    import pickle
    n_splits = cfg["features"]["test_folds"]; base_seed = cfg["features"]["random_seed"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed)
    win_all = build_windows(raw)
    out_dir = Path(save_dir) if save_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    def bt(p, yy):
        b, bf = 0.5, 0.0
        for t in np.arange(0.2, 0.8, 0.005):
            f = f1_score(yy, (p >= t).astype(int), zero_division=0)
            if f > bf:
                bf, b = f, float(t)
        return b

    oof = np.zeros(len(y)); rows = []
    for fi, (tr, te) in enumerate(skf.split(raw, y), 1):
        nval = max(int(len(tr) * 0.2), 10); vi, ti = tr[:nval], tr[nval:]
        tep = np.zeros(len(te)); vap = np.zeros(len(vi)); states = []
        for s in range(n_seeds):
            m = train_pinamx(win_all[ti], y[ti], win_all[vi], y[vi], spec,
                             lam_mono=lam_mono, hidden=hidden, dropout=dropout, seed=base_seed + s)
            tep += predict_pinamx(m, win_all[te]); vap += predict_pinamx(m, win_all[vi])
            states.append({k: v.cpu() for k, v in m.state_dict().items()})
        tep /= n_seeds; vap /= n_seeds; oof[te] = tep
        thr = bt(vap, y[vi]); pred = (tep >= thr).astype(int)
        rows.append({"fold": fi, "f1": f1_score(y[te], pred, zero_division=0),
                     "roc_auc": roc_auc_score(y[te], tep),
                     "accuracy": accuracy_score(y[te], pred),
                     "balanced_acc": balanced_accuracy_score(y[te], pred),
                     "mcc": matthews_corrcoef(y[te], pred), "threshold": thr})
        if out_dir:
            with open(out_dir / f"fold_{fi:02d}.pkl", "wb") as fh:
                pickle.dump({"fold_idx": fi, "train_idx": tr.tolist(), "test_idx": te.tolist(),
                             "seed_states": states, "n_seeds": n_seeds, "threshold": thr,
                             "spec": spec, "hidden": hidden, "dropout": dropout,
                             "window": WINDOW}, fh, protocol=4)
    return rows, oof


def _fold_seed_worker(win_ti, y_ti, win_vi, y_vi, win_te, spec, lam, hidden,
                      dropout, lr, wd, bs, seed, depth=2):
    """Train one (fold, seed) model in a worker process; return test/val probs + state."""
    import torch
    torch.set_num_threads(1)
    m = train_pinamx(win_ti, y_ti, win_vi, y_vi, spec, lam_mono=lam, hidden=hidden,
                     dropout=dropout, lr=lr, weight_decay=wd, batch_size=bs, seed=seed, depth=depth)
    return (predict_pinamx(m, win_te), predict_pinamx(m, win_vi),
            {k: v.cpu() for k, v in m.state_dict().items()})


def run_pinamx_cv_parallel(raw, y, cfg, spec, lam_mono=0.03, hidden=24, dropout=0.1,
                           lr=5e-4, weight_decay=3e-4, batch_size=32, n_seeds=5,
                           n_jobs=14, depth=2, save_dir: Optional[Path] = None):
    """Parallel 10-fold CV: distributes (fold, seed) trainings across cores."""
    from joblib import Parallel, delayed
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                  matthews_corrcoef, roc_auc_score)
    import pickle
    n_splits = cfg["features"]["test_folds"]; base_seed = cfg["features"]["random_seed"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed)
    win_all = build_windows(raw)
    out_dir = Path(save_dir) if save_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    folds = list(skf.split(raw, y))
    tasks, meta = [], []
    for fi, (tr, te) in enumerate(folds, 1):
        nval = max(int(len(tr) * 0.2), 10); vi, ti = tr[:nval], tr[nval:]
        for s in range(n_seeds):
            tasks.append(delayed(_fold_seed_worker)(
                win_all[ti], y[ti], win_all[vi], y[vi], win_all[te], spec,
                lam_mono, hidden, dropout, lr, weight_decay, batch_size, base_seed + s, depth))
            meta.append((fi, te, vi))
    results = Parallel(n_jobs=n_jobs, backend="loky")(tasks)

    # Aggregate by fold
    from collections import defaultdict
    fold_te = defaultdict(lambda: None); fold_va = defaultdict(lambda: None)
    fold_states = defaultdict(list); fold_idx_map = {}
    for (fi, te, vi), (tep, vap, st) in zip(meta, results):
        fold_te[fi] = (fold_te[fi][0] + tep, te) if fold_te[fi] else (tep, te)
        fold_va[fi] = (fold_va[fi][0] + vap, vi) if fold_va[fi] else (vap, vi)
        fold_states[fi].append(st); fold_idx_map[fi] = (te, vi)

    def bt(p, yy):
        b, bf = 0.5, 0.0
        for t in np.arange(0.2, 0.8, 0.005):
            f = f1_score(yy, (p >= t).astype(int), zero_division=0)
            if f > bf:
                bf, b = f, float(t)
        return b

    oof = np.zeros(len(y)); rows = []
    for fi in sorted(fold_te):
        tep, te = fold_te[fi]; vap, vi = fold_va[fi]
        tep = tep / n_seeds; vap = vap / n_seeds; oof[te] = tep
        thr = bt(vap, y[vi]); pred = (tep >= thr).astype(int)
        rows.append({"fold": fi, "f1": f1_score(y[te], pred, zero_division=0),
                     "roc_auc": roc_auc_score(y[te], tep),
                     "accuracy": accuracy_score(y[te], pred),
                     "balanced_acc": balanced_accuracy_score(y[te], pred),
                     "mcc": matthews_corrcoef(y[te], pred), "threshold": thr})
        if out_dir:
            with open(out_dir / f"fold_{fi:02d}.pkl", "wb") as fh:
                pickle.dump({"fold_idx": fi, "train_idx": fold_idx_map[fi][0].tolist(),
                             "test_idx": te.tolist(), "seed_states": fold_states[fi],
                             "n_seeds": n_seeds, "threshold": thr, "spec": spec,
                             "hidden": hidden, "dropout": dropout, "depth": depth,
                             "window": WINDOW}, fh, protocol=4)
    return rows, oof


def deployable_f1(oof, y, n_splits, seed):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    def bt(p, yy):
        b, bf = 0.5, 0.0
        for t in np.arange(0.2, 0.8, 0.005):
            f = f1_score(yy, (p >= t).astype(int), zero_division=0)
            if f > bf:
                bf, b = f, float(t)
        return b
    fs = []
    for tr, te in skf.split(oof, y):
        t = bt(oof[tr], y[tr]); fs.append(f1_score(y[te], (oof[te] >= t).astype(int), zero_division=0))
    return float(np.mean(fs))
