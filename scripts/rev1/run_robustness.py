"""Stage 7 robustness (R1): missing weather variables and record length, S0.

  inference  trained models (PI-NAM: saved lambda = 10 grid ensembles; tabular: refitted with
             the grid hyperparameters, identical to the grid fits) predict the test rows
             under every prediction-time corruption of src/revision/robustness.py
             (group masked by the training mean; 10/20/30 % random cells, mean or
             carry-forward imputation, 5 repetitions); thresholds stay those of the clean fit
  drop       every model re-fitted with one variable group removed (constant)
  curve      learning curve: every model re-fitted on a stratified 25/50/75 % of each
             outer training set (year: 3 repetitions; block: 1), hyperparameters unchanged
  reverse    train on 2024, test on 2023 (baselines and XGBoost-mono tuned on 2024 only;
             PI-NAM: V1 architecture, training hyperparameters tuned on 2024, lambda = 10,
             i.e. the mirror of PI-NAM-fixed)

PI-NAM is the V1 architecture with lambda = 10 throughout (block: grid PI-NAM; year:
PI-NAM-fixed; the nested year PI-NAM is added to the inference experiment, which needs no
training). Every PI-NAM fit is a 15-seed ensemble with the inner-CV threshold.

Outputs under outputs/rev1/robustness/: inference/, drop/, curve/, reverse/ (+ tuning/).

Run from the repository root:
  python3 scripts/rev1/run_robustness.py --smoke
  python3 scripts/rev1/run_robustness.py --experiments inference drop curve reverse
"""

import argparse
import dataclasses
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_extra import extra_params, grid_params  # noqa: E402
from run_grid import _slug, save_results  # noqa: E402
from src.revision.extra_models import (feature_fn, fit_tabular, fit_with_threshold,  # noqa: E402
                                       load_ensemble, tune_extra)
from src.revision.info_sets import load_inputs  # noqa: E402
from src.revision.models import BASELINES, PINAM_V1, PinamConfig, fit_pinam  # noqa: E402
from src.revision.protocols import make_splits, split_year  # noqa: E402
from src.revision.robustness import (GROUPS, drop_group_windows, inference_conditions,  # noqa: E402
                                     raw_series, subsample_splits)
from src.revision.tuning import load_tuning, save_tuning, tune_baselines, tune_pinam  # noqa: E402

logger = logging.getLogger("run_robustness")
GRID = ROOT / "outputs/rev1/grid"
FINAL_LAM = 10.0
TABULAR = BASELINES + ("EBM", "XGBoost-mono")
MODELS = ("PI-NAM",) + TABULAR
EXPERIMENTS = ("inference", "drop", "curve", "reverse")
CURVE_FRACS = (0.25, 0.5, 0.75)
CURVE_REPS = {"year": 3, "block": 1}
# grid file of the PI-NAM used per protocol (V1 architecture, lambda = 10)
PINAM_GRID = {"block": "pinam_thrcv_lam10", "year": "pinamfixed_thrcv_lam10"}


def _pinam_cfg(proto: str, stem: str = None) -> PinamConfig:
    meta = json.loads(str(np.load(GRID / f"S0_{proto}_{stem or PINAM_GRID[proto]}.npz")["meta"]))
    return PinamConfig(**meta["params"])


def tabular_params(proto: str, inp, args) -> Dict[str, Dict]:
    """Grid hyperparameters of the original baselines + stage 6 tuning of the extra models."""
    ns = argparse.Namespace(out=ROOT / "outputs/rev1/extra", models=["XGBoost-mono"],
                            trials=40, smoke=False)
    return grid_params("S0", proto) | {"EBM": {}} | extra_params("S0", proto, inp, ns)


# ── inference-time corruption ────────────────────────────────────────────────

def _tab_inference(name, params, win, y, split, conds: Dict[str, np.ndarray]):
    feat = feature_fn(name)
    model, thr, _ = fit_with_threshold(name, params, feat(win), y, split.train)
    return thr, np.stack([model.predict_proba(feat(w))[:, 1] for w in conds.values()])


def _save_inference(path: Path, splits, y, thr, P, names, meta) -> None:
    test = np.concatenate([s.test for s in splits])
    np.savez(path, test=test, y=y[test], conditions=np.array(names),
             p=np.concatenate(P, axis=1), thr=np.concatenate([np.full(len(s.test), t)
                                                               for s, t in zip(splits, thr)]),
             split=np.concatenate([np.full(len(s.test), s.name) for s in splits]),
             meta=json.dumps(meta))


def run_inference(inp, proto: str, splits, args) -> None:
    raw = raw_series(inp)
    conds = [dict(inference_conditions(raw, s.train, s.test, n_reps=args.reps)) for s in splits]
    names = list(conds[0])
    params = tabular_params(proto, inp, args)
    pinams = {"PI-NAM": PINAM_GRID[proto]}
    if proto == "year":
        pinams = {"PI-NAM-fixed": "pinamfixed_thrcv_lam10", "PI-NAM": "pinam_thrcv_lam10"}
    for model in [m for m in args.models if m in MODELS]:
        for label in (pinams if model == "PI-NAM" else [model]):
            out = args.out / "inference" / f"S0_{proto}_{_slug(label)}{args.tail}.npz"
            if out.exists() and not args.overwrite:
                logger.info("skip %s (exists)", out.name)
                continue
            t0 = time.time()
            if model == "PI-NAM":
                stem = pinams[label]
                cfg = _pinam_cfg(proto, stem)
                saved = torch.load(GRID / "states" / f"S0_{proto}_{stem}.pt", weights_only=False)
                d = np.load(GRID / f"S0_{proto}_{stem}.npz")
                thr, P = [], []
                for s, c in zip(splits, conds):
                    pred = load_ensemble(cfg, saved[s.name]["states"])
                    thr.append(float(d["thr"][d["split"] == s.name][0]))
                    P.append(np.stack([pred(w) for w in c.values()]))
                meta = {"model": label, "source": stem}
            else:
                jobs = 1 if model == "TabPFN" else min(args.n_jobs, len(splits))
                res = Parallel(n_jobs=jobs)(delayed(_tab_inference)(model, params.get(model, {}),
                                                                    inp.win, inp.y, s, c)
                                            for s, c in zip(splits, conds))
                thr, P = [r[0] for r in res], [r[1] for r in res]
                meta = {"model": model, "params": params.get(model, {})}
            _save_inference(out, splits, inp.y, thr, P, names, {"protocol": proto, **meta})
            logger.info("inference %s %-14s done in %.0fs", proto, label, time.time() - t0)


# ── re-fitting experiments ───────────────────────────────────────────────────

def _fit(model, cfg, params, win, y, splits, args):
    if model == "PI-NAM":
        return fit_pinam(cfg, win, y, splits, n_seeds=args.n_seeds, n_jobs=args.n_jobs,
                         keep_states=False, threshold="inner_cv"), cfg.__dict__
    res = fit_tabular(model, params.get(model, {}), win, y, splits)
    for r in res:
        r.extra.pop("fd")
    return res, params.get(model, {})


def _fit_and_save(out: Path, model, cfg, params, win, y, splits, args, meta) -> None:
    if out.exists() and not args.overwrite:
        logger.info("skip %s (exists)", out.name)
        return
    t0 = time.time()
    res, p = _fit(model, cfg, params, win, y, splits, args)
    save_results(out, res, y, {"model": model, "params": p, "threshold_policy": "inner_cv", **meta})
    logger.info("%s done in %.0fs", out.name, time.time() - t0)


def run_drop(inp, proto, splits, args) -> None:
    raw, params, cfg = raw_series(inp), tabular_params(proto, inp, args), _pinam_cfg(proto)
    for g in args.groups:
        win = drop_group_windows(raw, g)
        for model in args.models:
            _fit_and_save(args.out / "drop" / f"S0_{proto}_{_slug(model)}_no{g}{args.tail}.npz",
                          model, cfg, params, win, inp.y, splits, args,
                          {"protocol": proto, "dropped": g, "channels": GROUPS[g]})


def run_curve(inp, proto, splits, args) -> None:
    params, cfg = tabular_params(proto, inp, args), _pinam_cfg(proto)
    for frac in args.fracs:
        for k in range(1 if args.smoke else CURVE_REPS[proto]):
            sub = subsample_splits(splits, inp.y, frac, seed=42 + k)
            for model in args.models:
                _fit_and_save(args.out / "curve" /
                              f"S0_{proto}_{_slug(model)}_f{int(100 * frac)}_r{k}{args.tail}.npz",
                              model, cfg, params, inp.win, inp.y, sub, args,
                              {"protocol": proto, "frac": frac, "rep": k,
                               "n_train": [int(len(s.train)) for s in sub]})


def reverse_tuning(inp, rows, args):
    """Baselines, XGBoost-mono and PI-NAM (fixed V1 architecture) tuned on 2024 only."""
    path = args.out / "tuning" / f"S0_y2024{args.tail}.json"
    if path.exists():
        base, cfg = load_tuning(path)
        return base, dataclasses.replace(cfg, lam_mono=FINAL_LAM)
    t0 = time.time()
    trials_b, trials_p = (2, 2) if args.smoke else (40, 60)
    base = tune_baselines(inp.X, inp.y, rows, [m for m in BASELINES], n_trials=trials_b)
    base["EBM"] = {}
    base["XGBoost-mono"] = tune_extra("XGBoost-mono", inp.X[rows], inp.y[rows], n_trials=trials_b)
    cfg = tune_pinam(inp.win, inp.y, rows, n_trials=trials_p, n_jobs=args.n_jobs,
                     n_seeds=2 if args.smoke else 3, fixed_spec=PINAM_V1.spec)
    save_tuning(path, base, cfg, {"info_set": "S0", "scope": "y2024", "n_rows": int(len(rows)),
                                  "fixed_spec": PINAM_V1.spec, "trials_baseline": trials_b,
                                  "trials_pinam": trials_p, "lam_mono_used": FINAL_LAM,
                                  "minutes": round((time.time() - t0) / 60, 1)})
    return base, dataclasses.replace(cfg, lam_mono=FINAL_LAM)


def run_reverse(inp, args) -> None:
    splits = split_year(inp.dates, train_year=2024, test_year=2023)
    params, cfg = reverse_tuning(inp, splits[0].train, args)
    for model in args.models:
        _fit_and_save(args.out / "reverse" / f"S0_yearrev_{_slug(model)}{args.tail}.npz",
                      model, cfg, params, inp.win, inp.y, splits, args,
                      {"protocol": "yearrev", "train_year": 2024, "test_year": 2023})


def run(args) -> None:
    inp = load_inputs("S0", ROOT / "config.yaml")
    for sub in ("inference", "drop", "curve", "reverse", "tuning"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    for exp in args.experiments:
        if exp == "reverse":
            run_reverse(inp, args)
            continue
        for proto in args.protocols:
            splits = make_splits(proto, inp.y, inp.dates)
            {"inference": run_inference, "drop": run_drop, "curve": run_curve}[exp](
                inp, proto, splits, args)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=EXPERIMENTS)
    ap.add_argument("--protocols", nargs="+", default=["year", "block"], choices=["year", "block"])
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    ap.add_argument("--fracs", nargs="+", type=float, default=list(CURVE_FRACS))
    ap.add_argument("--reps", type=int, default=5, help="random-missingness repetitions")
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--n-jobs", type=int, default=28)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/rev1/robustness")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="year protocol, 2 seeds, 1 group, 1 fraction, 2 repetitions, 3 models")
    args = ap.parse_args()
    args.tail = "_smoke" if args.smoke else ""
    if args.smoke:
        args.protocols, args.n_seeds, args.reps = ["year"], 2, 2
        args.groups, args.fracs = ["wind"], [0.5]
        if args.models == list(MODELS):
            args.models = ["PI-NAM", "LightGBM", "EBM"]
    return args


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("src.feature_engineering", "src.revision.info_sets", "interpret"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run(parse_args())
