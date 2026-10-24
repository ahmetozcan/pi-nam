"""Stage 6: extra comparison models under the rev1 protocols (S0 by default).

  tabular  XGBoost-mono, LightGBM-mono (raw-16, monotone -1 on wind/gust/cloud/precip/snow),
           XGBoost-derived, LightGBM-derived (PI-NAM V1 factors + interaction products),
           EBM (raw-16, untuned defaults)
  neural   MLP-phys (V1 factors -> one MLP, lambda = 0) and MLP-phys-mono (lambda = 10, the
           final PI-NAM weight): same factors, training settings, 15-seed ensemble and
           threshold procedure as PI-NAM, without the additive structure
  fd-only  the six original baselines refitted with their grid hyperparameters, only to
           record the monotonicity check (predictions equal the grid files)

Tuning follows the grid policy: rand/block use one search on all rows, year searches on 2023
only (Optuna, 40 trials, the submitted search space of the base learner, constraints active).
Every model stores a finite-difference monotonicity check on its test rows in fd/.

Outputs under outputs/rev1/extra/: <info>_<protocol>_<model>.npz (+ aux/, states/, fd/),
tuning/<info>_<scope>.json.

Run from the repository root:
  python3 scripts/rev1/run_extra.py --smoke
  python3 scripts/rev1/run_extra.py
"""

import argparse
import dataclasses
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_grid import SUBMITTED_PARAMS, _slug, save_results  # noqa: E402
from src.revision.extra_models import (TABULAR_EXTRA, ensemble_fd, feature_fn, fit_tabular,  # noqa: E402
                                       tune_extra)
from src.revision.info_sets import load_inputs  # noqa: E402
from src.revision.models import BASELINES, PINAM_V1, fit_pinam  # noqa: E402
from src.revision.protocols import make_splits, split_year  # noqa: E402
from src.revision.tuning import load_tuning  # noqa: E402

logger = logging.getLogger("run_extra")
GRID = ROOT / "outputs/rev1/grid"
NEURAL = {"MLP-phys": dataclasses.replace(PINAM_V1, arch="mlp", lam_mono=0.0),
          "MLP-phys-mono": dataclasses.replace(PINAM_V1, arch="mlp", lam_mono=10.0)}
FD_ONLY = tuple(f"{b} (fd)" for b in BASELINES)
MODELS = TABULAR_EXTRA + tuple(NEURAL) + FD_ONLY


def _scope(protocol: str) -> str:
    return "y2023" if protocol == "year" else "full"


def _tuning_rows(protocol: str, inp) -> np.ndarray:
    return split_year(inp.dates)[0].train if protocol == "year" else np.arange(len(inp.y))


def extra_params(info: str, protocol: str, inp, args) -> Dict[str, Dict]:
    """Tuned hyperparameters of the tabular extra models, cached per (info set, scope)."""
    path = args.out / "tuning" / f"{info}_{_scope(protocol)}{'_smoke' if args.smoke else ''}.json"
    cached = json.loads(path.read_text()) if path.exists() else {}
    rows = _tuning_rows(protocol, inp)
    for name in [m for m in args.models if m in TABULAR_EXTRA and m not in cached]:
        t0 = time.time()
        cached[name] = tune_extra(name, feature_fn(name)(inp.win)[rows], inp.y[rows],
                                  n_trials=args.trials)
        logger.info("%s %s tuned %s in %.0fs", info, _scope(protocol), name, time.time() - t0)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached, indent=2))
    return cached


def grid_params(info: str, protocol: str) -> Dict[str, Dict]:
    """Hyperparameters the original baselines used in the grid."""
    if protocol != "year" and info == "S0":
        sub = json.loads(SUBMITTED_PARAMS.read_text())
        return {k: v["best_params"] for k, v in sub.items()} | {"TabPFN": {}}
    return load_tuning(GRID / "tuning" / f"{info}_{_scope(protocol)}.json")[0] | {"TabPFN": {}}


def _save_fd(out: Path, results, fd) -> None:
    (out.parent / "fd").mkdir(exist_ok=True)
    np.savez(out.parent / "fd" / out.name, test=np.concatenate([r.test for r in results]),
             fd=np.concatenate(fd, axis=1))


def run(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    for info in args.info_sets:
        inp = load_inputs(info, ROOT / "config.yaml")
        for protocol in args.protocols:
            splits = make_splits(protocol, inp.y, inp.dates)
            tuned = extra_params(info, protocol, inp, args)
            for name in args.models:
                out = args.out / f"{info}_{protocol}_{_slug(name.replace(' (fd)', '_fd'))}" \
                                 f"{'_smoke' if args.smoke else ''}.npz"
                if out.exists() and not args.overwrite:
                    logger.info("skip %s (exists)", out.name)
                    continue
                t0 = time.time()
                if name in NEURAL:
                    cfg = NEURAL[name]
                    res = fit_pinam(cfg, inp.win, inp.y, splits, n_seeds=args.n_seeds,
                                    n_jobs=args.n_jobs, keep_states=True, threshold="inner_cv")
                    fd, params, n_seeds = ensemble_fd(cfg, inp.win, res), cfg.__dict__, args.n_seeds
                else:
                    base = name.removesuffix(" (fd)")
                    params = grid_params(info, protocol).get(base, {}) if name in FD_ONLY else tuned[name]
                    res = fit_tabular(base, params, inp.win, inp.y, splits)
                    fd, n_seeds = [r.extra.pop("fd") for r in res], 1
                save_results(out, res, inp.y, {"info_set": info, "protocol": protocol, "model": name,
                                               "params": params, "n_seeds": n_seeds,
                                               "threshold_policy": "inner_cv"})
                _save_fd(out, res, fd)
                logger.info("%s %s %-18s done in %.0fs", info, protocol, name, time.time() - t0)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--info-sets", nargs="+", default=["S0"], choices=["S0", "S1"])
    ap.add_argument("--protocols", nargs="+", default=["year", "rand", "block"],
                    choices=["rand", "block", "year"])
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--n-jobs", type=int, default=28)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/rev1/extra")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="year protocol, 3 trials, 2 seeds")
    args = ap.parse_args()
    if args.smoke:
        args.protocols, args.trials, args.n_seeds = ["year"], 3, 2
    return args


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("src.feature_engineering", "src.revision.info_sets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run(parse_args())
