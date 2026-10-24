"""Run the rev1 evaluation grid: {S0, S1} x {rand, block, year} x models.

Tuning policy (agreed 2026-10-07):
  * year  -- fully nested: every model (incl. PI-NAM structure search) is tuned on 2023 only.
             "PI-NAM-fixed" (agreed 2026-10-08) keeps the submitted V1 architecture and tunes
             only its training hyperparameters on 2023; both PI-NAM rows are reported.
             Under rand/block, PI-NAM-fixed is the submitted V1 unchanged (agreed 2026-10-18).
  * rand, block -- one tuning run on all rows per information set, applied equally to every
    model (S0 reuses the submitted Optuna params and PI-NAM V1; S1 is tuned afresh).
Thresholds are always chosen on training-portion predictions (see src/revision/models.py).
  --pinam-threshold inner_cv (agreed 2026-10-09) gives PI-NAM the same threshold procedure
  as the baselines (stratified 5-fold inner CV, 15-seed ensemble per inner fold); its files
  carry the suffix _thrcv and leave the submitted-behaviour files untouched.
  --lam-mono X (agreed 2026-10-10) overrides the monotonicity weight of every PI-NAM config
  (tuned lambdas included) and adds the suffix _lam<X>. stage 5 showed lambda <= 0.2 leaves
  ~25-50% windspeed violations while lambda = 10 removes them at equal accuracy; lambda is
  then a consistency requirement (smallest lambda in the grid with ~0 violations), not tuned.

Outputs (per info set / protocol / model), all under outputs/rev1/grid/:
  <info>_<protocol>_<model>.npz      test indices, probabilities, per-row thresholds, split ids
  aux/<same name>.npz                threshold-source predictions (PI-NAM validation set,
                                     baseline inner CV) and PI-NAM per-seed test predictions
  states/<same name>.pt              PI-NAM state_dicts per split and seed (shape stability)

Examples (run from the repository root):
  python3 scripts/rev1/run_grid.py --smoke
  python3 scripts/rev1/run_grid.py --info-sets S0 S1 --protocols rand block year
"""

import argparse
import dataclasses
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.revision.info_sets import load_inputs  # noqa: E402
from src.revision.models import BASELINES, PINAM_V1, PinamConfig, fit_baseline, fit_pinam  # noqa: E402
from src.revision.protocols import PROTOCOLS, make_splits, split_year  # noqa: E402
from src.revision.tuning import load_tuning, save_tuning, tune_baselines, tune_pinam  # noqa: E402

logger = logging.getLogger("run_grid")
MODELS = ("PI-NAM",) + BASELINES
EXTRA_MODELS = ("PI-NAM-fixed",)      # opt-in via --models
SUBMITTED_PARAMS = ROOT / "outputs/results/optuna_best_params.json"


def _slug(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "")


def resolve_tuning(info_set: str, protocol: str, inp, models, args) -> Tuple[Dict, PinamConfig]:
    """Hyperparameters for one (info set, protocol), cached under <out>/tuning/."""
    scope = "y2023" if protocol == "year" else "full"
    if scope == "full" and info_set == "S0":
        sub = json.loads(SUBMITTED_PARAMS.read_text())
        return {k: v["best_params"] for k, v in sub.items()} | {"TabPFN": {}}, PINAM_V1
    path = args.out / "tuning" / f"{info_set}_{scope}{'_smoke' if args.smoke else ''}.json"
    if path.exists():
        return load_tuning(path)
    rows = split_year(inp.dates)[0].train if scope == "y2023" else np.arange(len(inp.y))
    t0 = time.time()
    base = tune_baselines(inp.X, inp.y, rows, [m for m in models if m != "PI-NAM"],
                          n_trials=args.trials_baseline)
    pinam = (tune_pinam(inp.win, inp.y, rows, n_trials=args.trials_pinam, n_jobs=args.n_jobs)
             if "PI-NAM" in models else PINAM_V1)
    save_tuning(path, base, pinam, {"info_set": info_set, "scope": scope, "n_rows": int(len(rows)),
                                    "trials_baseline": args.trials_baseline,
                                    "trials_pinam": args.trials_pinam,
                                    "minutes": round((time.time() - t0) / 60, 1)})
    return base, pinam


def resolve_pinam_fixed(info_set: str, protocol: str, inp, args) -> PinamConfig:
    """V1 architecture; year: training hyperparameters tuned on 2023, rand/block: submitted V1.

    Under rand/block the S0 PI-NAM already is V1, so the row matters for S1 (agreed 2026-10-18):
    it shows the single submitted configuration next to the configuration searched for S1.
    """
    if protocol != "year":
        return PINAM_V1
    path = args.out / "tuning" / f"{info_set}_y2023_fixedarch{'_smoke' if args.smoke else ''}.json"
    if path.exists():
        return load_tuning(path)[1]
    rows = split_year(inp.dates)[0].train
    t0 = time.time()
    cfg = tune_pinam(inp.win, inp.y, rows, n_trials=args.trials_pinam, n_jobs=args.n_jobs,
                     fixed_spec=PINAM_V1.spec)
    save_tuning(path, {}, cfg, {"info_set": info_set, "scope": "y2023", "fixed_spec": PINAM_V1.spec,
                                "n_rows": int(len(rows)), "trials_pinam": args.trials_pinam,
                                "minutes": round((time.time() - t0) / 60, 1)})
    return cfg


def save_results(path: Path, results, y: np.ndarray, meta: Dict) -> None:
    test = np.concatenate([r.test for r in results])
    np.savez(path, test=test, y=y[test],
             p=np.concatenate([r.p_test for r in results]),
             thr=np.concatenate([np.full(len(r.test), r.threshold) for r in results]),
             split=np.concatenate([np.full(len(r.test), r.split) for r in results]),
             meta=json.dumps(meta))
    _save_aux(path, results, y)


def _save_aux(path: Path, results, y: np.ndarray) -> None:
    """Threshold-source predictions, per-seed test predictions and PI-NAM weights."""
    aux_dir, st_dir = path.parent / "aux", path.parent / "states"
    aux_dir.mkdir(exist_ok=True)
    rows = np.concatenate([r.extra["thr_rows"] for r in results])
    aux = {"thr_rows": rows, "thr_y": y[rows],
           "thr_p": np.concatenate([r.extra["p_thr"] for r in results]),
           "thr_split": np.concatenate([np.full(len(r.extra["thr_rows"]), r.split) for r in results])}
    if "p_seeds" in results[0].extra:       # (n_seeds, n_test) aligned with the main file
        aux["p_seeds"] = np.concatenate([r.extra["p_seeds"] for r in results], axis=1)
    np.savez(aux_dir / path.name, **aux)
    if "states" in results[0].extra:
        import torch
        st_dir.mkdir(exist_ok=True)
        torch.save({r.split: {"test": r.test, "states": r.extra["states"]} for r in results},
                   st_dir / path.with_suffix(".pt").name)


def run(args) -> None:
    models: List[str] = args.models
    for info_set in args.info_sets:
        inp = load_inputs(info_set, ROOT / "config.yaml")
        for protocol in args.protocols:
            splits = make_splits(protocol, inp.y, inp.dates)
            base_params, pinam_cfg = resolve_tuning(info_set, protocol, inp,
                                                    [m for m in models if m in MODELS], args)
            for model in models:
                is_pinam = model.startswith("PI-NAM")
                tag = "_thrcv" if is_pinam and args.pinam_threshold == "inner_cv" else ""
                if is_pinam and args.lam_mono is not None:
                    tag += f"_lam{args.lam_mono:g}"
                out = args.out / f"{info_set}_{protocol}_{_slug(model)}{tag}{'_smoke' if args.smoke else ''}.npz"
                if out.exists() and not args.overwrite:
                    logger.info("skip %s (exists)", out.name)
                    continue
                t0 = time.time()
                if is_pinam:
                    cfg = (resolve_pinam_fixed(info_set, protocol, inp, args)
                           if model == "PI-NAM-fixed" else pinam_cfg)
                    if args.lam_mono is not None:
                        cfg = dataclasses.replace(cfg, lam_mono=args.lam_mono)
                    res = fit_pinam(cfg, inp.win, inp.y, splits, n_seeds=args.n_seeds,
                                    n_jobs=args.n_jobs, keep_states=True,
                                    threshold=args.pinam_threshold)
                    params = cfg.__dict__
                else:
                    params = base_params.get(model, {})
                    res = fit_baseline(model, params, inp.X, inp.y, splits)
                save_results(out, res, inp.y, {"info_set": info_set, "protocol": protocol,
                                               "model": model, "params": params,
                                               "threshold_policy": (args.pinam_threshold if is_pinam
                                                                    else "inner_cv"),
                                               "n_seeds": args.n_seeds if model.startswith("PI-NAM") else 1})
                logger.info("%s %s %-13s done in %.0fs", info_set, protocol, model, time.time() - t0)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--info-sets", nargs="+", default=["S0", "S1"], choices=["S0", "S1"])
    ap.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS + EXTRA_MODELS)
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--trials-baseline", type=int, default=40)
    ap.add_argument("--trials-pinam", type=int, default=60)
    ap.add_argument("--n-jobs", type=int, default=28)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/rev1/grid")
    ap.add_argument("--pinam-threshold", default="val", choices=["val", "inner_cv"])
    ap.add_argument("--lam-mono", type=float, default=None,
                    help="override the PI-NAM monotonicity weight (file suffix _lam<X>)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end check: 2 seeds, 2 tuning trials, PI-NAM + LightGBM")
    args = ap.parse_args()
    if args.smoke:
        args.n_seeds, args.trials_baseline, args.trials_pinam = 2, 2, 2
        if args.models == list(MODELS):
            args.models = ["PI-NAM", "LightGBM"]
    return args


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("src.feature_engineering", "src.revision.info_sets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    run(a)
