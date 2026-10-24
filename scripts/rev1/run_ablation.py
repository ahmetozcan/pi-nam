"""Stage 5 ablation of PI-NAM V1 (R2-4, R1 "Section ??").

All variants keep the V1 training hyperparameters and differ only in the component that is
switched; every variant is a 15-seed ensemble with the baselines' threshold procedure
(stratified 5-fold inner CV, threshold="inner_cv").

  cumulative   C1 raw-16 NAM -> C2 +physics -> C3 +temporal min@{3,7} -> C4 +synoptic@5
               -> C5 +interactions -> V1 (+monotonicity, lambda=0.081)
  leave-one-out  V1 without physics / temporal / synoptic / interactions (without
                 monotonicity = C5)
  sensitivity  synoptic window w in {3, 7} (5 = V1); lambda in {0.03, 0.2, 1, 10} (0 = C5, 0.081 = V1);
               1 and 10 added 2026-10-10 because windspeed violations barely moved
  ensemble     V1 as single models, seeds 42..46 (compared with the 15-seed V1)
  final (_l10) leave-one-out, synoptic window and single-seed variants of the final model
               (lambda = 10, agreed 2026-10-10; reference S_lam10); the cumulative chain ends
               C5 -> S_lam10

Outputs: outputs/rev1/ablation/<info>_<protocol>_<variant>.npz (+ aux/, states/), same
format as the grid. The S0-block V1 result is reused from the grid (identical config and
threshold procedure).

Run from the repository root:
  python3 scripts/rev1/run_ablation.py --protocols year block
  python3 scripts/rev1/run_ablation.py --smoke
"""

import argparse
import dataclasses
import logging
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_grid import save_results  # noqa: E402
from src.revision.info_sets import load_inputs  # noqa: E402
from src.revision.models import PINAM_V1, PinamConfig, fit_pinam  # noqa: E402
from src.revision.protocols import make_splits  # noqa: E402

logger = logging.getLogger("run_ablation")
OUT = ROOT / "outputs/rev1/ablation"
GRID = ROOT / "outputs/rev1/grid"
V1 = PINAM_V1.spec
NO_TEMP = {"stats": [], "scales": []}


def _cfg(lam: float = PINAM_V1.lam_mono, **spec) -> PinamConfig:
    return dataclasses.replace(PINAM_V1, spec={**V1, **spec}, lam_mono=lam)


FINAL_LAM = 10.0


# name -> (config, n_seeds, base_seed)
VARIANTS: Dict[str, Tuple[PinamConfig, int, int]] = {
    "C1_raw": (_cfg(0.0, physics=False, synoptic=False, interactions=False, **NO_TEMP), 15, 42),
    "C2_phys": (_cfg(0.0, synoptic=False, interactions=False, **NO_TEMP), 15, 42),
    "C3_temp": (_cfg(0.0, synoptic=False, interactions=False), 15, 42),
    "C4_syn": (_cfg(0.0, interactions=False), 15, 42),
    "C5_inter": (_cfg(0.0), 15, 42),
    "V1": (PINAM_V1, 15, 42),
    "L_phys": (_cfg(physics=False), 15, 42),
    "L_temp": (_cfg(**NO_TEMP), 15, 42),
    "L_syn": (_cfg(synoptic=False), 15, 42),
    "L_inter": (_cfg(interactions=False), 15, 42),
    "S_w3": (_cfg(syn_scale=3), 15, 42),
    "S_w7": (_cfg(syn_scale=7), 15, 42),
    "S_lam0.03": (_cfg(0.03), 15, 42),
    "S_lam0.2": (_cfg(0.2), 15, 42),
    "S_lam1": (_cfg(1.0), 15, 42),
    "S_lam10": (_cfg(10.0), 15, 42),
    **{f"E_seed{s}": (PINAM_V1, 1, s) for s in range(42, 47)},
    "L_phys_l10": (_cfg(FINAL_LAM, physics=False), 15, 42),
    "L_temp_l10": (_cfg(FINAL_LAM, **NO_TEMP), 15, 42),
    "L_syn_l10": (_cfg(FINAL_LAM, synoptic=False), 15, 42),
    "L_inter_l10": (_cfg(FINAL_LAM, interactions=False), 15, 42),
    "S_w3_l10": (_cfg(FINAL_LAM, syn_scale=3), 15, 42),
    "S_w7_l10": (_cfg(FINAL_LAM, syn_scale=7), 15, 42),
    **{f"E_seed{s}_l10": (_cfg(FINAL_LAM), 1, s) for s in range(42, 47)},
}
REUSE = {("S0", "block", "V1"): "S0_block_pinam_thrcv"}


def _reuse(info: str, proto: str, name: str, out: Path) -> bool:
    src = REUSE.get((info, proto, name))
    if src is None or not (GRID / f"{src}.npz").exists():
        return False
    for sub, ext in (("", ".npz"), ("aux", ".npz"), ("states", ".pt")):
        s = GRID / sub / f"{src}{ext}"
        if s.exists():
            (out.parent / sub).mkdir(exist_ok=True)
            shutil.copy2(s, out.parent / sub / f"{out.stem}{ext}")
    logger.info("%s %s %-10s reused from grid/%s", info, proto, name, src)
    return True


def run(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    for info in args.info_sets:
        inp = load_inputs(info, ROOT / "config.yaml")
        for proto in args.protocols:
            splits = make_splits(proto, inp.y, inp.dates)
            for name in args.variants:
                cfg, n_seeds, base_seed = VARIANTS[name]
                if args.smoke:
                    n_seeds = min(n_seeds, 2)
                out = args.out / f"{info}_{proto}_{name}{'_smoke' if args.smoke else ''}.npz"
                if out.exists() and not args.overwrite:
                    logger.info("skip %s (exists)", out.name)
                    continue
                if not args.smoke and _reuse(info, proto, name, out):
                    continue
                t0 = time.time()
                res = fit_pinam(cfg, inp.win, inp.y, splits, n_seeds=n_seeds, base_seed=base_seed,
                                n_jobs=args.n_jobs, keep_states=True, threshold="inner_cv")
                save_results(out, res, inp.y, {"info_set": info, "protocol": proto, "model": name,
                                               "params": cfg.__dict__, "n_seeds": n_seeds,
                                               "base_seed": base_seed, "threshold_policy": "inner_cv"})
                logger.info("%s %s %-10s done in %.0fs", info, proto, name, time.time() - t0)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--info-sets", nargs="+", default=["S0"], choices=["S0", "S1"])
    ap.add_argument("--protocols", nargs="+", default=["year", "block"], choices=["rand", "block", "year"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--n-jobs", type=int, default=28)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="year protocol, 2 seeds, three variants")
    args = ap.parse_args()
    if args.smoke:
        args.protocols = ["year"]
        if args.variants == list(VARIANTS):
            args.variants = ["C1_raw", "V1", "E_seed42"]
    return args


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("src.feature_engineering", "src.revision.info_sets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run(parse_args())
