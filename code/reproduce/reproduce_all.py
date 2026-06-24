"""
One-shot reproduction of all weight-backed PI-NAM paper artifacts.

Runs, in order:
  1. reproduce_pinam_v1.py     - rebuild PI-NAM OOF from saved fold weights (verify)
  2. paper_results_table.py    - Table 2 (deployable comparison) + Table 4 (Wilcoxon)
  3. make_season_table.py      - Table 3 (season-stratified)
  4. make_main_figures.py      - Figures 1-3 (F1/AUC, MCC, AUC trajectory)
  5. make_pinam_v1_architecture.py - architecture figure
  6. make_pinam_v1_shapes.py   - learned shape-function figure

All outputs land in outputs/results/ (*.tex, *.json) and outputs/figures/ (*.pdf,*.png).
Run from the package root:  python code/reproduce/reproduce_all.py
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STEPS = [
    "reproduce_pinam_v1.py",
    "paper_results_table.py",
    "make_season_table.py",
    "make_main_figures.py",
    "make_pinam_v1_architecture.py",
    "make_pinam_v1_shapes.py",
]
for s in STEPS:
    print(f"\n{'='*70}\n>>> {s}\n{'='*70}", flush=True)
    r = subprocess.run([sys.executable, str(HERE / s)], cwd=str(ROOT))
    if r.returncode != 0:
        print(f"!! {s} failed (exit {r.returncode})"); sys.exit(r.returncode)
print(f"\n{'='*70}\nAll artifacts regenerated in outputs/results/ and outputs/figures/\n{'='*70}")
