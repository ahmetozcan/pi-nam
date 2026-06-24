# Reproduce the paper from saved weights (no retraining)

All commands run from the package root (`pinam/`).

```bash
python code/reproduce/reproduce_all.py     # everything, in order
```

or individually:

| Script | Produces | Reads |
|--------|----------|-------|
| `reproduce_pinam_v1.py` | rebuilds PI-NAM OOF **from weights**, verifies deployable F1 = 0.877 | `reproducibility/folds/pinamx_v1/` |
| `paper_results_table.py` | Table 2 (`tab_main_deployable.tex`), Table 4 (`tab_wilcoxon_pinam.tex`), `paper_results.json` | baseline weights + PI-NAM OOF + TabPFN frozen OOF |
| `make_season_table.py` | Table 3 (`tab_seasonal.tex`) | PI-NAM OOF |
| `make_main_figures.py` | Figures 1–3 | baseline weights + PI-NAM OOF + TabPFN OOF |
| `make_pinam_v1_architecture.py` | architecture figure | — |
| `make_pinam_v1_shapes.py` | learned per-factor shape curves | `reproducibility/folds/pinamx_v1/` |

Outputs land in `outputs/results/` (`.tex`, `.json`) and `outputs/figures/`
(`.pdf`, `.png`). Copy the `.tex` tables and `.pdf` figures into the LaTeX
project to rebuild the paper.

**How the weight-backed guarantee works.** `reproduce_pinam_v1.py` instantiates a
fresh `PINAMX` model per fold, loads each of the 15 saved seed state-dicts, runs
the model forward on that fold's held-out window, averages the ensemble, and
assembles the out-of-fold prediction vector — entirely from stored weights. It
then writes that vector to `outputs/results/pinamx_v1_oof.npy`, which the other
reproduce scripts consume, so every downstream table and figure traces back to
the saved weights. Baselines are reconstructed analogously from their
`model + scaler` bundles; only TabPFN is supplied as a frozen OOF array.
