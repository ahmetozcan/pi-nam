# PI-NAM: Physics-Informed Neural Additive Model for Hot-Air Balloon Flight Prediction

Reproducible code and weights for the paper *A Data-Driven Decision Making Framework for Hot-Air Balloon1 Tourism Using a Physics-Informed Neural Additive Model*.

**PI-NAM** predicts daily meteorological suitability for hot-air balloon flights
in Cappadocia from 16 raw weather channels. It augments per-factor additive shape
functions with a differentiable aviation-physics layer that couples *instantaneous*
quantities (air density, lift, density altitude) with multi-day **synoptic dynamics**
(pressure tendency, frontal passage, fog risk, diurnal range), under monotonicity
priors. Under a realistic *deployable* thresholding protocol it attains the
**highest deployable F1 (0.877)** of all models — numerically topping every
Optuna-tuned tree baseline, statistically matching the best of them, and
**significantly** outperforming TabPFN v2 (Wilcoxon p < 0.01) — while remaining
fully interpretable and physically consistent.

---

## Repository layout

```
pinam/
├── README.md                 # this file
├── requirements.txt          # dependencies
├── config.yaml               # dataset path, seed, k-folds
├── data/                     # the 728-day Cappadocia dataset (.xlsx)
├── src/                      # shared library (model, physics, features)
│   ├── pinam_search.py       #   PI-NAM (PINAMX) + synoptic_physics + parallel CV
│   ├── pinn.py               #   differentiable instantaneous-physics layer
│   ├── pinam.py              #   shape networks
│   ├── feature_engineering.py
│   ├── optuna_tuning.py, models.py, ...
│
├── code/
│   ├── train/                # === TRAIN FROM SCRATCH (incl. Optuna search) ===
│   └── reproduce/            # === REPRODUCE PAPER FROM SAVED WEIGHTS ===
│
├── reproducibility/
│   ├── folds/
│   │   ├── pinamx_v1/        # final PI-NAM weights: 15 seed state-dicts × 10 folds
│   │   └── base_models/      # tuned baseline weights (SVM/XGB/LGBM/RF/CatBoost)
│   └── oof/
│       └── tabpfn_oof.npy    # TabPFN v2 frozen out-of-fold probs (weights omitted, see below)
│
└── outputs/
    ├── results/              # generated tables (.tex) + metrics (.json)  [+ seeded Optuna search artifacts]
    └── figures/              # generated figures (.pdf/.png)
```

The two `code/` sub-folders are intentionally separated:
**`code/train`** rebuilds everything from raw data (Optuna search → final model);
**`code/reproduce`** regenerates every paper table and figure from the saved
weights without any retraining.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -r requirements.txt
```

CPU is sufficient and recommended (the additive shape nets are tiny; GPU is
slower for them). Everything below runs from the **package root** (`pinam/`).

---

## Quick start — reproduce the paper from saved weights

```bash
python code/reproduce/reproduce_all.py
```

This regenerates, into `outputs/`:

| Paper artifact | Script | Source |
|----------------|--------|--------|
| Table 2 — deployable comparison (`tab_main_deployable.tex`) | `paper_results_table.py` | weights |
| Table 3 — season-stratified (`tab_seasonal.tex`) | `make_season_table.py` | weights |
| Table 4 — Wilcoxon (`tab_wilcoxon_pinam.tex`) | `paper_results_table.py` | weights |
| Figure 1 — F1/AUC boxplots | `make_main_figures.py` | weights |
| Figure 2 — MCC boxplot | `make_main_figures.py` | weights |
| Figure 3 — AUC trajectory | `make_main_figures.py` | weights |
| PI-NAM architecture figure | `make_pinam_v1_architecture.py` | — |
| Learned shape-function figure | `make_pinam_v1_shapes.py` | weights |

**Weight-backed guarantee.** `code/reproduce/reproduce_pinam_v1.py` rebuilds the
PI-NAM out-of-fold predictions *directly from the 15 saved seed state-dicts per
fold* (not from any cached array) and verifies the deployable F1 = **0.8770**.
The tuned baselines are likewise reconstructed from their saved `model + scaler`
bundles in `reproducibility/folds/base_models/`.

> **What is an OOF array?** "Out-of-fold" (OOF) predictions are the test-time
> outputs of cross-validation. In 10-fold CV every one of the 728 days lands in
> the held-out test fold exactly once; the OOF array stores, for each day, the
> probability predicted *while that day was held out* (by a model trained only on
> the other 9 folds). It is therefore a length-728 vector of honest, never-seen
> test predictions — and every reported metric (F1, ROC-AUC, the deployable
> threshold, the Wilcoxon test) is computed directly from it. Shipping a model's
> OOF array reproduces that model's paper numbers **exactly**, without needing the
> model itself.
>
> **TabPFN note.** The TabPFN v2 foundation-model weights (~450 MB, CUDA-serialised
> and fragile to move between machines) are **not** shipped; instead its exact
> frozen OOF probabilities are provided in `reproducibility/oof/tabpfn_oof.npy`,
> so the TabPFN row of every table and the Wilcoxon test reproduce identically.
> All other models (the five tuned baselines and PI-NAM) are rebuilt from their
> saved weights, and their OOF arrays are regenerated on the fly during reproduction.
>
> To regenerate the TabPFN OOF **from scratch**, run `code/train/tabpfn_oof.py`
> with `tabpfn==7.0.1` installed and a **CUDA GPU** (the configuration used in the
> paper: `n_estimators=32`, `random_state=0`, `StandardScaler`, seed-42 folds).
> On a CUDA machine with the same version this reproduces the shipped array; note
> that GPU floating-point non-determinism (different GPU/driver/library) can shift
> probabilities by ~1e-5 and move F1 by ~0.001-0.002, so the shipped frozen OOF
> remains the bit-exact reference.

---

## Train everything from scratch

See `code/train/README.md`. In short:

```bash
python code/train/retrain_tuned_baselines.py   # Optuna-tune baselines on raw-16, save weights
python code/train/search_pinamx.py             # Optuna search over PI-NAM (incl. synoptic physics)
python code/train/final_pinamx.py              # 15-seed deep ensemble of the top configs
python code/train/save_synoptic_winner.py      # persist the final model weights (pinamx_v1)
```

---

## Key numbers (10-fold stratified CV, raw-16, deployable threshold)

| Model | Deployable F1 | ROC-AUC | MCC | threshold (mean±std) |
|-------|:-:|:-:|:-:|:-:|
| **PI-NAM (proposed)** | **0.877** | 0.903 | 0.650 | **0.360 ± 0.002** |
| LightGBM* | 0.874 | 0.894 | 0.653 | 0.462 ± 0.013 |
| SVM* / Random Forest* | 0.872 | 0.896 / 0.892 | 0.641 / 0.646 | — |
| CatBoost* | 0.865 | 0.897 | 0.622 | — |
| TabPFN v2 | 0.846 | 0.908 | 0.572 | 0.425 ± 0.092 |

`*` Optuna-tuned. PI-NAM vs TabPFN: Wilcoxon p = 0.010; vs tuned trees: p > 0.05 (statistical tie).

---

## Citation

If you use this code, dataset, or the PI-NAM architecture in your research, please cite our paper:

```bibtex
@article{ozcan2026pinam,
  author  = {Ahmet ÖZCAN},
  title   = {A Data-Driven Decision Making Framework for Hot-Air Balloon Tourism Using a Physics-Informed Neural Additive Model},
  journal = {},
  year    = {},
  doi     = {10.1016/XXX.XXXX.XXXXXX}
}```

License

This project is licensed under the MIT License - see the LICENSE file for details.

The Cappadocia Hot-Air Balloon Dataset provided in the data/ directory is released under the same open-access terms for academic and research purposes. 
We strongly encourage reproducible research and welcome community extensions to the PI-NAM framework.
