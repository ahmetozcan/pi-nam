# PI-NAM: Physics-Informed Neural Additive Model for Hot-Air Balloon Flight Authorization

Code, data and saved predictions for the manuscript *A Data-Driven Decision Making Framework for
Hot-Air Balloon Tourism Using a Physics-Informed Neural Additive Model* (under review).

PI-NAM predicts the daily flight-authorization decision of the civil aviation authority for the
Göreme flight zone in Cappadocia, Türkiye, from 16 daily meteorological variables. It learns a
separate shape function for each raw variable, derived aviation-physics quantity and multi-day
synoptic feature, and a monotonicity penalty discourages predicted flight probabilities that
increase with wind, cloud cover or precipitation.

| Protocol | AUC of PI-NAM (same-day information) |
|---|---|
| Random 10-fold cross-validation | 0.907 |
| Blocked 10-fold cross-validation (14-day buffers) | 0.853 |
| Year-ahead (train 2023, test 2024, nested tuning) | 0.872 |

The discrimination of PI-NAM is statistically indistinguishable from that of the tuned
gradient-boosted trees, TabPFN v2 and an explainable boosting machine. See the manuscript for all
results and tests.

## Contents

```
├── data/datasetCappadociaBalloonFlight.xlsx   731 daily records, 2023–2024
├── config.yaml                                dataset path and seed
├── src/                                       model, physics layer and features
│   ├── pinam_search.py                        PI-NAM model, synoptic features, training
│   ├── pinn.py                                instantaneous aviation-physics quantities
│   ├── pinam.py                               shape networks
│   └── revision/                              information sets, protocols, metrics, tests,
│                                              comparison models, robustness, interpretation
├── scripts/rev1/                              experiments, analyses, tables and figures
├── tests/                                     unit tests (incl. the leakage test of Sec. 4.4)
└── outputs/
    ├── results/optuna_best_params.json        baseline hyperparameters of the original study
    └── rev1/
        ├── grid/          predictions of every model in the six settings (*.npz, aux/, tuning/),
        │                  trained PI-NAM ensembles (states/)
        ├── ablation/      predictions and summaries of the ablation variants
        ├── extra/         predictions and summaries of the additional models
        ├── robustness/    missing-variable, record-length and reversed-split results
        ├── stability/     stability of the learned contributions
        ├── analysis_thrcv_lam10/  calibration, safety operating point, seasonal results
        ├── class_metrics/ class-specific metrics and confusion matrices
        ├── tables/        manuscript tables (.tex) and the tests behind them (.csv)
        └── figures/       manuscript figures (.pdf, .png)
```

The dataset is also published on Mendeley Data:
Özcan, A. (2026). *Dataset on Weather Conditions and Flight Decision Records for Hot-Air Balloons
in Cappadocia Region (2023–2024)*. Mendeley Data, Version 2.
https://doi.org/10.17632/mjv3zmxwsn.2

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3.12
pip install -r requirements.txt
python3 -m pytest                                      # 52 tests, about 2 minutes
```

All commands run from the repository root. PI-NAM trains on the CPU; TabPFN v2 benefits from a GPU.

## Reproducing the tables and figures from the saved predictions

The `.npz` files hold, for every model, setting and split, the test-day probabilities and the
thresholds chosen on the training data. Every accuracy figure and significance test of the
manuscript is computed from them, without retraining:

```bash
python3 scripts/rev1/summarize_grid.py
python3 scripts/rev1/analyze_grid.py --pinam-threshold inner_cv --pinam-lam 10
python3 scripts/rev1/analyze_ablation.py
python3 scripts/rev1/analyze_extra.py
python3 scripts/rev1/analyze_robustness.py
python3 scripts/rev1/class_metrics.py
python3 scripts/rev1/make_tables.py          # -> outputs/rev1/tables/
python3 scripts/rev1/make_all_figures.py     # -> outputs/rev1/figures/
```

The trained PI-NAM ensembles of the main comparison are included in `outputs/rev1/grid/states/`
(ten files, about 490 MB). With them, `make_all_figures.py` also redraws the shape functions and
the stability analysis (Fig. 5 and Fig. S3). The networks of the ablation variants (about 2 GB)
are not included; their predictions are, and the monotonicity checks of the ablation are
provided as saved summaries in `outputs/rev1/ablation/summary/`.

## Retraining

```bash
python3 scripts/rev1/run_grid.py --info-sets S0 S1 --protocols rand block year \
        --pinam-threshold inner_cv --lam-mono 10        # main comparison
python3 scripts/rev1/run_ablation.py                    # ablation
python3 scripts/rev1/run_extra.py                       # EBM, monotone and derived-feature trees, MLPs
python3 scripts/rev1/run_robustness.py                  # missing variables, record length
python3 scripts/rev1/run_stability.py                   # stability of the contributions
```

Each script accepts `--smoke` for a quick check. Data splits use the random seed 42 and the
ensemble members the seeds 42 to 56. On an Intel Core i9-14900HX (32 threads), one evaluation of
PI-NAM under random cross-validation, with 15-member ensembles in every outer and inner fold, took
about two hours; TabPFN v2 ran on an NVIDIA RTX 5000 Ada Generation Laptop GPU.

## Citation

The manuscript is under review. Until it is published, please cite the dataset (see above).

## License

MIT, see `LICENSE`.
