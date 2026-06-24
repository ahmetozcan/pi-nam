# Train from scratch (Optuna search → final model)

All commands run from the package root (`pinam/`). Random seed = 42.

## Pipeline

```bash
# 1) Optuna-tune the baselines on the 16 raw weather channels, save weights
python code/train/retrain_tuned_baselines.py
#    -> reproducibility/folds/base_models/<model>/fold_XX.pkl  (model + scaler + indices)
#    -> outputs/results/<model>_folds.csv

# 2) Optuna structure + hyperparameter search for PI-NAM
#    Searches: instantaneous physics on/off, SYNOPTIC physics on/off + window {3,5,7},
#    temporal aggregates, pairwise interactions, monotonicity lambda, shape-net
#    width/depth, lr, batch size, weight decay.  Objective = 10-fold deployable F1.
python code/train/search_pinamx.py
#    -> outputs/results/pinamx_search.json   (full search space + per-trial log + best)

# 3) Validate the top-3 search configs at the full 15-seed deep ensemble
python code/train/final_pinamx.py
#    -> outputs/results/pinamx_final.json + per-config fold weights

# 4) (optional) Targeted experiments behind the final model
python code/train/test_synoptic.py    # synoptic on/off × window {3,5,7}
python code/train/test_combined.py    # synoptic + interactions -> SAVES the final
#    winning model V1 to reproducibility/folds/pinamx_v1/fold_XX.pkl (15 seed state-dicts/fold)

# (optional) persist the pre-interaction synoptic-only model used in the ablation
python code/train/save_synoptic_winner.py
#    -> reproducibility/folds/pinamx_synoptic5/fold_XX.pkl
```

## (optional) Regenerate the TabPFN baseline from scratch

```bash
pip install tabpfn==7.0.1        # CUDA GPU strongly recommended
python code/train/tabpfn_oof.py     # -> reproducibility/oof/tabpfn_oof.npy
```
TabPFN is zero-shot (no tuning): n_estimators=32, random_state=0, StandardScaler,
seed-42 folds. GPU float non-determinism may shift F1 by ~0.001-0.002 vs the
shipped frozen OOF (the bit-exact reference). Not required for reproduction.

## Notes
- Parallelism: `search_pinamx.py` / `final_pinamx.py` run folds × seeds across CPU
  cores via joblib (`n_jobs`); each worker uses 1 torch thread. CPU is faster than
  GPU for these tiny additive nets.
- The final PI-NAM configuration (V1): instantaneous + synoptic (window=5) physics,
  minimum temporal aggregates over {3,7} days, pairwise interactions, monotonicity
  lambda=0.08, hidden width 48, two-layer shape nets, lr 2e-4, batch 32, weight
  decay 1e-3, 15-seed deep ensemble.
- The full-feature (118-dim) baseline benchmark (paper Table 1) and the SHAP
  analysis use `src/optuna_tuning.py`, `src/models.py`, and `src/shap_analysis.py`;
  they require retraining and are independent of the raw-16 deployable comparison.
