# Notebooks

## `vietnam_real_estate_baseline.ipynb`

End-to-end LightGBM / CatBoost tabular baseline for Vietnam real-estate price
prediction on the [`tinixai/vietnam-real-estates`](https://huggingface.co/datasets/tinixai/vietnam-real-estates)
dataset (~3.5M listings).

**Run on Colab:** open the file in Google Colab and `Runtime → Run all`. The
first cell installs `lightgbm catboost datasets scikit-learn matplotlib seaborn`.
Defaults to a stratified 100k-row sample so it finishes on the free CPU tier;
set `SAMPLE_MODE = "metro"` (Hanoi / HCMC) or `"full"` in the **Configuration**
cell for larger runs.

Pipeline: load → clean anomalous records → engineer location encodings
(frequency + out-of-fold target encoding) and physical ratios → train on
`log1p(price)` with early stopping → evaluate (RMSLE, MAPE, MAE in tỷ VND, MedAE,
R²) for total price and price/m² → plot feature importance, actual-vs-predicted,
and residuals.

`_build_notebook.py` regenerates the `.ipynb` from plain Python
(`python notebooks/_build_notebook.py`) so it stays reviewable in diffs.
