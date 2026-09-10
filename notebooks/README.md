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

## `vietnam_real_estate_text_features.ipynb`

Text feature extraction from the `name` / `description` fields of the same
dataset, and an out-of-time measurement of what those features add to a tabular
price baseline. Companion to `vietnam_real_estate_baseline.ipynb`; the code lives
in `real_estate/text/` and the design notes in
`docs/real_estate_text_features.md`.

**Run on Colab:** open the file and `Runtime → Run all`. The first cell installs
`lightgbm scikit-learn pandas pyarrow numpy` and clones the repo at `REF` — set
`REF` to any branch containing `real_estate/text`. Defaults to 60k train / 25k
test rows so it finishes on the free CPU tier; the PhoBERT cell is opt-in and
needs `torch` + `transformers`.

Pipeline: load two shards (train `shard_0000` = 2025-06, test `shard_0009` =
2026-03) → check the three data traps before modelling → extract 47 domain
keyword flags → parse numeric entities and validate them against the structured
columns → demonstrate the price-leakage control → TF-IDF + SVD → ablation ladder
(tabular → +keywords → +entities → +tfidf) on RMSLE / MAPE / MAE (tỷ) / MedAE /
R² → compare TF-IDF analyzers → optional PhoBERT arm → integration snippet.

Headline result: **−10.07% RMSLE and −2.67 MAPE points** over the tabular
baseline, with every arm improving every metric.

`_build_text_features_notebook.py` regenerates the `.ipynb` from plain Python
(`python notebooks/_build_text_features_notebook.py`) so it stays reviewable in
diffs.
