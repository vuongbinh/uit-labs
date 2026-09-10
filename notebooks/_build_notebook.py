"""Generator for notebooks/vietnam_real_estate_baseline.ipynb.

Kept in the repo so the notebook can be regenerated / diffed as plain Python.
Run:  python notebooks/_build_notebook.py
"""
import json
from pathlib import Path

cells = []


def md(text: str):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    })


def code(text: str):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


md(r"""
# Vietnam Real Estate — Tabular Price Prediction Baseline

End-to-end **LightGBM / CatBoost** baseline for predicting residential real-estate
prices in Vietnam from the [`tinixai/vietnam-real-estates`](https://huggingface.co/datasets/tinixai/vietnam-real-estates)
dataset (~3.5M listings).

**Targets**

| | Target | Notes |
|---|---|---|
| Primary | Total price (`price`, VND) | modeled as `log1p(price)` |
| Secondary | Price per m² (`price / area`) | reported for reference |

**What this notebook does**

1. Installs dependencies (Colab-ready).
2. Loads the dataset — *fast prototyping mode* (stratified 100k sample or metro
   filter) or *full-dataset mode*.
3. Cleans anomalous records and engineers location + physical features.
4. Trains a gradient-boosted regressor on `log1p(price)` with early stopping.
5. Reports RMSLE, MAPE, MAE (tỷ VND), MedAE, R² — for total price and price/m².
6. Plots feature importance, actual-vs-predicted, and residual distribution.

> **1-click Colab:** `Runtime → Run all`. Runs on the free CPU tier in fast mode.
""")

md("## 1. Environment setup")

code(r"""
# Colab-ready dependency install. Safe to re-run; skip if already satisfied.
import importlib, subprocess, sys

REQUIRED = {
    "lightgbm": "lightgbm>=4.0",
    "catboost": "catboost>=1.2",
    "datasets": "datasets>=2.14",
    "sklearn": "scikit-learn>=1.3",
    "matplotlib": "matplotlib>=3.7",
    "seaborn": "seaborn>=0.12",
    "pandas": "pandas>=2.0",
    "pyarrow": "pyarrow>=12.0",
}

missing = [pip_name for mod, pip_name in REQUIRED.items()
           if importlib.util.find_spec(mod) is None]
if missing:
    print("Installing:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
else:
    print("All dependencies already present.")
""")

code(r"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)
""")

md("## 2. Configuration")

md(r"""
Everything you might want to tweak lives here.

* `SAMPLE_MODE`
  * `"sample"` — stratified random sample of `SAMPLE_SIZE` rows (default, fast).
  * `"metro"`  — keep only cities in `METRO_CITIES` (Hanoi / HCMC by default).
  * `"full"`   — the entire dataset (needs plenty of RAM; use a high-RAM runtime).
* `PRIMARY_TARGET` / `SECONDARY_TARGET` — never touch unless the schema changes.
* `MODEL_BACKEND` — `"lightgbm"`, `"catboost"`, or `"both"` (trains and compares).
""")

code(r"""
# ----- Data loading -----------------------------------------------------------
HF_DATASET = "tinixai/vietnam-real-estates"
HF_SPLIT = "train"

SAMPLE_MODE = "sample"          # "sample" | "metro" | "full"
SAMPLE_SIZE = 100_000
METRO_CITIES = ["Hà Nội", "Hồ Chí Minh", "Ha Noi", "Ho Chi Minh", "TP HCM", "Hanoi"]
STRATIFY_ON = "province_name"   # column used to stratify the fast-mode sample

# ----- Schema (edit here if the upstream column names change) -----------------
COL = {
    "price": "price",             # total price, VND
    "area": "area",               # usable area, m^2
    "province": "province_name",
    "district": "district_name",
    "ward": "ward_name",
    "street": "street_name",
    "bedrooms": "bedroom",
    "bathrooms": "bathroom",
    "floors": "floor",
    "road_width": "road_width",
    "frontage": "frontage_width",
    "length": "length",
    "width": "width",
    "house_direction": "house_direction",
    "legal": "legal_status",
    "property_type": "property_type",
    "lat": "latitude",
    "lon": "longitude",
    "post_date": "post_date",
}

PRIMARY_TARGET = "price"
SECONDARY_TARGET = "price_per_m2"

# ----- Cleaning bounds (VND / m^2) -------------------------------------------
MIN_PRICE_VND = 50_000_000            # 50 triệu — below this is almost always noise
MAX_PRICE_VND = 500_000_000_000       # 500 tỷ  — above this is almost always noise
MIN_AREA_M2 = 5
MAX_AREA_M2 = 10_000
MIN_PPM2_VND = 1_000_000             # 1 triệu / m^2
MAX_PPM2_VND = 2_000_000_000         # 2 tỷ / m^2

# ----- Modeling --------------------------------------------------------------
MODEL_BACKEND = "lightgbm"      # "lightgbm" | "catboost" | "both"
TEST_SIZE = 0.15
VALID_SIZE = 0.15
N_ESTIMATORS = 5_000
EARLY_STOPPING_ROUNDS = 200
VND_PER_TY = 1_000_000_000       # 1 tỷ = 1e9 VND
""")

md("## 3. Load the dataset")

md(r"""
The dataset is streamed from the Hugging Face Hub. In `sample` mode we stream and
reservoir-fill a stratified sample so we never materialize all 3.5M rows; in
`full` mode we download the Arrow shards and load them directly.
""")

code(r"""
from datasets import load_dataset


def _to_frame(ds) -> pd.DataFrame:
    df = ds.to_pandas() if not isinstance(ds, pd.DataFrame) else ds
    return df


def load_real_estate(mode: str) -> pd.DataFrame:
    if mode == "full":
        ds = load_dataset(HF_DATASET, split=HF_SPLIT)
        return _to_frame(ds)

    # Stream to avoid pulling the whole dataset into memory.
    stream = load_dataset(HF_DATASET, split=HF_SPLIT, streaming=True)

    if mode == "metro":
        wanted = {c.lower() for c in METRO_CITIES}
        rows, seen = [], 0
        for rec in stream:
            seen += 1
            prov = str(rec.get(COL["province"], "")).lower()
            if any(w in prov or prov in w for w in wanted):
                rows.append(rec)
            if seen % 250_000 == 0:
                print(f"  scanned {seen:,} rows, kept {len(rows):,}")
        return pd.DataFrame(rows)

    if mode == "sample":
        # Two-pass-free approximate stratified sample via per-key reservoirs.
        from collections import defaultdict
        reservoirs = defaultdict(list)
        counts = defaultdict(int)
        per_key_cap = max(200, SAMPLE_SIZE // 60)  # ~60 provinces
        seen = 0
        for rec in stream:
            seen += 1
            key = rec.get(STRATIFY_ON, "NA")
            counts[key] += 1
            res = reservoirs[key]
            if len(res) < per_key_cap:
                res.append(rec)
            else:
                j = np.random.randint(0, counts[key])
                if j < per_key_cap:
                    res[j] = rec
            if seen % 250_000 == 0:
                kept = sum(len(v) for v in reservoirs.values())
                print(f"  scanned {seen:,} rows, buffered {kept:,}")
            if seen >= 3_600_000:
                break
        pool = pd.DataFrame([r for res in reservoirs.values() for r in res])
        if len(pool) > SAMPLE_SIZE:
            pool = pool.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
        return pool.reset_index(drop=True)

    raise ValueError(f"Unknown SAMPLE_MODE: {mode!r}")


raw = load_real_estate(SAMPLE_MODE)
print(f"\nLoaded {len(raw):,} rows x {raw.shape[1]} columns  (mode={SAMPLE_MODE})")
raw.head()
""")

code(r"""
# Normalize the schema: keep only columns we know about, rename to canonical keys.
present = {k: v for k, v in COL.items() if v in raw.columns}
missing = {k: v for k, v in COL.items() if v not in raw.columns}
if missing:
    print("NOTE: columns not found in this dataset snapshot -> skipped:")
    for k, v in missing.items():
        print(f"  {k:16s} ({v})")

df = raw.rename(columns={v: k for k, v in present.items()}).copy()
df = df[[k for k in COL if k in df.columns]].copy()
df.info()
""")

md("## 4. Exploratory sanity checks")

code(r"""
numeric_cols = [c for c in ["price", "area", "bedrooms", "bathrooms", "floors",
                            "road_width", "frontage", "length", "width"]
                if c in df.columns]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

display(df[numeric_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T)
print("\nMissing values (%):")
print((df.isna().mean() * 100).round(2).sort_values(ascending=False))
""")

code(r"""
if "price" in df.columns and "area" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    valid = df[(df["price"] > 0) & (df["area"] > 0)]
    axes[0].hist(np.log1p(valid["price"]), bins=80, color="#4C72B0")
    axes[0].set_title("log1p(price) distribution")
    axes[0].set_xlabel("log1p(VND)")
    ppm2 = (valid["price"] / valid["area"]).clip(upper=valid["price"].quantile(0.999))
    axes[1].hist(np.log1p(ppm2), bins=80, color="#55A868")
    axes[1].set_title("log1p(price / m²) distribution")
    axes[1].set_xlabel("log1p(VND per m²)")
    plt.tight_layout()
    plt.show()
""")

md("## 5. Cleaning")

md(r"""
Drop records that cannot be real listings:

* non-positive / missing `price` or `area`,
* prices outside `[MIN_PRICE_VND, MAX_PRICE_VND]`,
* areas outside `[MIN_AREA_M2, MAX_AREA_M2]`,
* implied price-per-m² outside `[MIN_PPM2_VND, MAX_PPM2_VND]`,
* exact duplicate rows.
""")

code(r"""
def clean(frame: pd.DataFrame) -> pd.DataFrame:
    n0 = len(frame)
    f = frame.copy()

    f = f.dropna(subset=["price", "area"])
    f = f[(f["price"] > 0) & (f["area"] > 0)]
    f = f[f["price"].between(MIN_PRICE_VND, MAX_PRICE_VND)]
    f = f[f["area"].between(MIN_AREA_M2, MAX_AREA_M2)]

    ppm2 = f["price"] / f["area"]
    f = f[ppm2.between(MIN_PPM2_VND, MAX_PPM2_VND)]

    for c in ["bedrooms", "bathrooms", "floors"]:
        if c in f.columns:
            f.loc[f[c].notna() & ((f[c] < 0) | (f[c] > 50)), c] = np.nan
    for c in ["road_width", "frontage", "length", "width"]:
        if c in f.columns:
            f.loc[f[c].notna() & ((f[c] <= 0) | (f[c] > 200)), c] = np.nan

    f = f.drop_duplicates()
    print(f"Cleaning: {n0:,} -> {len(f):,} rows  ({100*(n0-len(f))/max(n0,1):.1f}% dropped)")
    return f.reset_index(drop=True)


clean_df = clean(df)
""")

md("## 6. Feature engineering")

md(r"""
* **Secondary target** `price_per_m2 = price / area`.
* **Location encoding** — for each of `province / district / ward`:
  * *frequency encoding* (share of listings), and
  * *out-of-fold target encoding* on `log1p(price)` (K-fold to avoid leakage),
    with a global-mean smoothing prior.
* **Physical ratios** — `road_frontage_ratio = road_width / frontage`,
  `aspect_ratio = length / width`, `plot_area ≈ length * width`,
  `floor_area ≈ area * floors`, `area_per_room`, and log transforms of the
  skewed size columns.
* **Temporal** — year / month from `post_date` if present.
""")

code(r"""
from sklearn.model_selection import KFold

GEO_COLS = [c for c in ["province", "district", "ward", "street"] if c in clean_df.columns]


def frequency_encode(frame, cols):
    for c in cols:
        freq = frame[c].value_counts(normalize=True)
        frame[f"{c}_freq"] = frame[c].map(freq).astype("float32")
    return frame


def kfold_target_encode(frame, cols, target_log, n_splits=5, smoothing=50):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    global_mean = target_log.mean()
    for c in cols:
        oof = np.full(len(frame), np.nan, dtype="float64")
        for tr_idx, val_idx in kf.split(frame):
            tr = frame.iloc[tr_idx]
            stats = target_log.iloc[tr_idx].groupby(tr[c]).agg(["mean", "count"])
            smooth = ((stats["mean"] * stats["count"] + global_mean * smoothing)
                      / (stats["count"] + smoothing))
            oof[val_idx] = frame.iloc[val_idx][c].map(smooth).values
        frame[f"{c}_te"] = pd.Series(oof, index=frame.index).fillna(global_mean).astype("float32")
    return frame


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    f[SECONDARY_TARGET] = (f["price"] / f["area"]).astype("float64")

    target_log = np.log1p(f["price"])
    f = frequency_encode(f, GEO_COLS)
    f = kfold_target_encode(f, GEO_COLS, target_log)

    if {"road_width", "frontage"}.issubset(f.columns):
        f["road_frontage_ratio"] = (f["road_width"] / f["frontage"]).replace([np.inf, -np.inf], np.nan)
    if {"length", "width"}.issubset(f.columns):
        f["aspect_ratio"] = (f["length"] / f["width"]).replace([np.inf, -np.inf], np.nan)
        f["plot_area"] = (f["length"] * f["width"]).clip(upper=MAX_AREA_M2)
    if "floors" in f.columns:
        f["floor_area"] = (f["area"] * f["floors"].fillna(1).clip(lower=1))
    rooms = f[[c for c in ["bedrooms", "bathrooms"] if c in f.columns]].sum(axis=1)
    if rooms.gt(0).any():
        f["area_per_room"] = f["area"] / rooms.replace(0, np.nan)

    for c in ["area", "road_width", "frontage", "length", "width", "plot_area", "floor_area"]:
        if c in f.columns:
            f[f"log_{c}"] = np.log1p(f[c].clip(lower=0))

    if "post_date" in f.columns:
        dt = pd.to_datetime(f["post_date"], errors="coerce")
        f["post_year"] = dt.dt.year
        f["post_month"] = dt.dt.month

    return f


feat_df = engineer(clean_df)
new_cols = sorted(set(feat_df.columns) - set(clean_df.columns))
print("Engineered columns:", new_cols)
feat_df[new_cols].head()
""")

code(r"""
# Assemble the model matrix.
DROP_FROM_X = {PRIMARY_TARGET, SECONDARY_TARGET, "post_date", "lat", "lon"}
RAW_GEO_STRINGS = set(GEO_COLS)  # keep encoded versions, drop raw high-cardinality strings

categorical_features = [c for c in ["house_direction", "legal", "property_type"]
                        if c in feat_df.columns]
for c in categorical_features:
    feat_df[c] = feat_df[c].astype("category")

feature_cols = [c for c in feat_df.columns
                if c not in DROP_FROM_X and c not in RAW_GEO_STRINGS]
numeric_features = [c for c in feature_cols if c not in categorical_features]
for c in numeric_features:
    feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce").astype("float32")

X = feat_df[feature_cols]
y_price = feat_df[PRIMARY_TARGET].astype("float64")
y_log = np.log1p(y_price)

print(f"X: {X.shape}  |  {len(categorical_features)} categorical, {len(numeric_features)} numeric")
feature_cols
""")

md("## 7. Train / validation / test split")

code(r"""
from sklearn.model_selection import train_test_split

idx = np.arange(len(X))
train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, valid_idx = train_test_split(
    train_idx, test_size=VALID_SIZE / (1 - TEST_SIZE), random_state=RANDOM_STATE)

X_tr, X_va, X_te = X.iloc[train_idx], X.iloc[valid_idx], X.iloc[test_idx]
yl_tr, yl_va, yl_te = y_log.iloc[train_idx], y_log.iloc[valid_idx], y_log.iloc[test_idx]
yp_te = y_price.iloc[test_idx]
area_te = feat_df["area"].iloc[test_idx].to_numpy()

print(f"train={len(X_tr):,}  valid={len(X_va):,}  test={len(X_te):,}")
""")

md("## 8. Model training")

md(r"""
Both backends regress on `log1p(price)` (equivalent to an RMSLE objective on the
original scale) with early stopping on the validation fold.
""")

code(r"""
def train_lightgbm():
    import lightgbm as lgb
    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=255,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_child_samples=40,
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, yl_tr,
        eval_set=[(X_va, yl_va)],
        eval_metric="rmse",
        categorical_feature=categorical_features or "auto",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(200)],
    )
    print(f"LightGBM best iteration: {model.best_iteration_}")
    return model


def train_catboost():
    from catboost import CatBoostRegressor, Pool
    cat_idx = [X_tr.columns.get_loc(c) for c in categorical_features]
    tr_pool = Pool(X_tr.fillna(-999), yl_tr, cat_features=cat_idx)
    va_pool = Pool(X_va.fillna(-999), yl_va, cat_features=cat_idx)
    model = CatBoostRegressor(
        loss_function="RMSE",
        learning_rate=0.03,
        depth=8,
        iterations=N_ESTIMATORS,
        random_seed=RANDOM_STATE,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=200,
    )
    model.fit(tr_pool, eval_set=va_pool)
    print(f"CatBoost best iteration: {model.get_best_iteration()}")
    return model


def predict(model, frame):
    name = type(model).__name__
    if name.startswith("CatBoost"):
        return model.predict(frame.fillna(-999))
    return model.predict(frame)


models = {}
if MODEL_BACKEND in ("lightgbm", "both"):
    models["LightGBM"] = train_lightgbm()
if MODEL_BACKEND in ("catboost", "both"):
    models["CatBoost"] = train_catboost()
""")

md("## 9. Evaluation")

md(r"""
Metrics on the held-out test set:

| Metric | Scale | Meaning |
|---|---|---|
| RMSLE | log VND | root mean squared log error (the training objective) |
| MAPE | % | mean absolute percentage error on total price |
| MAE | tỷ VND | mean absolute error |
| MedAE | tỷ VND | median absolute error (robust) |
| R² | — | variance explained on total price |

Reported for the **primary** target (total price) and the **secondary** target
(price per m², derived as `pred_price / area`).
""")

code(r"""
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error


def regression_report(y_true, y_pred, label):
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.clip(np.asarray(y_pred, dtype="float64"), 0, None)
    rmsle = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
    mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    mae_ty = mean_absolute_error(y_true, y_pred) / VND_PER_TY
    medae_ty = median_absolute_error(y_true, y_pred) / VND_PER_TY
    r2 = r2_score(y_true, y_pred)
    return {
        "target": label, "RMSLE": rmsle, "MAPE_%": mape,
        "MAE_tyVND": mae_ty, "MedAE_tyVND": medae_ty, "R2": r2,
    }


rows = []
predictions = {}
for name, model in models.items():
    pred_log = predict(model, X_te)
    pred_price = np.expm1(pred_log)
    predictions[name] = pred_price

    r_price = regression_report(yp_te, pred_price, "total_price")
    r_price["model"] = name
    rows.append(r_price)

    mask = area_te > 0
    r_ppm2 = regression_report(
        (yp_te.to_numpy()[mask] / area_te[mask]),
        (pred_price[mask] / area_te[mask]),
        "price_per_m2",
    )
    r_ppm2["model"] = name
    rows.append(r_ppm2)

report = pd.DataFrame(rows)[["model", "target", "RMSLE", "MAPE_%",
                             "MAE_tyVND", "MedAE_tyVND", "R2"]]
report = report.round({"RMSLE": 4, "MAPE_%": 2, "MAE_tyVND": 3,
                       "MedAE_tyVND": 3, "R2": 4})
display(report)

best_model_name = report.query("target == 'total_price'").sort_values("RMSLE").iloc[0]["model"]
print(f"\nBest by RMSLE (total price): {best_model_name}")
""")

md("## 10. Visualizations")

code(r"""
best_model = models[best_model_name]
best_pred = predictions[best_model_name]


def get_importance(model):
    name = type(model).__name__
    if name.startswith("CatBoost"):
        imp = model.get_feature_importance()
    else:
        imp = model.feature_importances_
    return pd.Series(imp, index=feature_cols).sort_values(ascending=False)


imp = get_importance(best_model).head(25)
plt.figure(figsize=(9, 8))
sns.barplot(x=imp.values, y=imp.index, color="#4C72B0")
plt.title(f"{best_model_name} — top {len(imp)} feature importances")
plt.xlabel("importance")
plt.tight_layout()
plt.show()
""")

code(r"""
# Actual vs predicted (log-log) with y = x reference.
lo = np.log1p(min(yp_te.min(), best_pred.min()))
hi = np.log1p(max(yp_te.max(), best_pred.max()))

plt.figure(figsize=(6.5, 6.5))
plt.scatter(np.log1p(yp_te), np.log1p(np.clip(best_pred, 0, None)),
            s=6, alpha=0.15, color="#4C72B0")
plt.plot([lo, hi], [lo, hi], "r--", lw=2, label="y = x")
plt.xlabel("log1p(actual price)")
plt.ylabel("log1p(predicted price)")
plt.title(f"{best_model_name} — actual vs predicted")
plt.legend()
plt.tight_layout()
plt.show()
""")

code(r"""
# Residual distribution (relative error on total price).
rel_err = (best_pred - yp_te.to_numpy()) / yp_te.to_numpy()
rel_err = np.clip(rel_err, -2, 2)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(rel_err, bins=100, color="#55A868")
axes[0].axvline(0, color="r", ls="--")
axes[0].set_title("Relative error  (pred - actual) / actual")
axes[0].set_xlabel("relative error")

resid_log = np.log1p(np.clip(best_pred, 0, None)) - np.log1p(yp_te.to_numpy())
axes[1].scatter(np.log1p(yp_te), resid_log, s=6, alpha=0.15, color="#C44E52")
axes[1].axhline(0, color="r", ls="--")
axes[1].set_title("Log-residual vs actual")
axes[1].set_xlabel("log1p(actual price)")
axes[1].set_ylabel("log residual")
plt.tight_layout()
plt.show()
""")

md("## 11. Summary")

code(r"""
print("=" * 64)
print(f"Dataset         : {HF_DATASET}  (mode={SAMPLE_MODE})")
print(f"Rows (clean)    : {len(feat_df):,}")
print(f"Features        : {len(feature_cols)}")
print(f"Best model      : {best_model_name}")
print("-" * 64)
display(report.set_index(["model", "target"]))
print("=" * 64)
print("Next steps: geo-spatial features from lat/lon, log-price quantile models,")
print("           hyper-parameter search, and full-dataset training on a GPU runtime.")
""")

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "vietnam_real_estate_baseline.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {out}  ({len(cells)} cells)")
