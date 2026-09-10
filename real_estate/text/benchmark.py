"""Marginal-uplift benchmark: how much do text features add to a tabular baseline?

This module owns a deliberately small *reference* tabular baseline so the text
features can be measured against something concrete before the project pipeline
(HYPE-12) lands.  It is not a competitor to that pipeline: the arms are defined
as "reference tabular block + text group", and :func:`run_uplift_benchmark`
accepts a pre-built tabular matrix so a richer baseline can be dropped in.

Evaluation protocol
-------------------
* Out-of-time: train on an early shard, test on a later shard (no random split,
  so no look-ahead across the listing timeline).
* Target is ``log1p(price)``; metrics are reported in natural units.
* Early stopping uses a time-ordered slice of the *training* period only.
* Every fitted encoder (target encoding, TF-IDF, SVD, PhoBERT reduction) sees
  training rows only.
* Each arm is run over several seeds and reported as mean ± spread, because an
  uplift smaller than the seed-to-seed noise is not a finding.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
    clean_frame,
    load_shard_sample,
    time_span,
)
from .entities import (
    ENTITY_COLUMNS,
    TARGET_LEAKING_ENTITIES,
    TEXT_IMPUTABLE_COLUMNS,
    impute_from_text,
)
from .keywords import KeywordExtractor
from .pipeline import TextFeatureConfig, TextFeaturePipeline

__all__ = [
    "VND_PER_TY",
    "regression_metrics",
    "TabularEncoder",
    "build_reference_tabular",
    "entity_agreement_report",
    "run_uplift_benchmark",
    "format_report",
    "main",
]

VND_PER_TY = 1e9  # one "tỷ" = 1e9 VND


# --------------------------------------------------------------------- metrics
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Price-prediction metrics in natural units.

    ``y_true`` / ``y_pred`` are VND prices.  RMSLE is the RMSE of ``log1p`` values,
    the primary objective; MAE and MedAE are additionally reported in tỷ VND
    because that is the unit Vietnamese listings quote in.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        metrics = {k: float("nan") for k in ("rmsle", "mape_pct", "mae_ty", "medae_ty", "r2")}
        metrics["n"] = 0
        return metrics

    log_true, log_pred = np.log1p(y_true), np.log1p(y_pred)
    abs_error = np.abs(y_pred - y_true)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmsle": float(np.sqrt(np.mean((log_true - log_pred) ** 2))),
        "mape_pct": float(np.mean(abs_error / y_true) * 100.0),
        "mae_ty": float(np.mean(abs_error) / VND_PER_TY),
        "medae_ty": float(np.median(abs_error) / VND_PER_TY),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n": int(y_true.size),
    }


# ------------------------------------------------------------ tabular encoder
@dataclass
class TabularEncoder:
    """Reference tabular feature block: numerics, ratios and geographic encodings.

    Fitted state is the category frequency map and the smoothed per-category mean
    of ``log1p(price)``.  Unknown categories at transform time fall back to the
    training global mean, which keeps the encoding leak-free and total.
    """

    geo_target_columns: tuple[str, ...] = ("province_name", "district_name", "ward_name")
    freq_columns: tuple[str, ...] = (
        "property_type_name",
        "province_name",
        "district_name",
        "ward_name",
        "street_name",
        "project_name",
        "house_direction",
        "balcony_direction",
    )
    smoothing: float = 50.0
    freq_prefix: str = "freq_"
    target_prefix: str = "tgt_"
    _freq: dict = field(default_factory=dict, init=False, repr=False)
    _target: dict = field(default_factory=dict, init=False, repr=False)
    _global_mean: float = field(default=float("nan"), init=False, repr=False)

    @staticmethod
    def _numeric_block(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for column in NUMERIC_COLUMNS:
            out[column] = (
                pd.to_numeric(df[column], errors="coerce").astype("float64")
                if column in df.columns
                else np.nan
            )
        area, frontage = out.get("area"), out.get("frontage_width")
        depth, road = out.get("house_depth"), out.get("road_width")
        if frontage is not None and depth is not None:
            out["ratio_frontage_to_depth"] = frontage / depth.replace(0, np.nan)
        if road is not None and frontage is not None:
            out["ratio_road_to_frontage"] = road / frontage.replace(0, np.nan)
        if area is not None and frontage is not None:
            out["implied_depth_from_area"] = area / frontage.replace(0, np.nan)
        if TIME_COLUMN in df.columns:
            stamps = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
            out["published_month"] = stamps.dt.year * 12 + stamps.dt.month
            out["published_day_of_week"] = stamps.dt.dayofweek
        return out

    def fit(self, df: pd.DataFrame) -> TabularEncoder:
        """Learn frequencies and smoothed target means from training rows."""
        log_price = np.log1p(pd.to_numeric(df[TARGET_COLUMN], errors="coerce").astype("float64"))
        self._global_mean = float(np.nanmean(log_price))
        for column in self.freq_columns:
            if column not in df.columns:
                continue
            values = df[column].astype("string").fillna("__missing__")
            counts = values.value_counts()
            self._freq[column] = counts
            if column in self.geo_target_columns:
                grouped = pd.DataFrame({"v": values, "y": log_price}).groupby("v")["y"]
                means, sizes = grouped.mean(), grouped.size()
                self._target[column] = (
                    (means * sizes + self._global_mean * self.smoothing) / (sizes + self.smoothing)
                )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode ``df`` using fitted state only."""
        if not self._freq and not np.isfinite(self._global_mean):
            raise RuntimeError("TabularEncoder.transform called before fit")
        out = self._numeric_block(df)
        for column in self.geo_target_columns:
            if column in df.columns and column in self._target:
                values = df[column].astype("string").fillna("__missing__")
                out[f"{self.target_prefix}{column}"] = (
                    values.map(self._target[column]).astype("float64").fillna(self._global_mean)
                )
        for column, counts in self._freq.items():
            if column not in df.columns:
                continue
            values = df[column].astype("string").fillna("__missing__")
            encoded = values.map(counts).astype("float64")
            out[f"{self.freq_prefix}{column}"] = np.log1p(encoded.fillna(0.0))
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``df`` and return its encoding."""
        return self.fit(df).transform(df)


def build_reference_tabular(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper returning fitted train/test reference tabular blocks."""
    encoder = TabularEncoder().fit(train)
    return encoder.transform(train), encoder.transform(test)


# ------------------------------------------------------- entity sanity report
def entity_agreement_report(df: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
    """How often does a text-parsed entity match the structured column?

    Reported per entity: coverage (text produced a value), overlap (both present),
    and agreement among overlapping rows.  ``price`` uses a 5% relative tolerance
    because listings round ("7.45 tỷ" against a recorded 7,450,000,000 VND);
    counts and dimensions must match exactly.
    """
    pairs = list(TEXT_IMPUTABLE_COLUMNS.items()) + [("price", "text_price_vnd")]
    rows = []
    for structured, parsed in pairs:
        if structured not in df.columns or parsed not in entities.columns:
            continue
        left = pd.to_numeric(df[structured], errors="coerce")
        right = entities[parsed]
        both = left.notna() & (left > 0) & right.notna()
        n_both = int(both.sum())
        if structured == "price":
            agree = (np.abs(left[both] - right[both]) / left[both].clip(lower=1)) <= 0.05
        else:
            agree = np.isclose(left[both], right[both], rtol=0.02, atol=0.51)
        rows.append(
            {
                "structured_column": structured,
                "text_entity": parsed,
                "text_coverage": float(right.notna().mean()),
                "structured_missing_rate": float((left.isna() | (left <= 0)).mean()),
                "overlap_rows": n_both,
                "agreement_rate": float(agree.mean()) if n_both else float("nan"),
                "recovered_when_missing": int(
                    ((left.isna() | (left <= 0)) & right.notna()).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- modelling
def _time_ordered_valid(train: pd.DataFrame, valid_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Split training rows by ``published_at`` so early stopping cannot see the end."""
    stamps = pd.to_datetime(train[TIME_COLUMN], errors="coerce")
    if stamps.isna().all():
        n_valid = max(1, int(len(train) * valid_fraction))
        return np.arange(len(train) - n_valid), np.arange(len(train) - n_valid, len(train))
    order = np.argsort(stamps.to_numpy(dtype="datetime64[ns]"), kind="stable")
    cut = int(len(order) * (1.0 - valid_fraction))
    return order[:cut], order[cut:]


def _fit_lgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
) -> np.ndarray:
    """Fit LightGBM on ``log1p(price)`` and return VND predictions for the test block."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=40,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return np.expm1(model.predict(x_test))


def _feature_importance_top(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    names: Sequence[str],
    *,
    seed: int,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    top_k: int = 25,
) -> list[tuple[str, float]]:
    """Top-``k`` gain-importance features for the richest arm (diagnostic only)."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=40,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        importance_type="gain",
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    gains = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(-gains)[:top_k]
    total = float(gains.sum()) or 1.0
    return [(str(names[i]), float(gains[i] / total)) for i in order]


# ------------------------------------------------------------------ the ladder
@dataclass
class BenchmarkConfig:
    """Everything the ablation ladder needs to run."""

    n_train: int = 100_000
    n_test: int = 40_000
    seed: int = 0
    n_seeds: int = 2
    valid_fraction: float = 0.15
    n_estimators: int = 1_200
    learning_rate: float = 0.06
    num_leaves: int = 63
    exclude_target_leaking_entities: bool = True
    redact_price_in_text: bool = True
    tfidf_mode: str = "word"
    tfidf_components: int = 96
    tfidf_max_features: int = 60_000
    tfidf_min_df: int = 5
    use_phobert: bool = False
    phobert: dict = field(default_factory=dict)
    # Mirrors clean_frame's defaults, which follow the EDA's recommended Rule 1.
    min_price: float = 1e8
    max_price: float = 2e11


def run_uplift_benchmark(
    train_source: str | Path,
    test_source: str | Path,
    config: BenchmarkConfig | None = None,
    cache_dir: str | Path | None = None,
    tabular_train: pd.DataFrame | None = None,
    tabular_test: pd.DataFrame | None = None,
) -> dict:
    """Run the ablation ladder and return a JSON-serialisable report.

    Arms, each a superset of the previous:

    ``tabular``      reference tabular block only
    ``+keywords``    adds the ``kw_*`` domain flags
    ``+entities``    adds parsed numeric entities, provenance flags and imputation
    ``+tfidf``       adds TF-IDF/SVD components
    ``+phobert``     adds PhoBERT embeddings (only when ``use_phobert``)
    """
    config = config or BenchmarkConfig()
    started = time.time()

    keyword_extractor = KeywordExtractor()
    raw_train = load_shard_sample(
        train_source, n_rows=config.n_train, seed=config.seed, cache_dir=cache_dir
    )
    raw_test = load_shard_sample(
        test_source, n_rows=config.n_test, seed=config.seed + 1, cache_dir=cache_dir
    )
    # Flags are computed on the raw rows so prevalence reporting below reflects
    # the full corpus rather than the cleaned subset.
    raw_train_flags = keyword_extractor.extract(raw_train)
    raw_test_flags = keyword_extractor.extract(raw_test)

    # Cleaning is by price/area bounds only. The min_price floor is what excludes
    # genuine rental listings (monthly rates of 15-60M VND); the kw_cho_thue text
    # flag must NOT be used for this, because 96% of the rows it fires on are sale
    # listings pitching rental yield at a sale-scale price. See clean_frame.
    train = clean_frame(raw_train, min_price=config.min_price, max_price=config.max_price)
    test = clean_frame(raw_test, min_price=config.min_price, max_price=config.max_price)
    # clean_frame preserves labels, so this selects exactly the surviving rows.
    train_flags = raw_train_flags.loc[train.index]
    test_flags = raw_test_flags.loc[test.index]

    text_config = TextFeatureConfig(
        use_keywords=False,  # flags already computed above, before cleaning
        use_entities=True,
        use_tfidf=True,
        use_phobert=config.use_phobert,
        tfidf_mode=config.tfidf_mode,
        tfidf_components=config.tfidf_components,
        tfidf_max_features=config.tfidf_max_features,
        tfidf_min_df=config.tfidf_min_df,
        redact_price=config.redact_price_in_text,
        phobert=config.phobert,
        seed=config.seed,
    )
    pipeline = TextFeaturePipeline(text_config)
    train_text = pipeline.fit_transform(train)  # encodes the train text exactly once
    test_text = pipeline.transform(test)

    entity_cols = [c for c in ENTITY_COLUMNS if c in train_text.columns]
    # `text_price_vnd` restates the regression target — the asking price is written
    # into the listing itself — so it is measured for extraction quality but never
    # handed to the model.
    model_entity_cols = (
        [c for c in entity_cols if c not in TARGET_LEAKING_ENTITIES]
        if config.exclude_target_leaking_entities
        else entity_cols
    )
    provenance_cols = [c for c in train_text.columns if c.endswith("__from_text")]
    tfidf_cols = [c for c in train_text.columns if c.startswith("tfidf_")]
    phobert_cols = [c for c in train_text.columns if c.startswith("phobert_")]

    tab_encoder = TabularEncoder().fit(train)
    tab_train = tab_encoder.transform(train)
    tab_test = tab_encoder.transform(test)

    def with_text_imputation(frame: pd.DataFrame, text: pd.DataFrame) -> pd.DataFrame:
        """Substitute text-recovered values into the sparse structured columns.

        Imputation happens *before* tabular encoding so derived ratios
        (frontage/depth, road/frontage) are computed from the recovered numbers
        rather than from the original nulls.
        """
        numeric_present = [c for c in NUMERIC_COLUMNS if c in frame.columns]
        if not numeric_present or not model_entity_cols:
            return frame
        out = frame.copy()
        filled = impute_from_text(frame[numeric_present], text[model_entity_cols])
        for column in numeric_present:
            out[column] = filled[column]
        return out

    tab_train_entities = tab_encoder.transform(with_text_imputation(train, train_text))
    tab_test_entities = tab_encoder.transform(with_text_imputation(test, test_text))
    entity_train = train_text[model_entity_cols + provenance_cols]
    entity_test = test_text[model_entity_cols + provenance_cols]

    arms: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "tabular": (tab_train, tab_test),
        "+keywords": (
            pd.concat([tab_train, train_flags], axis=1),
            pd.concat([tab_test, test_flags], axis=1),
        ),
        "+entities": (
            pd.concat([tab_train_entities, train_flags, entity_train], axis=1),
            pd.concat([tab_test_entities, test_flags, entity_test], axis=1),
        ),
    }
    if tfidf_cols:
        arms["+tfidf"] = (
            pd.concat([arms["+entities"][0], train_text[tfidf_cols]], axis=1),
            pd.concat([arms["+entities"][1], test_text[tfidf_cols]], axis=1),
        )
    if phobert_cols:
        base = arms.get("+tfidf", arms["+entities"])
        arms["+phobert"] = (
            pd.concat([base[0], train_text[phobert_cols]], axis=1),
            pd.concat([base[1], test_text[phobert_cols]], axis=1),
        )

    y_train_all = np.log1p(train[TARGET_COLUMN].to_numpy(dtype=np.float64))
    y_test = test[TARGET_COLUMN].to_numpy(dtype=np.float64)
    fit_idx, valid_idx = _time_ordered_valid(train, config.valid_fraction)

    results: dict[str, dict] = {}
    richest: str = list(arms)[-1]
    for arm_name, (x_tr, x_te) in arms.items():
        x_train_np = x_tr.to_numpy(dtype=np.float32, na_value=np.nan)
        x_test_np = x_te.to_numpy(dtype=np.float32, na_value=np.nan)
        per_seed = []
        for offset in range(config.n_seeds):
            seed = config.seed + offset
            predictions = _fit_lgbm(
                x_train_np[fit_idx],
                y_train_all[fit_idx],
                x_train_np[valid_idx],
                y_train_all[valid_idx],
                x_test_np,
                seed=seed,
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                num_leaves=config.num_leaves,
            )
            per_seed.append(regression_metrics(y_test, predictions))
        metrics = {
            key: float(np.mean([run[key] for run in per_seed]))
            for key in per_seed[0]
            if key != "n"
        }
        spread = {
            f"{key}_std": float(np.std([run[key] for run in per_seed]))
            for key in ("rmsle", "mape_pct")
        }
        importance = []
        if arm_name == richest:
            importance = _feature_importance_top(
                x_train_np[fit_idx],
                y_train_all[fit_idx],
                x_train_np[valid_idx],
                y_train_all[valid_idx],
                list(x_tr.columns),
                seed=config.seed,
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                num_leaves=config.num_leaves,
            )
        results[arm_name] = {
            "n_features": int(x_tr.shape[1]),
            "metrics": metrics,
            "seed_spread": spread,
            "per_seed": per_seed,
            "top_features": importance,
        }

    baseline = results["tabular"]["metrics"]
    deltas = {}
    for arm_name, payload in results.items():
        if arm_name == "tabular":
            continue
        metrics = payload["metrics"]
        deltas[arm_name] = {
            "rmsle_delta": metrics["rmsle"] - baseline["rmsle"],
            "rmsle_delta_pct": 100.0 * (metrics["rmsle"] - baseline["rmsle"]) / baseline["rmsle"],
            "mape_delta_pct_points": metrics["mape_pct"] - baseline["mape_pct"],
            "mae_ty_delta": metrics["mae_ty"] - baseline["mae_ty"],
            "r2_delta": metrics["r2"] - baseline["r2"],
        }

    agreement = entity_agreement_report(train, train_text[entity_cols])
    # Lexicon health is measured on the *raw* rows: cleaning removes price/area
    # outliers, and a prevalence table computed after that describes the surviving
    # subset rather than the corpus the lexicon has to cover.
    prevalence = keyword_extractor.prevalence(raw_train_flags)

    return {
        "config": asdict(config),
        "dataset": {
            "train_shard": str(train_source),
            "test_shard": str(test_source),
            "train_rows_raw": int(len(raw_train)),
            "train_rows_clean": int(len(train)),
            "test_rows_raw": int(len(raw_test)),
            "test_rows_clean": int(len(test)),
            "train_time_span": list(time_span(train)),
            "test_time_span": list(time_span(test)),
            "tfidf_vocabulary": pipeline.tfidf.vocabulary_sizes if pipeline.tfidf else {},
            "tfidf_explained_variance": pipeline.tfidf.explained_variance if pipeline.tfidf else {},
        },
        "arms": results,
        "deltas_vs_tabular": deltas,
        "entity_agreement": agreement.to_dict(orient="records"),
        "keyword_prevalence_basis": (
            f"raw training rows before cleaning (n={len(raw_train_flags):,}); cleaning drops "
            "price/area outliers, so prevalence is reported on the full corpus"
        ),
        "keyword_prevalence": prevalence.to_dict(orient="records"),
        "runtime_seconds": round(time.time() - started, 1),
    }


def format_report(report: dict) -> str:
    """Render a report as a compact markdown summary."""
    lines = ["## Text-feature uplift benchmark", ""]
    meta = report["dataset"]
    lines.append(
        f"- Train: {meta['train_rows_clean']:,} clean rows of {meta['train_rows_raw']:,} "
        f"({meta['train_time_span'][0][:10]} → {meta['train_time_span'][1][:10]})"
    )
    lines.append(
        f"- Test (out-of-time): {meta['test_rows_clean']:,} clean rows of {meta['test_rows_raw']:,} "
        f"({meta['test_time_span'][0][:10]} → {meta['test_time_span'][1][:10]})"
    )
    lines.append(f"- Runtime: {report['runtime_seconds']}s")
    lines += ["", "| arm | feats | RMSLE | MAPE % | MAE (tỷ) | MedAE (tỷ) | R² |",
              "|---|---|---|---|---|---|---|"]
    for arm_name, payload in report["arms"].items():
        m = payload["metrics"]
        lines.append(
            f"| {arm_name} | {payload['n_features']} | {m['rmsle']:.4f} | {m['mape_pct']:.2f} "
            f"| {m['mae_ty']:.3f} | {m['medae_ty']:.3f} | {m['r2']:.4f} |"
        )
    lines += ["", "### Marginal gain vs tabular-only", "",
              "| arm | ΔRMSLE | ΔRMSLE % | ΔMAPE (pts) | ΔMAE (tỷ) | ΔR² |", "|---|---|---|---|---|---|"]
    for arm_name, d in report["deltas_vs_tabular"].items():
        lines.append(
            f"| {arm_name} | {d['rmsle_delta']:+.4f} | {d['rmsle_delta_pct']:+.2f}% "
            f"| {d['mape_delta_pct_points']:+.2f} | {d['mae_ty_delta']:+.3f} | {d['r2_delta']:+.4f} |"
        )
    lines += ["", "### Entity extraction agreement (train rows)", "",
              "| structured | text entity | text coverage | col missing | overlap | agreement | recovered |",
              "|---|---|---|---|---|---|---|"]
    for row in report["entity_agreement"]:
        agree = row["agreement_rate"]
        agree_s = f"{agree:.3f}" if agree == agree else "n/a"
        lines.append(
            f"| {row['structured_column']} | {row['text_entity']} | {row['text_coverage']:.3f} "
            f"| {row['structured_missing_rate']:.3f} | {row['overlap_rows']:,} | {agree_s} "
            f"| {row['recovered_when_missing']:,} |"
        )
    top = report["arms"][list(report["arms"])[-1]].get("top_features") or []
    if top:
        lines += ["", "### Top-15 gain-importance features (richest arm)", ""]
        for name, share in top[:15]:
            lines.append(f"- `{name}` — {share * 100:.2f}% of total gain")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: run the uplift benchmark and write JSON + markdown reports."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-shard", default="shard_0000")
    parser.add_argument("--test-shard", default="shard_0009")
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-test", type=int, default=40_000)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tfidf-mode", default="word", choices=("word", "char", "both"))
    parser.add_argument("--tfidf-components", type=int, default=96)
    parser.add_argument("--tfidf-min-df", type=int, default=5)
    parser.add_argument("--use-phobert", action="store_true")
    parser.add_argument("--out-dir", default="outputs/text_features")
    args = parser.parse_args(argv)

    config = BenchmarkConfig(
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
        n_seeds=args.n_seeds,
        tfidf_mode=args.tfidf_mode,
        tfidf_components=args.tfidf_components,
        tfidf_min_df=args.tfidf_min_df,
        use_phobert=args.use_phobert,
        phobert={"n_components": 64, "max_length": 128, "batch_size": 16} if args.use_phobert else {},
    )
    report = run_uplift_benchmark(
        args.train_shard, args.test_shard, config=config, cache_dir=args.cache_dir
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "uplift_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (out_dir / "uplift_report.md").write_text(format_report(report) + "\n")
    print(format_report(report))
    print(f"\nwrote {out_dir/'uplift_report.json'} and {out_dir/'uplift_report.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
