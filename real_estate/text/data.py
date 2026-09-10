"""Loading, cleaning and splitting for ``tinixai/vietnam-real-estates``.

Two properties of the published dataset drive this module:

* ``price`` is stored as a **string** in the parquet shards even though the
  dataset card declares it ``float64``.  It is parsed here (see :func:`parse_price`)
  so nothing downstream has to cope with the mismatch.
* The shards are **chronologically ordered** — shard ``0000`` covers 2025-06 and
  shard ``0009`` covers 2026-03 — which makes a leakage-free out-of-time split as
  simple as "train on an early shard, test on a late one".
"""

from __future__ import annotations

import unicodedata
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

__all__ = [
    "DATASET_ID",
    "SHARD_URL_TEMPLATE",
    "TEXT_COLUMNS",
    "NUMERIC_COLUMNS",
    "CATEGORICAL_COLUMNS",
    "TARGET_COLUMN",
    "TIME_COLUMN",
    "resolve_shard",
    "parse_price",
    "normalize_unicode",
    "load_shard_sample",
    "clean_frame",
    "out_of_time_pair",
    "time_span",
]

DATASET_ID = "tinixai/vietnam-real-estates"
SHARD_URL_TEMPLATE = (
    "https://huggingface.co/datasets/" + DATASET_ID + "/resolve/main/{name}.parquet"
)

TEXT_COLUMNS: tuple[str, ...] = ("name", "description")
TARGET_COLUMN = "price"
TIME_COLUMN = "published_at"

NUMERIC_COLUMNS: tuple[str, ...] = (
    "area",
    "floor_count",
    "frontage_width",
    "house_depth",
    "road_width",
    "bedroom_count",
    "bathroom_count",
)

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "property_type_name",
    "province_name",
    "district_name",
    "ward_name",
    "street_name",
    "project_name",
    "house_direction",
    "balcony_direction",
)

DEFAULT_COLUMNS: tuple[str, ...] = (
    TEXT_COLUMNS + NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + (TARGET_COLUMN, TIME_COLUMN)
)


def resolve_shard(source: str | Path, cache_dir: str | Path | None = None) -> Path:
    """Return a local path for ``source``, downloading it when it names a shard.

    ``source`` may be an existing local path, a bare shard name such as
    ``"shard_0000"``, or a full URL.  Downloads land in ``cache_dir``.
    """
    text = str(source)
    if text.startswith(("http://", "https://")):
        url = text
        name = Path(text.split("?")[0]).stem
    else:
        candidate = Path(text)
        if candidate.exists():
            return candidate
        url = SHARD_URL_TEMPLATE.format(name=candidate.name)
        name = candidate.stem

    cache = Path(cache_dir) if cache_dir is not None else Path.cwd() / "data"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f"{name}.parquet"
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    tmp = destination.with_suffix(".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as handle:  # noqa: S310
        while chunk := response.read(1 << 22):
            handle.write(chunk)
    tmp.rename(destination)
    return destination


def parse_price(series: pd.Series) -> pd.Series:
    """Convert the string-typed ``price`` column to float VND.

    Parsing is deliberately strict: anything that is not a plain number becomes
    ``NaN`` rather than being coerced.  Stripping currency words and separators
    would silently turn ``"7.45 tỷ"`` into 7.45 **VND**, a billion-fold error, so
    ambiguous values are dropped for the caller to notice.  On the published
    shards every non-null value is a plain integer string.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    cleaned = series.astype("string").str.strip()
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def normalize_unicode(frame: pd.DataFrame) -> pd.DataFrame:
    """NFC-normalise every string column, in place, and return the frame.

    About 0.15% of ``street_name`` and ``project_name`` values are stored in
    NFD-decomposed form (a base letter plus separate combining marks).  They
    render identically but compare unequal, so the same street silently becomes
    two categories in any frequency or target encoding.  The geographic columns
    that matter most (``province_name``, ``district_name``, ``ward_name``) are
    already clean NFC; this is cheap insurance for the rest.
    """
    for column in frame.columns:
        if not pd.api.types.is_string_dtype(frame[column]):
            continue
        frame[column] = frame[column].map(
            lambda value: unicodedata.normalize("NFC", value) if isinstance(value, str) else value
        )
    return frame


def load_shard_sample(
    source: str | Path,
    n_rows: int | None = None,
    seed: int = 0,
    columns: Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load a uniform random sample of one parquet shard.

    Only ``columns`` are read off disk.  With ``n_rows`` set, a seeded random row
    subset is taken before conversion to pandas so peak memory stays bounded.
    """
    path = resolve_shard(source, cache_dir=cache_dir)
    wanted = list(columns) if columns is not None else list(DEFAULT_COLUMNS)
    table = pq.read_table(path, columns=wanted)
    total = table.num_rows
    if n_rows is not None and n_rows < total:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(total, size=int(n_rows), replace=False))
        table = table.take(indices)
    frame = normalize_unicode(table.to_pandas())
    # Column projection may exclude these; only parse what was actually read.
    if TARGET_COLUMN in frame.columns:
        frame[TARGET_COLUMN] = parse_price(frame[TARGET_COLUMN])
    if TIME_COLUMN in frame.columns:
        frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], errors="coerce", format="ISO8601")
    return frame.reset_index(drop=True)


def clean_frame(
    df: pd.DataFrame,
    *,
    min_price: float = 1e8,
    max_price: float = 2e11,
    min_area: float = 15.0,
    max_area: float = 3_000.0,
    min_unit_price: float = 3e6,
    max_unit_price: float = 8e8,
    drop_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Drop records that cannot support a sale-price regression.

    Defaults match **Rule 1 ("Recommended Baseline")** in
    ``reports/vietnam_real_estates_eda.md`` so this module and the baseline
    notebook model the same population: price in [100M, 200B] VND, area in
    [15, 3000] m², and unit price in [3M, 800M] VND/m². On the full corpus that
    rule retains ~89.6% of rows and leaves log-price skew at +0.19.

    The unit-price band is what catches price/area mismatches that neither bound
    catches alone — a 20 tỷ listing recorded as 3 m², or a 500 m² flat priced
    like a parking space.

    The ``min_price`` floor is also what excludes genuine rental listings, which
    quote a *monthly* rate (15-60M VND on ``shard_0000``), plus token deposits.

    Do **not** filter rentals with the ``kw_cho_thue`` text flag. It measures a
    *mention*, and 96% of the rows it fires on carry a sale-scale price (median
    10 tỷ VND) — they are sale listings pitching rental yield ("sẵn hợp đồng
    thuê", "tiện cho thuê"). Dropping on it discards ~19% of valid training rows
    and does so non-randomly, since yield-pitched properties form a coherent
    higher-priced segment. ``drop_mask`` stays available for a caller that has a
    trustworthy exclusion signal.

    The original index is **preserved** so feature blocks computed on ``df`` stay
    row-aligned with the returned subset; call ``reset_index`` explicitly if a
    clean 0..n-1 index is wanted.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"frame is missing the target column {TARGET_COLUMN!r}")
    price = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    keep = price.notna() & (price >= min_price) & (price <= max_price)
    if "area" in df.columns:
        area = pd.to_numeric(df["area"], errors="coerce")
        keep &= area.notna() & (area >= min_area) & (area <= max_area)
        unit_price = price / area.where(area > 0)
        keep &= unit_price.notna() & (unit_price >= min_unit_price) & (unit_price <= max_unit_price)
    if drop_mask is not None:
        if not drop_mask.index.equals(df.index):
            raise ValueError("drop_mask index does not align with the frame index")
        keep &= ~drop_mask.astype(bool)
    return df.loc[keep]


def time_span(df: pd.DataFrame) -> tuple[str, str]:
    """Return the inclusive ``(min, max)`` ISO range of ``published_at``."""
    stamps = df[TIME_COLUMN].dropna()
    if stamps.empty:
        return ("n/a", "n/a")
    return (str(stamps.min()), str(stamps.max()))


def out_of_time_pair(
    train_source: str | Path,
    test_source: str | Path,
    n_train: int,
    n_test: int,
    seed: int = 0,
    columns: Iterable[str] | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load an early shard for training and a later shard for testing.

    Rows are sampled *before* cleaning so the split has no look-ahead from the
    test period into the training period.
    """
    wanted = list(columns) if columns is not None else None
    train = load_shard_sample(
        train_source, n_rows=n_train, seed=seed, columns=wanted, cache_dir=cache_dir
    )
    test = load_shard_sample(
        test_source, n_rows=n_test, seed=seed + 1, columns=wanted, cache_dir=cache_dir
    )
    return train, test
