"""Tests for dataset loading, price parsing, cleaning and splitting."""

import unicodedata

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from real_estate.text.data import (
    TARGET_COLUMN,
    clean_frame,
    load_shard_sample,
    normalize_unicode,
    out_of_time_pair,
    parse_price,
    resolve_shard,
    time_span,
)


@pytest.fixture
def shard(tmp_path):
    """A miniature parquet shard that reproduces the real one's quirks."""
    frame = pd.DataFrame(
        {
            "name": [f"Nha {i}" for i in range(10)],
            "description": [f"Mo ta {i} so do" for i in range(10)],
            "property_type_name": ["Nhà"] * 10,
            "province_name": ["Hà Nội"] * 5 + ["Hồ Chí Minh"] * 5,
            "district_name": ["Cầu Giấy"] * 10,
            "ward_name": ["Dịch Vọng"] * 10,
            "street_name": ["Xuân Thủy"] * 10,
            "project_name": [None] * 10,
            # price is a *string* column, matching the published shards even
            # though the dataset card claims float64.
            "price": ["5000000000"] * 8 + [None, "300000"],
            "area": [40.0] * 8 + [np.nan, 50.0],
            "floor_count": [np.nan] * 10,
            "frontage_width": [4.0] * 10,
            "house_depth": [np.nan] * 10,
            "road_width": [np.nan] * 10,
            "bedroom_count": [3.0] * 10,
            "bathroom_count": [2.0] * 10,
            "house_direction": [None] * 10,
            "balcony_direction": [None] * 10,
            "published_at": [f"2025-06-{i + 1:02d}T10:00:00.000000" for i in range(10)],
        }
    )
    table = pa.Table.from_pandas(frame, preserve_index=False)
    path = tmp_path / "mini.parquet"
    pq.write_table(table, path)
    return path


def test_parse_price_converts_string_column():
    series = pd.Series(["7450000000", "1300000000", None, "  9100000000 "])
    parsed = parse_price(series)
    assert parsed.tolist()[:2] == [7450000000.0, 1300000000.0]
    assert np.isnan(parsed.iloc[2])
    assert parsed.iloc[3] == 9100000000.0


def test_parse_price_is_strict_about_units():
    """"7.45 tỷ" must become NaN, never a silently wrong 7.45 VND."""
    parsed = parse_price(pd.Series(["7.45 tỷ", "Thỏa thuận", "1.300.000"]))
    assert parsed.isna().all()


def test_parse_price_passes_numeric_through():
    series = pd.Series([1.5e9, 2.0e9])
    assert parse_price(series).tolist() == series.tolist()


def test_load_shard_sample_parses_price_and_time(shard):
    df = load_shard_sample(shard)
    assert len(df) == 10
    assert pd.api.types.is_float_dtype(df[TARGET_COLUMN])
    assert pd.api.types.is_datetime64_any_dtype(df["published_at"])
    assert df[TARGET_COLUMN].max() == 5.0e9


def test_load_shard_sample_respects_row_limit_and_seed(shard):
    a = load_shard_sample(shard, n_rows=4, seed=1)
    b = load_shard_sample(shard, n_rows=4, seed=1)
    c = load_shard_sample(shard, n_rows=4, seed=2)
    assert len(a) == 4
    assert a[TARGET_COLUMN].tolist() == b[TARGET_COLUMN].tolist()
    assert len(a) == len(c)


def test_load_shard_sample_column_projection(shard):
    df = load_shard_sample(shard, columns=["name", "price"])
    assert list(df.columns) == ["name", "price"]


def test_load_shard_sample_projection_without_target_or_time(shard):
    """Regression: parsing must be skipped for columns that were not projected."""
    df = load_shard_sample(shard, columns=["name", "description"])
    assert list(df.columns) == ["name", "description"]
    df = load_shard_sample(shard, columns=["area", "price"])
    assert list(df.columns) == ["area", "price"]


def test_normalize_unicode_merges_nfd_variants():
    """A street stored NFD must not read as a second category."""
    nfd = unicodedata.normalize("NFD", "Gò Đen")
    assert nfd != "Gò Đen" and unicodedata.normalize("NFC", nfd) == "Gò Đen"
    frame = pd.DataFrame(
        {"street_name": ["Gò Đen", nfd], "price": [1e9, 2e9], "area": [40.0, 50.0]}
    )
    out = normalize_unicode(frame)
    assert out["street_name"].nunique() == 1
    assert out["price"].tolist() == [1e9, 2e9]  # numeric columns untouched


def test_load_shard_sample_returns_nfc_strings(shard):
    df = load_shard_sample(shard)
    for column in df.columns:
        if not pd.api.types.is_string_dtype(df[column]):
            continue
        values = [v for v in df[column] if isinstance(v, str)]
        assert all(unicodedata.normalize("NFC", v) == v for v in values), column


def test_resolve_shard_returns_existing_local_path(shard):
    assert resolve_shard(shard) == shard


def test_time_span(shard):
    df = load_shard_sample(shard)
    start, end = time_span(df)
    assert start.startswith("2025-06-01") and end.startswith("2025-06-10")
    assert time_span(pd.DataFrame({"published_at": pd.Series([], dtype="datetime64[ns]")})) == (
        "n/a",
        "n/a",
    )


# ------------------------------------------------------------------ cleaning
def _cleanable_frame():
    return pd.DataFrame(
        {
            TARGET_COLUMN: [5e9, 1e5, 9e15, np.nan, 3e9],
            "area": [40.0, 50.0, 60.0, 70.0, np.nan],
            "district_name": ["A"] * 5,
        },
        index=[100, 101, 102, 103, 104],
    )


def test_clean_frame_trims_price_and_area_outliers():
    out = clean_frame(_cleanable_frame())
    # 1e5 below min_price, 9e15 above max_price, NaN price, NaN area all dropped.
    assert list(out.index) == [100]


def _valid_frame():
    """A frame where every row already satisfies the default cleaning bounds."""
    return pd.DataFrame(
        {
            TARGET_COLUMN: [5e9, 3e9, 4e9, 6e9, 7e9],
            "area": [40.0, 50.0, 60.0, 70.0, 80.0],
            "district_name": ["A"] * 5,
        },
        index=[100, 101, 102, 103, 104],
    )


def test_clean_frame_preserves_original_index():
    frame = _valid_frame()
    out = clean_frame(frame)
    assert list(out.index) == list(frame.index)


def test_clean_frame_applies_drop_mask():
    frame = _valid_frame()
    mask = pd.Series([True, False, False, False, False], index=frame.index)
    out = clean_frame(frame, drop_mask=mask)
    assert 100 not in out.index
    assert list(out.index) == [101, 102, 103, 104]


def test_clean_frame_rejects_misaligned_mask():
    frame = _cleanable_frame()
    with pytest.raises(ValueError, match="does not align"):
        clean_frame(frame, drop_mask=pd.Series([True] * 5))


def test_clean_frame_requires_target():
    with pytest.raises(KeyError, match="price"):
        clean_frame(pd.DataFrame({"area": [40.0]}))


def test_price_floor_excludes_monthly_rate_rentals():
    """Rentals quote a monthly rate; the floor separates them from sales.

    This is the mechanism that keeps rental listings out of the target — not the
    `kw_cho_thue` text flag, which mostly fires on sale listings pitching yield.
    """
    frame = pd.DataFrame(
        {
            TARGET_COLUMN: [
                15_500_000,   # "Cho thuê nhà chính chủ" — monthly rent
                60_000_000,   # "cho thuê nhà phố ... 60 triệu/tháng"
                9_000_000_000,  # sale listing that mentions an existing lease
                7_350_000_000,  # plain sale
            ],
            "area": [40.0, 80.0, 60.0, 35.0],
        }
    )
    out = clean_frame(frame)
    assert list(out.index) == [2, 3]


def test_clean_frame_enforces_the_area_band():
    """Defaults follow the EDA's Rule 1: area in [15, 3000] m².

    Prices are chosen per row so every candidate also sits inside the unit-price
    band — otherwise the area assertion would be testing the ratio instead.
    """
    frame = pd.DataFrame(
        {
            TARGET_COLUMN: [5e9, 5e9, 20e9, 20e9],
            "area": [10.0, 15.0, 3000.0, 5000.0],
        }
    )
    # 10 m² is below the floor; 5000 m² above the ceiling. The middle two pass
    # on unit price too (333M/m² and 6.7M/m²).
    assert list(clean_frame(frame).index) == [1, 2]


def test_unit_price_band_catches_price_area_mismatch():
    """Neither bound alone catches these; the ratio does."""
    frame = pd.DataFrame(
        {
            TARGET_COLUMN: [
                20e9,   # 20 tỷ on 3 m² -> 6.7B/m², far above any real unit price
                1e8,    # 100M on 500 m² -> 200k/m², below it
                5e9,    # 5 tỷ on 50 m² -> 100M/m², plausible
            ],
            "area": [3.0, 500.0, 50.0],
        }
    )
    assert list(clean_frame(frame).index) == [2]


def test_unit_price_bounds_are_overridable():
    frame = pd.DataFrame({TARGET_COLUMN: [20e9], "area": [3.0]})
    assert clean_frame(frame).empty
    # 3 m² also breaches the area floor, so both bounds need relaxing.
    assert len(clean_frame(frame, min_area=1.0, max_unit_price=1e10)) == 1


def test_clean_frame_keeps_rows_that_mention_renting():
    """A sale-scale listing mentioning rental yield is valid training data."""
    frame = pd.DataFrame(
        {TARGET_COLUMN: [9e9], "area": [60.0], "kw_cho_thue": [True]}
    )
    assert len(clean_frame(frame)) == 1


def test_out_of_time_pair_returns_both_frames(shard):
    train, test = out_of_time_pair(shard, shard, n_train=6, n_test=3, cache_dir=shard.parent)
    assert len(train) == 6 and len(test) == 3
