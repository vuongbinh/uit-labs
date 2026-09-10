"""Tests for the uplift benchmark harness: metrics, splitting and encoders."""

import numpy as np
import pandas as pd
import pytest

from real_estate.text.benchmark import (
    TabularEncoder,
    _time_ordered_valid,
    entity_agreement_report,
    regression_metrics,
)


# ------------------------------------------------------------------- metrics
def test_perfect_prediction_scores_zero_error():
    y = np.array([1e9, 2e9, 3e9])
    m = regression_metrics(y, y)
    assert m["rmsle"] == pytest.approx(0.0, abs=1e-12)
    assert m["mape_pct"] == pytest.approx(0.0)
    assert m["mae_ty"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["n"] == 3


def test_metrics_match_hand_computed_values():
    y_true = np.array([1e9, 3e9])
    y_pred = np.array([1e9, 2e9])
    m = regression_metrics(y_true, y_pred)
    expected_rmsle = np.sqrt((np.log1p(3e9) - np.log1p(2e9)) ** 2 / 2)
    assert m["rmsle"] == pytest.approx(expected_rmsle, rel=1e-9)
    # MAPE averages the relative error over both rows: (0 + 1/3) / 2 = 1/6.
    assert m["mape_pct"] == pytest.approx(100 / 6, rel=1e-9)
    # MAE averages |error| over both rows: (0 + 1e9) / 2 = 0.5 tỷ.
    assert m["mae_ty"] == pytest.approx(0.5)
    assert m["medae_ty"] == pytest.approx(0.5)
    assert m["r2"] == pytest.approx(0.5)


def test_rmsle_dampens_a_single_wild_outlier():
    """log-scale error absorbs a 1000x miss that linear MAE cannot."""
    y_true = np.array([1e9] * 50 + [1e9])
    y_pred = np.array([1e9] * 50 + [1e12])
    m = regression_metrics(y_true, y_pred)
    assert m["rmsle"] < 1.0
    assert m["mae_ty"] > 10.0


def test_metrics_ignore_invalid_entries():
    y_true = np.array([1e9, np.nan, 0.0, 2e9])
    y_pred = np.array([1e9, 1e9, 1e9, -5.0])
    m = regression_metrics(y_true, y_pred)
    assert m["n"] == 1  # only the first pair is usable


def test_metrics_on_no_valid_pairs_are_nan():
    m = regression_metrics(np.array([0.0, np.nan]), np.array([1.0, 1.0]))
    assert m["n"] == 0
    assert np.isnan(m["rmsle"])


# ------------------------------------------------------------- time splitting
def _time_frame(n=40):
    stamps = pd.date_range("2025-06-01", periods=n, freq="6h")
    return pd.DataFrame({"published_at": stamps, "price": np.linspace(1e9, 5e9, n)})


def test_time_ordered_valid_put_the_latest_rows_in_validation():
    frame = _time_frame(40)
    fit_idx, valid_idx = _time_ordered_valid(frame, valid_fraction=0.25)
    assert len(fit_idx) == 30 and len(valid_idx) == 10
    assert set(fit_idx).isdisjoint(valid_idx)
    assert frame["published_at"].iloc[fit_idx].max() < frame["published_at"].iloc[valid_idx].min()


def test_time_ordered_valid_survives_shuffled_input():
    frame = _time_frame(40).sample(frac=1.0, random_state=3).reset_index(drop=True)
    fit_idx, valid_idx = _time_ordered_valid(frame, valid_fraction=0.25)
    assert frame["published_at"].iloc[fit_idx].max() < frame["published_at"].iloc[valid_idx].min()


def test_time_ordered_valid_falls_back_when_timestamps_missing():
    frame = pd.DataFrame({"published_at": pd.Series([pd.NaT] * 20), "price": [1e9] * 20})
    fit_idx, valid_idx = _time_ordered_valid(frame, valid_fraction=0.2)
    assert len(fit_idx) == 16 and len(valid_idx) == 4
    assert set(fit_idx).isdisjoint(valid_idx)


# ---------------------------------------------------------- tabular encoding
def _tabular_frame():
    return pd.DataFrame(
        {
            "price": [5e9, 6e9, 7e9, 8e9],
            "area": [40.0, 50.0, np.nan, 70.0],
            "floor_count": [3.0, np.nan, 5.0, np.nan],
            "frontage_width": [4.0, 5.0, 6.0, 7.0],
            "house_depth": [10.0, 10.0, 12.0, 10.0],
            "road_width": [8.0, np.nan, 8.0, np.nan],
            "bedroom_count": [3.0, 3.0, 4.0, 4.0],
            "bathroom_count": [2.0, 2.0, 3.0, 3.0],
            "property_type_name": ["Nhà", "Nhà", "Căn hộ chung cư", "Căn hộ chung cư"],
            "province_name": ["Hà Nội", "Hà Nội", "Hồ Chí Minh", "Hồ Chí Minh"],
            "district_name": ["Cầu Giấy", "Cầu Giấy", "Quận 1", "Quận 1"],
            "ward_name": ["Dịch Vọng", "Dịch Vọng", "Bến Nghé", "Bến Nghé"],
            "street_name": ["Xuân Thủy", "Xuân Thủy", "Đồng Khởi", "Đồng Khởi"],
            "project_name": [None, None, None, None],
            "house_direction": ["Nam", None, "Đông", None],
            "balcony_direction": [None, None, None, None],
            "published_at": pd.to_datetime(
                ["2025-06-01", "2025-06-02", "2025-06-03", "2025-06-04"]
            ),
        }
    )


def test_tabular_encoder_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        TabularEncoder().transform(_tabular_frame())


def test_tabular_encoder_emits_numerics_ratios_and_encodings():
    frame = _tabular_frame()
    out = TabularEncoder().fit_transform(frame)
    assert len(out) == len(frame)
    for column in ("area", "frontage_width", "ratio_frontage_to_depth", "ratio_road_to_frontage",
                   "implied_depth_from_area", "published_month", "tgt_district_name",
                   "freq_province_name"):
        assert column in out.columns, f"missing {column}"
    # Encodings must be total; raw sparse numerics may legitimately stay NaN
    # because LightGBM handles missing values natively.
    encoded = [c for c in out.columns if c.startswith(("tgt_", "freq_"))]
    assert encoded and np.isfinite(out[encoded].to_numpy(dtype="float64")).all()


def test_tabular_encoder_smooths_target_encoding():
    """A single-row category must not receive its own log-price verbatim."""
    frame = _tabular_frame()
    encoder = TabularEncoder(smoothing=50.0).fit(frame)
    out = encoder.transform(frame)
    global_mean = np.log1p(frame["price"]).mean()
    for value in out["tgt_district_name"]:
        assert abs(value - global_mean) < 1.0


def test_tabular_encoder_handles_unseen_categories():
    train = _tabular_frame()
    encoder = TabularEncoder().fit(train)
    unseen = train.copy()
    unseen["district_name"] = ["Hoàn Kiếm"] * len(unseen)
    unseen["ward_name"] = ["Hàng Bạc"] * len(unseen)
    out = encoder.transform(unseen)
    global_mean = encoder._global_mean
    np.testing.assert_allclose(out["tgt_district_name"].to_numpy(), global_mean)
    encoded = [c for c in out.columns if c.startswith(("tgt_", "freq_"))]
    assert np.isfinite(out[encoded].to_numpy(dtype="float64")).all()


def test_tabular_encoder_missing_category_becomes_its_own_bucket():
    frame = _tabular_frame()
    encoder = TabularEncoder().fit(frame)
    out = encoder.transform(frame)
    # house_direction is null for half the rows; nulls must still be encoded.
    assert np.isfinite(out["freq_house_direction"].to_numpy()).all()


# ---------------------------------------------------- entity agreement report
def test_entity_agreement_report_measures_coverage_and_agreement():
    df = pd.DataFrame(
        {
            "price": [5e9, 5e9, 5e9, np.nan],
            "area": [40.0, np.nan, 60.0, 70.0],
            "floor_count": [np.nan, np.nan, 3.0, np.nan],
        }
    )
    entities = pd.DataFrame(
        {
            "text_price_vnd": [5e9, 4e9, np.nan, np.nan],
            "text_area_m2": [40.0, 55.0, 999.0, np.nan],
            "text_floor_count": [4.0, np.nan, 3.0, np.nan],
        }
    )
    report = entity_agreement_report(df, entities).set_index("structured_column")
    assert report.loc["area", "text_coverage"] == pytest.approx(0.75)
    assert report.loc["area", "structured_missing_rate"] == pytest.approx(0.25)
    assert report.loc["area", "recovered_when_missing"] == 1
    # overlap is rows 0 and 2 (row 1 has no structured area); only row 0 agrees.
    assert report.loc["area", "overlap_rows"] == 2
    assert report.loc["area", "agreement_rate"] == pytest.approx(0.5)
    assert report.loc["floor_count", "overlap_rows"] == 1
    assert report.loc["floor_count", "agreement_rate"] == pytest.approx(1.0)
    assert report.loc["price", "agreement_rate"] == pytest.approx(0.5)


def test_entity_agreement_report_skips_absent_columns():
    report = entity_agreement_report(pd.DataFrame({"price": [1e9]}), pd.DataFrame())
    assert report.empty
