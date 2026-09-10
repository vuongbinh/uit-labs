"""Tests for the composed text-feature pipeline."""

import numpy as np
import pandas as pd
import pytest

from real_estate.text.pipeline import FEATURE_GROUPS, TextFeatureConfig, TextFeaturePipeline

BASE = "Nhà mặt tiền sổ đỏ chính chủ, ô tô đỗ cửa, diện tích 40m2, 3 phòng ngủ"

FILLER = [
    "Căn hộ chung cư full nội thất, thang máy, sổ hồng riêng",
    "Nhà cấp 4 trong hẻm ba gác, gần chợ, pháp lý rõ ràng",
    "Biệt thự phân lô hai mặt thoáng, vỉa hè rộng, sân vườn",
    "Đất nền lô góc, không quy hoạch, không tranh chấp",
    "Nhà mới xây 5 tầng, đường 8m, tiện kinh doanh buôn bán",
    "Nhà 6 tầng thang máy, ngõ thông, ô tô đỗ cổng",
    "Cho thuê nguyên căn mặt tiền, giá tốt, có thương lượng",
    "Nhà phố 3 tấm, sân thượng, đang chờ sổ, chính chủ",
    "Căn hộ 2 phòng ngủ ban công đông nam, công chứng ngay",
    "Đất vườn rộng 500m2 sổ đỏ lâu dài, chính chủ bán",
    "Nhà hẻm xe hơi, 1 trệt 2 lầu, gần trường học các cấp",
    "Bán gấp nhà trong ngõ, xe hơi vào nhà, 4 phòng ngủ",
]


def make_frame(*, with_price_pair: bool = False, index=None) -> pd.DataFrame:
    """A listing frame with text plus a few sparse structured columns."""
    names = [f"Tin {i}" for i in range(len(FILLER))]
    descriptions = list(FILLER)
    if with_price_pair:
        # Identical listings that differ only in the advertised price.
        names += ["Tin A", "Tin B"]
        descriptions += [f"{BASE}, giá 5.5 tỷ", f"{BASE}, giá 7.25 tỷ"]
    n = len(descriptions)
    frame = pd.DataFrame(
        {
            "name": names,
            "description": descriptions,
            "price": np.linspace(2e9, 2e10, n),
            "area": np.where(np.arange(n) % 3 == 0, np.nan, 45.0),
            "floor_count": np.where(np.arange(n) % 2 == 0, np.nan, 4.0),
            "frontage_width": np.nan,
            "house_depth": np.nan,
            "road_width": np.nan,
            "bedroom_count": 3.0,
            "bathroom_count": 2.0,
        }
    )
    if index is not None:
        frame.index = index
    return frame


def test_fit_transform_emits_every_enabled_group():
    frame = make_frame()
    config = TextFeatureConfig(tfidf_components=4, tfidf_min_df=1)
    features = TextFeaturePipeline(config).fit_transform(frame)
    assert len(features) == len(frame)
    groups = TextFeaturePipeline(config).fit(frame).group_columns(features)
    assert set(groups) == {"keywords", "entities", "tfidf"}
    assert set(FEATURE_GROUPS) >= set(groups)
    # every emitted column belongs to exactly one group
    assigned = [c for cols in groups.values() for c in cols]
    assert sorted(assigned) == sorted(features.columns)
    assert len(assigned) == len(set(assigned))


def test_fit_transform_matches_fit_then_transform():
    """fit_transform reuses encoder output; it must not change the result."""
    frame = make_frame()
    config = TextFeatureConfig(tfidf_components=4, tfidf_min_df=1, seed=3)
    fused = TextFeaturePipeline(config).fit_transform(frame)
    split = TextFeaturePipeline(config).fit(frame).transform(frame)
    assert list(fused.columns) == list(split.columns)
    np.testing.assert_allclose(
        fused.to_numpy(dtype="float64"), split.to_numpy(dtype="float64"), rtol=1e-6, atol=1e-9
    )


def test_transform_before_fit_raises():
    pipeline = TextFeaturePipeline(TextFeatureConfig(tfidf_components=2, tfidf_min_df=1))
    with pytest.raises(RuntimeError, match="before fit"):
        pipeline.transform(make_frame())


def test_config_with_no_groups_rejected():
    config = TextFeatureConfig(use_keywords=False, use_entities=False, use_tfidf=False)
    with pytest.raises(ValueError, match="at least one"):
        TextFeaturePipeline(config)


def test_stateless_groups_work_without_fit():
    """Keywords and entities need no fitting, so no fit call is required."""
    config = TextFeatureConfig(use_tfidf=False)
    features = TextFeaturePipeline(config).transform(make_frame())
    assert any(c.startswith("kw_") for c in features.columns)
    assert any(c.startswith("text_") for c in features.columns)
    assert not any(c.startswith("tfidf_") for c in features.columns)


def test_transform_preserves_a_non_default_index():
    index = pd.Index([7, 19, 31, 42, 55, 61, 77, 83, 90, 96, 101, 111])
    frame = make_frame(index=index)
    config = TextFeatureConfig(tfidf_components=3, tfidf_min_df=1)
    features = TextFeaturePipeline(config).fit_transform(frame)
    assert list(features.index) == list(index)


def test_train_fitted_state_applies_to_a_held_out_frame():
    train = make_frame()
    held_out = pd.DataFrame(
        {
            "name": ["Tin mới"],
            "description": ["Nhà mặt tiền sổ đỏ, chưa từng xuất hiện trong tập huấn luyện xyz"],
            "price": [6e9],
            "area": [50.0],
            "floor_count": [np.nan],
            "frontage_width": [np.nan],
            "house_depth": [np.nan],
            "road_width": [np.nan],
            "bedroom_count": [np.nan],
            "bathroom_count": [np.nan],
        }
    )
    pipeline = TextFeaturePipeline(TextFeatureConfig(tfidf_components=4, tfidf_min_df=1)).fit(train)
    train_features = pipeline.transform(train)
    test_features = pipeline.transform(held_out)
    assert list(test_features.columns) == list(train_features.columns)
    assert len(test_features) == 1
    assert bool(test_features["kw_so_do"].iloc[0])
    assert bool(test_features["kw_mat_tien"].iloc[0])


def test_price_redaction_removes_the_target_from_tfidf():
    """Identical listings quoting different prices must encode identically.

    Without redaction the advertised price — which *is* the regression target —
    leaks into the bag-of-words features as digit tokens.
    """
    frame = make_frame(with_price_pair=True)
    i, j = len(frame) - 2, len(frame) - 1
    config = TextFeatureConfig(
        tfidf_components=4, tfidf_min_df=1, use_entities=False, use_keywords=False
    )

    redacted = TextFeaturePipeline(config).fit_transform(frame)
    np.testing.assert_allclose(
        redacted.iloc[i].to_numpy(dtype=float), redacted.iloc[j].to_numpy(dtype=float), atol=1e-6
    )

    leaky_config = TextFeatureConfig(
        tfidf_components=4,
        tfidf_min_df=1,
        use_entities=False,
        use_keywords=False,
        redact_price=False,
    )
    leaky = TextFeaturePipeline(leaky_config).fit_transform(frame)
    assert not np.allclose(
        leaky.iloc[i].to_numpy(dtype=float), leaky.iloc[j].to_numpy(dtype=float), atol=1e-6
    )


def test_provenance_flags_mark_text_recovered_values():
    frame = make_frame()
    config = TextFeatureConfig(use_tfidf=False, use_keywords=False)
    features = TextFeaturePipeline(config).fit_transform(frame)
    provenance = [c for c in features.columns if c.endswith("__from_text")]
    assert provenance, "expected at least one provenance flag"
    assert features[provenance].dtypes.eq(bool).all()


def test_entity_columns_helper_is_stable():
    assert list(TextFeaturePipeline.entity_columns()) == list(TextFeaturePipeline.entity_columns())
    assert "text_price_vnd" in TextFeaturePipeline.entity_columns()


def test_missing_text_column_raises():
    pipeline = TextFeaturePipeline(TextFeatureConfig(use_tfidf=False))
    with pytest.raises(KeyError, match="description"):
        pipeline.transform(pd.DataFrame({"name": ["a"], "price": [1e9]}))
