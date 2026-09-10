"""Tests for numeric entity extraction, imputation and price redaction."""

import numpy as np
import pandas as pd
import pytest

from real_estate.text.entities import (
    ENTITY_COLUMNS,
    TARGET_LEAKING_ENTITIES,
    TEXT_IMPUTABLE_COLUMNS,
    extract_entities,
    extract_entities_folded,
    find_price_spans,
    impute_from_text,
    parse_vn_number,
    prepare_for_embedding,
    prepare_for_tfidf,
    redact_spans,
)


def one(text: str, name: str = "") -> pd.Series:
    """Extract entities from a single listing and return the first row."""
    frame = pd.DataFrame({"name": [name], "description": [text]})
    return extract_entities(frame).iloc[0]


# ------------------------------------------------------------- number parsing
@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("4", 4.0),
        ("7.45", 7.45),
        ("1,3", 1.3),
        ("4,5", 4.5),
        ("1.300", 1300.0),  # 3-digit group reads as thousands, not a decimal
        ("1.300.000", 1300000.0),
        ("12,5", 12.5),
    ],
)
def test_parse_vn_number(token, expected):
    assert parse_vn_number(token) == pytest.approx(expected)


def test_parse_vn_number_rejects_empty():
    assert parse_vn_number("") is None
    assert parse_vn_number("   ") is None


# ------------------------------------------------------------- entity patterns
def test_area_from_square_metre_notation():
    assert one("Diện tích 300m2")["text_area_m2"] == pytest.approx(300.0)
    assert one("Diện tích: 44m²")["text_area_m2"] == pytest.approx(44.0)
    assert one("diện tích hơn 300 mét vuông")["text_area_m2"] == pytest.approx(300.0)


def test_area_rejects_implausible_values():
    assert pd.isna(one("Diện tích 999999m2")["text_area_m2"])


def test_dimensions_split_into_frontage_and_depth():
    row = one("Nhà 4m x 12m, kết cấu đẹp")
    assert row["text_frontage_m"] == pytest.approx(4.0)
    assert row["text_depth_m"] == pytest.approx(12.0)


def test_dimensions_with_decimal_comma_and_no_unit():
    row = one("Ngang 4,5x16")
    assert row["text_frontage_m"] == pytest.approx(4.5)
    assert row["text_depth_m"] == pytest.approx(16.0)


def test_ngang_dai_wording():
    row = one("ngang 5m dài 20m")
    assert row["text_frontage_m"] == pytest.approx(5.0)
    assert row["text_depth_m"] == pytest.approx(20.0)


def test_floor_count_from_tang_and_tam():
    assert one("Nhà 6 tầng thang máy")["text_floor_count"] == pytest.approx(6.0)
    assert one("Nhà đúc 3 tấm")["text_floor_count"] == pytest.approx(3.0)


def test_lau_is_not_conflated_with_tang():
    """"N lầu" is ambiguous, so it lands in its own column and never imputes."""
    row = one("Nhà 1 trệt 2 lầu")
    assert row["text_lau_count"] == pytest.approx(2.0)
    assert pd.isna(row["text_floor_count"])


def test_bedroom_and_bathroom_counts():
    assert one("Căn hộ 3 PN, 2 WC")["text_bedroom_count"] == pytest.approx(3.0)
    assert one("Căn hộ 3 PN, 2 WC")["text_bathroom_count"] == pytest.approx(2.0)
    assert one("4 phòng ngủ, 3 toilet")["text_bedroom_count"] == pytest.approx(4.0)
    assert one("4 phòng ngủ, 3 toilet")["text_bathroom_count"] == pytest.approx(3.0)


def test_road_width():
    assert one("MT đường 8m có lề")["text_road_width_m"] == pytest.approx(8.0)
    assert one("hẻm 3m thông")["text_road_width_m"] == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Giá bán: 20 tỷ", 20e9),
        ("giá: 7.45 tỷ", 7.45e9),
        ("Giá 1,3 tỷ", 1.3e9),
        ("cho thuê 15 triệu/tháng", 15e6),
    ],
)
def test_price_in_vnd(text, expected):
    assert one(text)["text_price_vnd"] == pytest.approx(expected, rel=1e-6)


def test_price_prefers_the_value_stated_next_to_gia():
    row = one(name="Bán nhà đẹp", text="Diện tích 40m2, giá 5.5 tỷ, hỗ trợ vay 2 tỷ")
    assert row["text_price_vnd"] == pytest.approx(5.5e9)


def test_price_absent_when_no_magnitude_word():
    assert pd.isna(one("Nhà đẹp, vị trí trung tâm")["text_price_vnd"])


def test_entity_columns_are_complete_and_ordered():
    frame = pd.DataFrame({"name": ["a"], "description": ["sổ đỏ"]})
    out = extract_entities(frame)
    assert list(out.columns) == list(ENTITY_COLUMNS)
    assert "text_lau_count" in ENTITY_COLUMNS


def test_extract_entities_preserves_index():
    frame = pd.DataFrame(
        {"name": ["a", "b"], "description": ["300m2", "6 tầng"]}, index=[7, 9]
    )
    out = extract_entities(frame)
    assert list(out.index) == [7, 9]
    assert out.loc[7, "text_area_m2"] == pytest.approx(300.0)
    assert out.loc[9, "text_floor_count"] == pytest.approx(6.0)


def test_extract_entities_rejects_missing_column():
    with pytest.raises(KeyError, match="description"):
        extract_entities(pd.DataFrame({"name": ["a"]}))


def test_folded_variant_matches_full_pipeline():
    folded = list(extract_entities_folded(["nha 6 tang, 300m2"]).itertuples(index=False, name=None))[0]
    assert folded[ENTITY_COLUMNS.index("text_floor_count")] == pytest.approx(6.0)
    assert folded[ENTITY_COLUMNS.index("text_area_m2")] == pytest.approx(300.0)


# ----------------------------------------------------------------- imputation
def _frames():
    tabular = pd.DataFrame(
        {
            "area": [40.0, np.nan],
            "floor_count": [np.nan, 3.0],
            "frontage_width": [np.nan, np.nan],
        },
        index=[0, 1],
    )
    entities = pd.DataFrame(
        {
            "text_area_m2": [44.0, 100.0],
            "text_floor_count": [6.0, np.nan],
            "text_frontage_m": [4.0, 5.0],
        },
        index=[0, 1],
    )
    return tabular, entities


def test_imputation_fills_nulls_without_overwriting():
    tabular, entities = _frames()
    out = impute_from_text(tabular, entities)
    assert out.loc[0, "area"] == pytest.approx(40.0)  # existing value kept
    assert out.loc[1, "area"] == pytest.approx(100.0)  # null filled from text
    assert out.loc[0, "floor_count"] == pytest.approx(6.0)
    assert out.loc[1, "floor_count"] == pytest.approx(3.0)


def test_imputation_records_provenance():
    tabular, entities = _frames()
    out = impute_from_text(tabular, entities)
    assert bool(out.loc[0, "area__from_text"]) is False
    assert bool(out.loc[1, "area__from_text"]) is True
    assert bool(out.loc[0, "floor_count__from_text"]) is True
    assert bool(out.loc[1, "floor_count__from_text"]) is False
    assert bool(out.loc[0, "frontage_width__from_text"]) is True


def test_imputation_never_uses_the_price_entity():
    """The target must not be reconstructible through an imputed column."""
    assert "price" not in TEXT_IMPUTABLE_COLUMNS
    assert set(TARGET_LEAKING_ENTITIES).isdisjoint(TEXT_IMPUTABLE_COLUMNS.values())


def test_imputation_realigns_when_indices_differ():
    tabular, entities = _frames()
    out = impute_from_text(tabular, entities.iloc[::-1])
    assert list(out.index) == [0, 1]


# ----------------------------------------------------------------- redaction
def test_find_price_spans_and_redact():
    folded = "nha dep, gia 7.45 ty, dien tich 40m2"
    spans = find_price_spans(folded)
    assert len(spans) == 1
    redacted = redact_spans(folded, spans)
    assert "7.45" not in redacted
    assert "<gia>" in redacted
    assert "40m2" in redacted


def test_find_price_spans_merges_overlaps_and_returns_sorted():
    spans = find_price_spans("gia 5 ty hoac 900 trieu")
    assert spans == sorted(spans)
    assert len(spans) == 2


def test_prepare_for_tfidf_folds_and_redacts():
    text = "Nhà đẹp, giá 7.45 tỷ, diện tích 40m²"
    prepared = prepare_for_tfidf(text)
    assert prepared == prepared.lower()
    assert "7.45" not in prepared
    assert "40m2" in prepared


def test_prepare_for_tfidf_can_keep_price():
    text = "giá 7.45 tỷ"
    assert "7.45" in prepare_for_tfidf(text, redact_price=False)
    assert "7.45" not in prepare_for_tfidf(text, redact_price=True)


def test_prepare_for_embedding_keeps_diacritics_but_drops_price():
    text = "Nhà đẹp, giá 7.45 tỷ, sổ đỏ chính chủ"
    prepared = prepare_for_embedding(text)
    assert "7.45" not in prepared
    assert "Nhà đẹp" in prepared
    assert "sổ đỏ" in prepared


@pytest.mark.parametrize(
    "noise",
    [
        "\ufe0f",       # emoji variation selector: category Mn, dropped when folding
        "\u0301",       # stray combining acute accent
        "\u0300\u0323",  # two stacked combining marks
    ],
)
def test_prepare_for_embedding_redacts_when_folding_changes_length(noise):
    """Real listing text contains marks that folding drops, so spans must be
    mapped per character rather than by assuming equal lengths."""
    text = f"Nhà{noise} đẹp, giá 7.45 tỷ, sổ đỏ{noise} chính chủ"
    prepared = prepare_for_embedding(text)
    assert "<gia>" in prepared
    assert "7.45" not in prepared
    # diacritics survive on this path, and text either side of the price is intact
    assert "sổ đỏ" in prepared
    assert "chính chủ" in prepared
    assert "đẹp" in prepared


def test_redact_spans_on_empty_input():
    assert find_price_spans("không có giá") == []
    assert redact_spans("giữ nguyên", []) == "giữ nguyên"
