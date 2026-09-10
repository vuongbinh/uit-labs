"""Tests for Vietnamese text normalisation."""

import pytest

from real_estate.text.normalize import (
    clean_listing_text,
    fold_diacritics,
    fold_with_alignment,
    join_text_columns,
    make_searchable,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Sổ đỏ chính chủ", "so do chinh chu"),
        ("sổ đỏ chính chủ", "so do chinh chu"),
        ("Ô tô đỗ cửa", "o to do cua"),
        ("ôtô đỗ cổng", "oto do cong"),
        ("Mặt tiền đường 8m", "mat tien duong 8m"),
        ("Nhà cấp 4", "nha cap 4"),
        ("Đường Đê La Thành", "duong de la thanh"),
        ("4m² x 12m", "4m2 x 12m"),
    ],
)
def test_fold_diacritics(raw, expected):
    assert fold_diacritics(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "Sổ đỏ chính chủ, ô tô đỗ cửa, 4m² x 12m",
        "Nhà 6 tầng thang máy, ngõ thông, Đê La Thành",
        "full nội thất – vào ở ngay — giá 7.45 tỷ",
        "Đường 8m, vỉa hè, 2 mặt thoáng",
        "không tranh chấp, không quy hoạch",
        "",
    ],
)
def test_fold_preserves_length_for_ordinary_vietnamese(raw):
    """Ordinary Vietnamese text folds 1:1 — but see the alignment tests below,
    which cover the combining marks that break that assumption."""
    assert len(fold_diacritics(raw)) == len(raw)


@pytest.mark.parametrize("noise", ["\ufe0f", "\u0301", "\u0300\u0323"])
def test_fold_shrinks_on_dropped_combining_marks(noise):
    """Folding is NOT length-preserving in general; marks in category Mn vanish."""
    text = f"Nhà{noise} đẹp"
    assert len(fold_diacritics(text)) < len(text)


def test_fold_with_alignment_maps_every_output_character():
    text = "Nhà\ufe0f đẹp, giá 7.45 tỷ"
    folded, origin = fold_with_alignment(text)
    assert folded == fold_diacritics(text)
    assert len(origin) == len(folded)
    # each origin index points at the source character that produced that output
    assert all(0 <= i < len(text) for i in origin)
    assert origin == sorted(origin)
    assert text[origin[0]].lower().startswith("n")


def test_alignment_span_maps_back_to_the_source_substring():
    text = "Nhà\ufe0f đẹp, giá 7.45 tỷ, sổ đỏ"
    folded, origin = fold_with_alignment(text)
    start = folded.index("7.45")
    end = start + len("7.45")
    assert text[origin[start] : origin[end - 1] + 1] == "7.45"


def test_clean_removes_redaction_placeholders():
    text = "Bán nhà 7.45 tỷ, lh [phone_number] gặp [email]"
    cleaned = clean_listing_text(text)
    assert "[phone_number]" not in cleaned
    assert "[email]" not in cleaned
    assert "7.45 tỷ" in cleaned


def test_clean_collapses_runs_of_spaces_and_keeps_diacritics():
    assert clean_listing_text("Sổ   đỏ \t  chính   chủ") == "Sổ đỏ chính chủ"


def test_clean_collapses_long_newline_runs():
    assert clean_listing_text("mô tả\n\n\n\n\nchi tiết") == "mô tả\n\nchi tiết"


def test_clean_preserves_the_d_stroke():
    """Regression: folding đ→d belongs to fold_diacritics only.

    Cleaning is used on the PhoBERT path, where "đỏ" and "dỏ" are different
    tokens; silently de-stroking them degraded the embeddings.
    """
    assert "đỏ" in clean_listing_text("Sổ   đỏ")
    assert "Đường" in clean_listing_text("Đường  Đê La Thành")
    assert fold_diacritics(clean_listing_text("Sổ đỏ")) == "so do"


@pytest.mark.parametrize("empty", [None, ""])
def test_clean_handles_missing(empty):
    assert clean_listing_text(empty) == ""
    assert make_searchable(empty) == ""


def test_make_searchable_composes_clean_then_fold():
    assert make_searchable("  Sổ   đỏ  [phone_number] ") == "so do"


def test_join_text_columns():
    assert join_text_columns(None) == ""
    assert join_text_columns("một mình") == "một mình"
    assert join_text_columns(["Bán nhà", None, "sổ đỏ"]) == "Bán nhà sổ đỏ"
    assert join_text_columns(("a", 3, "b")) == "a b"
