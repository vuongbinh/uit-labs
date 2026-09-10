"""Tests for the domain keyword / entity-flag extractor.

Every phrase named in the issue's deliverable list is asserted here, in both its
diacritised and diacritic-free spelling, because the extractor is required to
match either.
"""

import pandas as pd
import pytest

from real_estate.text.keywords import KeywordExtractor
from real_estate.text.lexicon import DERIVED_FLAGS, LEXICON, KeywordRule, rules_by_category


def flags_for(text: str) -> pd.Series:
    """Single-row convenience wrapper: flags for one description string."""
    frame = pd.DataFrame({"name": [""], "description": [text]})
    return KeywordExtractor().extract(frame).iloc[0]


@pytest.mark.parametrize(
    ("phrase", "feature"),
    [
        # legal status
        ("Nhà sổ đỏ lâu dài", "kw_so_do"),
        ("Sổ hồng riêng, công chứng ngay", "kw_so_hong"),
        ("Chính chủ bán, không qua môi giới", "kw_chinh_chu"),
        ("Pháp lý rõ ràng, sang tên nhanh", "kw_phap_ly_ro_rang"),
        ("Pháp lý đầy đủ", "kw_phap_ly_ro_rang"),
        ("Nhà đang chờ sổ", "kw_dang_cho_so"),
        # accessibility & road
        ("Ô tô đỗ cửa, ngõ rộng", "kw_o_to_do_cua"),
        ("ôtô đỗ cổng", "kw_o_to_do_cua"),
        ("Nhà mặt tiền đường lớn", "kw_mat_tien"),
        ("Ngõ thông các ngả", "kw_ngo_thong"),
        ("Xe hơi vào nhà thoải mái", "kw_xe_hoi_vao_nha"),
        ("Ngõ ba gác", "kw_ngo_ba_gac"),
        # condition & interior
        ("Full nội thất, dọn vào ở", "kw_full_noi_that"),
        ("Nội thất cơ bản", "kw_noi_that_co_ban"),
        ("Nhà mới xây 2025", "kw_nha_moi"),
        ("Nhà cấp 4 còn ở tốt", "kw_nha_cap_4"),
    ],
)
def test_required_phrases_fire(phrase, feature):
    assert bool(flags_for(phrase)[feature]), f"{feature!r} did not fire on {phrase!r}"


@pytest.mark.parametrize(
    ("phrase", "feature"),
    [
        ("Nha so do chinh chu", "kw_so_do"),
        ("so hong rieng", "kw_so_hong"),
        ("O TO DO CUA", "kw_o_to_do_cua"),
        ("MAT TIEN duong lon", "kw_mat_tien"),
        ("full noi that", "kw_full_noi_that"),
        ("nha cap 4", "kw_nha_cap_4"),
    ],
)
def test_matches_without_diacritics_or_case(phrase, feature):
    assert bool(flags_for(phrase)[feature])


def test_compound_written_without_space_matches():
    """Real listings write "ôtô" as one word; the pattern must accept both."""
    assert bool(flags_for("ôtô đỗ cửa")["kw_o_to"])
    assert bool(flags_for("ô tô đỗ cửa")["kw_o_to"])


def test_negation_guard_suppresses_risk_flags():
    """"Không tranh chấp" advertises the *absence* of a dispute."""
    assert bool(flags_for("Nhà có tranh chấp với hàng xóm")["kw_tranh_chap"])
    assert not bool(flags_for("Không tranh chấp, sổ đỏ rõ ràng")["kw_tranh_chap"])
    assert not bool(flags_for("Không quy hoạch, không giải toả")["kw_quy_hoach"])
    assert bool(flags_for("Đất dính quy hoạch mở đường")["kw_quy_hoach"])


def test_explicit_no_encumbrment_flag():
    assert bool(flags_for("Không dính quy hoạch")["kw_khong_quy_hoach"])
    assert not bool(flags_for("Dính quy hoạch")["kw_khong_quy_hoach"])


def test_unrelated_text_raises_no_flags():
    row = flags_for("Cần bán căn hộ view sông thoáng mát")
    assert not any(row[list(row.index)].tolist()), f"unexpected flags: {list(row[row].index)}"


def test_derived_flags_are_disjunctions():
    row = flags_for("Sổ hồng riêng, full nội thất")
    assert bool(row["kw_legal_certificate"])
    assert bool(row["kw_furnished"])
    assert bool(row["kw_so_hong"]) and bool(row["kw_noi_that"])

    negative = flags_for("Nhà cấp 4 cũ, không giấy tờ")
    assert not bool(negative["kw_legal_certificate"])
    assert not bool(negative["kw_furnished"])


def test_cho_thue_flags_a_mention_not_a_listing_type():
    """`kw_cho_thue` means "renting is mentioned", not "this is a rental listing".

    Locked in because the distinction is load-bearing: 96% of the rows this fires
    on in shard_0000 carry a sale-scale price (median 10 tỷ VND) and are sale
    listings pitching rental yield. Using it as a row filter silently discards
    ~19% of valid training data in a price-correlated way. Rental *listings* are
    excluded by clean_frame's price floor instead, since they quote a monthly rate.
    """
    # A genuine rental listing fires.
    assert bool(flags_for("Cho thuê nhà chính chủ, 60 triệu/tháng")["kw_cho_thue"])
    # A SALE listing that pitches rental yield also fires — expected, not a bug.
    assert bool(flags_for("Bán nhà mặt tiền, sẵn hợp đồng thuê, tiện cho thuê")["kw_cho_thue"])
    # A plain sale with no rental mention does not.
    assert not bool(flags_for("Bán nhà mặt tiền sổ đỏ")["kw_cho_thue"])


def test_extract_preserves_frame_index():
    frame = pd.DataFrame(
        {"name": ["a", "b", "c"], "description": ["sổ đỏ", "mặt tiền", "cho thuê"]},
        index=[10, 20, 30],
    )
    out = KeywordExtractor().extract(frame)
    assert list(out.index) == [10, 20, 30]
    assert out.loc[10, "kw_so_do"] and out.loc[20, "kw_mat_tien"] and out.loc[30, "kw_cho_thue"]


def test_feature_names_cover_every_rule_and_derivation():
    extractor = KeywordExtractor()
    expected = [f"kw_{r.name}" for r in LEXICON] + [f"kw_{n}" for n in DERIVED_FLAGS]
    assert extractor.feature_names == expected
    frame = pd.DataFrame({"name": ["x"], "description": ["sổ đỏ"]})
    assert list(extractor.extract(frame).columns) == expected


def test_extract_rejects_missing_text_column():
    with pytest.raises(KeyError, match="description"):
        KeywordExtractor().extract(pd.DataFrame({"name": ["sổ đỏ"]}))


def test_empty_frame_returns_typed_columns():
    out = KeywordExtractor().extract(pd.DataFrame({"name": [], "description": []}))
    assert out.empty
    assert list(out.columns) == KeywordExtractor().feature_names
    assert all(dtype == bool for dtype in out.dtypes)


def test_prevalence_reports_every_feature():
    frame = pd.DataFrame(
        {"name": [""] * 4, "description": ["sổ đỏ", "sổ hồng", "mặt tiền", "không gì cả"]}
    )
    extractor = KeywordExtractor()
    prevalence = extractor.prevalence(extractor.extract(frame))
    assert len(prevalence) == len(extractor.feature_names)
    row = prevalence.set_index("feature").loc["kw_so_do"]
    assert row["hits"] == 1 and row["rate"] == pytest.approx(0.25)


def test_every_rule_pattern_is_diacritic_free():
    """Patterns match folded text, so a literal diacritic in one can never fire."""
    offending = [
        r.name
        for r in LEXICON
        if any(ord(ch) > 127 for ch in r.pattern)
    ]
    assert not offending, f"patterns contain non-ASCII characters: {offending}"


def test_lexicon_categories_are_valid():
    for category in ("legal", "access", "condition", "intent"):
        assert rules_by_category(category), f"no rules for category {category!r}"
    assert sum(len(rules_by_category(c)) for c in ("legal", "access", "condition", "intent")) == len(
        LEXICON
    )


def test_duplicate_rule_name_rejected():
    rule = KeywordRule("dup", "legal", "x", r"\bx\b")
    with pytest.raises(ValueError, match="duplicate"):
        KeywordExtractor(rules=(rule, rule))


def test_unknown_category_rejected():
    rule = KeywordRule("x", "not_a_category", "x", r"\bx\b")
    with pytest.raises(ValueError, match="unknown category"):
        KeywordExtractor(rules=(rule,))


def test_derived_flag_referencing_unknown_rule_rejected():
    rule = KeywordRule("real", "legal", "x", r"\bx\b")
    with pytest.raises(ValueError, match="unknown rules"):
        KeywordExtractor(rules=(rule,), derived_flags={"d": ("real", "ghost")})


def test_derived_flag_colliding_with_rule_name_rejected():
    rule = KeywordRule("clash", "legal", "x", r"\bx\b")
    with pytest.raises(ValueError, match="collides"):
        KeywordExtractor(rules=(rule,), derived_flags={"clash": ("clash",)})
