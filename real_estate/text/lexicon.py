"""Domain lexicon for Vietnamese real-estate listings.

Patterns are written against *diacritic-folded, lower-cased* text (see
:mod:`real_estate.text.normalize`), so one rule matches both ``"ô tô đỗ cửa"``
and ``"ôtô đỗ cổng"`` / ``"o to do cua"``.

Every rule was checked for prevalence on a 60k-row sample of
``tinixai/vietnam-real-estates`` before being kept here; ``label`` records the
canonical Vietnamese phrase for documentation and review.

Categories
----------
``legal``      certificate / title / planning status
``access``     road, alley and frontage accessibility
``condition``  building age, interior fit-out, amenities
``intent``     transaction intent and marketing framing
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["KeywordRule", "LEXICON", "DERIVED_FLAGS", "rules_by_category", "CATEGORIES"]


@dataclass(frozen=True)
class KeywordRule:
    """One matchable domain concept.

    Attributes
    ----------
    name:
        Feature suffix.  The extractor emits ``kw_<name>``.
    category:
        One of :data:`CATEGORIES`.
    label:
        Human-readable Vietnamese phrase(s) this rule stands for.
    pattern:
        Regex matched against folded text.  Use ``\\s+`` between syllables and
        ``o\\s?to`` style optionals for compounds written with or without a space.
    negation_guard:
        When true the match is discarded if a negation cue (``không``, ``chưa``,
        ``ko``, ``kg``...) immediately precedes it.  Set for terms where the
        negated form is common and means the *opposite* — e.g. ``không tranh
        chấp`` ("no dispute") must not flag a legal risk.
    """

    name: str
    category: str
    label: str
    pattern: str
    negation_guard: bool = False


CATEGORIES: tuple[str, ...] = ("legal", "access", "condition", "intent")

LEXICON: tuple[KeywordRule, ...] = (
    # ---------------------------------------------------------------- legal
    KeywordRule("so_do", "legal", "sổ đỏ", r"\bso\s+do\b"),
    KeywordRule("so_hong", "legal", "sổ hồng", r"\bso\s+hong\b"),
    KeywordRule(
        "so_do_hong",
        "legal",
        "sổ đỏ / sổ hồng (either certificate)",
        r"\bso\s+(do|hong)\b",
    ),
    KeywordRule("chinh_chu", "legal", "chính chủ", r"\bchinh\s+chu\b"),
    KeywordRule("phap_ly", "legal", "pháp lý", r"\bphap\s+ly\b"),
    KeywordRule(
        "phap_ly_ro_rang",
        "legal",
        "pháp lý rõ ràng / đầy đủ",
        r"\bphap\s+ly\s+(ro\s+rang|day\s+du|ro|chuan|minh\s+bach|hoan\s+chinh)\b",
    ),
    KeywordRule(
        "dang_cho_so",
        "legal",
        "đang chờ sổ",
        r"\b(dang\s+)?cho\s+so\b|\bso\s+(dang\s+)?(cho|ve|lam)\b|\bchua\s+co\s+so\b",
    ),
    KeywordRule("giay_to", "legal", "giấy tờ (hợp lệ)", r"\bgiay\s+to\b"),
    KeywordRule("cong_chung", "legal", "công chứng", r"\bcong\s+chung\b"),
    KeywordRule("hop_dong", "legal", "hợp đồng (mua bán)", r"\bhop\s+dong\b"),
    KeywordRule(
        "tranh_chap",
        "legal",
        "tranh chấp",
        r"\btranh\s+chap\b",
        negation_guard=True,
    ),
    # Raw mention is ambiguous: "không quy hoạch" advertises the *absence* of a
    # planning encumbrance.  The guard turns the bare mention into a risk signal
    # and a separate rule captures the positive claim.
    KeywordRule(
        "quy_hoach",
        "legal",
        "(dính) quy hoạch",
        r"\bquy\s+hoach\b",
        negation_guard=True,
    ),
    KeywordRule(
        "khong_quy_hoach",
        "legal",
        "không quy hoạch / không tranh chấp",
        r"\b(khong|ko|kg)\s+(dinh\s+)?(quy\s+hoach|tranh\s+chap)\b",
    ),
    # --------------------------------------------------------------- access
    KeywordRule("mat_tien", "access", "mặt tiền", r"\bmat\s+tien\b"),
    KeywordRule(
        "o_to_do_cua",
        "access",
        "ô tô đỗ cửa / đỗ cổng",
        r"\bo\s?to\s+(do|dau|ghe)\s+(cua|cong|nha|cua\s+nha)\b",
    ),
    KeywordRule(
        "xe_hoi_vao_nha",
        "access",
        "xe hơi vào nhà",
        r"\b(xe\s+hoi|o\s?to)\s+(vao|do|chay|den)\s+(nha|trong\s+nha|tan\s+nha|trong\s+trong)\b",
    ),
    KeywordRule("o_to", "access", "ô tô (bất kỳ ngữ cảnh nào)", r"\bo\s?to\b"),
    KeywordRule("xe_hoi", "access", "xe hơi", r"\bxe\s+hoi\b"),
    KeywordRule("ngo_thong", "access", "ngõ thông", r"\bngo\s+thong\b|\bhem\s+thong\b"),
    KeywordRule(
        "ngo_ba_gac",
        "access",
        "ngõ ba gác",
        r"\b(ngo|hem|ngo\s+hem)\s+(ba\s+gac|3\s+gac|bagac)\b|\bxe\s+ba\s+gac\b|\bba\s+gac\b",
    ),
    KeywordRule("hem", "access", "hẻm", r"\bhem\b"),
    KeywordRule("hem_xe_hoi", "access", "hẻm xe hơi", r"\bhem\s+(xe\s+hoi|o\s?to)\b"),
    KeywordRule("via_he", "access", "vỉa hè", r"\bvia\s+he\b|\ble\s+duong\b"),
    KeywordRule("phan_lo", "access", "khu phân lô", r"\bphan\s+lo\b"),
    KeywordRule("hai_mat_thoang", "access", "2 mặt thoáng", r"\b(2|hai)\s+mat\s+thoang\b"),
    KeywordRule("lo_goc", "access", "lô góc / căn góc", r"\b(lo|can|nha|dat)\s+goc\b"),
    KeywordRule("duong_rong", "access", "đường rộng / đường lớn", r"\bduong\s+(rong|lon|to)\b|\bduong\s+\d+\s*m\b"),
    # ------------------------------------------------------------ condition
    KeywordRule(
        "full_noi_that",
        "condition",
        "full nội thất / nội thất đầy đủ",
        r"\b(full|day\s+du|tron)\s+noi\s+that\b|\bnoi\s+that\s+(day\s+du|full|cao\s+cap|sang\s+trong|xach\s+vali)\b",
    ),
    KeywordRule("noi_that_co_ban", "condition", "nội thất cơ bản", r"\bnoi\s+that\s+co\s+ban\b"),
    KeywordRule("noi_that", "condition", "nội thất (bất kỳ)", r"\bnoi\s+that\b"),
    KeywordRule(
        "nha_moi",
        "condition",
        "nhà mới / xây mới",
        r"\bnha\s+moi\b|\bmoi\s+xay\b|\bxay\s+moi\b|\bmoi\s+coong\b|\bmoi\s+tinh\b",
    ),
    KeywordRule("nha_cap_4", "condition", "nhà cấp 4", r"\bnha\s+cap\s*4\b|\bcap\s*4\b"),
    KeywordRule("nha_cu", "condition", "nhà cũ / xuống cấp", r"\bnha\s+(cu|tho|nat)\b|\bxuong\s+cap\b"),
    KeywordRule("o_ngay", "condition", "vào ở ngay", r"\b(vao\s+)?o\s+ngay\b|\bdon\s+vao\s+o\b"),
    KeywordRule("thang_may", "condition", "thang máy", r"\bthang\s+may\b"),
    KeywordRule("san_thuong", "condition", "sân thượng", r"\bsan\s+thuong\b"),
    KeywordRule("tang_ham", "condition", "tầng hầm / bán hầm", r"\btang\s+ham\b|\bban\s+ham\b|\bham\s+(xe|o\s?to|gara)\b"),
    # --------------------------------------------------------------- intent
    # MEASURES A MENTION, NOT A LISTING TYPE. On shard_0000 this fires on 18.8%
    # of rows, but 96% of those carry a sale-scale price (median 10 tỷ VND): they
    # are sale listings pitching rental yield ("sẵn hợp đồng thuê", "tiện cho
    # thuê"), not rentals. Genuine rental listings quote a monthly rate (15-60M
    # VND) and are excluded by clean_frame's price floor instead. Use this as a
    # feature, never as a row filter.
    KeywordRule(
        "cho_thue",
        "intent",
        "nhắc đến cho thuê (không phân loại được tin thuê)",
        r"\bcho\s+thue\b|\bgia\s+thue\b|\bthue\s+nguyen\s+can\b|\bcan\s+ho\s+cho\s+thue\b",
    ),
    KeywordRule("kinh_doanh", "intent", "tiện kinh doanh", r"\bkinh\s+doanh\b|\bbuon\s+ban\b|\bcho\s+thue\s+kinh\s+doanh\b"),
    KeywordRule("ban_gap", "intent", "bán gấp / bán nhanh", r"\bban\s+(gap|nhanh|lo)\b|\bcan\s+ban\s+(gap|nhanh)\b|\bcan\s+tien\s+ban\s+gap\b"),
    KeywordRule("gia_re", "intent", "giá rẻ / giá tốt", r"\bgia\s+(re|tot|mem|nhe)\b"),
    KeywordRule("thuong_luong", "intent", "có thương lượng", r"\bthuong\s+luong\b|\bco\s+thuong\s+luong\b|\bgia\s+co\s+the\s+thuong\b"),
)

#: Aggregate flags derived from primitive rules.  Keys are emitted feature
#: suffixes; values are the primitive rule names they combine (OR semantics).
DERIVED_FLAGS: dict[str, tuple[str, ...]] = {
    "legal_certificate": ("so_do", "so_hong"),
    "legal_risk": ("tranh_chap", "quy_hoach", "dang_cho_so"),
    "legal_clean": ("phap_ly_ro_rang", "so_do_hong", "cong_chung"),
    "road_car_access": ("o_to_do_cua", "xe_hoi_vao_nha", "hem_xe_hoi"),
    "furnished": ("full_noi_that", "noi_that"),
}


def rules_by_category(category: str) -> tuple[KeywordRule, ...]:
    """Return the rules belonging to ``category``."""
    return tuple(r for r in LEXICON if r.category == category)
