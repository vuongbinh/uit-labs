"""Numeric entity extraction from listing text.

The structured columns of ``tinixai/vietnam-real-estates`` are heavily sparse
(``floor_count``, ``frontage_width``, ``house_depth`` and ``road_width`` are
routinely null) while the free text routinely states those same values — e.g.
``"nhà 6 tầng"``, ``"4m x 12m"``, ``"MT đường 8m"``.  This module parses them
back out so they can impute the tabular frame.

Values are extracted from diacritic-folded text.  Numbers follow Vietnamese
listing convention, where both ``.`` and ``,`` appear as decimal separators
(``"7.45 tỷ"``, ``"1,3 tỷ"``) and ``.`` also groups thousands (``"1.300.000"``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .normalize import clean_listing_text, fold_diacritics, fold_with_alignment, join_text_columns

__all__ = [
    "parse_vn_number",
    "extract_entities_folded",
    "extract_entities",
    "impute_from_text",
    "find_price_spans",
    "redact_spans",
    "prepare_for_tfidf",
    "prepare_for_embedding",
    "ENTITY_COLUMNS",
    "TEXT_IMPUTABLE_COLUMNS",
    "TARGET_LEAKING_ENTITIES",
]

#: Entities that restate the regression target and must not be fed to a model
#: trained on that target.  ``price`` is written verbatim into most listing
#: descriptions ("giá: 7.45 tỷ"), so parsing it back out and offering it as a
#: feature is circular — it measures transcription, not valuation.
TARGET_LEAKING_ENTITIES: tuple[str, ...] = ("text_price_vnd",)

#: Replacement token for redacted price mentions.
PRICE_REDACTION = " <gia> "

#: Columns emitted by :func:`extract_entities`.
ENTITY_COLUMNS: tuple[str, ...] = (
    "text_price_vnd",
    "text_area_m2",
    "text_frontage_m",
    "text_depth_m",
    "text_floor_count",
    "text_lau_count",
    "text_bedroom_count",
    "text_bathroom_count",
    "text_road_width_m",
)

#: Structured columns that text entities may fill, mapped to their entity source.
TEXT_IMPUTABLE_COLUMNS: dict[str, str] = {
    "area": "text_area_m2",
    "frontage_width": "text_frontage_m",
    "house_depth": "text_depth_m",
    "floor_count": "text_floor_count",
    "bedroom_count": "text_bedroom_count",
    "bathroom_count": "text_bathroom_count",
    "road_width": "text_road_width_m",
}

# A number token using either separator style: 4 | 7.45 | 1,3 | 1.300.000
_NUM = r"\d+(?:[.,]\d+)+"
_INT = r"\d+"
_ANY_NUM = rf"(?:{_NUM}|{_INT})"

_AREA_RE = re.compile(
    rf"\b({_ANY_NUM})\s*(?:m2|mv|met\s+vuong|metro\s+vuong)\b", re.IGNORECASE
)
# "4m x 12m", "4 x 12", "4,5x16m"
_DIMS_RE = re.compile(
    rf"\b({_ANY_NUM})\s*m?\s*[x*]\s*({_ANY_NUM})\s*m?\b"
)
# "ngang 4m ... dai 12m"
_NGANG_DAI_RE = re.compile(rf"\bngang\s+({_ANY_NUM})\s*m?\b[^.]{{0,40}}?\bdai\s+({_ANY_NUM})\s*m?\b")
_TANG_RE = re.compile(rf"\b({_INT})\s*(?:tang|tam)\b")
_LAU_RE = re.compile(rf"\b({_INT})\s+lau\b")
_BED_RE = re.compile(rf"\b({_INT})\s*(?:pn|phong\s+ngu|bedroom)\b")
_BATH_RE = re.compile(rf"\b({_INT})\s*(?:wc|toilet|phong\s+tam|nvs|vs)\b")
# "duong 8m", "hem 3m", "ngo 2.5m", "mat duong 12m"
_ROAD_RE = re.compile(rf"\b(?:duong|hem|ngo|mat\s+duong|mt\s+duong|lo\s+gioi)\s+({_ANY_NUM})\s*(?:m|met)\b")

# Price with an explicit Vietnamese magnitude word.  "ty"/"ti" = 1e9, "tr" = 1e6.
_PRICE_UNITS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(rf"\b({_ANY_NUM})\s*(?:ty|ti)\b"), 1e9),
    (re.compile(rf"\b({_ANY_NUM})\s*(?:trieu|tr)\b"), 1e6),
    (re.compile(rf"\b({_ANY_NUM})\s*(?:nghin|ngan|k)\b"), 1e3),
)
_GIA_RE = re.compile(r"\bgia\b")

# Sanity bounds: reject parses that cannot describe a real listing.
_MAX_AREA_M2 = 100_000.0
_MAX_DIM_M = 500.0
_MAX_FLOORS = 200.0
_MAX_ROOMS = 200.0
_MAX_ROAD_M = 200.0
_MAX_PRICE_VND = 1e15
_MIN_PRICE_VND = 1e5


def parse_vn_number(token: str) -> float | None:
    """Parse a Vietnamese-styled numeric token into a float.

    ``"7.45"`` -> 7.45, ``"1,3"`` -> 1.3, ``"1.300.000"`` -> 1300000.0,
    ``"1.300"`` -> 1300.0 (a 3-digit group is treated as thousands), ``"4"`` -> 4.0.
    """
    token = token.strip()
    if not token:
        return None
    parts = re.split(r"[.,]", token)
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        whole, frac = parts
        # 1-2 digits after a single separator is a decimal; 3 digits is a
        # thousands group ("1.300" = one thousand three hundred).
        if 1 <= len(frac) <= 2:
            return float(f"{whole}.{frac}")
        return float(whole + frac)
    # Multiple separators can only be thousands grouping.
    return float("".join(parts))


def _first(pattern: re.Pattern[str], text: str, *, limit: float | None = None) -> float | None:
    match = pattern.search(text)
    if not match:
        return None
    value = parse_vn_number(match.group(1))
    if value is None:
        return None
    if limit is not None and not (0 < value <= limit):
        return None
    return value


def _extract_price(folded: str) -> float | None:
    """Return the advertised price in VND, preferring a value stated near "giá"."""
    best: tuple[float, float] | None = None  # (distance penalty, value)
    gia = _GIA_RE.search(folded)
    anchor = gia.start() if gia else None
    for pattern, multiplier in _PRICE_UNITS:
        for match in pattern.finditer(folded):
            amount = parse_vn_number(match.group(1))
            if amount is None:
                continue
            value = amount * multiplier
            if not (_MIN_PRICE_VND <= value <= _MAX_PRICE_VND):
                continue
            penalty = abs(match.start() - anchor) if anchor is not None else 1e6
            candidate = (penalty, value)
            if best is None or candidate[0] < best[0]:
                best = candidate
            if anchor is None:
                # Without a "giá" anchor the first stated amount wins.
                return value
    return best[1] if best else None


def _extract_floors(folded: str) -> float | None:
    """Return the level count from an explicit "N tầng" / "N tấm".

    ``N lầu`` is deliberately *not* converted here: southern listings use it
    both for "N levels above ground" and for "N levels total", so adding a
    ground floor would bake in an unfounded assumption.  ``text_lau_count`` is
    exposed as its own raw signal instead, and only the unambiguous
    ``tầng``/``tấm`` form imputes the structured ``floor_count`` column.
    """
    return _first(_TANG_RE, folded, limit=_MAX_FLOORS)


# ------------------------------------------------------------- price redaction
def find_price_spans(folded: str) -> list[tuple[int, int]]:
    """Locate advertised-price mentions in diacritic-folded text.

    Returns merged, left-to-right ``(start, end)`` spans covering each numeric
    amount and its magnitude word (``"7.45 ty"``, ``"900 trieu"``).
    """
    spans: list[tuple[int, int]] = []
    for pattern, _ in _PRICE_UNITS:
        spans.extend((m.start(), m.end()) for m in pattern.finditer(folded))
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def redact_spans(
    text: str, spans: Iterable[tuple[int, int]], replacement: str = PRICE_REDACTION
) -> str:
    """Replace each span in ``text`` with ``replacement``, right to left."""
    out = text
    for start, end in sorted(spans, reverse=True):
        out = out[:start] + replacement + out[end:]
    return out


def prepare_for_tfidf(text: str | None, redact_price: bool = True) -> str:
    """Clean, diacritic-fold and price-redact text for the TF-IDF view.

    Folding is applied because the TF-IDF vocabulary benefits from unifying
    spelling variants (``ô tô`` / ``ôtô``), and redaction removes the advertised
    price so bag-of-words features cannot smuggle the regression target in as
    digit n-grams.
    """
    folded = fold_diacritics(clean_listing_text(text))
    if redact_price:
        folded = redact_spans(folded, find_price_spans(folded))
    return folded


def _map_spans(spans: Iterable[tuple[int, int]], origin: list[int]) -> list[tuple[int, int]]:
    """Translate spans in folded text back into spans in the source string."""
    mapped = []
    for start, end in spans:
        if start < 0 or start >= end or end > len(origin):
            continue
        mapped.append((origin[start], origin[end - 1] + 1))
    return mapped


def prepare_for_embedding(text: str | None, redact_price: bool = True) -> str:
    """Clean and price-redact text for a diacritic-sensitive encoder (PhoBERT).

    PhoBERT was trained on properly diacritised Vietnamese, so diacritics are
    *kept* here; price spans are located on the folded form and mapped back onto
    the cleaned string per character, because folding is not length-preserving
    (stray combining marks and emoji variation selectors are dropped).
    """
    cleaned = clean_listing_text(text)
    if not redact_price:
        return cleaned
    folded, origin = fold_with_alignment(cleaned)
    return redact_spans(cleaned, _map_spans(find_price_spans(folded), origin))


def extract_entities_folded(folded_texts: Iterable[str]) -> pd.DataFrame:
    """Extract numeric entities from already clean, diacritic-folded text."""
    rows: list[dict[str, float]] = []
    for folded in folded_texts:
        area = _first(_AREA_RE, folded, limit=_MAX_AREA_M2)
        dims = _DIMS_RE.search(folded) or _NGANG_DAI_RE.search(folded)
        frontage = depth = None
        if dims:
            f = parse_vn_number(dims.group(1))
            d = parse_vn_number(dims.group(2))
            if f is not None and 0 < f <= _MAX_DIM_M:
                frontage = f
            if d is not None and 0 < d <= _MAX_DIM_M:
                depth = d
        rows.append(
            {
                "text_price_vnd": _extract_price(folded),
                "text_area_m2": area,
                "text_frontage_m": frontage,
                "text_depth_m": depth,
                "text_floor_count": _extract_floors(folded),
                "text_lau_count": _first(_LAU_RE, folded, limit=_MAX_FLOORS),
                "text_bedroom_count": _first(_BED_RE, folded, limit=_MAX_ROOMS),
                "text_bathroom_count": _first(_BATH_RE, folded, limit=_MAX_ROOMS),
                "text_road_width_m": _first(_ROAD_RE, folded, limit=_MAX_ROAD_M),
            }
        )
    frame = pd.DataFrame(rows, columns=list(ENTITY_COLUMNS), dtype="float64")
    return frame


def extract_entities(df: pd.DataFrame, text_columns: tuple[str, ...] = ("name", "description")) -> pd.DataFrame:
    """Extract numeric entities for ``df``, preserving its index."""
    missing = [c for c in text_columns if c not in df.columns]
    if missing:
        raise KeyError(f"text column(s) not present in frame: {missing}")
    folded = (
        fold_diacritics(clean_listing_text(join_text_columns(row)))
        for row in df[list(text_columns)].itertuples(index=False, name=None)
    )
    return extract_entities_folded(folded).set_axis(df.index)


def impute_from_text(tabular: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
    """Fill null structured columns from text-derived entities.

    Returns a copy containing the imputed structured columns plus, for every
    imputed column, a ``<column>__from_text`` boolean recording that the value
    came from free text rather than the source record.  Existing non-null values
    are never overwritten, so the transformation is monotone in information.
    """
    if not tabular.index.equals(entities.index):
        entities = entities.reindex(tabular.index)
    out = tabular.copy()
    for column, source in TEXT_IMPUTABLE_COLUMNS.items():
        if column not in out.columns or source not in entities.columns:
            continue
        original = pd.to_numeric(out[column], errors="coerce").astype("float64")
        fill = entities[source].astype("float64")
        filled = np.where(original.notna() & (original > 0), original, fill)
        out[column] = pd.Series(filled, index=out.index, dtype="float64")
        out[f"{column}__from_text"] = pd.Series(
            (original.isna() | (original <= 0)) & fill.notna(),
            index=out.index,
            dtype=bool,
        )
    return out
