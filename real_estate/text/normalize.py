"""Vietnamese text normalisation shared by the keyword and TF-IDF layers.

Listing text on Vietnamese property portals is noisy: mixed diacritic conventions
(``ô tô`` / ``ôtô``), unicode superscripts (``m²``), assorted dash and multiplication
characters, and redaction placeholders such as ``[phone_number]``.  Everything
downstream matches against text passed through :func:`make_searchable`, so the
rules live in exactly one place.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "fold_diacritics",
    "fold_with_alignment",
    "clean_listing_text",
    "make_searchable",
    "join_text_columns",
]

# Redaction placeholders emitted by the source crawler, e.g. `[phone_number]`.
_PLACEHOLDER_RE = re.compile(r"\[[a-z_]+\]")

# Symbol variants that are safe to normalise while *keeping* Vietnamese letters
# intact: superscripts, multiplication signs, dashes, quotes, ellipsis, NBSP.
# All entries are one character to one character so string length is preserved.
_SYMBOL_MAP = str.maketrans(
    {
        "\u00b2": "2",  # ²  (m² -> m2)
        "\u00b3": "3",  # ³
        "\u00d7": "x",  # ×  (4 × 12 -> 4 x 12)
        "\u2013": "-",  # –
        "\u2014": "-",  # —
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": " ",  # …
        "\u00a0": " ",  # non-breaking space
    }
)

# đ/Đ have no decomposed form, so they need an explicit map.  Applied only when
# folding — cleaning must leave them alone, since "đỏ" and "dỏ" are different
# words to any diacritic-sensitive model (PhoBERT included).
_D_MAP = str.maketrans({"\u0111": "d", "\u0110": "D"})

_MULTI_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _fold_char(ch: str) -> str:
    """Fold a single character to its ASCII base form (possibly empty)."""
    decomposed = unicodedata.normalize("NFD", ch.translate(_SYMBOL_MAP).translate(_D_MAP))
    kept = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", kept).lower()


def fold_with_alignment(text: str) -> tuple[str, list[int]]:
    """Fold ``text`` and record which source character produced each output character.

    Folding is *not* length-preserving in general: combining marks are dropped,
    so a stray ``U+0301`` or an emoji variation selector (``U+FE0F``, category
    ``Mn``) makes the result shorter than its input, and real listing text does
    contain both.  Returning the per-character origin lets a span found in folded
    text be mapped back onto the original string exactly, which is what price
    redaction on the diacritic-preserving PhoBERT path needs.

    Returns ``(folded, origin)`` where ``origin[i]`` is the index in ``text`` of
    the character that produced ``folded[i]``.
    """
    pieces: list[str] = []
    origin: list[int] = []
    for index, ch in enumerate(text):
        folded = _fold_char(ch)
        pieces.append(folded)
        origin.extend([index] * len(folded))
    return "".join(pieces), origin


def fold_diacritics(text: str) -> str:
    """Lower-case and strip Vietnamese tone marks and diacritics.

    ``"Sổ đỏ chính chủ"`` and ``"so do chinh chu"`` both map to
    ``"so do chinh chu"``, which lets a single pattern match listings written
    with or without diacritics.  Use :func:`fold_with_alignment` when the folded
    offsets must be mapped back onto the original string.
    """
    return fold_with_alignment(text)[0]


def clean_listing_text(text: str | None) -> str:
    """Tidy a raw listing field without touching its Vietnamese letters.

    Diacritics — including ``đ`` — are preserved because they carry lexical
    meaning (``số`` vs ``sổ``, ``đỏ`` vs ``dỏ``); only whitespace, redaction
    placeholders and symbol variants are normalised.
    """
    if not text:
        return ""
    cleaned = _PLACEHOLDER_RE.sub(" ", text.translate(_SYMBOL_MAP))
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def make_searchable(text: str | None) -> str:
    """Clean and diacritic-fold ``text`` for regex matching."""
    return fold_diacritics(clean_listing_text(text))


def join_text_columns(values: object) -> str:
    """Join the text columns of one row into a single searchable string.

    Accepts ``None``, a plain string, or any iterable of them.  Columns are
    separated by a space so that n-gram features can span the title/description
    boundary, while regex matches stay anchored to real words.
    """
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    return " ".join(v for v in values if isinstance(v, str) and v)
