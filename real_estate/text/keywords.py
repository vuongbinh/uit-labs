"""Domain keyword / entity flags for Vietnamese real-estate listings.

Turns the raw ``name`` + ``description`` fields into a boolean feature block
(``kw_*``) covering legal status, road accessibility, building condition and
transaction intent.  See :mod:`real_estate.text.lexicon` for the vocabulary.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import pandas as pd

from .lexicon import CATEGORIES, DERIVED_FLAGS, LEXICON, KeywordRule
from .normalize import clean_listing_text, fold_diacritics, join_text_columns

__all__ = ["KeywordExtractor", "DEFAULT_TEXT_COLUMNS"]

DEFAULT_TEXT_COLUMNS: tuple[str, ...] = ("name", "description")

_FEATURE_PREFIX = "kw_"

# Negation cues that must appear immediately before a guarded match to cancel it.
# Matched against the tail of the preceding window, so "không dính quy hoạch"
# suppresses the `quy_hoach` risk flag.
_NEGATION_TAIL_RE = re.compile(
    r"\b(khong|ko|kg|k0|chang|chua|khong\s+he|khong\s+bi|khong\s+dinh|mie[n]?)\b\s*(\w{1,6}\s*)?$"
)


class KeywordExtractor:
    """Extract boolean domain flags from listing text.

    Parameters
    ----------
    rules:
        Lexicon entries to apply.  Defaults to the full :data:`LEXICON`.
    derived_flags:
        OR-aggregates of primitive rule names, e.g. ``legal_certificate``.
    negation_window:
        Number of trailing folded characters inspected for a negation cue.
    text_columns:
        Columns concatenated per row before matching.
    """

    def __init__(
        self,
        rules: Sequence[KeywordRule] = LEXICON,
        derived_flags: dict[str, tuple[str, ...]] | None = None,
        negation_window: int = 28,
        text_columns: Sequence[str] = DEFAULT_TEXT_COLUMNS,
    ) -> None:
        self.rules: tuple[KeywordRule, ...] = tuple(rules)
        self.derived_flags = dict(DERIVED_FLAGS if derived_flags is None else derived_flags)
        self.negation_window = int(negation_window)
        self.text_columns: tuple[str, ...] = tuple(text_columns)

        self._validate()
        self._compiled = {r.name: re.compile(r.pattern) for r in self.rules}

    # ------------------------------------------------------------------ setup
    def _validate(self) -> None:
        seen: set[str] = set()
        known = {r.name for r in self.rules}
        for rule in self.rules:
            if rule.name in seen:
                raise ValueError(f"duplicate lexicon rule name: {rule.name!r}")
            seen.add(rule.name)
            if rule.category not in CATEGORIES:
                raise ValueError(
                    f"rule {rule.name!r} has unknown category {rule.category!r}; "
                    f"expected one of {CATEGORIES}"
                )
            try:
                re.compile(rule.pattern)
            except re.error as exc:  # pragma: no cover - lexicon is static data
                raise ValueError(f"rule {rule.name!r} has an invalid pattern: {exc}") from exc
        for flag, sources in self.derived_flags.items():
            if flag in known:
                raise ValueError(f"derived flag {flag!r} collides with a primitive rule name")
            missing = [s for s in sources if s not in known]
            if missing:
                raise ValueError(f"derived flag {flag!r} references unknown rules: {missing}")

    @property
    def feature_names(self) -> list[str]:
        """Ordered output column names, primitive rules first."""
        return [f"{_FEATURE_PREFIX}{r.name}" for r in self.rules] + [
            f"{_FEATURE_PREFIX}{name}" for name in self.derived_flags
        ]

    # -------------------------------------------------------------- matching
    def _rule_hits(self, rule: KeywordRule, folded: str) -> bool:
        rx = self._compiled[rule.name]
        if not rule.negation_guard:
            return rx.search(folded) is not None
        if rx.search(folded) is None:
            return False
        for match in rx.finditer(folded):
            window = folded[max(0, match.start() - self.negation_window) : match.start()]
            if not _NEGATION_TAIL_RE.search(window):
                return True
        return False

    def extract_folded(self, folded_texts: Iterable[str]) -> pd.DataFrame:
        """Build the flag block from text that is already clean and folded."""
        rows: list[dict[str, bool]] = []
        for folded in folded_texts:
            values = {rule.name: self._rule_hits(rule, folded) for rule in self.rules}
            for flag, sources in self.derived_flags.items():
                values[flag] = any(values[s] for s in sources)
            rows.append({f"{_FEATURE_PREFIX}{k}": v for k, v in values.items()})
        flags = pd.DataFrame(rows, dtype=bool)
        if flags.empty:
            return pd.DataFrame({name: pd.Series(dtype=bool) for name in self.feature_names})
        return flags[self.feature_names]

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the flag block for ``df`` using :attr:`text_columns`.

        The returned frame keeps ``df``'s index so it can be concatenated with
        tabular features via ``pd.concat(..., axis=1)``.
        """
        missing = [c for c in self.text_columns if c not in df.columns]
        if missing:
            raise KeyError(f"text column(s) not present in frame: {missing}")
        folded = (
            fold_diacritics(clean_listing_text(join_text_columns(row)))
            for row in df[list(self.text_columns)].itertuples(index=False, name=None)
        )
        return self.extract_folded(folded).set_axis(df.index)

    # -------------------------------------------------------------- reporting
    def prevalence(self, flags: pd.DataFrame) -> pd.DataFrame:
        """Per-flag hit counts and rates — used to validate the lexicon on real data."""
        labels = {r.name: r.label for r in self.rules}
        categories = {r.name: r.category for r in self.rules}
        n = len(flags)
        records = []
        for column in flags.columns:
            name = column[len(_FEATURE_PREFIX) :]
            hits = int(flags[column].sum())
            records.append(
                {
                    "feature": column,
                    "category": categories.get(name, "derived"),
                    "label": labels.get(name, " / ".join(self.derived_flags.get(name, ()))),
                    "hits": hits,
                    "rate": (hits / n) if n else float("nan"),
                }
            )
        return pd.DataFrame(records).sort_values("rate", ascending=False).reset_index(drop=True)
