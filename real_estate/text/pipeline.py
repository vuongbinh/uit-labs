"""One-call text feature pipeline for listing ``name`` / ``description``.

Composes the keyword flags (:mod:`real_estate.text.keywords`), numeric entities
(:mod:`real_estate.text.entities`) and a fitted text representation
(:mod:`real_estate.text.representation`) behind a single scikit-learn-style
``fit`` / ``transform`` object.

Leakage rule: keyword and entity extraction are stateless, so they cannot leak.
TF-IDF vocabularies, SVD bases and any PhoBERT reduction **are** fitted, and are
therefore learned from the training frame only — ``transform`` on a held-out
frame reuses that state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from .entities import (
    ENTITY_COLUMNS,
    TEXT_IMPUTABLE_COLUMNS,
    extract_entities,
    prepare_for_embedding,
    prepare_for_tfidf,
)
from .keywords import DEFAULT_TEXT_COLUMNS, KeywordExtractor
from .representation import PhoBERTEmbedder, TfidfTextEncoder

__all__ = ["TextFeatureConfig", "TextFeaturePipeline", "FEATURE_GROUPS"]

#: Named feature groups, used for ablation in the benchmark.
FEATURE_GROUPS: tuple[str, ...] = ("keywords", "entities", "tfidf", "phobert")


@dataclass
class TextFeatureConfig:
    """Knobs for :class:`TextFeaturePipeline`."""

    text_columns: tuple[str, ...] = DEFAULT_TEXT_COLUMNS
    use_keywords: bool = True
    use_entities: bool = True
    use_tfidf: bool = True
    use_phobert: bool = False
    tfidf_mode: str = "word"
    tfidf_components: int = 96
    tfidf_max_features: int = 60_000
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    tfidf_min_df: int = 5
    fold_for_tfidf: bool = True
    redact_price: bool = True
    phobert: dict = field(default_factory=dict)
    seed: int = 0

    def groups(self) -> list[str]:
        """Enabled feature groups, in canonical order."""
        return [
            name
            for name, enabled in (
                ("keywords", self.use_keywords),
                ("entities", self.use_entities),
                ("tfidf", self.use_tfidf),
                ("phobert", self.use_phobert),
            )
            if enabled
        ]


class TextFeaturePipeline:
    """Fit text-feature state on a training frame and apply it anywhere.

    Examples
    --------
    >>> pipeline = TextFeaturePipeline(TextFeatureConfig(tfidf_components=8))
    >>> _ = pipeline.fit(train_df)
    >>> test_features = pipeline.transform(test_df)
    """

    def __init__(self, config: TextFeatureConfig | None = None) -> None:
        self.config = config or TextFeatureConfig()
        if not self.config.groups():
            raise ValueError("at least one text feature group must be enabled")
        self.keywords = (
            KeywordExtractor(text_columns=self.config.text_columns)
            if self.config.use_keywords
            else None
        )
        self.tfidf: TfidfTextEncoder | None = None
        self.phobert: PhoBERTEmbedder | None = None
        self._fitted = False

    # ------------------------------------------------------------------ setup
    def _joined_texts(self, df: pd.DataFrame) -> list[str]:
        missing = [c for c in self.config.text_columns if c not in df.columns]
        if missing:
            raise KeyError(f"text column(s) not present in frame: {missing}")
        joined = (
            df[list(self.config.text_columns)].astype("string").fillna("").agg(" ".join, axis=1)
        )
        return list(joined)

    def _tfidf_texts(self, df: pd.DataFrame) -> list[str]:
        """Text for the TF-IDF view: folded by default, price mentions redacted."""
        prepare = prepare_for_tfidf if self.config.fold_for_tfidf else prepare_for_embedding
        return [prepare(t, redact_price=self.config.redact_price) for t in self._joined_texts(df)]

    def _embedding_texts(self, df: pd.DataFrame) -> list[str]:
        """Text for PhoBERT: diacritics preserved, price mentions redacted."""
        return [
            prepare_for_embedding(t, redact_price=self.config.redact_price)
            for t in self._joined_texts(df)
        ]

    # ---------------------------------------------------------------- fitting
    def _build_encoders(self) -> None:
        """Instantiate the fitted encoders without fitting them."""
        if self.config.use_tfidf:
            self.tfidf = TfidfTextEncoder(
                mode=self.config.tfidf_mode,
                n_components=self.config.tfidf_components,
                max_features=self.config.tfidf_max_features,
                ngram_range=self.config.tfidf_ngram_range,
                min_df=self.config.tfidf_min_df,
                seed=self.config.seed,
            )
        if self.config.use_phobert:
            self.phobert = PhoBERTEmbedder(seed=self.config.seed, **self.config.phobert)

    def fit(self, df: pd.DataFrame) -> TextFeaturePipeline:
        """Fit TF-IDF / PhoBERT state on ``df``."""
        self._build_encoders()
        if self.tfidf is not None:
            self.tfidf.fit(self._tfidf_texts(df))
        if self.phobert is not None:
            self.phobert.fit(self._embedding_texts(df))
        self._fitted = True
        return self

    # -------------------------------------------------------------- transform
    def _assemble(
        self,
        df: pd.DataFrame,
        tfidf_block: pd.DataFrame | None = None,
        phobert_block: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Column-bind the stateless blocks with any precomputed representations."""
        blocks: list[pd.DataFrame] = []
        if self.keywords is not None:
            blocks.append(self.keywords.extract(df))

        if self.config.use_entities:
            entities = extract_entities(df, text_columns=self.config.text_columns)
            blocks.append(entities)
            blocks.append(self._provenance_flags(df, entities))

        if tfidf_block is not None:
            blocks.append(tfidf_block)
        if phobert_block is not None:
            blocks.append(phobert_block)

        out = pd.concat(blocks, axis=1)
        if len(out) != len(df):
            raise AssertionError(
                f"text feature block has {len(out)} rows for a {len(df)}-row frame; "
                "a sub-block is not index-aligned"
            )
        out.index = df.index
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return all enabled text-derived features for ``df``, index-aligned.

        Output column groups:

        ``kw_*``
            boolean domain flags.
        ``text_*``
            numeric entities parsed from the listing text.
        ``<col>__from_text``
            boolean provenance, true where a structured column was null and the
            text supplied a value (see :func:`real_estate.text.entities.impute_from_text`).
        ``tfidf_*`` / ``phobert_*``
            dense representation components.
        """
        if (self.config.use_tfidf or self.config.use_phobert) and not self._fitted:
            raise RuntimeError("TextFeaturePipeline.transform called before fit")

        # Representation encoders are index-agnostic and return a fresh
        # RangeIndex; re-attach df's labels so the axis=1 concat in _assemble
        # aligns row-for-row instead of taking an index union.
        tfidf_block = (
            self.tfidf.transform(self._tfidf_texts(df)).set_axis(df.index)
            if self.tfidf is not None
            else None
        )
        phobert_block = (
            self.phobert.transform(self._embedding_texts(df)).set_axis(df.index)
            if self.phobert is not None
            else None
        )
        return self._assemble(df, tfidf_block, phobert_block)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``df`` and return its features without encoding its text twice.

        PhoBERT encoding dominates this pipeline's cost, so the fitted frame's
        representation is taken from the encoders' own ``fit_transform`` rather
        than recomputed by a following ``transform`` pass.
        """
        self._build_encoders()
        tfidf_block = (
            self.tfidf.fit_transform(self._tfidf_texts(df)).set_axis(df.index)
            if self.tfidf is not None
            else None
        )
        phobert_block = (
            self.phobert.fit_transform(self._embedding_texts(df)).set_axis(df.index)
            if self.phobert is not None
            else None
        )
        self._fitted = True
        return self._assemble(df, tfidf_block, phobert_block)

    # ------------------------------------------------------------------ utils
    def _provenance_flags(self, df: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
        """Mark structured columns whose text-derived replacement would be used."""
        flags = {}
        for column, source in TEXT_IMPUTABLE_COLUMNS.items():
            if column not in df.columns or source not in entities.columns:
                continue
            original = pd.to_numeric(df[column], errors="coerce")
            flags[f"{column}__from_text"] = (original.isna() | (original <= 0)) & entities[source].notna()
        return pd.DataFrame(flags, index=df.index, dtype=bool)

    def group_columns(self, features: pd.DataFrame) -> dict[str, list[str]]:
        """Partition an emitted feature frame into its ablation groups."""
        columns = list(features.columns)
        groups: dict[str, list[str]] = {name: [] for name in FEATURE_GROUPS}
        for column in columns:
            if column.startswith("kw_"):
                groups["keywords"].append(column)
            elif column.startswith("tfidf_"):
                groups["tfidf"].append(column)
            elif column.startswith("phobert_"):
                groups["phobert"].append(column)
            elif column.startswith("text_") or column.endswith("__from_text"):
                groups["entities"].append(column)
        return {name: cols for name, cols in groups.items() if cols}

    @staticmethod
    def entity_columns() -> Sequence[str]:
        """Entity columns parsed from text, in canonical order."""
        return ENTITY_COLUMNS
