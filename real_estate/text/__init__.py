"""Text feature extraction for Vietnamese real-estate listings.

Public surface:

* :class:`KeywordExtractor` — boolean domain flags (legal / access / condition / intent)
* :func:`extract_entities`, :func:`impute_from_text` — numeric entities parsed from text
* :class:`TfidfTextEncoder`, :class:`PhoBERTEmbedder` — dense representations
* :class:`TextFeaturePipeline` — one-call composition of all of the above

Typical use::

    from real_estate.text import TextFeatureConfig, TextFeaturePipeline

    pipeline = TextFeaturePipeline(TextFeatureConfig(tfidf_components=96))
    train_text = pipeline.fit_transform(train_df)   # fits TF-IDF on train only
    test_text = pipeline.transform(test_df)
"""

from __future__ import annotations

from .data import (
    CATEGORICAL_COLUMNS,
    DATASET_ID,
    NUMERIC_COLUMNS,
    SHARD_URL_TEMPLATE,
    TARGET_COLUMN,
    TEXT_COLUMNS,
    TIME_COLUMN,
    clean_frame,
    load_shard_sample,
    normalize_unicode,
    out_of_time_pair,
    parse_price,
    resolve_shard,
    time_span,
)
from .entities import (
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
from .keywords import KeywordExtractor
from .lexicon import CATEGORIES, DERIVED_FLAGS, LEXICON, KeywordRule, rules_by_category
from .normalize import (
    clean_listing_text,
    fold_diacritics,
    fold_with_alignment,
    join_text_columns,
    make_searchable,
)
from .pipeline import FEATURE_GROUPS, TextFeatureConfig, TextFeaturePipeline
from .representation import (
    TFIDF_MODES,
    PhoBERTEmbedder,
    TfidfTextEncoder,
    ensure_dense,
    hstack_blocks,
)

__all__ = [
    # data
    "DATASET_ID",
    "SHARD_URL_TEMPLATE",
    "TEXT_COLUMNS",
    "NUMERIC_COLUMNS",
    "CATEGORICAL_COLUMNS",
    "TARGET_COLUMN",
    "TIME_COLUMN",
    "resolve_shard",
    "parse_price",
    "normalize_unicode",
    "load_shard_sample",
    "clean_frame",
    "out_of_time_pair",
    "time_span",
    # normalize
    "fold_diacritics",
    "fold_with_alignment",
    "clean_listing_text",
    "make_searchable",
    "join_text_columns",
    # lexicon + keywords
    "KeywordRule",
    "LEXICON",
    "DERIVED_FLAGS",
    "CATEGORIES",
    "rules_by_category",
    "KeywordExtractor",
    # entities
    "ENTITY_COLUMNS",
    "TEXT_IMPUTABLE_COLUMNS",
    "TARGET_LEAKING_ENTITIES",
    "parse_vn_number",
    "extract_entities",
    "extract_entities_folded",
    "impute_from_text",
    "find_price_spans",
    "redact_spans",
    "prepare_for_tfidf",
    "prepare_for_embedding",
    # representation
    "TfidfTextEncoder",
    "PhoBERTEmbedder",
    "TFIDF_MODES",
    "hstack_blocks",
    "ensure_dense",
    # pipeline
    "TextFeatureConfig",
    "TextFeaturePipeline",
    "FEATURE_GROUPS",
]
