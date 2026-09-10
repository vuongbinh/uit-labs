"""Dense text representations of listing descriptions.

Two extractors are provided:

* :class:`TfidfTextEncoder` — TF-IDF over word or character n-grams, reduced with
  Truncated SVD.  Cheap, deterministic, no GPU, and the default for the pipeline.
* :class:`PhoBERTEmbedder` — mean-pooled ``vinai/phobert-base`` embeddings.  Heavier
  and optional: ``torch`` / ``transformers`` are imported lazily inside
  :meth:`PhoBERTEmbedder.encode` so importing this module never requires them.

Both expose the same ``fit`` / ``transform`` / ``fit_transform`` contract and
return dense ``numpy`` arrays with stable column names, so either can be
concatenated with the tabular block.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalize import clean_listing_text

__all__ = [
    "TfidfTextEncoder",
    "PhoBERTEmbedder",
    "TFIDF_MODES",
    "hstack_blocks",
    "ensure_dense",
]

TFIDF_MODES: tuple[str, ...] = ("word", "char", "both")


def _as_text_list(texts: Iterable[str | None]) -> list[str]:
    return [clean_listing_text(t) for t in texts]


class TfidfTextEncoder:
    """TF-IDF + Truncated SVD encoder for Vietnamese listing text.

    Vietnamese is syllable-delimited, so word n-grams are needed to capture
    multi-syllable domain phrases (``sổ đỏ``, ``mặt tiền``); character n-grams add
    robustness to abbreviation and teencode variants (``ô tô`` / ``ôtô`` / ``oto``).

    Parameters
    ----------
    mode:
        ``"word"``, ``"char"`` or ``"both"`` (independent SVD per view, concatenated).
    n_components:
        SVD dimensions kept per view.
    max_features, ngram_range, min_df, sublinear_tf:
        Passed through to :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
        ``ngram_range`` applies to the active analyzer's own units.
    """

    def __init__(
        self,
        mode: str = "word",
        n_components: int = 96,
        max_features: int = 60_000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 5,
        sublinear_tf: bool = True,
        seed: int = 0,
    ) -> None:
        if mode not in TFIDF_MODES:
            raise ValueError(f"mode must be one of {TFIDF_MODES}, got {mode!r}")
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.mode = mode
        self.n_components = int(n_components)
        self.max_features = int(max_features)
        self.ngram_range = tuple(ngram_range)
        self.min_df = int(min_df)
        self.sublinear_tf = bool(sublinear_tf)
        self.seed = int(seed)

        self._views: dict[str, TfidfVectorizer] = {}
        self._svd: dict[str, TruncatedSVD] = {}
        self._fitted = False

    # ------------------------------------------------------------------ setup
    def _view_specs(self) -> dict[str, dict[str, object]]:
        specs: dict[str, dict[str, object]] = {}
        if self.mode in ("word", "both"):
            specs["w"] = {
                "analyzer": "word",
                "token_pattern": r"(?u)\b\w+\b",
                "ngram_range": self.ngram_range,
            }
        if self.mode in ("char", "both"):
            specs["c"] = {"analyzer": "char_wb", "ngram_range": (3, 5)}
        return specs

    @property
    def feature_prefixes(self) -> list[str]:
        """Column-name prefixes produced by this encoder, in output order."""
        if not self._fitted:
            return []
        return [f"tfidf_{key}" for key in self._views]

    @property
    def vocabulary_sizes(self) -> dict[str, int]:
        """Fitted vocabulary size per view (empty before :meth:`fit`)."""
        return {key: len(vec.vocabulary_) for key, vec in self._views.items()}

    @property
    def explained_variance(self) -> dict[str, float]:
        """Total SVD explained-variance ratio per view."""
        return {
            key: float(np.sum(svd.explained_variance_ratio_))
            for key, svd in self._svd.items()
        }

    # ---------------------------------------------------------------- fitting
    def fit(self, texts: Iterable[str | None]) -> TfidfTextEncoder:
        """Fit every TF-IDF view and its SVD on ``texts``."""
        rows = _as_text_list(texts)
        if not rows:
            raise ValueError("cannot fit TfidfTextEncoder on an empty corpus")
        self._views, self._svd = {}, {}
        for key, extra in self._view_specs().items():
            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                sublinear_tf=self.sublinear_tf,
                lowercase=True,
                strip_accents="unicode",
                **extra,  # type: ignore[arg-type]
            )
            matrix = vectorizer.fit_transform(rows)
            n_components = min(self.n_components, matrix.shape[1] - 1, matrix.shape[0] - 1)
            if n_components < 1:
                raise ValueError(
                    f"corpus too small for SVD on view {key!r}: "
                    f"{matrix.shape[0]} documents x {matrix.shape[1]} terms"
                )
            svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
            svd.fit(matrix)
            self._views[key] = vectorizer
            self._svd[key] = svd
        self._fitted = True
        return self

    def transform(self, texts: Iterable[str | None], frame: bool = True) -> pd.DataFrame | np.ndarray:
        """Project ``texts`` into the fitted dense representation."""
        if not self._fitted:
            raise RuntimeError("TfidfTextEncoder.transform called before fit")
        rows = _as_text_list(texts)
        blocks = []
        for key, vectorizer in self._views.items():
            matrix = vectorizer.transform(rows)
            blocks.append(self._svd[key].transform(matrix))
        if not blocks:  # pragma: no cover - guarded by fit()
            raise RuntimeError("no TF-IDF views were fitted")
        dense = np.hstack(blocks).astype(np.float32)
        if not frame:
            return dense
        columns = [
            f"tfidf_{key}_{i}"
            for key in self._views
            for i in range(self._svd[key].n_components)
        ]
        return pd.DataFrame(dense, columns=columns)

    def fit_transform(self, texts: Iterable[str | None], frame: bool = True) -> pd.DataFrame | np.ndarray:
        """Fit on ``texts`` and return their representation."""
        rows = _as_text_list(texts)
        self.fit(rows)
        return self.transform(rows, frame=frame)

    # ------------------------------------------------------------------ utils
    def top_terms(self, view: str = "w", component: int = 0, k: int = 12) -> list[tuple[str, float]]:
        """Highest-loading terms of an SVD component — for qualitative inspection."""
        if view not in self._views:
            raise KeyError(f"unknown TF-IDF view {view!r}; fitted views: {list(self._views)}")
        vectorizer, svd = self._views[view], self._svd[view]
        if not (0 <= component < svd.n_components):
            raise IndexError(f"component {component} out of range for {svd.n_components} components")
        names = np.array(vectorizer.get_feature_names_out())
        weights = svd.components_[component]
        order = np.argsort(-np.abs(weights))[:k]
        return [(str(names[i]), float(weights[i])) for i in order]


class PhoBERTEmbedder:
    """Mean-pooled PhoBERT embeddings for listing text.

    ``torch`` and ``transformers`` are imported on first use, so the rest of the
    package works without them installed.

    Parameters
    ----------
    model_name:
        Hugging Face id of a PhoBERT-style encoder.
    max_length, batch_size:
        Token budget and inference batch size (CPU-bound: keep ``batch_size`` modest).
    pooling:
        ``"mean"`` over unmasked tokens, or ``"cls"`` for the ``<s>`` vector.
    n_components:
        When set, embeddings are reduced with Truncated SVD fitted on the first
        :meth:`fit` corpus, which keeps the downstream tabular model tractable.
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        max_length: int = 128,
        batch_size: int = 16,
        pooling: str = "mean",
        n_components: int | None = None,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        if pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got {pooling!r}")
        self.model_name = model_name
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.pooling = pooling
        self.n_components = None if n_components is None else int(n_components)
        self.seed = int(seed)
        self.device = device

        self._tokenizer = None
        self._model = None
        self._svd: TruncatedSVD | None = None
        self.hidden_size_: int | None = None

    # ------------------------------------------------------------------ setup
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401  (imported for the availability check)
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "PhoBERTEmbedder requires 'torch' and 'transformers'. Install them with "
                "`uv pip install torch transformers` (CPU wheel: "
                "`--index-url https://download.pytorch.org/whl/cpu`), or use "
                "TfidfTextEncoder instead."
            ) from exc
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)
        self.hidden_size_ = int(self._model.config.hidden_size)

    # ---------------------------------------------------------------- encode
    def encode(self, texts: Iterable[str | None]) -> np.ndarray:
        """Return raw pooled embeddings, shape ``(len(texts), hidden_size)``."""
        self._ensure_model()
        import torch

        rows = _as_text_list(texts)
        out = np.zeros((len(rows), self.hidden_size_), dtype=np.float32)
        for start in range(0, len(rows), self.batch_size):
            chunk = rows[start : start + self.batch_size]
            encoded = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden = self._model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            if self.pooling == "mean":
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden[:, 0]
            out[start : start + len(chunk)] = pooled.cpu().numpy()
        return out

    # ---------------------------------------------------------------- fitting
    def fit(self, texts: Iterable[str | None]) -> PhoBERTEmbedder:
        """Fit the optional SVD reduction on ``texts``."""
        embeddings = self.encode(texts)
        if self.n_components is not None:
            n_components = min(self.n_components, embeddings.shape[1], embeddings.shape[0] - 1)
            if n_components < 1:
                raise ValueError("corpus too small for PhoBERT SVD reduction")
            self._svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
            self._svd.fit(embeddings)
        return self

    def transform(self, texts: Iterable[str | None], frame: bool = True) -> pd.DataFrame | np.ndarray:
        """Embed ``texts`` (and reduce them when ``n_components`` is set)."""
        embeddings = self.encode(texts)
        if self._svd is not None:
            embeddings = self._svd.transform(embeddings).astype(np.float32)
        if not frame:
            return embeddings
        return pd.DataFrame(embeddings, columns=[f"phobert_{i}" for i in range(embeddings.shape[1])])

    def fit_transform(self, texts: Iterable[str | None], frame: bool = True) -> pd.DataFrame | np.ndarray:
        """Fit on ``texts`` and return their embedding representation."""
        rows = _as_text_list(texts)
        embeddings = self.encode(rows)
        if self.n_components is not None:
            n_components = min(self.n_components, embeddings.shape[1], len(rows) - 1)
            if n_components < 1:
                raise ValueError("corpus too small for PhoBERT SVD reduction")
            self._svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
            embeddings = self._svd.fit_transform(embeddings).astype(np.float32)
        if not frame:
            return embeddings
        return pd.DataFrame(embeddings, columns=[f"phobert_{i}" for i in range(embeddings.shape[1])])


def hstack_blocks(blocks: Sequence[pd.DataFrame | np.ndarray]) -> pd.DataFrame:
    """Column-bind feature blocks, accepting sparse matrices, arrays or frames."""
    frames: list[pd.DataFrame] = []
    for block in blocks:
        if block is None:
            continue
        if isinstance(block, pd.DataFrame):
            frames.append(block)
        elif issparse(block):
            frames.append(pd.DataFrame(block.toarray()))
        else:
            frames.append(pd.DataFrame(np.asarray(block)))
    if not frames:
        raise ValueError("no feature blocks to concatenate")
    return pd.concat(frames, axis=1)


def ensure_dense(matrix: object) -> np.ndarray:
    """Return ``matrix`` as a dense float array (accepts sparse input)."""
    if issparse(matrix):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)
