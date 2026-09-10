"""Tests for TF-IDF / PhoBERT representation encoders."""

import numpy as np
import pytest

from real_estate.text.representation import (
    TFIDF_MODES,
    PhoBERTEmbedder,
    TfidfTextEncoder,
    ensure_dense,
    hstack_blocks,
)

# A small but varied Vietnamese corpus: enough documents for a stable SVD and
# enough distinct domain phrases for the vocabulary to be non-trivial.
CORPUS = [
    "Bán nhà mặt tiền sổ đỏ chính chủ, ô tô đỗ cửa",
    "Nhà trong ngõ ba gác, nội thất cơ bản, gần chợ",
    "Căn hộ chung cư full nội thất, có thang máy, sổ hồng",
    "Nhà cấp 4 cũ, hẻm xe hơi, pháp lý rõ ràng",
    "Biệt thự phân lô, hai mặt thoáng, vỉa hè rộng",
    "Nhà mới xây 5 tầng, đường 8m, tiện kinh doanh",
    "Đất nền lô góc, không quy hoạch, không tranh chấp",
    "Nhà mặt phố 3 tấm, sân thượng, đang chờ sổ",
    "Cho thuê nguyên căn mặt tiền, giá tốt, có thương lượng",
    "Bán gấp nhà trong hẻm, xe hơi vào nhà, 4 phòng ngủ",
    "Nhà phố phân lô, ô tô đỗ cổng, nội thất đầy đủ",
    "Căn hộ 2 phòng ngủ, ban công đông nam, công chứng ngay",
    "Nhà cấp 4 mặt tiền đường lớn, tiện kinh doanh buôn bán",
    "Đất vườn rộng 500m2, sổ đỏ lâu dài, chính chủ bán",
    "Nhà 6 tầng thang máy, ngõ thông, ô tô đỗ cửa",
    "Biệt thự sân vườn, full nội thất cao cấp, hồ bơi",
    "Nhà hẻm ba gác, 1 trệt 2 lầu, gần trường học",
    "Căn hộ cho thuê, nội thất cơ bản, vào ở ngay",
    "Nhà mặt tiền kinh doanh, vỉa hè 5m, sổ hồng riêng",
    "Đất nền dự án, pháp lý đầy đủ, không dính quy hoạch",
]

TEST_CORPUS = [
    "Nhà mặt tiền sổ đỏ, ô tô đỗ cửa, kinh doanh tốt",
    "Căn hộ chưa từng thấy trong dữ liệu huấn luyện xyz",
    "",
]


@pytest.fixture
def encoder() -> TfidfTextEncoder:
    return TfidfTextEncoder(mode="word", n_components=6, min_df=1, max_features=5000).fit(CORPUS)


def test_fit_transform_shape(encoder):
    out = encoder.transform(CORPUS)
    assert out.shape == (len(CORPUS), 6)
    assert np.isfinite(out.to_numpy()).all()


def test_columns_are_named_and_unique(encoder):
    out = encoder.transform(CORPUS)
    assert list(out.columns) == [f"tfidf_w_{i}" for i in range(6)]
    assert len(set(out.columns)) == len(out.columns)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        TfidfTextEncoder().transform(CORPUS)


def test_fit_on_empty_corpus_raises():
    with pytest.raises(ValueError, match="empty corpus"):
        TfidfTextEncoder(min_df=1).fit([])


@pytest.mark.parametrize("mode", ["sparse", "hybrid", "dense"])
def test_invalid_mode_rejected(mode):
    with pytest.raises(ValueError, match="mode must be one of"):
        TfidfTextEncoder(mode=mode)


def test_fit_is_deterministic_for_a_seed():
    a = TfidfTextEncoder(n_components=4, min_df=1, seed=7).fit(CORPUS).transform(CORPUS)
    b = TfidfTextEncoder(n_components=4, min_df=1, seed=7).fit(CORPUS).transform(CORPUS)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


def test_unseen_text_reuses_the_fitted_vocabulary(encoder):
    """No refitting at transform time: the test block must have the same width."""
    out = encoder.transform(TEST_CORPUS)
    assert out.shape == (len(TEST_CORPUS), 6)
    vocab_before = dict(encoder.vocabulary_sizes)
    encoder.transform(CORPUS)
    assert encoder.vocabulary_sizes == vocab_before


def test_blank_text_maps_to_a_finite_vector(encoder):
    out = encoder.transform([""]).to_numpy()
    assert np.isfinite(out).all()


def test_both_mode_produces_two_views():
    enc = TfidfTextEncoder(mode="both", n_components=4, min_df=1).fit(CORPUS)
    assert set(enc.vocabulary_sizes) == {"w", "c"}
    out = enc.transform(CORPUS)
    assert out.shape[1] == 8
    assert list(out.columns)[:4] == [f"tfidf_w_{i}" for i in range(4)]
    assert list(out.columns)[4:] == [f"tfidf_c_{i}" for i in range(4)]


def test_char_mode_brings_spelling_variants_closer_than_word_mode():
    """Char n-grams must treat "ô tô"/"ôtô" as more alike than word n-grams do.

    The two spellings tokenize differently as words (two tokens vs one), so a
    word-level view separates them; a char-level view shares most of their
    trigrams and should land nearer.
    """
    variants = ["ô tô đỗ cửa", "ôtô đỗ cửa"]
    corpus = CORPUS + variants

    def cosine(mode: str) -> float:
        out = TfidfTextEncoder(mode=mode, n_components=6, min_df=1).fit(corpus).transform(
            variants, frame=False
        )
        a, b = out[0], out[1]
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert cosine("char") > cosine("word")


def test_svd_components_are_clamped_to_the_corpus_rank():
    """Requesting more components than the corpus supports degrades gracefully."""
    encoder = TfidfTextEncoder(n_components=50, min_df=1).fit(CORPUS[:4])
    assert encoder.transform(CORPUS[:4]).shape[1] <= 3


def test_single_document_corpus_cannot_be_reduced():
    with pytest.raises(ValueError, match="too small for SVD"):
        TfidfTextEncoder(n_components=4, min_df=1).fit(["một tài liệu duy nhất"])


def test_modes_are_all_supported():
    assert set(TFIDF_MODES) == {"word", "char", "both"}


def test_explained_variance_is_a_fraction(encoder):
    variance = encoder.explained_variance["w"]
    assert 0.0 < variance <= 1.0 + 1e-9


def test_top_terms_returns_weighted_terms(encoder):
    terms = encoder.top_terms("w", component=0, k=5)
    assert len(terms) == 5
    assert all(isinstance(term, str) and isinstance(weight, float) for term, weight in terms)


def test_top_terms_rejects_unknown_view_and_component(encoder):
    with pytest.raises(KeyError, match="unknown TF-IDF view"):
        encoder.top_terms("zzz")
    with pytest.raises(IndexError, match="out of range"):
        encoder.top_terms("w", component=99)


def test_frame_false_returns_a_bare_array(encoder):
    out = encoder.transform(CORPUS, frame=False)
    assert isinstance(out, np.ndarray) and out.shape == (len(CORPUS), 6)


# ------------------------------------------------------------------- PhoBERT
def test_phobert_construction_does_not_import_torch():
    """The embedder must be importable and constructible without torch installed."""
    embedder = PhoBERTEmbedder(n_components=8)
    assert embedder._model is None
    assert embedder.hidden_size_ is None


def test_phobert_rejects_unknown_pooling():
    with pytest.raises(ValueError, match="pooling must be"):
        PhoBERTEmbedder(pooling="max")


# --------------------------------------------------------------------- helpers
def test_hstack_blocks_mixes_frames_arrays_and_sparse():
    import pandas as pd
    from scipy.sparse import csr_matrix

    frame = pd.DataFrame({"a": [1, 2]})
    array = np.zeros((2, 2))
    sparse = csr_matrix(np.eye(2))
    out = hstack_blocks([frame, array, sparse])
    assert out.shape == (2, 5)


def test_hstack_blocks_rejects_empty_input():
    with pytest.raises(ValueError, match="no feature blocks"):
        hstack_blocks([])


def test_ensure_dense_handles_sparse_and_dense():
    from scipy.sparse import csr_matrix

    sparse = csr_matrix(np.eye(2))
    np.testing.assert_allclose(ensure_dense(sparse), np.eye(2))
    np.testing.assert_allclose(ensure_dense(np.eye(2)), np.eye(2))
