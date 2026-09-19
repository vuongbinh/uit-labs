"""Shared fixtures: a tiny random-weight 3-class model, so no test downloads anything.

Predictions from it are meaningless; the point is exercising the real load/predict path.
"""

from pathlib import Path

import pytest

from model import LABELS, MAX_LENGTH

# Benign Vietnamese; every word is in the fixture vocab so nothing degenerates to UNK.
BENIGN_TEXTS = (
    "hôm nay thời tiết đẹp",
    "xin chào bạn",
    "cảm ơn đội ngũ hỗ trợ",
)

_VOCAB = (
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "hôm",
    "nay",
    "thời",
    "tiết",
    "đẹp",
    "xin",
    "chào",
    "bạn",
    "tôi",
    "rất",
    "cảm",
    "ơn",
    "đội",
    "ngũ",
    "hỗ",
    "trợ",
)


def make_tiny_model(model_dir: Path, num_labels: int = len(LABELS)) -> Path:
    """Save a tiny classifier + word-level tokenizer into model_dir (save_pretrained layout)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        XLMRobertaConfig,
        XLMRobertaForSequenceClassification,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    config = XLMRobertaConfig(
        vocab_size=len(_VOCAB),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=MAX_LENGTH + 8,
        pad_token_id=0,
        num_labels=num_labels,
    )
    XLMRobertaForSequenceClassification(config).save_pretrained(model_dir)

    tokenizer = Tokenizer(
        models.WordLevel({word: i for i, word in enumerate(_VOCAB)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=MAX_LENGTH,
    ).save_pretrained(model_dir)
    return model_dir


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A saved tiny model, session-scoped because saving is the slow part."""
    return make_tiny_model(tmp_path_factory.mktemp("tiny") / "model")
