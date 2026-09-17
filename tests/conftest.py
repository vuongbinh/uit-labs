"""Shared fixtures for the demo-bundle tests: a tiny random-weight ViHSD-shaped bundle.

No downloads, no GPU, no ViHSD data and no hate-speech text. The fixture model is a
2-layer XLM-RoBERTa with a 3-class head and a hand-built word-level tokenizer, which
is enough to exercise every path the real ViSoBERT bundle would: artifact layout,
config validation, threshold logic, batch parsing, Gradio wiring and HTTP serving.

Predictions from it are meaningless -- the point is plumbing, not accuracy.

The heavy imports (torch, transformers, gradio) stay inside the fixtures and helper
bodies, matching the package convention, so `tests/test_labels.py` and friends still
collect without the `demo` extra installed.
"""

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from vihate.demo_bundle import DemoBundleSpec, write_demo_bundle

if TYPE_CHECKING:
    from vihate.serve_demo import HateSpeechPredictor

LABELS = ("CLEAN", "OFFENSIVE", "HATE")
T_STAR = 0.19
MAX_LENGTH = 128
BACKBONE = "uitnlp/visobert"
REPO_ID = "fixture/vihsd-visobert"

# Benign Vietnamese, and every word is in the fixture vocab so nothing degenerates to UNK.
BENIGN_TEXTS = (
    "hôm nay thời tiết đẹp",
    "xin chào bạn",
    "cảm ơn đội ngũ hỗ trợ",
)

_VOCAB = (
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "hôm", "nay", "thời", "tiết", "đẹp", "xin", "chào", "bạn",
    "tôi", "rất", "cảm", "ơn", "đội", "ngũ", "hỗ", "trợ",
)

SPEC = DemoBundleSpec(
    labels=LABELS,
    backbone=BACKBONE,
    max_length=MAX_LENGTH,
    t_star=T_STAR,
    hate_recall_target=0.85,
    calibrated_on="r2_visobert",
    training={"learning_rate": 2e-5, "epochs_budget": 6.0, "seed": 42},
    metrics={"test_macro_f1": 0.6834, "test_hate_recall": 0.8612},
)


def variant(**overrides: object) -> DemoBundleSpec:
    """Return SPEC with individual fields replaced."""
    return replace(SPEC, **overrides)  # type: ignore[arg-type]


def make_tiny_model(model_dir: Path) -> Path:
    """Save a tiny 3-class model + tokenizer into model_dir, laid out like save_pretrained()."""
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
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={name: index for index, name in enumerate(LABELS)},
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
    return make_tiny_model(tmp_path_factory.mktemp("src-model") / "model")


@pytest.fixture(scope="session")
def bundle_dir(tiny_model_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A complete bundle written by the canonical writer, sample CSV included."""
    root = tmp_path_factory.mktemp("bundle-root")
    sample = root / "sample_source.csv"
    sample.write_text(
        "text,true_label\n" + "".join(f"{line},CLEAN\n" for line in BENIGN_TEXTS),
        encoding="utf-8-sig",
    )
    return write_demo_bundle(
        root / "demo_bundle", tiny_model_dir, SPEC, sample_csv=sample, repo_id=REPO_ID
    )


@pytest.fixture(scope="session")
def predictor(
    bundle_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> "HateSpeechPredictor":
    """A loaded predictor, session-scoped because model loading dominates the runtime."""
    from vihate.serve_demo import HateSpeechPredictor

    return HateSpeechPredictor(bundle_dir, device="cpu",
                              output_dir=tmp_path_factory.mktemp("out"))
