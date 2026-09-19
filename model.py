"""Load the fine-tuned classifier and score Vietnamese text. No Gradio in here."""

import os
from collections.abc import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_MODEL_ID = "bvuong/nlp-vihate"
# The checkpoint only stores LABEL_0..LABEL_2; this is the ViHSD class order.
LABELS = ("CLEAN", "OFFENSIVE", "HATE")
MAX_LENGTH = 160
BATCH_SIZE = 32


class Classifier:
    """A model loaded once and reused for every request."""

    def __init__(self, model_id: str | None = None, device: str | None = None) -> None:
        """Load from `model_id`, else $MODEL_ID, else the published Hub repo (or a local path)."""
        model_id = model_id or os.environ.get("MODEL_ID") or DEFAULT_MODEL_ID
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = _load_tokenizer(model_id)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device).eval()
        )
        head_size = int(self.model.config.num_labels)
        if head_size != len(LABELS):
            message = f"{model_id} has {head_size} output labels; this demo expects {len(LABELS)}."
            raise ValueError(message)

    def predict(self, texts: Sequence[str], batch_size: int = BATCH_SIZE) -> list[dict[str, float]]:
        """Return one {label: probability} dict per text, in input order."""
        results: list[dict[str, float]] = []
        for start in range(0, len(texts), batch_size):
            inputs = self.tokenizer(
                list(texts[start : start + batch_size]),
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                probs = torch.softmax(self.model(**inputs).logits, dim=-1).cpu()
            results.extend(dict(zip(LABELS, row.tolist(), strict=True)) for row in probs)
        return results


def _load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    """Prefer the slow tokenizer (what training used); fall back to the fast one."""
    try:
        return AutoTokenizer.from_pretrained(model_id, use_fast=False)
    except (ValueError, OSError, ImportError):
        return AutoTokenizer.from_pretrained(model_id, use_fast=True)
