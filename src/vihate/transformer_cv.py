"""Transformer cross-validation baselines."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from vihate.data import TextExample
from vihate.metrics import FoldResult, evaluate_fold

if TYPE_CHECKING:
    from datasets import Dataset
    from torch import Tensor
    from torch.nn import Module
    from torch.utils.data import Dataset as TorchDataset
    from transformers import BatchEncoding, PreTrainedTokenizerBase, Trainer, TrainingArguments


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    """Configuration for Hugging Face transformer CV."""

    model_name: str = "vinai/phobert-base"
    folds: int = 5
    seed: int = 13
    epochs: float = 3.0
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 160


def run_transformer_cv(
    examples: list[TextExample],
    config: TransformerConfig,
    out_dir: Path,
) -> list[FoldResult]:
    """Run stratified CV for a sequence-classification transformer."""
    import numpy as np
    import torch
    from datasets import Dataset
    from sklearn.model_selection import StratifiedKFold
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
    )

    texts = [example.text for example in examples]
    labels = [example.label for example in examples]
    splitter = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False)
    fold_results: list[FoldResult] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(texts, labels), start=1):
        train_texts = [texts[idx] for idx in train_idx]
        train_labels = [labels[idx] for idx in train_idx]
        test_texts = [texts[idx] for idx in test_idx]
        test_labels = [labels[idx] for idx in test_idx]
        train_raw = Dataset.from_dict({"text": train_texts, "label": train_labels})
        test_raw = Dataset.from_dict({"text": test_texts, "label": test_labels})
        train_dataset = _tokenized_dataset(train_raw, tokenizer, config)
        test_dataset = _tokenized_dataset(test_raw, tokenizer, config)
        model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=3)
        class_weights = _class_weights(train_labels)
        args = TrainingArguments(
            output_dir=str(out_dir / f"fold_{fold}"),
            eval_strategy="no",
            save_strategy="no",
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.epochs,
            seed=config.seed,
            report_to=[],
        )
        trainer = _build_trainer(class_weights, model, args, train_dataset)
        trainer.train()
        output = trainer.predict(cast("TorchDataset[Any]", test_dataset))
        probabilities = torch.softmax(torch.tensor(output.predictions), dim=1).numpy()
        predictions = np.argmax(probabilities, axis=1)
        fold_results.append(
            evaluate_fold(fold, test_labels, predictions.tolist(), probabilities.tolist(), out_dir),
        )

    return fold_results


def _tokenized_dataset(
    dataset: "Dataset",
    tokenizer: "PreTrainedTokenizerBase",
    config: TransformerConfig,
) -> "Dataset":
    def tokenize(batch: dict[str, list[str]]) -> "BatchEncoding":
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
        )

    return dataset.map(tokenize, batched=True).with_format("torch")


def _class_weights(labels: list[int]) -> "Tensor":
    import torch

    counts = torch.bincount(torch.tensor(labels), minlength=3).float()
    return counts.sum() / (counts.clamp_min(1.0) * 3.0)


def _build_trainer(
    class_weights: "Tensor",
    model: "Module",
    args: "TrainingArguments",
    train_dataset: "Dataset",
) -> "Trainer":
    """Build a `Trainer` whose loss is weighted by training-fold class frequencies."""
    import torch
    from transformers import Trainer

    class _WeightedLossTrainer(Trainer):
        def compute_loss(
            self,
            model: "Module",
            inputs: dict[str, "Tensor"],
            return_outputs: bool = False,
            num_items_in_batch: "Tensor | int | None" = None,
        ) -> "Tensor | tuple[Tensor, object]":
            """Compute weighted cross-entropy loss for the current batch."""
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fn(logits.view(-1, 3), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    return _WeightedLossTrainer(model=model, args=args, train_dataset=train_dataset)
