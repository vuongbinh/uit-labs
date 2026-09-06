"""Transformer cross-validation baselines."""

from dataclasses import dataclass
from pathlib import Path

from vihate.data import TextExample
from vihate.metrics import FoldResult, evaluate_fold


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


def run_transformer_cv(examples: list[TextExample], config: TransformerConfig, out_dir: Path) -> list[FoldResult]:
    """Run stratified CV for a sequence-classification transformer."""
    import numpy as np
    import torch
    from datasets import Dataset
    from sklearn.model_selection import StratifiedKFold
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
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
        train_dataset = _tokenized_dataset(Dataset.from_dict({"text": train_texts, "label": train_labels}), tokenizer, config)
        test_dataset = _tokenized_dataset(Dataset.from_dict({"text": test_texts, "label": test_labels}), tokenizer, config)
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
        trainer = WeightedLossTrainer(class_weights=class_weights, model=model, args=args, train_dataset=train_dataset)
        trainer.train()
        output = trainer.predict(test_dataset)
        probabilities = torch.softmax(torch.tensor(output.predictions), dim=1).numpy()
        predictions = np.argmax(probabilities, axis=1)
        fold_results.append(evaluate_fold(fold, test_labels, predictions, probabilities, out_dir))

    return fold_results


def _tokenized_dataset(dataset, tokenizer, config: TransformerConfig):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=config.max_length)

    return dataset.map(tokenize, batched=True).with_format("torch")


def _class_weights(labels: list[int]):
    import torch

    counts = torch.bincount(torch.tensor(labels), minlength=3).float()
    weights = counts.sum() / (counts.clamp_min(1.0) * 3.0)
    return weights


class WeightedLossTrainer:
    """Factory wrapper that returns a Trainer subclass with weighted loss."""

    def __new__(cls, class_weights, *args, **kwargs):
        import torch
        from transformers import Trainer

        class _Trainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
                loss = loss_fn(logits.view(-1, 3), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        return _Trainer(*args, **kwargs)
