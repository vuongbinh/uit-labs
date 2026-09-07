"""Transformer cross-validation baselines."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from vihate.data import TextExample
from vihate.metrics import FoldResult, evaluate_fold
from vihate.reporting import write_json

if TYPE_CHECKING:
    from datasets import Dataset
    from torch import Tensor
    from torch.nn import Module
    from torch.utils.data import Dataset as TorchDataset
    from transformers import BatchEncoding, PreTrainedTokenizerBase, Trainer


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    """Configuration for Hugging Face transformer CV."""

    model_name: str = "vinai/phobert-base"
    folds: int = 5
    seed: int = 13
    epochs: float = 3.0
    batch_size: int = 16
    grad_accum_steps: int = 1
    learning_rate: float = 2e-5
    max_length: int = 160
    warmup_ratio: float = 0.1
    # Any `transformers` optimizer name. `adamw_torch` is full-precision AdamW;
    # `adamw_8bit` (needs `bitsandbytes`) keeps AdamW math but quantizes the
    # optimizer state, which is what lets XLM-R-base fine-tune inside a 4 GB GPU.
    optim: str = "adamw_torch"
    # Recompute activations in the backward pass instead of storing them: ~25%
    # slower per step but roughly halves activation memory, which lets larger
    # backbones use a usable batch size on a small GPU.
    gradient_checkpointing: bool = False
    # Freeze the input word-embedding table. For a big multilingual vocab
    # (xlm-roberta-base: ~192M of 278M params sit in embeddings) this drops
    # trainable params and optimizer memory to encoder-only scale, trading a
    # small amount of accuracy for a several-fold speed-up on a small GPU.
    freeze_embeddings: bool = False


def run_transformer_cv(
    examples: list[TextExample],
    config: TransformerConfig,
    out_dir: Path,
) -> list[FoldResult]:
    """Run stratified CV for a sequence-classification transformer.

    Each fold fine-tunes with AdamW and a linear warmup schedule, a
    cross-entropy loss weighted by the training-fold class frequencies, and
    records validation Macro F1 after every epoch to
    ``training_history_fold_<n>.json``.
    """
    import numpy as np
    import torch
    from datasets import Dataset
    from sklearn.model_selection import StratifiedKFold
    from transformers import (
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        TrainingArguments,
    )

    texts = [example.text for example in examples]
    labels = [example.label for example in examples]
    splitter = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
    tokenizer = _load_tokenizer(config.model_name)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    use_fp16 = torch.cuda.is_available()
    fold_results: list[FoldResult] = []
    history: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(texts, labels), start=1):
        train_labels = [labels[idx] for idx in train_idx]
        test_labels = [labels[idx] for idx in test_idx]
        train_raw = Dataset.from_dict(
            {
                "text": [texts[idx] for idx in train_idx],
                "label": train_labels,
            }
        )
        test_raw = Dataset.from_dict(
            {
                "text": [texts[idx] for idx in test_idx],
                "label": test_labels,
            }
        )
        train_dataset = _tokenized_dataset(train_raw, tokenizer, config)
        test_dataset = _tokenized_dataset(test_raw, tokenizer, config)
        model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=3)
        if config.freeze_embeddings:
            for parameter in model.get_input_embeddings().parameters():
                parameter.requires_grad = False
        args = TrainingArguments(
            output_dir=str(out_dir / f"fold_{fold}"),
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="no",
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.grad_accum_steps,
            num_train_epochs=config.epochs,
            lr_scheduler_type="linear",
            warmup_ratio=config.warmup_ratio,
            optim=config.optim,
            gradient_checkpointing=config.gradient_checkpointing,
            gradient_checkpointing_kwargs=(
                {"use_reentrant": False} if config.gradient_checkpointing else None
            ),
            fp16=use_fp16,
            seed=config.seed,
            report_to=[],
        )
        trainer = _weighted_loss_trainer_cls(_class_weights(train_labels))(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=collator,
            compute_metrics=_macro_f1_metric,
        )
        trainer.train()
        fold_history = _epoch_history(trainer)
        write_json(
            out_dir / f"training_history_fold_{fold}.json",
            {"fold": fold, "epochs": fold_history},
        )
        history.append({"fold": fold, "epochs": fold_history})
        output = trainer.predict(cast("TorchDataset[Any]", test_dataset))
        probabilities = torch.softmax(torch.tensor(output.predictions), dim=1).numpy()
        predictions = np.argmax(probabilities, axis=1)
        fold_results.append(
            evaluate_fold(fold, test_labels, predictions.tolist(), probabilities.tolist(), out_dir),
        )
        del model, trainer
        if use_fp16:
            torch.cuda.empty_cache()

    write_json(out_dir / "training_history.json", history)
    return fold_results


def _load_tokenizer(model_name: str) -> "PreTrainedTokenizerBase":
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except (ValueError, OSError, ImportError):
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def _tokenized_dataset(
    dataset: "Dataset",
    tokenizer: "PreTrainedTokenizerBase",
    config: TransformerConfig,
) -> "Dataset":
    def tokenize(batch: dict[str, list[str]]) -> "BatchEncoding":
        return tokenizer(batch["text"], truncation=True, max_length=config.max_length)

    # No fixed padding here: `DataCollatorWithPadding` pads each batch to its
    # own longest sequence (p50 length is ~12 subwords vs a 160 cap), which
    # cuts wall time several-fold at zero accuracy cost.
    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def _class_weights(labels: list[int]) -> "Tensor":
    import torch

    counts = torch.bincount(torch.tensor(labels), minlength=3).float()
    return counts.sum() / (counts.clamp_min(1.0) * 3.0)


def _macro_f1_metric(eval_pred: Any) -> dict[str, float]:  # noqa: ANN401
    """Compute validation Macro F1 for the `Trainer`'s per-epoch evaluation."""
    import numpy as np
    from sklearn.metrics import f1_score

    preds = np.argmax(eval_pred.predictions, axis=-1)
    macro_f1 = f1_score(
        eval_pred.label_ids,
        preds,
        average="macro",
        zero_division=0,  # pyright: ignore[reportArgumentType]
    )
    return {"macro_f1": float(macro_f1)}


def _epoch_history(trainer: "Trainer") -> list[dict[str, float]]:
    """Collapse the `Trainer` log history into one row per epoch."""
    by_epoch: dict[float, dict[str, float]] = {}
    for record in trainer.state.log_history:
        epoch = record.get("epoch")
        if epoch is None:
            continue
        row = by_epoch.setdefault(round(float(epoch), 4), {"epoch": round(float(epoch), 4)})
        for key in ("loss", "eval_loss", "eval_macro_f1"):
            if key in record:
                row[key] = float(record[key])
    return [by_epoch[key] for key in sorted(by_epoch)]


def _weighted_loss_trainer_cls(class_weights: "Tensor") -> type["Trainer"]:
    """Return a `Trainer` subclass whose loss is weighted by ``class_weights``."""
    import torch
    from transformers import Trainer

    class _WeightedLossTrainer(Trainer):
        def compute_loss(
            self,
            model: "Module",
            inputs: dict[str, "Tensor"],
            return_outputs: bool = False,  # noqa: FBT001, FBT002
            num_items_in_batch: "Tensor | int | None" = None,  # noqa: ARG002
        ) -> "Tensor | tuple[Tensor, object]":
            """Compute weighted cross-entropy loss for the current batch."""
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fn(logits.view(-1, 3), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    return _WeightedLossTrainer
