"""Dataset loading utilities."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from vihate.labels import normalize_label

type RawValue = int | str


@dataclass(frozen=True, slots=True)
class TextExample:
    """A single labeled Vietnamese social-media text."""

    text: str
    label: int


class DatasetColumnError(Exception):
    """Raised when a dataset split lacks usable text or label columns."""

    def __init__(self, columns: Sequence[str]) -> None:
        """Record the dataset columns that could not be mapped."""
        joined = ", ".join(columns)
        super().__init__(f"could not infer text/label columns from: {joined}")
        self.columns = tuple(columns)


def load_vihsd_examples(split: str = "train", sample_size: int | None = None) -> list[TextExample]:
    """Load ViHSD examples from Hugging Face datasets."""
    from datasets import Dataset, load_dataset

    # `split` is a plain (non-streaming) split name, so this always returns a
    # single `Dataset`, never a `DatasetDict`/`IterableDataset*`.
    dataset = cast("Dataset", load_dataset("uitnlp/vihsd", split=split))
    text_candidates = ("text", "comment", "content", "sentence", "free_text")
    label_candidates = ("label", "labels", "class", "category", "label_id")
    text_column = _first_present(dataset.column_names, text_candidates)
    label_column = _first_present(dataset.column_names, label_candidates)

    if text_column is None or label_column is None:
        raise DatasetColumnError(tuple(dataset.column_names))

    rows = dataset if sample_size is None else dataset.select(range(min(sample_size, len(dataset))))
    return [
        _row_to_example(cast("Mapping[str, RawValue]", row), text_column, label_column)
        for row in rows
    ]


def _row_to_example(
    row: Mapping[str, RawValue], text_column: str, label_column: str
) -> TextExample:
    return TextExample(
        text=str(row[text_column]),
        label=normalize_label(row[label_column]),
    )


def _first_present(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None
