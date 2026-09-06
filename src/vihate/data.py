"""Dataset loading utilities."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from vihate.labels import normalize_label

RawValue: TypeAlias = int | str
RawRow: TypeAlias = Mapping[str, RawValue]


@dataclass(frozen=True, slots=True)
class TextExample:
    """A single labeled Vietnamese social-media text."""

    text: str
    label: int


class DatasetColumnError(Exception):
    """Raised when a dataset split lacks usable text or label columns."""

    def __init__(self, columns: Sequence[str]) -> None:
        joined = ", ".join(columns)
        super().__init__(f"could not infer text/label columns from: {joined}")
        self.columns = tuple(columns)


def load_vihsd_examples(split: str = "train", sample_size: int | None = None) -> list[TextExample]:
    """Load ViHSD examples from Hugging Face datasets."""
    from datasets import load_dataset

    dataset = load_dataset("uitnlp/vihsd", split=split)
    text_column = _first_present(dataset.column_names, ("text", "comment", "content", "sentence", "free_text"))
    label_column = _first_present(dataset.column_names, ("label", "labels", "class", "category"))

    if text_column is None or label_column is None:
        raise DatasetColumnError(tuple(dataset.column_names))

    rows = dataset if sample_size is None else dataset.select(range(min(sample_size, len(dataset))))
    return [
        TextExample(text=str(row[text_column]), label=normalize_label(_raw_value(row[label_column])))
        for row in rows
    ]


def _first_present(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _raw_value(value: RawValue) -> RawValue:
    return value
