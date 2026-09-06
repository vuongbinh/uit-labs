"""Label parsing for ViHSD."""

from enum import IntEnum
from typing import Final, assert_never


class HateLabel(IntEnum):
    """Canonical ViHSD labels."""

    CLEAN = 0
    OFFENSIVE = 1
    HATE = 2


LABEL_NAMES: Final[tuple[str, str, str]] = ("CLEAN", "OFFENSIVE", "HATE")
_LABEL_BY_NAME: Final[dict[str, HateLabel]] = {name: HateLabel(idx) for idx, name in enumerate(LABEL_NAMES)}


class UnknownLabelError(Exception):
    """Raised when a dataset label cannot be mapped to the canonical label set."""

    def __init__(self, raw_label: str) -> None:
        super().__init__(f"unknown ViHSD label: {raw_label}")
        self.raw_label = raw_label


def normalize_label(raw_label: int | str) -> int:
    """Return canonical integer id for a ViHSD label."""
    match raw_label:
        case int() as value:
            if value in {label.value for label in HateLabel}:
                return value
            raise UnknownLabelError(str(value))
        case str() as value:
            key = value.strip().upper()
            if key in _LABEL_BY_NAME:
                return int(_LABEL_BY_NAME[key])
            if key.isdigit():
                return normalize_label(int(key))
            raise UnknownLabelError(value)
        case unreachable:
            assert_never(unreachable)


def label_name(label_id: int) -> str:
    """Return display name for a canonical label id."""
    return LABEL_NAMES[normalize_label(label_id)]
