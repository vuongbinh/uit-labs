import pytest

from vihate.labels import UnknownLabelError, label_name, normalize_label


def test_normalize_label_when_string_label() -> None:
    # Given
    raw_label = " offensive "

    # When
    label = normalize_label(raw_label)

    # Then
    assert label == 1


def test_normalize_label_when_unknown_label() -> None:
    # Given
    raw_label = "spam"

    # When / Then
    with pytest.raises(UnknownLabelError) as exc_info:
        normalize_label(raw_label)
    assert exc_info.value.raw_label == raw_label


def test_label_name_when_integer_id() -> None:
    # Given
    label_id = 2

    # When
    name = label_name(label_id)

    # Then
    assert name == "HATE"
