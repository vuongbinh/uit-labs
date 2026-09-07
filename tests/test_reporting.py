import pytest

from vihate.reporting import summarize_folds


def test_summarize_folds_when_two_folds() -> None:
    # Given
    folds = [{"macro_f1": 0.4, "weighted_f1": 0.6}, {"macro_f1": 0.8, "weighted_f1": 0.8}]

    # When
    summary = summarize_folds(folds)

    # Then
    assert summary["macro_f1"]["mean"] == pytest.approx(0.6)
    assert summary["macro_f1"]["std"] == pytest.approx(0.2)
