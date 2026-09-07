import json
from pathlib import Path

import pytest

from vihate.metrics import FoldResult
from vihate.reporting import persist_fold_results, summarize_folds


def test_summarize_folds_when_two_folds() -> None:
    # Given
    folds = [{"macro_f1": 0.4, "weighted_f1": 0.6}, {"macro_f1": 0.8, "weighted_f1": 0.8}]

    # When
    summary = summarize_folds(folds)

    # Then
    assert summary["macro_f1"]["mean"] == pytest.approx(0.6)
    assert summary["macro_f1"]["std"] == pytest.approx(0.2)


def _fold(fold: int, macro_f1: float) -> FoldResult:
    return FoldResult(
        fold=fold,
        scalars={"macro_f1": macro_f1},
        confusion_matrix=[[fold, 0, 0], [0, fold, 0], [0, 0, fold]],
    )


def test_persist_fold_results_writes_the_three_run_artifacts(tmp_path: Path) -> None:
    # Given
    fold_results = [_fold(1, 0.4), _fold(2, 0.8)]

    # When
    summary = persist_fold_results(tmp_path, fold_results)

    # Then
    assert summary["macro_f1"]["mean"] == pytest.approx(0.6)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "fold_metrics.json",
        "summary.json",
        "summary.md",
    ]


def test_persist_fold_results_when_json_round_tripped(tmp_path: Path) -> None:
    # Given
    fold_results = [_fold(1, 0.4), _fold(2, 0.8)]

    # When
    persist_fold_results(tmp_path, fold_results)

    # Then
    fold_metrics = json.loads((tmp_path / "fold_metrics.json").read_text(encoding="utf-8"))
    assert [entry["fold"] for entry in fold_metrics] == [1, 2]
    assert fold_metrics[0]["metrics"] == {"macro_f1": 0.4}
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == {
        "macro_f1": {"mean": pytest.approx(0.6), "std": pytest.approx(0.2)},
    }
    assert (tmp_path / "summary.md").read_text(encoding="utf-8").startswith("# Evaluation Summary")
