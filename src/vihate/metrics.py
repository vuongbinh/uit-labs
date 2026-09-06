"""Evaluation metrics for 3-class hate-speech classification."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from vihate.labels import LABEL_NAMES
from vihate.reporting import write_json


@dataclass(frozen=True, slots=True)
class FoldResult:
    """Metrics and confusion matrix for one CV fold."""

    fold: int
    scalars: dict[str, float]
    confusion_matrix: list[list[int]]


def evaluate_fold(
    fold: int,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[Sequence[float]] | None,
    out_dir: Path,
) -> FoldResult:
    """Compute required scalar metrics and persist the confusion matrix."""
    import numpy as np
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize

    labels = list(range(len(LABEL_NAMES)))
    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    scalars = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    for idx, label_name in enumerate(LABEL_NAMES):
        scalars[f"{label_name.lower()}_precision"] = float(precision[idx])
        scalars[f"{label_name.lower()}_recall"] = float(recall[idx])
        scalars[f"{label_name.lower()}_f1"] = float(f1[idx])

    if y_prob is not None:
        truth = label_binarize(y_true, classes=labels)
        probabilities = np.array(y_prob, dtype=float)
        scalars["roc_auc_ovr_weighted"] = float(
            roc_auc_score(truth, probabilities, average="weighted", multi_class="ovr"),
        )
        scalars["pr_auc_macro"] = float(average_precision_score(truth, probabilities, average="macro"))

    matrix = confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist()
    write_json(out_dir / f"confusion_matrix_fold_{fold}.json", {"matrix": matrix})
    return FoldResult(fold=fold, scalars=scalars, confusion_matrix=matrix)
