"""Classical TF-IDF baselines."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from vihate.data import TextExample
from vihate.metrics import FoldResult, evaluate_fold

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin

ClassicalModel = Literal["logreg", "svm"]


@dataclass(frozen=True, slots=True)
class ClassicalConfig:
    """Configuration for the classical baseline."""

    model: ClassicalModel = "logreg"
    folds: int = 5
    seed: int = 13


def run_classical_cv(
    examples: list[TextExample],
    config: ClassicalConfig,
    out_dir: Path,
) -> list[FoldResult]:
    """Run stratified cross-validation for TF-IDF word and char n-grams."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import FeatureUnion, Pipeline

    texts = [example.text for example in examples]
    labels = [example.label for example in examples]
    splitter = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
    fold_results: list[FoldResult] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(texts, labels), start=1):
        classifier = _build_classifier(config.model, config.seed)
        pipeline = Pipeline(
            steps=[
                (
                    "features",
                    FeatureUnion(
                        [
                            (
                                "word",
                                TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2),
                            ),
                            (
                                "char",
                                TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2),
                            ),
                        ],
                    ),
                ),
                ("classifier", classifier),
            ],
        )
        train_texts = [texts[idx] for idx in train_idx]
        train_labels = [labels[idx] for idx in train_idx]
        test_texts = [texts[idx] for idx in test_idx]
        test_labels = [labels[idx] for idx in test_idx]

        pipeline.fit(train_texts, train_labels)
        predictions = pipeline.predict(test_texts)
        has_predict_proba = hasattr(pipeline, "predict_proba")
        probabilities = pipeline.predict_proba(test_texts) if has_predict_proba else None
        fold_results.append(evaluate_fold(fold, test_labels, predictions, probabilities, out_dir))

    return fold_results


def _build_classifier(model: ClassicalModel, seed: int) -> "ClassifierMixin":
    match model:
        case "logreg":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=seed)
        case "svm":
            from sklearn.svm import LinearSVC

            return LinearSVC(class_weight="balanced", random_state=seed)
