import re
from pathlib import Path

import pytest
from conftest import BENIGN_TEXTS, make_tiny_model

from model import DEFAULT_MODEL_ID, LABELS, Classifier


@pytest.fixture(scope="module")
def classifier(tiny_model_dir: Path) -> Classifier:
    return Classifier(str(tiny_model_dir), device="cpu")


def test_predict_returns_one_distribution_per_text(classifier: Classifier) -> None:
    results = classifier.predict(BENIGN_TEXTS)
    assert len(results) == len(BENIGN_TEXTS)
    for scores in results:
        assert tuple(scores) == LABELS
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_empty_list(classifier: Classifier) -> None:
    assert classifier.predict([]) == []


def test_batch_size_does_not_change_scores(classifier: Classifier) -> None:
    one_by_one = classifier.predict(BENIGN_TEXTS, batch_size=1)
    together = classifier.predict(BENIGN_TEXTS, batch_size=32)
    for single, batched in zip(one_by_one, together, strict=True):
        for label in LABELS:
            assert single[label] == pytest.approx(batched[label], abs=1e-4)


def test_long_text_is_truncated_not_crashing(classifier: Classifier) -> None:
    assert len(classifier.predict(["xin chào " * 500])) == 1


def test_model_id_comes_from_environment(
    tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_ID", str(tiny_model_dir))
    assert len(Classifier(device="cpu").predict(["xin chào"])) == 1


def test_empty_model_id_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    def stop(model_id: str) -> None:
        raise LookupError(model_id)

    monkeypatch.setenv("MODEL_ID", "")
    monkeypatch.setattr("model._load_tokenizer", stop)
    with pytest.raises(LookupError, match=re.escape(DEFAULT_MODEL_ID)):
        Classifier(device="cpu")


def test_default_model_id_is_the_published_repo() -> None:
    assert DEFAULT_MODEL_ID == "bvuong/visoBert-ensemble"


def test_wrong_head_size_is_rejected(tmp_path: Path) -> None:
    two_way = make_tiny_model(tmp_path / "two-way", num_labels=2)
    with pytest.raises(ValueError, match="3"):
        Classifier(str(two_way), device="cpu")
