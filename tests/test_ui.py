from collections.abc import Sequence
from pathlib import Path

import gradio as gr
import pandas as pd
import pytest

from batch import MAX_ROWS
from ui import analyze_file, analyze_text, build_demo


class StubClassifier:
    """HATE for texts containing 'xấu', otherwise CLEAN. Records what it was asked."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def predict(self, texts: Sequence[str]) -> list[dict[str, float]]:
        self.calls.append(list(texts))
        return [
            {"CLEAN": 0.1, "OFFENSIVE": 0.1, "HATE": 0.8}
            if "xấu" in text
            else {"CLEAN": 0.8, "OFFENSIVE": 0.1, "HATE": 0.1}
            for text in texts
        ]


@pytest.fixture
def stub() -> StubClassifier:
    return StubClassifier()


def test_analyze_text_returns_label_and_probabilities(stub: StubClassifier) -> None:
    label, scores = analyze_text(stub, "  xin chào  ")
    assert label == "CLEAN"
    assert scores["CLEAN"] == 0.8
    assert stub.calls == [["xin chào"]]


@pytest.mark.parametrize("text", ["", "   ", None])
def test_analyze_text_empty_input_does_not_call_model(
    stub: StubClassifier, text: str | None
) -> None:
    assert analyze_text(stub, text) == ("Please enter some text.", {})
    assert stub.calls == []


def test_analyze_file_returns_table_csv_and_note(stub: StubClassifier, tmp_path: Path) -> None:
    upload = tmp_path / "in.txt"
    upload.write_text("xin chào\nbạn xấu\n", encoding="utf-8")
    table, csv_path, note = analyze_file(stub, str(upload))
    assert table["predicted_class"].tolist() == ["CLEAN", "HATE"]
    assert "2" in note
    exported = pd.read_csv(csv_path, encoding="utf-8-sig")
    assert exported["text"].tolist() == ["xin chào", "bạn xấu"]


def test_analyze_file_caps_rows(stub: StubClassifier, tmp_path: Path) -> None:
    upload = tmp_path / "big.txt"
    upload.write_text("\n".join(f"dòng {i}" for i in range(MAX_ROWS + 1)), encoding="utf-8")
    table, _, note = analyze_file(stub, str(upload))
    assert len(table) == MAX_ROWS
    assert str(MAX_ROWS) in note
    assert len(stub.calls[0]) == MAX_ROWS


def test_analyze_file_without_upload_raises_gradio_error(stub: StubClassifier) -> None:
    with pytest.raises(gr.Error):
        analyze_file(stub, None)


def test_analyze_file_bad_file_raises_gradio_error(stub: StubClassifier, tmp_path: Path) -> None:
    upload = tmp_path / "in.pdf"
    upload.write_text("x", encoding="utf-8")
    with pytest.raises(gr.Error, match=r"\.pdf"):
        analyze_file(stub, str(upload))
    assert stub.calls == []


def test_build_demo_returns_blocks(stub: StubClassifier) -> None:
    assert isinstance(build_demo(stub), gr.Blocks)
