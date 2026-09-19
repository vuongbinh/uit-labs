"""The Gradio layout and its handlers. Takes a classifier, so tests can pass a stub."""

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import gradio as gr
import pandas as pd

from batch import MAX_ROWS, BatchFileError, read_comments, results_table, top_label
from model import LABELS

TITLE = "Vietnamese Hate Speech Detection"
EMPTY_MESSAGE = "Please enter some text."
EXAMPLES = [
    "Hôm nay thời tiết rất đẹp",
    "Cảm ơn đội ngũ hỗ trợ rất nhiệt tình",
    "Bộ phim này dở tệ, đạo diễn làm ăn thật thiếu trách nhiệm",
]
DESCRIPTION = (
    f"# {TITLE}\n"
    "Classifies Vietnamese text as `CLEAN`, `OFFENSIVE` or `HATE` using "
    "[bvuong/nlp-vihate](https://huggingface.co/bvuong/nlp-vihate).\n\n"
    "**Content warning:** the model scores real social-media text, so examples and "
    "predictions can involve slurs. Predictions can be wrong; do not use them as the "
    "only basis for moderation decisions."
)


class Predictor(Protocol):
    """Anything that scores texts; `model.Classifier` in production, a stub in tests."""

    def predict(self, texts: Sequence[str]) -> list[dict[str, float]]:
        """Return one {label: probability} dict per text."""
        ...


def analyze_text(classifier: Predictor, text: str | None) -> tuple[str, dict[str, float]]:
    """Score one comment; returns (predicted label, probabilities)."""
    cleaned = (text or "").strip()
    if not cleaned:
        return EMPTY_MESSAGE, {}
    scores = classifier.predict([cleaned])[0]
    return top_label(scores), scores


def analyze_file(classifier: Predictor, path: str | None) -> tuple[pd.DataFrame, str, str]:
    """Score every comment in an uploaded file; returns (table, csv path, status note)."""
    if path is None:
        message = "Please upload a file first."
        raise gr.Error(message)
    try:
        comments = read_comments(path)
    except BatchFileError as exc:
        raise gr.Error(str(exc)) from exc

    total = len(comments)
    comments = comments[:MAX_ROWS]
    table = results_table(comments, classifier.predict(comments))

    csv_path = Path(tempfile.mkdtemp(prefix="vihate-")) / "predictions.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    note = (
        f"Showing the first {MAX_ROWS} of {total} comments."
        if total > MAX_ROWS
        else f"Analyzed {total} comment{'s' if total != 1 else ''}."
    )
    return table, str(csv_path), note


def build_layout(classifier: Predictor) -> None:
    """Render the two-tab app into the enclosing `gr.Blocks` around a loaded classifier."""
    gr.Markdown(DESCRIPTION)

    with gr.Tab("Single comment"):
        text_input = gr.Textbox(
            label="Vietnamese text", placeholder="Nhập nội dung cần kiểm tra...", lines=5
        )
        analyze_button = gr.Button("Analyze", variant="primary")
        predicted = gr.Textbox(label="Predicted class")
        probabilities = gr.Label(label="Class probabilities", num_top_classes=len(LABELS))
        gr.Examples(examples=EXAMPLES, inputs=text_input)
        analyze_button.click(
            lambda text: analyze_text(classifier, text),
            inputs=text_input,
            outputs=[predicted, probabilities],
        )

    with gr.Tab("Batch file"):
        gr.Markdown(
            "Upload a `.txt` file (one comment per line) or a `.csv`/`.tsv` file with a "
            "`free_text`, `text` or `comment` column. "
            f"At most {MAX_ROWS} comments are analyzed. Files up to 10 MB."
        )
        file_input = gr.File(label="Comments file", file_types=[".txt", ".csv", ".tsv"])
        batch_button = gr.Button("Analyze file", variant="primary")
        status = gr.Markdown()
        table = gr.Dataframe(label="Predictions", wrap=True)
        download = gr.File(label="Download predictions (.csv)")
        batch_button.click(
            lambda path: analyze_file(classifier, path),
            inputs=file_input,
            outputs=[table, download, status],
        )
