"""Gradio demo served from a persisted bundle -- no training, no notebook.

Cold start is one model download (cached after the first run) instead of the
~3 hours of cross-validation plus final training that Part 8 of the notebook used
to need before it could launch. Everything the demo depends on beyond the weights
-- label order, `max_length`, the calibrated `t*` -- comes from `demo_config.json`,
so a deployed demo cannot silently drift from the notebook that produced it.

Prediction maths are deliberately identical to the notebook's Part 8 (same
one-at-a-time loop, same threshold rule, same output columns) so a number shown
here matches the number the notebook showed.

Run it with `uv run vihate demo --model <repo-id-or-dir>`; see cli.py.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from vihate.demo_bundle import MODEL_DIRNAME, SAMPLE_NAME, resolve_bundle

if TYPE_CHECKING:
    from gradio import Blocks
    from torch import Tensor
    from transformers import PreTrainedTokenizerBase

FLAG_NOTE: Final = "ưu tiên kiểm duyệt"
TEXT_COLUMNS: Final = ("free_text", "text", "comment")
TABLE_EXTENSIONS: Final = (".csv", ".tsv")
EXPORT_NAME: Final = "demo_batch_predictions.csv"
APP_TITLE: Final = "Vietnamese Hate Speech Detection"
DEFAULT_PORT: Final = 7860


def load_tokenizer(path: Path) -> "PreTrainedTokenizerBase":
    """Load the slow tokenizer when available, falling back to the fast one.

    Same order Part 4 of the notebook uses: the fast PhoBERT/XLM-R tokenizers are
    why this project pins `transformers<5`.
    """
    try:
        return AutoTokenizer.from_pretrained(path, use_fast=False)
    except (ValueError, OSError, ImportError):
        return AutoTokenizer.from_pretrained(path, use_fast=True)


@dataclass(frozen=True, slots=True)
class ServeConfig:
    """How to reach a bundle, and where to serve it from."""

    source: str
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    share: bool = False
    device: str | None = None
    output_dir: Path | None = None


class HateSpeechPredictor:
    """A persisted bundle, loaded once, ready to score Vietnamese text."""

    def __init__(
        self, source: str | Path, *, device: str | None = None, output_dir: Path | None = None
    ) -> None:
        """Resolve the bundle, then load its tokenizer and weights onto `device`."""
        bundle_dir, config = resolve_bundle(source)
        self.bundle_dir = bundle_dir
        self.config = config
        self.labels = config.labels
        self.hate_label = config.hate_label
        self.max_length = config.max_length
        self.t_star = config.t_star
        # Derived from the bundle rather than hard-coded, so a wrong label order in
        # demo_config.json surfaces as a KeyError instead of silent 0.0 columns.
        self.prob_columns = tuple(f"P({name})" for name in self.labels)
        self.batch_columns = ("text", "predicted_class", *self.prob_columns, "flagged_for_review")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model_dir = bundle_dir / MODEL_DIRNAME
        self.tokenizer = load_tokenizer(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

        fallback_dir = Path(tempfile.mkdtemp(prefix="vihsd-demo-"))
        self.output_dir = Path(output_dir) if output_dir else fallback_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_path = bundle_dir / SAMPLE_NAME if config.has_sample else None

    def probabilities(self, text: str) -> dict[str, float]:
        """Return softmax probabilities keyed by label name for one comment."""
        inputs = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            logits: Tensor = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu()
        return {self.labels[index]: float(value) for index, value in enumerate(probs)}

    def predict_detailed(
        self, text: str | None, threshold: float | None = None
    ) -> tuple[str, dict[str, float]]:
        """Classify one comment; `threshold=None` means the bundle's calibrated t*."""
        cutoff = self.t_star if threshold is None else float(threshold)
        cleaned = (text or "").strip()
        if not cleaned:
            return "Please enter some text.", {}
        scores = self.probabilities(cleaned)
        prediction = max(scores, key=lambda label: scores[label])
        if scores[self.hate_label] >= cutoff:
            # Appended, not overwritten: predicted_class stays the argmax label and the
            # flag rides along in the string, matching the notebook's Part 8 exactly.
            note = f"{FLAG_NOTE}: P({self.hate_label}) >= {cutoff:.3f}"
            prediction = f"{prediction} ({note})"
        return prediction, scores

    def _read_texts(self, file_path: Path) -> list[str]:
        """Load comments from a .txt (one per line) or a .csv/.tsv with a text column."""
        if file_path.suffix.lower() in TABLE_EXTENSIONS:
            sep = "\t" if file_path.suffix.lower() == ".tsv" else ","
            table = pd.read_csv(file_path, sep=sep)
            column = next(
                (name for name in TEXT_COLUMNS if name in table.columns), str(table.columns[0])
            )
            return [str(value) for value in table[column].tolist()]
        raw = file_path.read_text(encoding="utf-8")
        return [line.strip() for line in raw.splitlines() if line.strip()]

    def predict_batch(
        self, file: str | Path | None, threshold: float | None = None
    ) -> tuple[pd.DataFrame, str | None]:
        """Score every comment in an uploaded file and export the table as CSV."""
        if file is None:
            return pd.DataFrame(columns=pd.Index(self.batch_columns)), None
        # str/Path only: a Path also has a `.name`, and taking that would silently
        # resolve to a bare filename in the current directory.
        rows = []
        for text in self._read_texts(Path(file)):
            prediction, scores = self.predict_detailed(text, threshold)
            scored = scores or dict.fromkeys(self.labels, 0.0)
            row: dict[str, object] = {
                "text": text,
                "predicted_class": prediction.split(" (", maxsplit=1)[0],
            }
            rounded = {
                column: round(scored[name], 4)
                for name, column in zip(self.labels, self.prob_columns, strict=True)
            }
            row.update(rounded)
            row["flagged_for_review"] = FLAG_NOTE in prediction
            rows.append(row)
        result_table = pd.DataFrame(rows, columns=pd.Index(self.batch_columns))
        out_path = self.output_dir / EXPORT_NAME
        result_table.to_csv(out_path, index=False, encoding="utf-8-sig")
        return result_table, str(out_path)


def build_app(predictor: HateSpeechPredictor) -> "Blocks":
    """Construct the Gradio Blocks app for an already-loaded predictor."""
    import gradio as gr

    t_star = predictor.t_star
    config = predictor.config
    subtitle = f"backbone `{config.backbone}` · calibrated `t*` = {t_star:.3f}"
    macro_f1 = config.metrics.get("test_macro_f1")
    if macro_f1 is not None:
        subtitle += f" · test macro F1 {macro_f1:.4f}"

    sample = predictor.sample_path
    has_sample = sample is not None and sample.is_file()
    sample_hint = (
        f"\n\nNo file handy? A ready-made sample is bundled at `{sample}` — download it "
        "below and upload it straight back."
        if has_sample
        else ""
    )
    headers = list(predictor.batch_columns[:-1])
    threshold_label = (
        "Decision threshold t — flag HATE if P(HATE) >= t "
        "(starts at the calibrated t*; move it to see the effect)"
    )

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n{subtitle}\n\n"
            "Served from a persisted model bundle — nothing was trained to start this app.\n\n"
            "**Content warning:** this classifier scores real Vietnamese social-media text, so "
            "predictions can quote slurs. The `HATE` threshold is calibrated for recall: expect "
            "false positives in the review queue by design."
        )
        with gr.Tab("Single comment"):
            text_input = gr.Textbox(
                label="Vietnamese text", placeholder="Nhập nội dung cần kiểm tra...", lines=5
            )
            threshold_slider = gr.Slider(0.0, 1.0, value=t_star, step=0.01, label=threshold_label)
            predict_button = gr.Button("Analyze")
            predicted_class = gr.Textbox(label="Predicted class")
            probabilities = gr.Label(
                label="Class probabilities", num_top_classes=len(predictor.labels)
            )
            predict_button.click(
                fn=predictor.predict_detailed,
                inputs=[text_input, threshold_slider],
                outputs=[predicted_class, probabilities],
            )
        with gr.Tab("Batch file"):
            gr.Markdown(
                "Upload a `.txt` file (one Vietnamese comment per line) or a `.csv`/`.tsv` "
                "file with a text column (`free_text`, `text`, or `comment`)." + sample_hint
            )
            if has_sample and sample is not None:
                gr.File(value=str(sample), label="Ready-made sample (download)")
            file_input = gr.File(label="Test file", file_types=[".txt", ".csv", ".tsv"])
            batch_threshold_slider = gr.Slider(
                0.0, 1.0, value=t_star, step=0.01,
                label="Decision threshold t — flag HATE if P(HATE) >= t",
            )
            batch_button = gr.Button("Analyze file")
            results_table = gr.Dataframe(headers=headers, label="Predictions", wrap=True)
            download_file = gr.File(label="Download predictions (.csv)")
            batch_button.click(
                fn=predictor.predict_batch,
                inputs=[file_input, batch_threshold_slider],
                outputs=[results_table, download_file],
            )
    return demo


def launch_app(predictor: HateSpeechPredictor, config: ServeConfig) -> None:
    """Build the app for an already-loaded predictor and block while serving it."""
    # show_error: a malformed upload otherwise surfaces as a bare "Error" in the UI,
    # which is useless mid-demo. allowed_paths: Gradio only serves files under the cwd
    # or the system temp dir, so an explicit output_dir would break the CSV download.
    build_app(predictor).launch(
        server_name=config.host,
        server_port=config.port,
        share=config.share,
        show_error=True,
        allowed_paths=[str(predictor.output_dir)],
    )


def serve(config: ServeConfig) -> HateSpeechPredictor:
    """Load a bundle and serve the demo. Returns the predictor once serving stops."""
    predictor = HateSpeechPredictor(
        config.source, device=config.device, output_dir=config.output_dir
    )
    launch_app(predictor, config)
    return predictor
