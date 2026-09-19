# Gradio Demo Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the repo to a small Gradio demo that serves `bvuong/visoBert-ensemble` from the Hugging Face Hub and deploys as a Gradio Space.

**Architecture:** Four flat root modules with one job each: `model.py` (load + predict, no Gradio), `batch.py` (file parsing + result tables, no torch/Gradio), `ui.py` (Gradio layout and handlers, takes a classifier as an argument), `app.py` (5-line entrypoint that loads the real model and exposes `demo`). Tests exercise `model.py` with a tiny random-weight model and `ui.py`/`batch.py` with a stub classifier, so no test downloads anything.

**Tech Stack:** Python 3.12, Gradio 5.50 (`>=5.50,<6`), torch (CPU wheel), transformers 4.57 (`<5`), sentencepiece, pandas, uv, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-09-19-gradio-demo-refactor-design.md`

**Deviation from spec (deliberate):** the spec lists `app.py` (UI) + `model.py`. This plan splits the UI into `ui.py` and keeps `app.py` as a thin entrypoint, plus a pure `batch.py`. Reason: `gradio app.py` needs a module-level `demo`, which forces a model load at import, so anything importable from `app.py` would make tests load the real model. Task 5 updates the spec to match.

## Global Constraints

- Model id from env `MODEL_ID`, default `bvuong/visoBert-ensemble`; a local path also works. No bundle/fallback logic.
- Labels constant `("CLEAN", "OFFENSIVE", "HATE")`; `MAX_LENGTH = 160`; batch size 32; `MAX_ROWS = 500`.
- No decision threshold `t*` and no "flagged for review" output.
- `pyproject.toml` + `uv.lock` are the source of truth; `requirements.txt` is generated with `uv export --no-hashes --no-dev` at deploy time and is gitignored.
- Space README front matter: `sdk: gradio`, `sdk_version: 5.50.0`, `python_version: "3.12"`, `app_file: app.py`.
- Leave `data/`, `cyber_bullying/`, `docs/` and untracked `src/wecode/` untouched. Delete `src/vihate/`, old tests, `README_vihsd.md`, and the two ViHate notebooks (`notebooks/vihate_project_run.ipynb`, `notebooks/vihate_project_run_fnal.ipynb`).
- Ruff `select = ["ALL"]`, line length 100. Commit messages end with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
- Branch: current branch `nlp/vihsd-demo-persistence`. Uncommitted edits to the old code (loader fallback, `tests/conftest.py`) are discarded by this refactor.

---

### Task 1: Clean slate and project config

**Files:**
- Delete: `src/vihate/` (tracked files), `tests/*` (tracked), `README_vihsd.md`, `notebooks/vihate_project_run.ipynb`, `notebooks/vihate_project_run_fnal.ipynb`, `app.py`
- Modify: `pyproject.toml` (full rewrite), `.gitignore`, `uv.lock` (regenerated)
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: `tests/conftest.py` fixture `tiny_model_dir(tmp_path_factory) -> Path` and helper `make_tiny_model(model_dir: Path, num_labels: int = 3) -> Path`; constant `BENIGN_TEXTS: tuple[str, ...]`. Tasks 3 and 4 rely on these names.

- [ ] **Step 1: Remove the old code**

```bash
cd /home/binhv/@uit/uit-labs
git checkout -- tests/conftest.py            # drop the unrelated uncommitted T_STAR edit
git rm -r -q src/vihate tests README_vihsd.md app.py \
  notebooks/vihate_project_run.ipynb notebooks/vihate_project_run_fnal.ipynb
```
`src/wecode/` is untracked, so `git rm` does not touch it. Verify: `ls src` shows only `wecode`.

- [ ] **Step 2: Rewrite `pyproject.toml`**

```toml
[project]
name = "vihate-demo"
version = "0.2.0"
description = "Gradio demo for Vietnamese hate-speech detection (bvuong/visoBert-ensemble)."
requires-python = ">=3.12"
dependencies = [
  "gradio>=5.50,<6",
  "pandas>=2.2",
  "sentencepiece>=0.2",
  "torch>=2.4",
  "transformers>=4.44,<5",
]

[dependency-groups]
dev = [
  "basedpyright>=1.20",
  "pytest>=8.3",
  "ruff>=0.6",
]

[tool.uv]
# Not a distributable package: the root modules are the app.
package = false

# CPU-only torch keeps the Space build small (the default Linux wheel drags in
# several GB of CUDA libraries).
[tool.uv.sources]
torch = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.ruff]
line-length = 100
target-version = "py312"
# The repo root also holds unrelated lab/homework content.
extend-exclude = ["cyber_bullying", "data", "docs", "notebooks", "src"]

[tool.ruff.lint]
select = ["ALL"]
ignore = ["COM812", "D203", "D213", "CPY001"]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["D", "INP001", "PLR2004", "S101"]

[tool.basedpyright]
include = ["app.py", "ui.py", "model.py", "batch.py", "tests"]
pythonVersion = "3.12"
typeCheckingMode = "standard"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-q"
```

- [ ] **Step 3: Update `.gitignore`**

Append:
```
# Generated at deploy time for the Hugging Face Space (see README)
requirements.txt
.gradio/
```

- [ ] **Step 4: Write `tests/conftest.py`**

```python
"""Shared fixtures: a tiny random-weight 3-class model, so no test downloads anything.

Predictions from it are meaningless; the point is exercising the real load/predict path.
"""

from pathlib import Path

import pytest

from model import LABELS, MAX_LENGTH

# Benign Vietnamese; every word is in the fixture vocab so nothing degenerates to UNK.
BENIGN_TEXTS = (
    "hôm nay thời tiết đẹp",
    "xin chào bạn",
    "cảm ơn đội ngũ hỗ trợ",
)

_VOCAB = (
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "hôm", "nay", "thời", "tiết", "đẹp", "xin", "chào", "bạn",
    "tôi", "rất", "cảm", "ơn", "đội", "ngũ", "hỗ", "trợ",
)


def make_tiny_model(model_dir: Path, num_labels: int = len(LABELS)) -> Path:
    """Save a tiny classifier + word-level tokenizer into model_dir (save_pretrained layout)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        XLMRobertaConfig,
        XLMRobertaForSequenceClassification,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    config = XLMRobertaConfig(
        vocab_size=len(_VOCAB),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=MAX_LENGTH + 8,
        pad_token_id=0,
        num_labels=num_labels,
    )
    XLMRobertaForSequenceClassification(config).save_pretrained(model_dir)

    tokenizer = Tokenizer(
        models.WordLevel({word: i for i, word in enumerate(_VOCAB)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=MAX_LENGTH,
    ).save_pretrained(model_dir)
    return model_dir


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A saved tiny model, session-scoped because saving is the slow part."""
    return make_tiny_model(tmp_path_factory.mktemp("tiny") / "model")
```
It imports `model`, which does not exist yet; that is fixed in Task 3, so collection fails until then. That is expected.

- [ ] **Step 5: Re-lock and sync**

Run: `uv lock && uv sync`
Expected: succeeds; `uv run python -c "import gradio, torch, transformers; print(gradio.__version__, torch.__version__)"` prints `5.50.x` and a `+cpu` torch.

- [ ] **Step 6: Commit**

```bash
git add -A pyproject.toml uv.lock .gitignore tests
git status --short   # confirm only deletions + these files; src/wecode/ stays untracked
git commit -m "refactor: strip repo down to a Gradio demo skeleton

Remove the training pipeline, CLI, bundle/publish code, old tests and the
two ViHate notebooks. New pyproject for a non-package uv project with
CPU-only torch.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: `batch.py` (file parsing and result tables)

**Files:**
- Create: `batch.py`
- Test: `tests/test_batch.py`

**Interfaces:**
- Produces:
  - `MAX_ROWS: int = 500`
  - `class BatchFileError(ValueError)`: message is user-readable
  - `read_comments(path: str | Path) -> list[str]`: non-empty stripped comments; raises `BatchFileError` for unsupported type, unreadable file, or no comments
  - `top_label(scores: dict[str, float]) -> str`
  - `results_table(texts: Sequence[str], scores: Sequence[dict[str, float]]) -> pd.DataFrame` with columns `text`, `predicted_class`, then `P(<label>)` per label in `scores` order, probabilities rounded to 4 places

- [ ] **Step 1: Write the failing tests** (`tests/test_batch.py`)

```python
from pathlib import Path

import pandas as pd
import pytest

from batch import BatchFileError, read_comments, results_table, top_label


def test_txt_one_comment_per_line_skips_blanks(tmp_path: Path) -> None:
    path = tmp_path / "in.txt"
    path.write_text("xin chào\n\n  cảm ơn  \n", encoding="utf-8")
    assert read_comments(path) == ["xin chào", "cảm ơn"]


@pytest.mark.parametrize(
    ("suffix", "sep"),
    [(".csv", ","), (".tsv", "\t")],
)
def test_table_prefers_known_text_column(tmp_path: Path, suffix: str, sep: str) -> None:
    path = tmp_path / f"in{suffix}"
    path.write_text(f"id{sep}text{sep}label\n1{sep}xin chào{sep}0\n2{sep}cảm ơn{sep}0\n", "utf-8")
    assert read_comments(path) == ["xin chào", "cảm ơn"]


def test_table_falls_back_to_first_column(tmp_path: Path) -> None:
    path = tmp_path / "in.csv"
    path.write_text("noidung,other\nxin chào,1\n", encoding="utf-8")
    assert read_comments(path) == ["xin chào"]


def test_table_drops_missing_values(tmp_path: Path) -> None:
    path = tmp_path / "in.csv"
    path.write_text("text\nxin chào\n\ncảm ơn\n", encoding="utf-8")
    assert read_comments(path) == ["xin chào", "cảm ơn"]


def test_utf8_bom_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "in.csv"
    path.write_text("text\nxin chào\n", encoding="utf-8-sig")
    assert read_comments(path) == ["xin chào"]


@pytest.mark.parametrize("name", ["empty.txt", "empty.csv"])
def test_empty_file_raises(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_text("", encoding="utf-8")
    with pytest.raises(BatchFileError):
        read_comments(path)


def test_unsupported_suffix_raises(tmp_path: Path) -> None:
    path = tmp_path / "in.pdf"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(BatchFileError, match=r"\.pdf"):
        read_comments(path)


def test_non_utf8_bytes_raise_readable_error(tmp_path: Path) -> None:
    path = tmp_path / "in.txt"
    path.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(BatchFileError, match="in.txt"):
        read_comments(path)


def test_top_label_is_argmax() -> None:
    assert top_label({"CLEAN": 0.1, "OFFENSIVE": 0.2, "HATE": 0.7}) == "HATE"


def test_results_table_shape_and_rounding() -> None:
    scores = [
        {"CLEAN": 0.123456, "OFFENSIVE": 0.3, "HATE": 0.576544},
        {"CLEAN": 0.9, "OFFENSIVE": 0.05, "HATE": 0.05},
    ]
    table = results_table(["a", "b"], scores)
    assert list(table.columns) == [
        "text", "predicted_class", "P(CLEAN)", "P(OFFENSIVE)", "P(HATE)",
    ]
    assert table["predicted_class"].tolist() == ["HATE", "CLEAN"]
    assert table["P(CLEAN)"].tolist() == [0.1235, 0.9]
    assert isinstance(table, pd.DataFrame)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_batch.py --noconftest`
(`--noconftest` is needed because `tests/conftest.py` imports `model`, which does not exist until Task 3.)
Expected: FAIL with `ModuleNotFoundError: No module named 'batch'`.

- [ ] **Step 3: Implement `batch.py`**

```python
"""Read an uploaded comment file and shape prediction results into a table.

Pure pandas/stdlib: no torch, no Gradio, so it is trivial to test.
"""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

MAX_ROWS = 500
TEXT_COLUMNS = ("free_text", "text", "comment")
TABLE_SEPARATORS = {".csv": ",", ".tsv": "\t"}


class BatchFileError(ValueError):
    """The uploaded file cannot be analyzed; the message is safe to show to the user."""


def read_comments(path: str | Path) -> list[str]:
    """Return the non-empty comments in a .txt (one per line) or .csv/.tsv file."""
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in TABLE_SEPARATORS:
            table = pd.read_csv(path, sep=TABLE_SEPARATORS[suffix], encoding="utf-8-sig")
            column = next((name for name in TEXT_COLUMNS if name in table.columns), None)
            values = table[column if column else table.columns[0]].dropna().astype(str)
            lines = values.tolist()
        elif suffix == ".txt":
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        else:
            message = f"Unsupported file type '{suffix}'. Upload a .txt, .csv or .tsv file."
            raise BatchFileError(message)  # noqa: TRY301
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        message = f"Could not read {path.name}: {exc}"
        raise BatchFileError(message) from exc

    comments = [line.strip() for line in lines if line.strip()]
    if not comments:
        message = f"{path.name} has no comments to analyze."
        raise BatchFileError(message)
    return comments


def top_label(scores: dict[str, float]) -> str:
    """Return the label with the highest probability."""
    return max(scores, key=lambda label: scores[label])


def results_table(texts: Sequence[str], scores: Sequence[dict[str, float]]) -> pd.DataFrame:
    """One row per comment: the text, the predicted class and every class probability."""
    rows = []
    for text, text_scores in zip(texts, scores, strict=True):
        row: dict[str, object] = {"text": text, "predicted_class": top_label(text_scores)}
        row.update({f"P({label})": round(prob, 4) for label, prob in text_scores.items()})
        rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_batch.py --noconftest`
Expected: all pass. If the `.pdf` case fails because the message differs, fix the message, not the test regex.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check batch.py tests/test_batch.py && uv run ruff format batch.py tests/test_batch.py
git add batch.py tests/test_batch.py
git commit -m "feat: add batch file parsing and result tables

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```
Fix any ruff findings before committing (the `noqa: TRY301` may be unnecessary; remove it if ruff reports it as unused).

---

### Task 3: `model.py` (load and predict)

**Files:**
- Create: `model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `tests/conftest.py` (`tiny_model_dir`, `make_tiny_model`, `BENIGN_TEXTS`) from Task 1.
- Produces:
  - `DEFAULT_MODEL_ID = "bvuong/visoBert-ensemble"`, `LABELS: tuple[str, str, str]`, `MAX_LENGTH = 160`, `BATCH_SIZE = 32`
  - `class Classifier`: `__init__(self, model_id: str | None = None, device: str | None = None)`; `predict(self, texts: Sequence[str], batch_size: int = BATCH_SIZE) -> list[dict[str, float]]`
  - Task 4 relies on `Classifier.predict` and `LABELS`.

- [ ] **Step 1: Write the failing tests** (`tests/test_model.py`)

```python
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


def test_default_model_id_is_the_published_repo() -> None:
    assert DEFAULT_MODEL_ID == "bvuong/visoBert-ensemble"


def test_wrong_head_size_is_rejected(tmp_path: Path) -> None:
    two_way = make_tiny_model(tmp_path / "two-way", num_labels=2)
    with pytest.raises(ValueError, match="3"):
        Classifier(str(two_way), device="cpu")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_model.py`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'model'`.

- [ ] **Step 3: Implement `model.py`**

```python
"""Load the fine-tuned classifier and score Vietnamese text. No Gradio in here."""

import os
from collections.abc import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_ID = "bvuong/visoBert-ensemble"
# The checkpoint only stores LABEL_0..LABEL_2; this is the ViHSD class order.
LABELS = ("CLEAN", "OFFENSIVE", "HATE")
MAX_LENGTH = 160
BATCH_SIZE = 32


class Classifier:
    """A model loaded once and reused for every request."""

    def __init__(self, model_id: str | None = None, device: str | None = None) -> None:
        """Load from `model_id`, else $MODEL_ID, else the published Hub repo (or a local path)."""
        model_id = model_id or os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = _load_tokenizer(model_id)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device).eval()
        )
        head_size = int(self.model.config.num_labels)
        if head_size != len(LABELS):
            message = f"{model_id} has {head_size} output labels; this demo expects {len(LABELS)}."
            raise ValueError(message)

    def predict(self, texts: Sequence[str], batch_size: int = BATCH_SIZE) -> list[dict[str, float]]:
        """Return one {label: probability} dict per text, in input order."""
        results: list[dict[str, float]] = []
        for start in range(0, len(texts), batch_size):
            inputs = self.tokenizer(
                list(texts[start : start + batch_size]),
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                probs = torch.softmax(self.model(**inputs).logits, dim=-1).cpu()
            results.extend(dict(zip(LABELS, row.tolist(), strict=True)) for row in probs)
        return results


def _load_tokenizer(model_id: str) -> "AutoTokenizer":
    """Prefer the slow tokenizer (what training used); fall back to the fast one."""
    try:
        return AutoTokenizer.from_pretrained(model_id, use_fast=False)
    except (ValueError, OSError, ImportError):
        return AutoTokenizer.from_pretrained(model_id, use_fast=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model.py tests/test_batch.py`
Expected: all pass (conftest now imports cleanly). If `_load_tokenizer`'s return annotation trips the type checker, annotate with `PreTrainedTokenizerBase` imported from `transformers`.

- [ ] **Step 5: Smoke-test against the real Hub model**

Run:
```bash
uv run python -c "
from model import Classifier
c = Classifier(device='cpu')
print(c.predict(['Hôm nay thời tiết rất đẹp', 'mày ngu quá']))"
```
Expected: two dicts with keys CLEAN/OFFENSIVE/HATE summing to about 1. Weights are cached from earlier, so this is fast.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check model.py tests && uv run ruff format model.py tests
git add model.py tests/test_model.py
git commit -m "feat: add Classifier that loads bvuong/visoBert-ensemble and scores text

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: `ui.py` and `app.py`

**Files:**
- Create: `ui.py`, `app.py`
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: `model.LABELS`, `model.Classifier.predict(texts) -> list[dict[str, float]]`; `batch.MAX_ROWS`, `BatchFileError`, `read_comments`, `results_table`, `top_label`.
- Produces:
  - `analyze_text(classifier, text: str | None) -> tuple[str, dict[str, float]]`: `(predicted label, probabilities)`; empty input gives `("Please enter some text.", {})`
  - `analyze_file(classifier, path: str | None) -> tuple[pd.DataFrame, str, str]`: `(table, csv_path, note)`; raises `gr.Error` on a bad or missing file
  - `build_demo(classifier) -> gr.Blocks`
  - `app.py` exposes module-level `demo`.
  - `classifier` parameters are typed with a `Predictor` Protocol (`predict(texts) -> list[dict[str, float]]`) so tests can pass a stub.

- [ ] **Step 1: Write the failing tests** (`tests/test_ui.py`)

```python
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

    def predict(self, texts: list[str]) -> list[dict[str, float]]:
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ui.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'ui'`.

- [ ] **Step 3: Implement `ui.py`**

```python
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
    "[bvuong/visoBert-ensemble](https://huggingface.co/bvuong/visoBert-ensemble).\n\n"
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


def analyze_file(
    classifier: Predictor, path: str | None
) -> tuple[pd.DataFrame, str, str]:
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


def build_demo(classifier: Predictor) -> gr.Blocks:
    """Assemble the two-tab app around an already-loaded classifier."""
    with gr.Blocks(title=TITLE) as demo:
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
                f"`free_text`, `text` or `comment` column. At most {MAX_ROWS} comments are analyzed."
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
    return demo
```

- [ ] **Step 4: Implement `app.py`**

```python
"""Entrypoint for the ViHate demo: `uv run app.py` locally, or a Hugging Face Space.

`demo` is exposed at module level so `uv run gradio app.py` (hot reload) works too.
Set MODEL_ID to a Hub repo id or local path to serve a different checkpoint.
"""

from model import Classifier
from ui import build_demo

demo = build_demo(Classifier())

if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest`
Expected: all tests in `test_batch.py`, `test_model.py`, `test_ui.py` pass. `analyze_file` output ordering `(table, csv_path, note)` must match `outputs=[table, download, status]`: a `gr.Dataframe`, a `gr.File` and a `gr.Markdown`.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check . && uv run ruff format .
uv run basedpyright 2>&1 | tail -15
git add ui.py app.py tests/test_ui.py
git commit -m "feat: add Gradio UI with single-text and batch tabs

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```
Fix real ruff findings. For pyright, only fix errors in our four modules; third-party `Unknown` noise is acceptable in `standard` mode.

---

### Task 5: Space README, spec update, and end-to-end verification

**Files:**
- Modify: `README.md` (rewrite as the Space README), `docs/superpowers/specs/2026-09-19-gradio-demo-refactor-design.md` (layout section)

**Interfaces:** none (documentation and verification).

- [ ] **Step 1: Rewrite `README.md`**

```markdown
---
title: Vietnamese Hate Speech Detection
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Vietnamese Hate Speech Detection

Gradio demo for [`bvuong/visoBert-ensemble`](https://huggingface.co/bvuong/visoBert-ensemble): classifies
Vietnamese text as `CLEAN`, `OFFENSIVE` or `HATE`, one comment at a time or from a
`.txt`/`.csv`/`.tsv` file.

## Run locally

```bash
uv sync
uv run app.py                       # plain launch, http://127.0.0.1:7860
uv run gradio app.py                # hot-reloading dev server
MODEL_ID=path/or/repo uv run app.py # serve a different checkpoint
uv run pytest                       # tests (no downloads)
```

## Layout

| File | Job |
| --- | --- |
| `app.py` | Entrypoint: loads the model, exposes `demo` |
| `ui.py` | Gradio layout and handlers |
| `model.py` | Load the checkpoint, `predict(texts)` |
| `batch.py` | Parse uploaded files, build result tables |

To add a feature, add a tab in `ui.py`; model changes stay in `model.py`.

## Deploy to a Hugging Face Space

Gradio Spaces install only from `requirements.txt`, so generate it from the lockfile
(it is gitignored and never edited by hand):

```bash
uv export --no-hashes --no-dev -o requirements.txt
```

Then push `app.py ui.py model.py batch.py requirements.txt README.md` to a Space with
SDK **Gradio**. `bvuong/visoBert-ensemble` is public, so no token is needed; the first start
downloads the weights and later starts use the cache. The sdk version in the header above
must match the `gradio` version in `uv.lock`.
```

- [ ] **Step 2: Update the spec layout section**

In `docs/superpowers/specs/2026-09-19-gradio-demo-refactor-design.md`, replace the Layout code block with:
```
app.py            entrypoint: loads the model, exposes `demo` (5 lines)
ui.py             Gradio layout + handlers; takes a classifier, so tests pass a stub
model.py          load model; Classifier.predict(texts) -> list[dict[label, prob]]
batch.py          parse uploaded files, build result tables (no torch/Gradio)
pyproject.toml    runtime deps + dev group; CPU-only torch index
uv.lock
README.md         HF Space front matter + run/deploy instructions
tests/            no real model downloads
```
and add one sentence under it: "`app.py` is split from `ui.py` because `gradio app.py` needs a module-level `demo`, which loads the model at import; keeping the UI in `ui.py` lets tests import it without loading a model."

- [ ] **Step 3: Verify the Space dependency export**

```bash
uv export --no-hashes --no-dev -o requirements.txt
grep -c "nvidia" requirements.txt          # expect 0
grep -E "^(--|torch|gradio|transformers)" requirements.txt
```
Expected: no `nvidia-*` packages; a `+cpu` torch pin with an index line for the PyTorch CPU index. Then prove pip can resolve it under the Space's Python without installing:
```bash
uv venv --python 3.12 "$TMPDIR/space-check" 2>/dev/null || uv venv --python 3.12 /tmp/claude-1000/-home-binhv--uit-uit-labs/9ab2e7ee-ceb3-4214-9018-7d4a96a06952/scratchpad/space-check
uv pip install --dry-run --python <that venv>/bin/python -r requirements.txt 2>&1 | tail -5
```
Expected: resolves without error. If the `+cpu` torch pin fails to resolve, report it: the fallback is to drop the `[tool.uv.sources]` CPU pin and accept the larger build.

- [ ] **Step 4: Manual end-to-end check with the real model**

```bash
uv run app.py &        # or run_in_background
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:7860/   # expect 200
```
Then, with the Playwright MCP tools or `gradio_client`, exercise both tabs: submit "Hôm nay thời tiết rất đẹp" (expect a label and three probabilities) and upload a small `.txt` and `.csv` (expect a table, a status note and a downloadable CSV). Stop the server afterwards.

Expected: no tracebacks in the server log; the Batch tab also shows a readable error for an unsupported `.pdf` upload.

- [ ] **Step 5: Final full check and commit**

```bash
uv run pytest && uv run ruff check . && git status --short
git add README.md docs/superpowers/specs/2026-09-19-gradio-demo-refactor-design.md
git commit -m "docs: Space README, run/deploy instructions, sync spec layout

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```
Confirm `requirements.txt` is untracked/ignored (`git check-ignore requirements.txt` prints it).

---

## Self-Review

**Spec coverage:** model id via env + default + local path → Task 3. Labels/MAX_LENGTH/3-way check/batched inference/no threshold → Task 3. Single tab (textbox, examples, probabilities, empty message) → Task 4. Batch tab (txt/csv/tsv, column choice, 500 cap + note, results table, temp-file CSV, `gr.Error`) → Tasks 2 and 4. Content warning → Task 4 `DESCRIPTION`. `demo` at module level → Task 4 `app.py`. README front matter, deploy via `uv export`, gitignored `requirements.txt` → Tasks 1 and 5. Removals and left-alone folders → Task 1. Testing (pure tests, real-model manual check; the "opt-in slow real-model test" from the spec is replaced by the Task 3 Step 5 smoke script and Task 5 Step 4, to keep the suite offline) → all tasks. CPU-torch pin is an addition to the spec, verified in Task 5 Step 3.

**Placeholders:** none. Every code step has full code.

**Type consistency:** `Classifier.predict(texts, batch_size) -> list[dict[str, float]]` matches `Predictor.predict` (Task 4) and the stub. `analyze_file` returns `(DataFrame, str, str)` and is wired to `[table, download, status]` in that order. `results_table`, `read_comments`, `top_label`, `BatchFileError`, `MAX_ROWS` names are identical in Tasks 2 and 4. `make_tiny_model(model_dir, num_labels)` matches its use in `test_model.py`.

**Known risks:** the Space build itself cannot be verified without pushing; Task 5 Step 3 covers dependency resolution only. Task 2 Step 2 uses `--noconftest` because the shared conftest imports `model.py` before it exists.
