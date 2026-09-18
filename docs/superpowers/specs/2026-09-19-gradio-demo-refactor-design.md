# Gradio demo refactor: design

## Goal
Reduce the repo to a small Gradio demo that serves `bvuong/nlp-vihate` from the
Hugging Face Hub, easy to read, easy to extend, and deployable as a Gradio Space.

## Decisions
- Demo-only repo: training pipeline, CLI, bundle and publish code are removed.
- Features: single-text analysis and batch file analysis.
- No decision threshold (`t*`) and no "flagged for review" column: the checkpoint has no
  calibrated threshold. Prediction is the argmax label with per-class probabilities.
- `pyproject.toml` + `uv.lock` are the source of truth (local dev uses `uv`).
  `requirements.txt` is generated for the Space at deploy time and gitignored here,
  because Gradio Spaces install only from `requirements.txt`.

## Layout
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

`app.py` is split from `ui.py` because `gradio app.py` needs a module-level `demo`, which loads the model at import; keeping the UI in `ui.py` lets tests import it without loading a model.

## model.py
- `MODEL_ID = os.environ.get("MODEL_ID", "bvuong/nlp-vihate")`; a local path also works
  since `from_pretrained` accepts both. No bundle or fallback logic.
- Loaded once at import: tokenizer + `AutoModelForSequenceClassification`, `.eval()`,
  CUDA if available else CPU.
- `LABELS = ("CLEAN", "OFFENSIVE", "HATE")` (checkpoint stores only `LABEL_0..2`);
  `MAX_LENGTH = 160`. Load fails with a clear error if the head is not 3-way.
- `predict(texts, batch_size=32)`: batched tokenization, `torch.inference_mode()`,
  softmax; returns one `{label: prob}` dict per text.

## app.py
- Single tab: textbox, Analyze button, `gr.Label` (probabilities), predicted-class
  output, a few clickable Vietnamese example sentences. Empty input returns a friendly
  message.
- Batch tab: upload `.txt` (one comment per line) or `.csv`/`.tsv` (column `free_text`,
  `text` or `comment`, else the first column). Capped at 500 rows with a visible note.
  Shows a results table and a CSV download written to a temp file (no `allowed_paths`).
  Unreadable or empty files raise `gr.Error` with a readable message.
- Content warning: text may quote slurs.
- `demo` exposed at module level; `demo.launch()` under `__main__`.

## Deployment
- README front matter: `sdk: gradio`, `sdk_version` pinned to the tested Gradio version,
  `python_version: "3.12"`, `app_file: app.py`.
- Deploy step: `{ echo "--extra-index-url https://download.pytorch.org/whl/cpu"; uv export --no-hashes --no-dev --no-emit-project; } > requirements.txt`, then push to the Space.
  The `--extra-index-url` line points pip at the PyTorch CPU wheel index, because `uv export` omits the explicit index.
  Documented in the README; `requirements.txt` gitignored.
  The Space is a separate git repo: `requirements.txt` is copied into a clone of it
  (with the app files) and pushed from there, never from this repo.
- Public model: no token needed. First start downloads weights, then cached.

## Removals
Delete `src/vihate/`, old tests, `README_vihsd.md`, the two ViHate notebooks and the
training-only extras in `pyproject.toml`. Leave `data/`, `cyber_bullying/`, `docs/` and
untracked `src/wecode/` untouched. Everything deleted remains in git history.

## Testing
- Unit tests for file parsing (txt/csv/tsv, column choice, row cap, empty file) and result
  formatting, using a stub predictor.
- Real-model check is a manual smoke run (`Classifier()` against the Hub repo, then both tabs over HTTP) rather than a test, so the suite stays offline.
- Manual check: run `uv run app.py` with the real model and exercise both tabs before
  calling it done. The Space build itself cannot be verified without pushing.
