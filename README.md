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

Gradio demo for [`bvuong/nlp-vihate`](https://huggingface.co/bvuong/nlp-vihate): classifies
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
{ echo "--extra-index-url https://download.pytorch.org/whl/cpu"; uv export --no-hashes --no-dev --no-emit-project; } > requirements.txt
```

The first line points pip at the PyTorch CPU wheel index, because `uv export` omits the explicit index.

A Space is its own git repo, so copy the files into a clone of it and push from there:

```bash
git clone https://huggingface.co/spaces/<user>/<space>
cp app.py ui.py model.py batch.py requirements.txt README.md <space-clone>/
cd <space-clone> && git add -A && git commit -m "Update demo" && git push
```

`requirements.txt` is intentionally gitignored here, so never rely on pushing this repo directly.
Create the Space with SDK **Gradio**. `bvuong/nlp-vihate` is public, so no token is needed; the
first start downloads the weights and later starts use the cache. The sdk version in the header above
must match the `gradio` version in `uv.lock`.
