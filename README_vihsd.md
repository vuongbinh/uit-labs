# Vietnamese Hate-Speech Pipeline

Reproducible training and evaluation for ViHSD 3-class classification:
`CLEAN`, `OFFENSIVE`, and `HATE`. The pipeline keeps raw Vietnamese social
text intact, including diacritics, emoji, profanity, slang, repeated
characters, and punctuation. Normalization is limited to label parsing.

ViHSD is loaded from Hugging Face dataset `uitnlp/vihsd`. The dataset is
documented as roughly 33K Vietnamese social comments with the three labels
above.

## Setup

```bash
uv sync --extra dev
```

The current runtime used to build this scaffold did not have `uv` or ML
packages installed, so dependency installation is expected before full runs.

## Notebook Run

The whole project run is also available as one executable notebook,
`notebooks/vihate_project_run.ipynb`: dataset load, fold-plan audit, classical
CV, transformer CV, metrics, confusion matrices, and the summary artifacts —
one stage per cell, with the configuration in a single cell at the top.

```bash
uv sync --extra dev --extra notebook
uv run jupyter lab notebooks/vihate_project_run.ipynb
```

The notebook calls the same `vihate` functions the CLI calls — no pipeline
logic is duplicated — and writes the same artifacts to `outputs/notebook/`, so
a notebook run and a CLI run are byte-identical. Stage 3 (transformer) is off
by default behind `RUN_TRANSFORMER`, since it needs a GPU. Headless execution:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/vihate_project_run.ipynb
```

The notebook's last stage compares its Macro F1 against the accepted classical
baseline (`docs/project_overview_and_handover.md` §4.2) and reports whether the
run reproduced it.

## Classical 5-Fold CV

```bash
uv run vihate run --experiment classical --out-dir outputs/classical
```

The classical baseline uses a TF-IDF feature union of word n-grams and
character n-grams with class-balanced logistic regression by default. Use
`--classical-model svm` for a Linear SVM run; AUC metrics are omitted for SVM
because `LinearSVC` does not expose probabilities.

## Transformer 5-Fold CV

```bash
uv run vihate run --experiment transformer --model-name uitnlp/visobert --out-dir outputs/visobert
uv run vihate run --experiment transformer --model-name xlm-roberta-base --out-dir outputs/xlm-roberta
```

Transformer folds use stratified 5-fold CV, fixed seed, AdamW (`adamw_torch`)
with a linear warmup schedule (`--warmup-ratio`, default 0.1), weighted
cross-entropy loss from training-fold class frequencies, dynamic per-batch
padding, `fp16` when CUDA is available, and Hugging Face `Trainer`. Validation
Macro F1 is evaluated after every epoch and written to
`training_history_fold_<n>.json` (and an aggregated `training_history.json`).

Optional knobs: `--epochs`, `--batch-size`, `--grad-accum-steps` (effective
batch = `batch_size * grad_accum_steps`), `--learning-rate`, `--max-length`,
`--warmup-ratio`, `--sample-size` (smoke runs), `--optim` (any `transformers`
optimizer name), `--gradient-checkpointing`, and `--freeze-embeddings`.

`uitnlp/visobert` fine-tunes fully with the defaults on a 4 GB GPU. For
`xlm-roberta-base` (its ~192M-parameter multilingual embedding table does not
fit alongside full-precision AdamW state) add `--freeze-embeddings`, which
keeps that table fixed and trains the encoder at ViSoBERT-comparable speed;
`--optim adamw_bnb_8bit --gradient-checkpointing` (extra: `uv sync --extra bnb`)
is the slower alternative that keeps every parameter trainable.

## Persisting and Serving the Demo

Training writes weights into the run's output directory, which on Colab or Kaggle is
session-local disk and disappears with the session. The demo also needs Part 6's
calibrated `HATE` threshold `t*`, which costs about an hour of cross-validation to
compute. Reaching the demo therefore used to mean re-running everything above it.

Part 7B of `notebooks/vihate_project_run_fnal.ipynb` closes that gap by writing one
self-describing **demo bundle**:

```
demo_bundle/
├── model/              weights + tokenizer, as save_pretrained() wrote them
├── demo_config.json    labels, max_length, calibrated t*, training recipe, test metrics
├── README.md           model card (this is what the Hub renders)
└── demo_test_set.csv   optional batch-tab sample
```

Persisting `t*` alongside the weights is the load-bearing part: without it a
training-free demo is impossible in principle, not merely inconvenient.
`load_demo_config()` validates `schema_version` and every field's type, so a stale or
hand-edited bundle fails at load time rather than mis-scoring text later. Part 7B also
reloads its own copy and scores a line before the session ends, so a broken artifact
is caught while the weights still exist.

Serve it (extra: `uv sync --extra demo`):

```bash
uv run vihate demo --model outputs/demo_bundle        # a local bundle directory
uv run vihate demo --model yourname/vihsd-visobert    # or a Hub repo id
```

Flags: `--share` (public tunnel, expires in about a week), `--host`, `--port`,
`--device`, `--out-dir`. A cold start is one model download, cached after the first
run — no training, no notebook. Part 8 of the notebook cold-starts the same way: in a
fresh session, run just the config cell and Part 8, and set `DEMO_SOURCE` to the bundle
directory or repo id.

If you prefer the conventional Gradio entrypoint, the repo root has an `app.py`:

```bash
uv run gradio app.py                                # hot-reloading dev server
uv run python app.py                                # plain launch
DEMO_MODEL=you/vihsd-visobert uv run gradio app.py  # serve a Hub bundle
```

It exposes `demo` at module level, which is what `gradio app.py` resolves, and reads
`DEMO_MODEL` (default `outputs/demo_bundle`), `DEMO_DEVICE`, `DEMO_HOST`, `DEMO_PORT`
and `DEMO_SHARE` from the environment. `vihate demo` remains the richer entrypoint;
`app.py` exists for the Gradio workflow and for hosts that expect that filename.

Both paths load the bundle through `HateSpeechPredictor`, which now refuses a bundle
whose model head label count disagrees with `demo_config.json` — that mismatch used to
load fine and then die at predict time with a bare `KeyError`. `vihate publish-model`
runs the same check (`serve_demo.verify_bundle`, which also scores one benign line)
before uploading, so a bundle that cannot serve never reaches the Hub.

Publish it, for a URL that does not expire:

```bash
uv run vihate publish-model --bundle outputs/demo_bundle --repo yourname/vihsd-visobert
uv run vihate publish-space --space yourname/vihsd-demo --model yourname/vihsd-visobert
```

Both need a Hugging Face **write** token (`hf auth login`). `publish-space` creates a
Gradio Space whose generated `app.py` has the model repo id baked in, so there are no
Space variables to configure; the free CPU tier is enough. `publish-model` deliberately
excludes `demo_test_set.csv` — it is real ViHSD validation text containing slurs, and
the weights are the part worth publishing (`--include-sample` overrides). Add
`--private` to either command to keep it unlisted.

The demo's maths match Part 8 of the notebook exactly, and
`tests/test_notebook_contract.py` enforces that: it executes the real Part 7B and
Part 8 cells against a fixture model and asserts they agree with `serve_demo`
prediction-for-prediction, so the notebook's inline writer cannot drift away from the
package's reader.

One deliberate quirk. The threshold rule **appends** the moderation flag instead of
overwriting the class, so a row can read `predicted_class = CLEAN` alongside
`flagged_for_review = True`. That is Part 8's existing behaviour and it is *not* Part 6's
`apply_hate_rule()`, which overwrites the prediction with `HATE` — so the demo's
`predicted_class` column will not reconcile with the Part 7 table on flagged rows.
`test_flag_does_not_change_predicted_class` pins it, making any change a decision
rather than an accident.

Only the final single transformer is persisted; the Part 6 ensembles and the classical
TF-IDF baselines are not, because Part 8 never served them. A bundle is a snapshot, not
a registry — re-running Part 7B overwrites it, so tag the Hub revision or keep the zip
if you need to roll back.

## Outputs

Each run writes:

- `fold_metrics.json`: machine-readable fold metrics.
- `summary.json`: mean and standard deviation for scalar metrics.
- `summary.md`: human-readable report.
- `confusion_matrix_fold_<n>.json`: per-fold confusion matrices.
- `training_history_fold_<n>.json` / `training_history.json`: per-epoch
  training loss and validation Macro F1 (transformer runs only).

Reported metrics include macro F1, weighted F1, balanced accuracy, MCC,
per-class precision/recall/F1, confusion matrix, and ROC-AUC/PR-AUC when
class probabilities are available.

A notebook run writes the same set under `outputs/notebook/<run>/`; the files
are byte-identical to the CLI's for the same configuration.

## Local Checks

```bash
uv run ruff check .
uv run basedpyright
uv run pytest
```

Without `uv`, the stdlib-only syntax check still works:

```bash
python3 -m compileall src tests
```

`ruff` lints `.ipynb` cells, so `notebooks/` is excluded alongside
`cyber_bullying/`: library-oriented rules such as `T201` (`print`) contradict
notebook idiom, where printing progress and displaying tables is the output.
