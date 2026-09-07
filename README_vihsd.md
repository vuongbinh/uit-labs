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
