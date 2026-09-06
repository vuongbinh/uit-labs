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
uv run vihate run --experiment transformer --model-name vinai/phobert-base --out-dir outputs/phobert
uv run vihate run --experiment transformer --model-name xlm-roberta-base --out-dir outputs/xlm-roberta
```

Transformer folds use stratified 5-fold CV, fixed seed, weighted loss from
training-fold class frequencies, and Hugging Face `Trainer`. Optional knobs:
`--epochs`, `--batch-size`, `--learning-rate`, `--max-length`, and
`--sample-size` for smoke runs.

## Outputs

Each run writes:

- `fold_metrics.json`: machine-readable fold metrics.
- `summary.json`: mean and standard deviation for scalar metrics.
- `summary.md`: human-readable report.
- `confusion_matrix_fold_<n>.json`: per-fold confusion matrices.

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
