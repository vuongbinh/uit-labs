# Vietnamese Hate-Speech Detection (`vihate`) — Project Overview & Handover Guide

**Audience:** stakeholders who need to understand how this project runs (inputs, outputs,
workflow), and any engineer or team taking over development.

**Status date:** 2026-09-07 · **Repository:** `github.com/vuongbinh/uit-labs` ·
**Project branch:** `nlp/train` · **Tracking:** Multica workspace, project
*"NLP Research, Training & Analysis"* (issues HYPE-1 … HYPE-7)

---

## 1. Executive summary

This project builds a **reproducible machine-learning pipeline for Vietnamese hate-speech
detection** on the ViHSD benchmark. Given a Vietnamese social-media comment, the pipeline
classifies it into one of three categories:

| Label | Meaning |
|---|---|
| `CLEAN` (0) | Non-hateful, constructive text |
| `OFFENSIVE` (1) | Rude/vulgar language not targeting a protected group |
| `HATE` (2) | Targeted hostility or dehumanization toward a protected characteristic |

What has been delivered so far:

1. **Research foundation** (HYPE-1, done): literature review, dataset profiling, and model
   selection. Chose `uitnlp/visobert` as the primary transformer backbone (published ViHSD
   Macro F1 ≈ 0.677, SOTA), with `xlm-roberta-base` as a robust fallback and TF-IDF + linear
   models as fast classical baselines.
2. **Working pipeline** (HYPE-2, merged via PR #1): the `vihate` Python package — dataset
   loading, classical and transformer 5-fold cross-validation, metrics, and JSON/Markdown
   reporting, driven by a single CLI command.
3. **QA suite** (HYPE-3, branch pushed, PR pending): 18 fast + 1 slow automated tests
   covering labels, data loading, metric correctness, reporting determinism, and a CI
   workflow (`.github/workflows/tests.yml`).
4. **Classical baseline results** (HYPE-4, in review): full 5-fold CV on the canonical
   24,048-example train split — **Macro F1 = 0.6475 ± 0.0069 (LogReg)** and
   **0.6501 ± 0.0095 (LinearSVC)** — independently audited and accepted as the project's
   reference baseline.

What remains: **HYPE-5** (transformer fine-tuning & benchmarking vs. the classical baseline)
and **HYPE-6** (qualitative error analysis & synthesis report), both in backlog, plus the
merge/push loose ends listed in §7.2.

---

## 2. How the project runs

### 2.1 Operating model (Multica workspace)

Work is organized as issues in the Multica project *"NLP Research, Training & Analysis"*
and executed by a team of AI agents, with a human workspace owner as reviewer and merge
authority:

| Role | Who | Responsibility so far |
|---|---|---|
| Coordinator | agent **Mika** | Decomposed the initiative into issues HYPE-1…HYPE-7 |
| Research | agent **Researcher - Sofia** | HYPE-1: literature review, dataset profiling, model survey |
| Development | agent **Dev - Cindy** | HYPE-2: pipeline implementation & hardening (PR #1); HYPE-4: baseline experiment runs |
| Development | agent **Dev - Gwen** | HYPE-4 review/audit of baseline results; this handover document (HYPE-7) |
| QA | agent **QA - Moly** | HYPE-3: test suite, metric verification, CI workflow |
| Owner / reviewer | human workspace owner (GitHub `vuongbinh`) | Reviews PRs, sets merge policy, unblocks credentials, marks issues `done` |

**Work-item flow:** an issue is created and assigned to an agent → the agent runs in an
isolated workspace, checks out the repo on a dedicated branch (`agent/<name>/<id>`) →
implements, verifies locally, pushes the branch → opens a PR **targeting `nlp/train`** →
human (and peer-agent) review → merge → issue moves through
`todo → in_progress → in_review → done` (`blocked` when external help is needed, e.g. the
GitHub credential expiry on 2026-09-07, which the owner resolved the same day).

Discussion, experiment results, and report attachments live in **issue comments** — they
are the system of record for anything not committed to git (see §8.3 for why this matters).

### 2.2 Repository & branch model

`uit-labs` is a shared lab-exercises repo; this project is one slice of it.

| Path / branch | Content |
|---|---|
| `nlp/train` (branch) | **The project's mainline.** All PRs target this branch, never `master` (explicit owner instruction) |
| `src/vihate/` | The pipeline package (7 modules, ~530 lines) |
| `tests/` | Pytest suite (2 files on `nlp/train`; 6 files on the QA branch) |
| `README_vihsd.md` | Pipeline usage guide |
| `pyproject.toml` + `uv.lock` | Dependencies & tooling config, locked for reproducibility |
| `data/*.csv`, `cyber_bullying/` | Pre-existing, **unrelated** lab content (local ViHSD subset + notebook). Not used by the `vihate` CLI; excluded from linting. Do not modify |
| `docs/research/vihsd_literature_review_and_baselines.md` | HYPE-1 research report — **not yet on `nlp/train`**; exists only on the local (unpushed) branch `agent/researcher-sofia/ace161d64951` and as a HYPE-1 comment attachment |

Branch state on origin as of 2026-09-07:

- `nlp/train` @ `30882a5` — pipeline + PR #1 fixes + `pytest.approx` assertion update.
- PR #1 (`agent/dev-cindy/ec5afd969673` → `nlp/train`) — **MERGED** 2026-09-07.
- `agent/qa-moly/hype-3-nlp-tests` @ `b88403f` — pushed, **no PR opened yet** (GitHub App
  returned 403 at the time). Based on the older commit `54ed34e`; needs a rebase onto
  current `nlp/train` before merging (overlapping edits in `tests/` and
  `src/vihate/reporting.py` are likely).
- `agent/researcher-sofia/ace161d64951` @ `4c6348d` — **local only, never pushed**; carries
  the research document.

### 2.3 The pipeline (data flow)

```
Hugging Face dataset "uitnlp/vihsd" (split: train by default)
        │  data.py: load + infer text/label columns (free_text / label_id),
        │           keep raw Vietnamese text intact (diacritics, emoji, slang)
        ▼
list[TextExample(text, label 0|1|2)]          labels.py: strict label normalization
        │
        ▼
StratifiedKFold(n_splits=5, shuffle=True, random_state=13)   ← identical folds everywhere
        │
        ├── experiment=classical ──► per fold: sklearn Pipeline
        │       TF-IDF words (1–2 grams, min_df=2) ∪ TF-IDF chars (3–5 grams, min_df=2)
        │       → LogisticRegression(class_weight="balanced")  or  LinearSVC(balanced)
        │
        └── experiment=transformer ──► per fold: HuggingFace Trainer
                AutoTokenizer (max_length=160) + AutoModelForSequenceClassification
                + weighted cross-entropy (weights from TRAINING-fold frequencies only)
        │
        ▼
metrics.py per fold: Macro F1, Weighted F1, Balanced Accuracy, MCC,
per-class P/R/F1, ROC-AUC (OvR, weighted) & PR-AUC when probabilities exist,
confusion matrix (written per fold)
        │
        ▼
reporting.py: summarize folds (mean ± population std) and write artifacts to --out-dir
```

**Leak-free by construction:** feature vectorizers, class balancing, and loss weights are
fit/computed inside each fold from training data only — verified in the HYPE-4 audit.

---

## 3. Inputs

### 3.1 Dataset

**Canonical benchmark (used by all official runs):** Hugging Face `uitnlp/vihsd`,
~33,400 Vietnamese social-media comments; the CLI loads the `train` split (24,048 examples)
by default.

| Split | Total | CLEAN | OFFENSIVE | HATE |
|---|---:|---:|---:|---:|
| train (70%) | 24,048 | 19,886 (82.69%) | 1,606 (6.68%) | 2,556 (10.63%) |
| dev (10%) | 2,672 | 2,190 (81.96%) | 212 (7.93%) | 270 (10.10%) |
| test (20%) | 6,680 | 5,548 (83.05%) | 444 (6.65%) | 688 (10.30%) |

The ~12:1:1.6 class imbalance is why class weighting and **Macro F1** (not accuracy) are
the primary success measures.

**Local subset (NOT used by the pipeline):** `data/train_df.csv`, `val_df.csv`,
`test_df.csv` — a 7,385-example, re-balanced (~40/34/26) Kaggle-derived subset belonging to
the pre-existing `cyber_bullying/` lab notebook. Do not confuse it with the canonical
benchmark; results on it are not comparable to anything reported here.

Linguistic characteristics that shape modeling decisions (from HYPE-1 profiling): teencode
in 45% of HATE samples, accent-less text in ~5%, deliberate obfuscation in ~7% of hate
comments, emoji in ~4%; 99.8% of comments fit within 160 subword tokens.

### 3.2 Environment

- **Python ≥ 3.12**, managed with **uv** (`uv sync --extra dev`; `uv.lock` committed).
- Runtime deps: `datasets`, `scikit-learn`, `numpy`, `torch`, `transformers >=4.44,<5`
  (the `<5` pin is deliberate — 5.x pre-releases broke slow tokenizers used by
  PhoBERT/XLM-R), `accelerate`, `sentencepiece`, `typer`, `rich`.
- Dev deps: `pytest`, `ruff`, `basedpyright`.
- **Hardware:** classical runs complete in minutes on CPU. Transformer runs need a GPU
  (literature-review estimates: ~3–8 min per fold; 5 folds per model).
- **Network:** first run downloads `uitnlp/vihsd` from Hugging Face and model weights.

### 3.3 Commands & configuration

All experiments go through one CLI (also documented in `README_vihsd.md`):

```bash
# Classical baselines (the ones already executed for HYPE-4)
uv run vihate run --experiment classical --out-dir outputs/classical
uv run vihate run --experiment classical --classical-model svm --out-dir outputs/classical_svm

# Transformer runs (HYPE-5 — note: pass the model explicitly; the default is PhoBERT)
uv run vihate run --experiment transformer --model-name uitnlp/visobert   --out-dir outputs/visobert
uv run vihate run --experiment transformer --model-name xlm-roberta-base  --out-dir outputs/xlm-roberta
```

| Option | Default | Meaning |
|---|---|---|
| `--experiment` | `classical` | `classical` or `transformer` |
| `--split` | `train` | ViHSD split to load |
| `--folds` / `--seed` | `5` / `13` | Stratified CV geometry — **keep fixed for comparability** |
| `--sample-size` | all | Limit examples (smoke runs only; official results use the full split) |
| `--classical-model` | `logreg` | `logreg` or `svm` (SVM yields no AUC metrics — no probabilities) |
| `--model-name` | `vinai/phobert-base` | Transformer backbone. Research recommends **`uitnlp/visobert` first**; the default predates that decision |
| `--epochs` / `--batch-size` / `--learning-rate` / `--max-length` | `3` / `16` / `2e-5` / `160` | Training configuration agreed in HYPE-1 |

---

## 4. Outputs

### 4.1 Artifacts per run (written to `--out-dir`)

| File | Content |
|---|---|
| `fold_metrics.json` | All metrics for each of the 5 folds (machine-readable) |
| `summary.json` | Mean ± std across folds for every scalar metric |
| `summary.md` | Human-readable report of the same summary |
| `confusion_matrix_fold_<n>.json` | 3×3 confusion matrix per fold (rows = true, cols = predicted) |

JSON is written with sorted keys and unescaped Unicode (Vietnamese text stays readable).
Note: `outputs/` directories are run artifacts — they are **not committed to git** (and are
not currently gitignored either). The official HYPE-4 `summary.json`/`summary.md` files are
attached to the HYPE-4 issue comment.

### 4.2 Established baseline results (HYPE-4 — the project's reference numbers)

Classical 5-fold CV, full `train` split (24,048 examples), mean ± std over folds:

| Metric | TF-IDF + LogReg (balanced) | TF-IDF + LinearSVC (balanced) |
|---|---:|---:|
| **Macro F1 (primary)** | **0.6475 ± 0.0069** | **0.6501 ± 0.0095** |
| Weighted F1 | 0.8553 ± 0.0028 | 0.8677 ± 0.0032 |
| Balanced Accuracy | 0.6725 ± 0.0054 | 0.6236 ± 0.0076 |
| MCC | 0.5387 ± 0.0077 | 0.5524 ± 0.0108 |
| ROC-AUC (OvR, weighted) | 0.9108 ± 0.0028 | n/a (no probabilities) |
| PR-AUC (macro) | 0.6696 ± 0.0108 | n/a |
| CLEAN P/R/F1 | 0.9431 / 0.9054 / 0.9238 | 0.9208 / 0.9573 / 0.9387 |
| OFFENSIVE P/R/F1 | 0.4130 / 0.4440 / 0.4277 | 0.5119 / 0.3381 / 0.4072 |
| HATE P/R/F1 | 0.5299 / 0.6682 / 0.5909 | 0.6363 / 0.5755 / 0.6043 |

Interpretation: the two models are statistically tied on Macro F1. `CLEAN` is near-solved
(F1 ≈ 0.92–0.94); the bottleneck is `OFFENSIVE` (F1 ≈ 0.41), which is heavily confused with
both `CLEAN` and `HATE` — a boundary the literature review flags as having the lowest
inter-annotator agreement, so part of it is irreducible. Fold stability is high
(σ ≤ 0.01 on all headline metrics). These numbers already match published PhoBERT-level
performance (0.6476) and sit ~9–13 points above published classical ViHSD results.

### 4.3 Success criteria for the transformer phase (HYPE-5)

Published ViHSD references: classical TF-IDF ≈ 0.52–0.56 · PhoBERT-base 0.6476 ·
XLM-R base 0.6550 · **ViSoBERT 0.6771 (SOTA)**.

Original HYPE-4 gates: Macro F1 > 0.68, Weighted F1 > 0.87, MCC > 0.56, OFFENSIVE F1 > 0.45,
HATE F1 > 0.62, CLEAN F1 ≥ 0.94 (no regression), ROC-AUC > 0.91.

**Recalibration adopted in the HYPE-4 review** (because "> 0.68" exceeds published SOTA):

- Two tiers: **"beats classical" ≥ 0.67 Macro F1**; **"SOTA-level" ≥ 0.68** as a stretch goal.
- Decide winners with a **paired per-fold significance test** (Wilcoxon / paired t-test over
  the 5 fold Macro F1 values) — valid because every run shares the seed-13 folds.
- Run **ViSoBERT first** (must pass `--model-name uitnlp/visobert` explicitly), XLM-R as
  fallback comparison.

---

## 5. Quality assurance

- **On `nlp/train`:** `tests/test_labels.py`, `tests/test_reporting.py` (label parsing,
  report formatting).
- **On the QA branch (pending merge):** 18 fast tests + 1 `slow`-marked test — exact-value
  metric assertions against hand-built confusion matrices, zero-division degenerate folds,
  AUC-skip behavior when probabilities are absent, byte-for-byte JSON determinism
  (Vietnamese text unescaped), `normalize_label` guards (bool/negative/out-of-range),
  offline mocked dataset-loader tests (column inference, diacritics/teencode/emoji
  preservation), a tiny classical-CV regression, and `.github/workflows/tests.yml`
  (ruff + fast/slow pytest on every push & PR). **CI does not run until this branch merges.**
- **Static checks:** `ruff check .` (select ALL, scoped to the `vihate` package —
  `cyber_bullying/` and `data/` are excluded on purpose), `basedpyright` in `standard` mode,
  both clean on `nlp/train`.
- **Verification commands:**

```bash
uv sync --extra dev
uv run ruff check . && uv run basedpyright && uv run pytest      # full local gate
uv run pytest -m slow                                            # slow regression (QA branch)
python3 -m compileall src tests                                  # stdlib-only smoke check
```

- **Baseline audit:** the HYPE-4 results were independently reviewed (fold-level leak-free
  fitting, protocol conformance, artifact-vs-comment consistency all verified) before being
  accepted as the reference.

---

## 6. Project status snapshot (2026-09-07)

| Issue | Scope | Status | Outcome |
|---|---|---|---|
| HYPE-1 | Literature review & baseline model selection | **done** | Research report (attachment + unpushed branch), methodology decisions (§3.3, §4.3) |
| HYPE-2 | Pipeline implementation & training framework | in_review | `vihate` package working end-to-end; **PR #1 merged** into `nlp/train` |
| HYPE-3 | Evaluation QA suite & test automation | in_review | QA branch `agent/qa-moly/hype-3-nlp-tests` pushed; **PR not yet opened**; needs rebase |
| HYPE-4 | Classical 5-fold CV baseline execution | in_review | Baseline established (§4.2) + peer review with recalibrated transformer gates (§4.3) |
| HYPE-5 | Transformer fine-tuning & benchmarking (ViSoBERT, XLM-R) | backlog | Not started — next major experiment |
| HYPE-6 | Qualitative error analysis & benchmark synthesis | backlog | Not started — depends on HYPE-5 outputs |
| HYPE-7 | Stakeholder documentation & handover | in_review | This document |

---

## 7. How to continue development (handover)

### 7.1 Onboarding quickstart

```bash
git clone https://github.com/vuongbinh/uit-labs.git
cd uit-labs
git checkout nlp/train                 # project mainline — do NOT base work on master
uv sync --extra dev
uv run pytest && uv run ruff check . && uv run basedpyright    # should be green

# Smoke-test the real pipeline (small sample, CPU, a few minutes):
uv run vihate run --experiment classical --sample-size 2000 --out-dir outputs/smoke
```

Work is coordinated through Multica issues (HYPE-*); each new work item gets an issue, an
agent or engineer assignment, a branch `agent/<name>/<id>` or a personal branch, and a PR
into `nlp/train`.

### 7.2 Immediate next actions (priority order)

1. **Land the QA suite (HYPE-3).** Rebase `agent/qa-moly/hype-3-nlp-tests` onto current
   `nlp/train` (it predates PR #1 and the `pytest.approx` commit; expect small conflicts in
   `tests/test_labels.py`, `tests/test_reporting.py`, `src/vihate/reporting.py`), open the
   PR into `nlp/train`, review, merge. This activates CI for all future work. Keep
   `min_df=2` defaults so the HYPE-4 numbers stay reproducible.
2. **Rescue the research document (HYPE-1).** `docs/research/vihsd_literature_review_and_baselines.md`
   exists only on the unpushed local branch `agent/researcher-sofia/ace161d64951` (in agent
   workspace clones) and as a HYPE-1 comment attachment. Recover it (download the attachment
   or push the branch) and merge it into `nlp/train` so the methodology rationale is durable.
3. **Run HYPE-5 (transformer benchmarks), GPU machine required:**

   ```bash
   uv run vihate run --experiment transformer --model-name uitnlp/visobert  --out-dir outputs/visobert
   uv run vihate run --experiment transformer --model-name xlm-roberta-base --out-dir outputs/xlm-roberta
   ```

   Evaluate against §4.3 (two-tier Macro F1 gates + paired per-fold significance test vs.
   the classical runs). Post the results table and attach `summary.md`/`summary.json` to the
   issue, as done for HYPE-4.
4. **Run HYPE-6 (error analysis & synthesis)** once HYPE-5 artifacts exist: quantify
   `CLEAN↔OFFENSIVE` and `OFFENSIVE↔HATE` confusion from the per-fold confusion matrices,
   sample failure modes (teencode, accent-less text, obfuscation, emoji), and produce the
   comparative benchmark table + next-phase recommendations.
5. **Housekeeping:** add `outputs/` to `.gitignore`; archive run artifacts somewhere durable
   (issue attachments today) rather than only in ephemeral agent workspaces.

### 7.3 Known limitations & improvement backlog

From the HYPE-4 code review — none block HYPE-5, all are candidate follow-ups:

- `transformer_cv.py` trains a fixed 3 epochs with `eval_strategy="no"` — no early stopping
  on validation Macro F1; watch for over/undertraining.
- `padding="max_length"` (160) on every sample although the median comment is ~9 words —
  dynamic padding would speed transformer folds several times at zero accuracy cost.
- CLI default `--model-name` is `vinai/phobert-base`, but the research decision is
  ViSoBERT-first — either always pass the flag or change the default.
- ROC-AUC (0.911) ≫ Macro F1 (0.647) for LogReg means ranking is better than argmax
  extraction: per-class **decision-threshold calibration** on an inner validation slice of
  each training fold (never the test fold) is cheap headroom (~1–2 Macro F1 points). If
  adopted, it must be done **before** freezing a new reference baseline.
- Avoid naïve text augmentation (EDA/synonym replacement) — it flips toxicity labels.

### 7.4 Conventions & invariants to preserve

These are what make results comparable across the project; changing any of them invalidates
the baseline comparison:

- **PRs target `nlp/train`**, never `master` (owner policy). `master` and
  `cyber_bullying/`+`data/` lab content are out of the project's scope.
- **Raw text is never modified** — no stripping of diacritics, emoji, punctuation, slang;
  normalization is limited to label parsing.
- **`StratifiedKFold(n_splits=5, shuffle=True, random_state=13)`** everywhere, classical and
  transformer alike.
- **Leak-free folds:** vectorizers, `class_weight="balanced"`, and transformer loss weights
  are computed from training-fold data only, inside the CV loop.
- **Macro F1 is the primary metric**; report the full diagnostic set (§4.1/§5) with mean ±
  population std across folds; mark unavailable metrics `n/a` rather than approximating.
- Classical feature defaults stay `min_df=2`, word (1–2) ∪ char (3–5) n-grams.
- Quality gates before merge: `ruff check .`, `basedpyright`, `pytest` (plus `-m slow` once
  the QA branch lands).
- Dependencies: keep `transformers>=4.44,<5`; install via `uv` against the committed
  `uv.lock`.

### 7.5 Operational notes & past incidents

- **GitHub credentials:** on 2026-09-07 the workspace's GitHub OAuth token expired mid-run
  ("OAuth session expired and could not be refreshed"), blocking pushes. Fix path: the
  workspace owner re-authenticates the GitHub connection in Multica workspace settings.
  Agents cannot refresh it themselves. A git bundle attached to HYPE-2 served as the
  fallback transfer mechanism.
- **PR creation permission:** the GitHub App integration returned 403 when QA tried to open
  a PR (`gh` push over SSH worked). If this recurs, push the branch and open the PR from the
  GitHub compare URL, or re-check the installation's permission scope.
- **Where knowledge lives:** durable code/docs in git (`nlp/train`); experiment results and
  reports as issue-comment attachments (HYPE-1 research report, HYPE-4 summaries); decisions
  and review rationale in issue comment threads. When taking over, read the HYPE-1…HYPE-4
  threads — they contain the audit trail behind every number in this document.

---

## 8. Glossary (for non-specialist stakeholders)

| Term | Plain meaning |
|---|---|
| 5-fold cross-validation | Split data into 5 parts; train 5 times, each time holding out a different part for scoring. Gives a mean ± spread instead of one lucky/unlucky number |
| Macro F1 | Per-class quality score averaged equally over CLEAN/OFFENSIVE/HATE — unlike accuracy, it cannot be inflated by the 83% majority class |
| Weighted F1 / Balanced Accuracy / MCC | Companion quality scores that weight classes differently; MCC is a single-number correlation between predictions and truth |
| Confusion matrix | 3×3 table of true vs. predicted labels; shows *which* classes get mixed up |
| TF-IDF | Classical text featurization: score words/character-fragments by how distinctive they are for a document |
| Transformer / fine-tuning | Modern pretrained language models (ViSoBERT, XLM-RoBERTa, PhoBERT) adapted to this task by further training on ViHSD |
| Class weighting / balanced | Countermeasure for the 12:1:1.6 imbalance: minority classes count more in training so they aren't ignored |
| ROC-AUC / PR-AUC | How well the model *ranks* classes by confidence, independent of the final decision threshold |
| Data leakage | Any accidental use of held-out information during training, which inflates scores; this pipeline is audited leak-free |
