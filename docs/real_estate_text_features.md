# Text Feature Extraction — Vietnamese Real-Estate Listings

Domain keyword flags, numeric entity recovery and text representations for the
`name` / `description` fields of
[`tinixai/vietnam-real-estates`](https://huggingface.co/datasets/tinixai/vietnam-real-estates),
plus an out-of-time benchmark of what they add to a tabular price baseline.

Code: `real_estate/text/` · Tests: `tests/text_features/` · Notebook:
`notebooks/vietnam_real_estate_text_features.ipynb`

---

## 1. Quick start

```python
from real_estate.text import TextFeatureConfig, TextFeaturePipeline

pipeline = TextFeaturePipeline(TextFeatureConfig(
    tfidf_mode="word",      # "word" | "char" | "both"
    tfidf_components=128,
    redact_price=True,      # keep the regression target out of the text features
))

train_features = pipeline.fit_transform(train_df)   # fits TF-IDF/SVD on TRAIN only
test_features  = pipeline.transform(test_df)        # reuses that fitted state
```

`transform` returns a frame index-aligned to its input, so it can be bound onto
any tabular matrix with `pd.concat([tabular, text_features], axis=1)`.

Install: `uv sync` (or `uv pip install -e .`); PhoBERT additionally needs
`uv pip install torch transformers`.

---

## 2. Module map

| Module | Responsibility |
|---|---|
| `normalize.py` | Diacritic folding, cleaning, and **fold-with-alignment** for mapping offsets back to the original string |
| `lexicon.py` | The domain vocabulary as data: `KeywordRule` entries + derived aggregates |
| `keywords.py` | `KeywordExtractor` → `kw_*` boolean flags, with a negation guard |
| `entities.py` | `extract_entities` → `text_*` numerics; `impute_from_text`; price redaction |
| `representation.py` | `TfidfTextEncoder` (TF-IDF + Truncated SVD), `PhoBERTEmbedder` (lazy torch import) |
| `pipeline.py` | `TextFeaturePipeline` — composition, fit/transform, group partitioning |
| `data.py` | Shard loading, strict `price` parsing, NFC normalisation, cleaning, out-of-time split |
| `benchmark.py` | Reference tabular block, metrics, ablation ladder, report writer, CLI |

---

## 3. Feature families

### 3.1 Domain keyword flags (`kw_*`)

47 boolean features across four categories — `legal`, `access`, `condition`,
`intent` — covering every phrase named in the requirements:

- **Legal status**: `sổ đỏ`, `sổ hồng`, `chính chủ`, `pháp lý rõ ràng`, `đang chờ sổ`
- **Accessibility / road**: `ô tô đỗ cửa`, `mặt tiền`, `ngõ thông`, `xe hơi vào nhà`, `ngõ ba gác`
- **Condition / interior**: `full nội thất`, `nội thất cơ bản`, `nhà mới`, `nhà cấp 4`

Patterns are matched against **diacritic-folded** text, so `"Ô Tô Đỗ Cửa"`,
`"ôtô đỗ cổng"` and `"o to do cua"` all hit the same rule. Two properties are
enforced by tests:

- No pattern may contain a literal non-ASCII character — it could never match
  folded text. (Two such patterns were caught during development.)
- Every rule must fire on real data; `KeywordExtractor.prevalence` reports hit
  rates, and no rule is currently dead.

**Negation guard.** `"không tranh chấp"` / `"không quy hoạch"` advertise the
*absence* of an encumbrance. Guarded rules inspect a 28-character window before
the match for a negation cue (`không`, `ko`, `kg`, `chưa`, `chẳng`, …) and skip
it; a separate `kw_khong_quy_hoach` rule captures the positive claim. Without
this, the `legal_risk` aggregate inverts its meaning on a meaningful share of
listings.

### 3.2 Numeric entities (`text_*`)

Parsed from prose: advertised price, area (`300m2`, `44m²`, `300 mét vuông`),
frontage × depth (`4m x 12m`, `ngang 5m dài 20m`), floor count (`6 tầng`,
`3 tấm`), `N lầu`, bedrooms (`3 PN`), bathrooms (`3 WC`) and road width
(`đường 8m`).

The value here is **recovery**, not novelty: the structured columns are sparse
and the text usually states the same number. See §5 for measured agreement.

`N lầu` is deliberately kept separate from `floor_count`. Southern listings use
it both for "levels above ground" and "levels total", so folding it into a floor
count would bake in an unfounded assumption; it ships as its own raw signal and
lets the model decide.

### 3.3 Text representations

- `TfidfTextEncoder` — TF-IDF reduced with Truncated SVD. Word bigrams are what
  capture multi-syllable Vietnamese phrases (`sổ đỏ` is two space-delimited
  syllables); a `char_wb` 3–5-gram view is available for spelling-variant
  robustness, and `"both"` concatenates the two. Fitted on training text only.
- `PhoBERTEmbedder` — mean-pooled `vinai/phobert-base`, optionally SVD-reduced.
  `torch`/`transformers` are imported inside `encode()`, so the rest of the
  package never requires them. Diacritics are **preserved** on this path because
  PhoBERT's vocabulary distinguishes them.

---

## 4. Leakage controls — read this before adding features

The asking price is written into the listing text itself (`"giá: 7.45 tỷ"`), and
`price` is the regression target. Left alone this destroys the benchmark: in a
first run `text_price_vnd` alone took **57% of total split gain** and made the
entity arm look ~26% better on RMSLE than it is.

Two independent controls:

1. `TARGET_LEAKING_ENTITIES = ("text_price_vnd",)` is excluded from every model
   feature block. It is still extracted, because it is useful for the *other*
   task — repairing records where the structured price is null — and it is still
   reported in the agreement table as an extraction-quality measure.
2. Price mentions are **redacted from the TF-IDF and PhoBERT input**, otherwise
   digit n-grams reintroduce the same information. `prepare_for_tfidf` and
   `prepare_for_embedding` do this; `redact_price=False` opts out.

Both are covered by tests. The pipeline test asserts that two listings identical
except for their quoted price encode to *identical* TF-IDF vectors with redaction
on, and to *different* vectors with it off.

Other protocol choices in `benchmark.py`:

- **Out-of-time evaluation.** Shards are chronological (`shard_0000` = 2025-06,
  `shard_0009` = 2026-03), so train and test are nine months apart. No random
  split, hence no look-ahead.
- **Early stopping** uses the latest 15% of the *training* period by
  `published_at`, never the test period.
- **All fitted state** (target encodings, TF-IDF vocabulary, SVD basis, PhoBERT
  reduction) is learned from training rows only.
- **Multiple seeds** per arm, with spread reported — an uplift smaller than the
  seed-to-seed noise is not a finding.

---

## 5. Dataset findings that affect any baseline

Verified on the published shards; each one changes how the target should be modelled.

1. **`price` is a string column.** The dataset card declares `float64`; the
   parquet schema says `string`. 97.1% of values are plain integer VND strings
   and 2.9% are null. `log1p(price)` on the raw column will not do what you
   expect. `parse_price` converts it **strictly** — anything that is not a plain
   number becomes `NaN` rather than being silently coerced, because stripping
   currency words would turn `"7.45 tỷ"` into `7.45` **VND**, a billion-fold error.

2. **Rentals are mixed in with sales.** ~17–19% of rows are `cho thuê`, whose
   monthly asking price is orders of magnitude below a sale price. Leaving them
   in corrupts the log-price target. `kw_cho_thue` is the flag, and `clean_frame`
   drops those rows. Removing them plus the price/area bounds costs ~22% of rows.

3. **Price outliers span 15 orders of magnitude.** The minimum recorded price on
   `shard_0000` is 300,000 VND and the maximum exceeds 800 *trillion* VND.
   `clean_frame` bounds the target; RMSLE is the primary metric because it
   damps this tail. Raw-scale R² is correspondingly unstable and should not be
   read as the headline number.

4. **~0.7% of text rows are NFD-decomposed.** Stray combining marks
   (`U+0301`, `U+0300`, `U+0323`, `U+0309`) and emoji variation selectors
   (`U+FE0F`) mean diacritic folding is **not** length-preserving, so price spans
   are mapped back per character via `fold_with_alignment` rather than by index
   arithmetic.

5. **`street_name` and `project_name` carry NFD variants** (30 and 20 distinct
   values; 313 and 198 rows). They render identically but compare unequal, so
   the same street becomes two categories in any frequency or target encoding.
   `province_name`, `district_name` and `ward_name` are clean NFC.
   `load_shard_sample` NFC-normalises every string column on the way in.

---

## 6. Integration contract for the baseline pipeline

The reference tabular block in `benchmark.py` exists so text features can be
measured before the project baseline lands. It is **not** the project baseline —
replace it, don't extend it.

`run_uplift_benchmark` accepts a pre-built tabular matrix
(`tabular_train` / `tabular_test`) for exactly this reason.

Order of operations matters:

```python
from real_estate.text import (ENTITY_COLUMNS, NUMERIC_COLUMNS, TARGET_LEAKING_ENTITIES,
                              TextFeatureConfig, TextFeaturePipeline, impute_from_text)

pipeline = TextFeaturePipeline(TextFeatureConfig(tfidf_components=128))
train_text = pipeline.fit_transform(train_df)     # fit text state on TRAIN only
test_text  = pipeline.transform(test_df)

# 1. Impute the sparse structured columns from text FIRST ...
numeric = [c for c in NUMERIC_COLUMNS if c in train_df.columns]
train_imp = train_df.copy()
train_imp[numeric] = impute_from_text(train_df[numeric], train_text[ENTITY_COLUMNS])[numeric]

# 2. ... THEN build the tabular block, so derived ratios (frontage/depth,
#    road/frontage, area/frontage) are computed from recovered values, not nulls.
tabular_train = your_encoder.fit_transform(train_imp)

# 3. Bind text features on, minus the target-restating entity.
drop = [c for c in TARGET_LEAKING_ENTITIES if c in train_text.columns]
X_train = pd.concat([tabular_train, train_text.drop(columns=drop)], axis=1)
```

Two invariants worth keeping in any downstream refactor:

- **Index alignment.** `clean_frame` deliberately preserves the input index
  rather than resetting it, so feature blocks computed before and after cleaning
  stay row-aligned. A `reset_index` in the middle silently selects the wrong
  rows. `TextFeaturePipeline.transform` asserts its output length matches its
  input for this reason.
- **`impute_from_text` never overwrites** a non-null structured value; it only
  fills, and records what it filled in a `<column>__from_text` boolean.

`benchmark.py` also implements RMSLE / MAPE / MAE(tỷ) / MedAE / R² locally so it
can run standalone. When the shared metrics module lands, that module should
supersede these — they are intentionally thin.

---

## 7. Reproducing

```bash
uv sync                                    # or: uv pip install -e ".[dev]"
bash scripts/run_text_feature_benchmark.sh data
```

Shards are downloaded on first use into the data directory (each is ~165 MB).
Reports are written to `reports/text_features/<arm>/uplift_report.{json,md}`.

Run the tests:

```bash
uv run pytest            # 173 tests
```

---

## 8. Limitations and next steps

- The keyword lexicon is hand-built from observed prevalence, not learned. It
  covers the required concepts well but will miss paraphrase; a phrase-mining
  pass over high-error residuals is the obvious extension.
- `text_price_vnd` agrees with the structured `price` on only ~75% of rows
  (5% tolerance), versus 88–95% for the physical attributes. Likely causes are
  price ranges, per-m² quotes and multi-unit listings; it needs work before it
  can be trusted for the null-price repair use case.
- PhoBERT is mean-pooled with no fine-tuning, at a subsample size set by CPU
  throughput. A fine-tuned encoder, or embeddings cached once at full scale,
  would be the fair next comparison.
- Only two shards are used. A multi-fold walk-forward across all ten would give
  tighter error bars on the marginal gains.
