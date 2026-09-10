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

Verified on the published shards. Items 1 and 2 are already documented in
`reports/vietnam_real_estates_eda.md`; they are restated here because this module
has to handle them, and item 2 records where an earlier reading of the data was
**wrong**. Items 3–5 are new.

1. **`price` is a string column.** The dataset card declares `float64`; the
   parquet schema says `string` (97.1% plain integer VND strings, 2.9% null).
   Already noted in the EDA's schema table. `parse_price` converts it
   **strictly** — anything that is not a plain number becomes `NaN` rather than
   being coerced, because stripping currency words would turn `"7.45 tỷ"` into
   `7.45` **VND**, a billion-fold error.

2. **`cho thuê` in the text does not mean the listing is a rental.** This one was
   read wrong initially, and the error is worth recording. `kw_cho_thue` fires on
   18.76% of rows, which initially read as "18.76% of the corpus is rentals" and
   was used as a row filter. Measured against price, that is false: **96.1% of
   the flagged rows carry a sale-scale price** (median 10 tỷ VND, versus 6.8 tỷ
   for unflagged rows). They are sale listings *pitching rental yield* — "sẵn hợp
   đồng thuê" (sold with an existing lease), "tiện cho thuê". Genuine rental
   listings do exist but quote a monthly rate (15–60M VND) and are rare.

   Consequences: the flag is a legitimate **feature** (a property marketed on
   yield differs from one that isn't) but an invalid **filter**. Dropping on it
   discards ~19% of valid training rows, and non-randomly — the dropped segment
   is systematically higher-priced, which biases both the model and any uplift
   measured against it. `clean_frame`'s `min_price` floor is what excludes real
   rentals, and `kw_cho_thue` is no longer used for cleaning.

3. **Cleaning now follows the EDA's Rule 1.** `clean_frame` defaults to price
   ∈ [100M, 200B] VND, area ∈ [15, 3000] m², unit price ∈ [3M, 800M] VND/m², so
   this module and the baseline notebook model the same population. Measured
   retention is 92.9% (`shard_0000`) and 90.2% (`shard_0009`) against the EDA's
   89.57% corpus-wide, and log-price skew comes out at +0.179 / +0.273 versus
   their +0.186 — close enough to confirm the rule is implemented faithfully.
   The unit-price band is the part that matters most: it catches price/area
   mismatches neither bound catches alone. Raw prices otherwise span 300,000 VND
   to over 800 *trillion*, which is also why raw-scale R² is unstable (§6.1).

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

## 6. Benchmark results

Every number below is **out-of-time**: training rows come from `shard_0000`
(2025-06-01 → 2025-07-30) and test rows from `shard_0009`
(2026-03-01 → 2026-03-30). Primary metric is RMSLE (lower is better). Raw JSON
and markdown reports are committed under `reports/text_features/`.

### 6.1 Marginal uplift ladder — 92,855 train / 36,118 test rows, 2 seeds

Each arm is a strict superset of the one above it, so every row is a marginal gain.

| arm | features | RMSLE | seed std | MAPE % | MAE (tỷ) | MedAE (tỷ) | R² |
|---|---|---|---|---|---|---|---|
| tabular | 23 | 0.3973 | 0.0003 | 28.59 | 3.998 | 1.428 | 0.7470 |
| + keywords | 70 | 0.3765 | 0.0002 | 27.01 | 3.801 | 1.366 | 0.7635 |
| + entities | 85 | 0.3678 | 0.0002 | 26.33 | 3.723 | 1.341 | 0.7665 |
| + tfidf | 213 | **0.3573** | 0.0006 | **25.92** | **3.679** | **1.330** | **0.7754** |

**Cumulative gain over tabular-only: −0.0400 RMSLE (−10.07%), −2.67 MAPE points,
+0.0284 R².** Every arm improves every metric — there is no trade-off to argue
about, which is worth noting because an earlier, mis-cleaned version of this
benchmark did show one (see the note below).

Seed spread is at most 0.0006 RMSLE per arm, so each step is 30–100× the
seed-to-seed noise; the ordering is not a sampling artifact.

> **Note on why these numbers moved.** An earlier run used looser price bounds
> (max 100 *trillion* VND) and dropped rows on the `kw_cho_thue` flag. Both were
> wrong: the outlier tail crushed raw-scale R² to 0.045 and made MAE uninformative,
> and the flag removed ~19% of valid *sales* rows in a price-correlated way
> (§5, item 2). Aligning `clean_frame` with the EDA's Rule 1 raised retention to
> 92.9% and lifted R² to 0.747. The text-feature ordering was the same either
> way; the absolute numbers were not.

### 6.2 TF-IDF analyzer — matched scale, 55,713 train / 22,557 test rows

All three modes share an identical tabular baseline (RMSLE 0.4144, MAPE 30.34%,
R² 0.7283), so the deltas are directly comparable.

| analyzer | features | RMSLE | Δ RMSLE | MAPE % | Δ MAPE | R² | SVD expl. var |
|---|---|---|---|---|---|---|---|
| `word` (1,2)-grams | 213 | **0.3694** | **−0.0450 (−10.85%)** | **26.99** | **−3.35** | **0.7573** | 0.132 |
| `both` | 341 | 0.3710 | −0.0434 (−10.47%) | 27.07 | −3.27 | 0.7522 | 0.362 |
| `char_wb` (3,5)-grams | 213 | 0.3774 | −0.0370 (−8.93%) | 27.51 | −2.83 | 0.7537 | 0.230 |

**Word bigrams win**, and concatenating the character view does not beat them
despite 60% more features. The character view explains more variance per
component (0.230 vs 0.132), but that variance is less useful for price. The
default is `word`; `char` remains available as a robustness option for
teencode-heavy sources.

### 6.3 PhoBERT — 4,642 train / 2,254 test rows

| arm | features | RMSLE | MAPE % | MAE (tỷ) | R² |
|---|---|---|---|---|---|
| tabular | 23 | 0.4989 | 39.90 | 4.973 | 0.6815 |
| + tfidf | 213 | **0.4591** | **36.09** | 4.586 | **0.7063** |
| + phobert | 277 | 0.4624 | 36.35 | 4.750 | 0.6825 |

Adding mean-pooled, unfine-tuned PhoBERT on top of TF-IDF makes every headline
metric **worse**: RMSLE +0.0033, MAPE +0.26 points, R² −0.0238. Only 2
`phobert_*` components reach the top-25 by split gain (0.62% of total) against 8
`tfidf_*` components (9.35%).

The cost comparison is the decisive part. This arm spent 787 s to process 4,642
training rows; the word TF-IDF arm spent 320 s on 55,713 — roughly **30× more
compute per row**, to end up slightly behind.

Conclusion: **TF-IDF is the right default.** PhoBERT is worth revisiting as a
fine-tuned encoder, or with embeddings computed once and cached — not as frozen
pooled features on a few thousand listings. See §9.

### 6.4 Entity extraction quality — validated against the structured columns

Where a parsed entity and the recorded value both exist, do they agree?
Measured on the 92,855 training rows.

| structured column | text entity | text coverage | column missing | overlap | agreement | cells recovered |
|---|---|---|---|---|---|---|
| area | `text_area_m2` | 0.783 | 0.000 | 72,733 | 0.926 | 0 |
| frontage_width | `text_frontage_m` | 0.304 | 0.459 | 19,065 | 0.885 | 9,179 |
| house_depth | `text_depth_m` | 0.304 | 0.970 | 1,408 | 0.953 | **26,841** |
| floor_count | `text_floor_count` | 0.402 | 0.827 | 11,187 | 0.916 | **26,179** |
| bedroom_count | `text_bedroom_count` | 0.399 | 0.477 | 32,595 | 0.884 | 4,486 |
| bathroom_count | `text_bathroom_count` | 0.258 | 0.511 | 21,329 | 0.930 | 2,630 |
| road_width | `text_road_width_m` | 0.087 | 0.892 | 1,447 | 0.858 | 6,597 |
| price *(excluded from features)* | `text_price_vnd` | 0.848 | 0.000 | 78,719 | 0.737 | 0 |

86–95% agreement on the physical attributes, against columns that are 46–97%
null in the source. `house_depth`, `floor_count` and `road_width` gain 26,841,
26,179 and 6,597 previously-empty cells respectively. That **recovery** — not
the raw entity values — is where most of the entity arm's gain comes from.

`text_price_vnd` agrees on only ~74%, consistent with price ranges, per-m²
quotes and multi-unit listings; it is excluded from model features regardless (§4).

### 6.5 Lexicon coverage — measured on 100,000 raw training rows

Prevalence is reported on rows **before** cleaning, so the table describes the
corpus the lexicon has to cover rather than the cleaned subset. **No rule is
dead** — all 47 fire on real data, and the test suite fails any pattern that
could never match folded text.

| concept | flag | hits | rate |
|---|---|---|---|
| sổ đỏ | `kw_so_do` | 23,895 | 23.89% |
| sổ hồng | `kw_so_hong` | 20,394 | 20.39% |
| chính chủ | `kw_chinh_chu` | 31,864 | 31.86% |
| pháp lý rõ ràng / đầy đủ | `kw_phap_ly_ro_rang` | 21,839 | 21.84% |
| đang chờ sổ | `kw_dang_cho_so` | 414 | 0.41% |
| mặt tiền | `kw_mat_tien` | 38,049 | 38.05% |
| ô tô đỗ cửa / cổng | `kw_o_to_do_cua` | 2,611 | 2.61% |
| ngõ thông | `kw_ngo_thong` | 5,413 | 5.41% |
| xe hơi vào nhà | `kw_xe_hoi_vao_nha` | 1,849 | 1.85% |
| ngõ ba gác | `kw_ngo_ba_gac` | 1,229 | 1.23% |
| full nội thất | `kw_full_noi_that` | 16,568 | 16.57% |
| nội thất cơ bản | `kw_noi_that_co_ban` | 2,769 | 2.77% |
| nhà mới / xây mới | `kw_nha_moi` | 9,908 | 9.91% |
| nhà cấp 4 | `kw_nha_cap_4` | 2,692 | 2.69% |
| *mentions* cho thuê † | `kw_cho_thue` | 18,760 | **18.76%** |
| certificate (derived) | `kw_legal_certificate` | 43,241 | 43.24% |
| car access (derived) | `kw_road_car_access` | 9,231 | 9.23% |
| legal risk (derived) | `kw_legal_risk` | 5,045 | 5.04% |

† `kw_cho_thue` counts *mentions* of renting, not rental listings: 96% of the rows
it fires on are sales pitching rental yield (§5, item 2). Use it as a feature,
never as a row filter.

The two rare required phrases — `đang chờ sổ` (0.41%) and `ngõ ba gác` (1.23%) —
are genuinely rare in the corpus rather than badly matched; both are verified
against synthetic listings in `tests/text_features/test_keywords.py`.

---

## 7. Integration contract for the baseline pipeline

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

## 8. Reproducing

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

## 9. Limitations and next steps

- The keyword lexicon is hand-built from observed prevalence, not learned. It
  covers the required concepts well but will miss paraphrase; a phrase-mining
  pass over high-error residuals is the obvious extension.
- `text_price_vnd` agrees with the structured `price` on only ~74% of rows
  (5% tolerance), versus 86–95% for the physical attributes. Likely causes are
  price ranges, per-m² quotes and multi-unit listings; it needs work before it
  can be trusted for the null-price repair use case.
- **PhoBERT was measured and lost** (§6.3): as frozen mean-pooled features it
  makes RMSLE, MAPE and R² all *worse* than TF-IDF alone, at ~30× the compute
  per row. This is a statement about *pooled, unfine-tuned* embeddings on ~5k
  listings — not about PhoBERT in general. A fine-tuned encoder, or embeddings
  computed once and cached, is the fair next comparison; do not pay for pooled
  PhoBERT by default.
- The TF-IDF representation is capacity-limited, not saturated: `max_features`
  hits its 60,000 cap and 128 SVD components explain only ~13% of word-view
  variance. More components or a larger vocabulary is untested headroom.
- Only two of the ten shards are used. A walk-forward across all ten would give
  tighter error bars and would test whether the text gain holds as the market
  moves further from the training window.
- `kw_cho_thue` is a blunt instrument. It reliably reports that renting is
  *mentioned*, but separating an actual rental listing from a sale pitched on
  yield needs more than a phrase match — combining the marker with a monthly-rate
  price band, or with `property_type_name`, would make it a classifier. Until
  then treat it as a feature only (§5, item 2).
