"""Generator for notebooks/vietnam_real_estate_text_features.ipynb.

Kept in the repo so the notebook can be regenerated / diffed as plain Python,
matching notebooks/_build_notebook.py.
Run:  python notebooks/_build_text_features_notebook.py

The notebook is **standalone**: the "Bundled library" cells below inline the
reachable subset of ``real_estate/text/`` verbatim (this script reads those
modules at build time and strips only the package plumbing), so the notebook
runs on a bare kernel with no project checkout and no ``real_estate`` import.
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEXT_PKG = REPO / "real_estate" / "text"

cells = []


def _strip_module(name: str) -> str:
    """Load ``real_estate/text/<name>.py`` and remove only the package plumbing.

    Dropped: the module docstring, ``from __future__`` imports, intra-package
    relative imports (``from .x import ...``), ``__all__`` assignments and the
    ``if __name__ == "__main__"`` block. Every definition is kept byte-for-byte
    so the notebook and the package cannot drift.
    """
    src = (TEXT_PKG / f"{name}.py").read_text(encoding="utf-8")
    # Drop the leading module docstring.
    src = re.sub(r'\A\s*""".*?"""\n', "", src, count=1, flags=re.DOTALL)
    # Drop the `if __name__ == "__main__":` block (to EOF).
    src = re.sub(r"\nif __name__ == .__main__.:.*\Z", "\n", src, flags=re.DOTALL)
    # Drop `__all__ = [ ... ]` (single- or multi-line).
    src = re.sub(r"\n__all__\s*=\s*\[.*?\]\n", "\n", src, flags=re.DOTALL)
    # Drop intra-package relative imports, parenthesised multi-line form first.
    src = re.sub(r"^from \.\S* import \([^)]*\)\n", "", src, flags=re.MULTILINE | re.DOTALL)
    src = re.sub(r"^from \.\S* import .*\n", "", src, flags=re.MULTILINE)
    src = re.sub(r"^from __future__ import .*\n", "", src, flags=re.MULTILINE)
    return src.strip("\n")


def library_cell(cid: str, header: str, modules: list[str]) -> None:
    """Emit one 'Bundled library' code cell concatenating several stripped modules."""
    body = [
        "from __future__ import annotations",
        "",
        f"# == Bundled library: {header} ==",
        "# Copied verbatim from real_estate/text/ by notebooks/_build_text_features_notebook.py.",
        "# Edit the source modules there, not here, then regenerate the notebook.",
    ]
    for module in modules:
        body += ["", f"# ---- real_estate/text/{module}.py " + "-" * (58 - len(module)), ""]
        body.append(_strip_module(module))
    code(cid, "\n".join(body))


def md(cid: str, text: str):
    cells.append({
        "cell_type": "markdown",
        "id": cid,
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    })


def code(cid: str, text: str):
    cells.append({
        "cell_type": "code",
        "id": cid,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


md("a1", r"""
# Text Feature Extraction for Vietnamese Real-Estate Listings

Turns the unstructured `name` / `description` fields of
[`tinixai/vietnam-real-estates`](https://huggingface.co/datasets/tinixai/vietnam-real-estates)
into model-ready features, and measures how much they add to a tabular price baseline.

**Three feature families**

1. **Domain keyword flags** (`kw_*`) — legal status (`sổ đỏ`, `sổ hồng`, `chính chủ`, `pháp lý rõ ràng`, `đang chờ sổ`), road accessibility (`ô tô đỗ cửa`, `mặt tiền`, `ngõ thông`, `xe hơi vào nhà`, `ngõ ba gác`) and condition/interior (`full nội thất`, `nội thất cơ bản`, `nhà mới`, `nhà cấp 4`). Matched diacritic-insensitively, with a negation guard so `không tranh chấp` does not read as a legal risk.
2. **Numeric entities** (`text_*`) — area, frontage × depth, floor count, bedrooms, bathrooms, road width and the advertised price, parsed out of free text and used to *impute* the heavily-sparse structured columns.
3. **Text representations** (`tfidf_*`, `phobert_*`) — TF-IDF + Truncated SVD by default, mean-pooled PhoBERT as an opt-in heavier path.

**Evaluation is out-of-time.** The parquet shards are chronologically ordered, so this trains on `shard_0000` (June 2025) and tests on `shard_0009` (March 2026) — a nine-month gap with no look-ahead.

Runtime: ~15 min on a Colab CPU runtime at the default sample sizes.
""")

md("a2", r"""
## 0. Setup
""")

code("a3", r"""
%pip install -q lightgbm scikit-learn pandas pyarrow numpy
""")

md("a3a", r"""
### 0.1 Bundled library

The four cells below inline the reachable subset of the project's
`real_estate/text/` package — text normalisation, the domain lexicon, entity
parsing, the TF-IDF / PhoBERT encoders, data loading and the uplift benchmark.
They are copied **verbatim** from those modules by
`notebooks/_build_text_features_notebook.py`, so this notebook runs on a bare
kernel with no project checkout and nothing to `pip install` from Git. Run them
once, top to bottom, then collapse the section.
""")

library_cell("a3b", "text normalisation, domain lexicon, keyword flags",
             ["normalize", "lexicon", "keywords"])
library_cell("a3c", "numeric entity parsing and dense text representations",
             ["entities", "representation"])
library_cell("a3d", "dataset loading / cleaning and the feature pipeline",
             ["data", "pipeline"])
library_cell("a3e", "marginal-uplift benchmark (reference tabular baseline + LightGBM)",
             ["benchmark"])

code("a4", r"""
import numpy as np
import pandas as pd

# ENTITY_COLUMNS, TARGET_LEAKING_ENTITIES, KeywordExtractor, TextFeatureConfig,
# TextFeaturePipeline, TfidfTextEncoder, clean_frame, extract_entities,
# load_shard_sample, parse_vn_number, prepare_for_tfidf, time_span, resolve_shard,
# NUMERIC_COLUMNS, impute_from_text, BenchmarkConfig, entity_agreement_report,
# format_report, run_uplift_benchmark, TabularEncoder
# are all defined by the "Bundled library" cells above.

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 60)
""")

md("a5", r"""
## 1. Load a sample and check the data before modelling

`resolve_shard` downloads on first use and caches under `DATA_DIR`.
""")

code("a6", r"""
DATA_DIR = "data"
TRAIN_SHARD, TEST_SHARD = "shard_0000", "shard_0009"

raw_train = load_shard_sample(TRAIN_SHARD, n_rows=60_000, seed=0, cache_dir=DATA_DIR)
raw_test = load_shard_sample(TEST_SHARD, n_rows=25_000, seed=1, cache_dir=DATA_DIR)
print("train", len(raw_train), time_span(raw_train))
print("test ", len(raw_test), time_span(raw_test))
raw_train.head(3)
""")

md("a7", r"""
### Three things the raw data does not tell you

Each of these was verified on the published shards and each one changes how you should model the target.
""")

code("a8", r"""
import pyarrow.parquet as pq

schema = pq.ParquetFile(resolve_shard(TRAIN_SHARD, cache_dir=DATA_DIR)).schema_arrow
print("1) `price` is declared float64 on the dataset card but stored as", schema.field("price").type)
print("   parse_price() converts it; non-numeric values become NaN, never a wrong number.\n")

flags = KeywordExtractor().extract(raw_train)
mentions = flags["kw_cho_thue"]
p = raw_train["price"]
print(f"2) {mentions.mean():.1%} of rows MENTION 'cho thuê' -- that does not make them rentals:")
print(f"   median price, mentioning rows : {p[mentions].median():,.0f} VND")
print(f"   median price, other rows      : {p[~mentions].median():,.0f} VND")
print(f"   share of mentioning rows at sale scale (>= 500M VND): {(p[mentions] >= 5e8).mean():.1%}")
print("   -> kw_cho_thue is a FEATURE, not a filter. Real rentals quote a monthly rate")
print("      (15-60M VND) and are excluded by clean_frame's price floor instead.\n")

clean = clean_frame(raw_train)
print(f"3) clean_frame applies the EDA's Rule 1 (price 100M-200B, area 15-3000 m2,")
print(f"   unit price 3M-800M/m2): kept {len(clean):,}/{len(raw_train):,} = {len(clean)/len(raw_train):.2%}\n")

sparse = raw_train[["floor_count", "frontage_width", "house_depth", "road_width",
                    "bedroom_count", "bathroom_count"]].isna().mean().sort_values(ascending=False)
print("4) structured columns are heavily sparse -- but the text usually states these values:")
print(sparse.map(lambda v: f"{v:.1%}").to_string())
print(f"\nraw price range on this shard: {p.min():,.0f} .. {p.max():,.0f} VND")
""")

md("a9", r"""
## 2. Domain keyword flags

Patterns run against diacritic-folded text, so `"Ô Tô Đỗ Cửa"`, `"ô tô đỗ cửa"`, `"ôtô đỗ cổng"` and `"o to do cua"` all match the same rule.

The prevalence table below doubles as a lexicon health check: a rule that never fires on real data is dead weight, and the test suite fails if any pattern contains a literal diacritic (which could never match folded text).
""")

code("b1", r"""
extractor = KeywordExtractor()
kw_train = extractor.extract(raw_train)
print(f"{kw_train.shape[1]} flags, {int((kw_train.sum(axis=1) == 0).mean() * 100)}% of rows match nothing\n")
extractor.prevalence(kw_train).head(28)
""")

code("b2", r"""
# Spot-check the phrases named in the requirements.
demo = pd.DataFrame({"description": [
    "Bán nhà mặt tiền, sổ đỏ chính chủ, ô tô đỗ cửa, pháp lý rõ ràng",
    "Nhà trong ngõ ba gác, nội thất cơ bản, đang chờ sổ",
    "Không tranh chấp, không quy hoạch, xe hơi vào nhà, full nội thất",
    "Nhà cấp 4 cũ, hẻm nhỏ",
]})
cols = ["kw_so_do", "kw_chinh_chu", "kw_mat_tien", "kw_o_to_do_cua", "kw_ngo_ba_gac",
        "kw_xe_hoi_vao_nha", "kw_noi_that_co_ban", "kw_dang_cho_so", "kw_nha_cap_4",
        "kw_tranh_chap", "kw_quy_hoach", "kw_khong_quy_hoach"]
KeywordExtractor().extract(demo)[cols].assign(description=demo["description"])
""")

md("b3", r"""
Row 3 is the reason the negation guard exists: `không tranh chấp` / `không quy hoạch` advertise the *absence* of an encumbrance, so `kw_tranh_chap` and `kw_quy_hoach` stay `False` while `kw_khong_quy_hoach` turns on.

Note that `kw_cho_thue` is a **mention** flag, not a listing-type classifier — see §1. Do not use it to filter rows.
""")

md("b4", r"""
## 3. Numeric entities and text-driven imputation

`extract_entities` parses the numbers listings state in prose. Because the structured columns are sparse, most of the value here is **recovery**: filling `house_depth`, `floor_count` and `road_width` where the source record has nothing.
""")

code("b5", r"""
ent_train = extract_entities(raw_train)
ent_train.describe().T[["count", "mean", "50%"]]
""")

code("b6", r"""
# Validation: where the structured column and the parsed entity both exist, do they agree?
entity_agreement_report(raw_train, ent_train)
""")

md("b7", r"""
Reading the table: `text_coverage` is how often the text yields a value, `col missing` is how often the structured column is empty, `agreement` is exact-match rate where both exist, and `recovered` is the number of previously-null cells the text fills.

`N lầu` is deliberately kept in its own `text_lau_count` column instead of being added to `floor_count`: southern listings use it both for "levels above ground" and "levels total", so converting it would bake in an unfounded assumption.
""")

code("b8", r"""
# Vietnamese number parsing: both separators, and 3-digit groups read as thousands.
for token in ["4", "7.45", "1,3", "1.300", "1.300.000"]:
    print(f"  {token:>12}  ->  {parse_vn_number(token):,.2f}")
""")

md("b9", r"""
## 4. Why `text_price_vnd` is measured but never modelled

The advertised price is written into the listing itself (`"giá: 7.45 tỷ"`), so parsing it back out and feeding it to a model trained on `price` is circular — it scores transcription, not valuation. In a first run this single column took **57% of total split gain** and flattered the whole entities arm by ~26% RMSLE.

Two controls are in place:

- `TARGET_LEAKING_ENTITIES` is excluded from every model feature block.
- Price mentions are **redacted from the TF-IDF/PhoBERT input** too, otherwise digit n-grams smuggle the same information back in.

The cell below is the direct check: two identical listings quoting different prices must encode *identically* once redaction is on.
""")

code("c1", r"""
BASE = "Nhà mặt tiền sổ đỏ chính chủ, ô tô đỗ cửa, diện tích 40m2, 3 phòng ngủ"
print("redacted :", prepare_for_tfidf(f"{BASE}, giá 5.5 tỷ"))
print("unredacted:", prepare_for_tfidf(f"{BASE}, giá 5.5 tỷ", redact_price=False))

demo_frame = pd.DataFrame({"name": ["A", "B"], "description": [
    f"{BASE}, giá 5.5 tỷ", f"{BASE}, giá 7.25 tỷ"]})
for redact in (True, False):
    cfg = TextFeatureConfig(use_keywords=False, use_entities=False,
                            tfidf_components=2, tfidf_min_df=1, redact_price=redact)
    vec = TextFeaturePipeline(cfg).fit_transform(demo_frame).to_numpy(dtype=float)
    identical = np.allclose(vec[0], vec[1], atol=1e-6)
    print(f"redact_price={redact!s:5} -> price-only difference visible to TF-IDF: {not identical}")
""")

md("c2", r"""
## 5. TF-IDF representation

Vietnamese is syllable-delimited, so word **bigrams** are what capture domain phrases like `sổ đỏ` and `mặt tiền`. The character view is a robustness alternative; §7 compares them.

The encoder is fitted on training text only and reused unchanged on the test period.
""")

code("c3", r"""
corpus = [prepare_for_tfidf(t) for t in
          (raw_train["name"].fillna("") + " " + raw_train["description"].fillna(""))]

tfidf = TfidfTextEncoder(mode="word", n_components=32, min_df=10).fit(corpus[:20_000])
print("vocabulary:", tfidf.vocabulary_sizes)
print("explained variance:", {k: round(v, 3) for k, v in tfidf.explained_variance.items()})

for component in (0, 1):
    terms = ", ".join(t for t, _ in tfidf.top_terms("w", component, k=10))
    print(f"\ncomponent {component}: {terms}")
""")

md("c4", r"""
## 6. Marginal uplift benchmark

One reference tabular block (numerics, ratios, frequency + smoothed target encodings of geography) trained with LightGBM on `log1p(price)`, then the same block plus each text family in turn. Every arm is a superset of the previous one, so each row of the delta table is a **marginal** gain.

Early stopping uses the latest 15% of the *training* period, never the test period. Two seeds per arm, because an uplift smaller than the seed-to-seed spread is not a finding.

The tabular block here is a reference harness, not the project baseline — `run_uplift_benchmark` accepts a pre-built matrix so the HYPE-12 pipeline can be dropped in.
""")

code("c5", r"""
config = BenchmarkConfig(
    n_train=60_000, n_test=25_000, n_seeds=2,
    tfidf_mode="word", tfidf_components=128, tfidf_min_df=5,
    exclude_target_leaking_entities=True, redact_price_in_text=True,
)
report = run_uplift_benchmark(TRAIN_SHARD, TEST_SHARD, config=config, cache_dir=DATA_DIR)
print(format_report(report))
""")

md("c6", r"""
## 7. Comparing TF-IDF analyzers

Word bigrams against character 3–5-grams against both concatenated, at matched sample sizes.
""")

code("c7", r"""
rows = []
for mode in ("word", "char", "both"):
    cfg = BenchmarkConfig(n_train=40_000, n_test=20_000, n_seeds=1,
                          tfidf_mode=mode, tfidf_components=128)
    rep = run_uplift_benchmark(TRAIN_SHARD, TEST_SHARD, config=cfg, cache_dir=DATA_DIR)
    for arm in ("tabular", "+tfidf"):
        rows.append({"mode": mode, "arm": arm, **rep["arms"][arm]["metrics"]})

comp = pd.DataFrame(rows).drop(columns=["n"])
comp
""")

code("c8", r"""
gain = (comp[comp.arm == "+tfidf"].set_index("mode")["rmsle"]
        - comp[comp.arm == "tabular"].set_index("mode")["rmsle"])
print("RMSLE change from adding TF-IDF, by analyzer (negative is better):")
print(gain.map(lambda v: f"{v:+.4f}").to_string())
""")

md("c9", r"""
## 8. Optional: PhoBERT embeddings

Heavier and opt-in. `torch`/`transformers` are imported lazily, so nothing above needs them. Mean-pooled `vinai/phobert-base`, reduced with SVD, on a subsample — CPU encoding is the bottleneck.

Diacritics are preserved on this path (unlike TF-IDF) because PhoBERT's vocabulary distinguishes them; price mentions are still redacted.
""")

code("d1", r"""
%pip install -q torch --index-url https://download.pytorch.org/whl/cpu
%pip install -q transformers

cfg = BenchmarkConfig(
    n_train=4_000, n_test=2_000, n_seeds=1,
    tfidf_mode="word", tfidf_components=128, use_phobert=True,
    phobert={"n_components": 64, "max_length": 128, "batch_size": 16},
)
phobert_report = run_uplift_benchmark(TRAIN_SHARD, TEST_SHARD, config=cfg, cache_dir=DATA_DIR)
print(format_report(phobert_report))
""")

md("d2", r"""
## 9. Using these features in the baseline pipeline

Fit the pipeline on training rows only, transform both sides, and column-bind onto the tabular block. The index contract is what keeps this safe: `transform` returns rows aligned to the input frame, and imputation must happen **before** tabular encoding so derived ratios use the recovered numbers.
""")

code("d3", r"""
# NUMERIC_COLUMNS, impute_from_text and TabularEncoder come from the bundled
# library cells in section 0.1.

# Clean with the EDA's Rule 1 bounds. No rental flag: kw_cho_thue marks a mention,
# and the price floor is what actually excludes monthly-rate rental listings.
train_df = clean_frame(raw_train)
test_df = clean_frame(raw_test)
print("clean rows:", len(train_df), len(test_df))

pipeline = TextFeaturePipeline(TextFeatureConfig(
    tfidf_mode="word", tfidf_components=128, tfidf_min_df=5,
    redact_price=True,          # keep the regression target out of the text features
))
train_features = pipeline.fit_transform(train_df)   # fits TF-IDF/SVD on TRAIN only
test_features = pipeline.transform(test_df)         # reuses that fitted state

# 1. Impute the sparse structured columns from text, THEN encode the tabular
#    block, so derived ratios use the recovered numbers rather than the nulls.
numeric = [c for c in NUMERIC_COLUMNS if c in train_df.columns]
train_imp, test_imp = train_df.copy(), test_df.copy()
train_imp[numeric] = impute_from_text(train_df[numeric], train_features[ENTITY_COLUMNS])[numeric]
test_imp[numeric] = impute_from_text(test_df[numeric], test_features[ENTITY_COLUMNS])[numeric]

encoder = TabularEncoder().fit(train_imp)
tabular_train = encoder.transform(train_imp)
tabular_test = encoder.transform(test_imp)

# 2. Bind the text features on, minus the target-restating entity.
drop = [c for c in TARGET_LEAKING_ENTITIES if c in train_features.columns]
X_train = pd.concat([tabular_train, train_features.drop(columns=drop)], axis=1)
X_test = pd.concat([tabular_test, test_features.drop(columns=drop)], axis=1)

assert list(X_train.columns) == list(X_test.columns)
assert len(X_train) == len(train_df) and len(X_test) == len(test_df)
print("tabular:", tabular_train.shape[1], "-> with text:", X_train.shape[1], "features")
""")

md("d4", r"""
## 10. Takeaways

Measured out-of-time (train 2025-06 → test 2026-03). Full tables in `reports/text_features/` and `docs/real_estate_text_features.md`.

- **Cumulative text gain over the tabular baseline: −0.0400 RMSLE (−10.07%), −2.67 MAPE points, +0.028 R²** on 92,855 train / 36,118 test rows. Ladder: tabular 0.3973 → +keywords 0.3765 → +entities 0.3678 → +tfidf **0.3573**. Every arm improves every metric. Seed spread is ≤ 0.0006 RMSLE, so each step is 30–100× the noise.
- **The biggest win is entity recovery, not bag-of-words.** `house_depth` is 97.0% null in the source and text fills 26,841 cells; `floor_count` is 82.7% null and text fills 26,179. Parsed values agree with the recorded ones in 86–95% of overlapping rows.
- **Keyword flags are cheap and additive** (−5.22% RMSLE) but individually weak: geographic target encodings already absorb much of the same signal.
- **Word bigrams beat character n-grams.** At matched scale (55,713/22,557, same tabular baseline 0.4144): word 0.3694, both 0.3710, char 0.3774. Concatenating the char view does *not* beat word alone despite 60% more features.
- **PhoBERT was measured and lost.** Frozen mean-pooled embeddings make RMSLE, MAPE and R² all *worse* than TF-IDF alone, at ~30× the compute per row. Worth revisiting fine-tuned; do not pay for pooled PhoBERT by default.
- **`cho thuê` in the text is a mention, not a listing type.** It fires on 18.8% of rows, but 96% of those are sales at a sale-scale median price (10 tỷ) pitching rental yield — "sẵn hợp đồng thuê". Use it as a feature; never as a row filter. Real rentals quote a monthly rate and are removed by the price floor.
- **Leakage is the main hazard of text features here.** The asking price is in the prose; unredacted it took 57% of total split gain. Redact it or the benchmark measures transcription, not valuation.
- **Clean with the EDA's Rule 1** (price 100M–200B, area 15–3000 m², unit price 3M–800M/m²). That retains ~93% of rows and lifts raw-scale R² to ~0.75; looser bounds leave an outlier tail past 800 trillion VND that crushes R² to ~0.05 and makes MAE meaningless.
""")

nb = {
    "cells": cells,
    "metadata": {
        "colab": {
            "name": "vietnam_real_estate_text_features.ipynb",
            "provenance": [],
            "toc_visible": True,
        },
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "vietnam_real_estate_text_features.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {out}  ({len(cells)} cells)")
