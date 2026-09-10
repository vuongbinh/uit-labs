## Text-feature uplift benchmark

- Train: 92,855 clean rows of 100,000 (2025-06-01 → 2025-07-30)
- Test (out-of-time): 36,118 clean rows of 40,000 (2026-03-01 → 2026-03-30)
- Runtime: 527.4s

| arm | feats | RMSLE | MAPE % | MAE (tỷ) | MedAE (tỷ) | R² |
|---|---|---|---|---|---|---|
| tabular | 23 | 0.3973 | 28.59 | 3.998 | 1.428 | 0.7470 |
| +keywords | 70 | 0.3765 | 27.01 | 3.801 | 1.366 | 0.7635 |
| +entities | 85 | 0.3678 | 26.33 | 3.723 | 1.341 | 0.7665 |
| +tfidf | 213 | 0.3573 | 25.92 | 3.679 | 1.330 | 0.7754 |

### Marginal gain vs tabular-only

| arm | ΔRMSLE | ΔRMSLE % | ΔMAPE (pts) | ΔMAE (tỷ) | ΔR² |
|---|---|---|---|---|---|
| +keywords | -0.0207 | -5.22% | -1.58 | -0.197 | +0.0166 |
| +entities | -0.0294 | -7.41% | -2.26 | -0.276 | +0.0195 |
| +tfidf | -0.0400 | -10.07% | -2.67 | -0.320 | +0.0284 |

### Entity extraction agreement (train rows)

| structured | text entity | text coverage | col missing | overlap | agreement | recovered |
|---|---|---|---|---|---|---|
| area | text_area_m2 | 0.783 | 0.000 | 72,733 | 0.926 | 0 |
| frontage_width | text_frontage_m | 0.304 | 0.459 | 19,065 | 0.885 | 9,179 |
| house_depth | text_depth_m | 0.304 | 0.970 | 1,408 | 0.953 | 26,841 |
| floor_count | text_floor_count | 0.402 | 0.827 | 11,187 | 0.916 | 26,179 |
| bedroom_count | text_bedroom_count | 0.399 | 0.477 | 32,595 | 0.884 | 4,486 |
| bathroom_count | text_bathroom_count | 0.258 | 0.511 | 21,329 | 0.930 | 2,630 |
| road_width | text_road_width_m | 0.087 | 0.892 | 1,447 | 0.858 | 6,597 |
| price | text_price_vnd | 0.848 | 0.000 | 78,719 | 0.737 | 0 |

### Top-15 gain-importance features (richest arm)

- `area` — 21.81% of total gain
- `tgt_district_name` — 19.93% of total gain
- `tgt_ward_name` — 9.79% of total gain
- `tfidf_w_5` — 9.12% of total gain
- `freq_district_name` — 3.70% of total gain
- `tgt_province_name` — 3.61% of total gain
- `bedroom_count` — 3.10% of total gain
- `floor_count` — 2.87% of total gain
- `freq_property_type_name` — 2.32% of total gain
- `text_area_m2` — 2.27% of total gain
- `frontage_width` — 2.18% of total gain
- `bathroom_count` — 1.52% of total gain
- `tfidf_w_1` — 1.48% of total gain
- `freq_province_name` — 1.19% of total gain
- `freq_ward_name` — 0.98% of total gain
