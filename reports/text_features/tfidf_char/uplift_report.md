## Text-feature uplift benchmark

- Train: 46,815 clean rows of 60,000 (2025-06-01 → 2025-07-30)
- Test (out-of-time): 18,624 clean rows of 25,000 (2026-03-01 → 2026-03-30)
- Runtime: 305.7s

| arm | feats | RMSLE | MAPE % | MAE (tỷ) | MedAE (tỷ) | R² |
|---|---|---|---|---|---|---|
| tabular | 23 | 0.4618 | 33.01 | 6.651 | 1.477 | 0.0694 |
| +keywords | 70 | 0.4454 | 32.02 | 6.460 | 1.414 | 0.0760 |
| +entities | 85 | 0.4400 | 31.27 | 6.376 | 1.381 | 0.0749 |
| +tfidf | 213 | 0.4270 | 30.63 | 6.428 | 1.390 | 0.0745 |

### Marginal gain vs tabular-only

| arm | ΔRMSLE | ΔRMSLE % | ΔMAPE (pts) | ΔMAE (tỷ) | ΔR² |
|---|---|---|---|---|---|
| +keywords | -0.0164 | -3.55% | -0.98 | -0.191 | +0.0066 |
| +entities | -0.0218 | -4.72% | -1.74 | -0.274 | +0.0055 |
| +tfidf | -0.0349 | -7.55% | -2.38 | -0.223 | +0.0051 |

### Entity extraction agreement (train rows)

| structured | text entity | text coverage | col missing | overlap | agreement | recovered |
|---|---|---|---|---|---|---|
| area | text_area_m2 | 0.791 | 0.000 | 37,019 | 0.924 | 0 |
| frontage_width | text_frontage_m | 0.297 | 0.458 | 9,544 | 0.883 | 4,345 |
| house_depth | text_depth_m | 0.297 | 0.964 | 865 | 0.953 | 13,024 |
| floor_count | text_floor_count | 0.369 | 0.839 | 5,259 | 0.920 | 12,008 |
| bedroom_count | text_bedroom_count | 0.388 | 0.489 | 16,353 | 0.884 | 1,808 |
| bathroom_count | text_bathroom_count | 0.256 | 0.525 | 10,837 | 0.935 | 1,149 |
| road_width | text_road_width_m | 0.086 | 0.889 | 754 | 0.850 | 3,256 |
| price | text_price_vnd | 0.835 | 0.000 | 39,101 | 0.749 | 0 |

### Top-15 gain-importance features (richest arm)

- `tgt_district_name` — 23.67% of total gain
- `area` — 20.61% of total gain
- `tgt_ward_name` — 7.46% of total gain
- `tfidf_c_5` — 5.98% of total gain
- `frontage_width` — 4.79% of total gain
- `freq_district_name` — 4.34% of total gain
- `bedroom_count` — 3.21% of total gain
- `freq_property_type_name` — 2.66% of total gain
- `tgt_province_name` — 2.07% of total gain
- `freq_province_name` — 2.06% of total gain
- `floor_count` — 1.68% of total gain
- `text_area_m2` — 1.50% of total gain
- `bathroom_count` — 1.31% of total gain
- `freq_ward_name` — 1.29% of total gain
- `implied_depth_from_area` — 0.77% of total gain
