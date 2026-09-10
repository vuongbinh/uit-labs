## Text-feature uplift benchmark

- Train: 46,815 clean rows of 60,000 (2025-06-01 → 2025-07-30)
- Test (out-of-time): 18,624 clean rows of 25,000 (2026-03-01 → 2026-03-30)
- Runtime: 346.4s

| arm | feats | RMSLE | MAPE % | MAE (tỷ) | MedAE (tỷ) | R² |
|---|---|---|---|---|---|---|
| tabular | 23 | 0.4618 | 33.01 | 6.651 | 1.477 | 0.0694 |
| +keywords | 70 | 0.4454 | 32.02 | 6.460 | 1.414 | 0.0760 |
| +entities | 85 | 0.4400 | 31.27 | 6.376 | 1.381 | 0.0749 |
| +tfidf | 341 | 0.4219 | 30.10 | 6.415 | 1.390 | 0.0686 |

### Marginal gain vs tabular-only

| arm | ΔRMSLE | ΔRMSLE % | ΔMAPE (pts) | ΔMAE (tỷ) | ΔR² |
|---|---|---|---|---|---|
| +keywords | -0.0164 | -3.55% | -0.98 | -0.191 | +0.0066 |
| +entities | -0.0218 | -4.72% | -1.74 | -0.274 | +0.0055 |
| +tfidf | -0.0399 | -8.64% | -2.91 | -0.235 | -0.0008 |

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

- `area` — 21.75% of total gain
- `tgt_district_name` — 19.68% of total gain
- `tfidf_w_5` — 10.23% of total gain
- `tgt_ward_name` — 7.93% of total gain
- `freq_district_name` — 5.17% of total gain
- `bedroom_count` — 2.78% of total gain
- `frontage_width` — 2.60% of total gain
- `tgt_province_name` — 2.33% of total gain
- `freq_property_type_name` — 2.08% of total gain
- `freq_province_name` — 1.56% of total gain
- `floor_count` — 1.37% of total gain
- `tfidf_w_1` — 1.33% of total gain
- `freq_ward_name` — 1.15% of total gain
- `text_area_m2` — 1.12% of total gain
- `bathroom_count` — 0.91% of total gain
