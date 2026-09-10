## Text-feature uplift benchmark

- Train: 4,642 clean rows of 5,000 (2025-06-01 → 2025-07-30)
- Test (out-of-time): 2,254 clean rows of 2,500 (2026-03-01 → 2026-03-30)
- Runtime: 787.3s

| arm | feats | RMSLE | MAPE % | MAE (tỷ) | MedAE (tỷ) | R² |
|---|---|---|---|---|---|---|
| tabular | 23 | 0.4989 | 39.90 | 4.973 | 1.823 | 0.6815 |
| +keywords | 70 | 0.4850 | 38.86 | 4.789 | 1.780 | 0.6934 |
| +entities | 85 | 0.4815 | 38.72 | 4.579 | 1.717 | 0.7173 |
| +tfidf | 213 | 0.4591 | 36.09 | 4.586 | 1.700 | 0.7063 |
| +phobert | 277 | 0.4624 | 36.35 | 4.750 | 1.697 | 0.6825 |

### Marginal gain vs tabular-only

| arm | ΔRMSLE | ΔRMSLE % | ΔMAPE (pts) | ΔMAE (tỷ) | ΔR² |
|---|---|---|---|---|---|
| +keywords | -0.0139 | -2.79% | -1.03 | -0.184 | +0.0119 |
| +entities | -0.0174 | -3.48% | -1.17 | -0.394 | +0.0358 |
| +tfidf | -0.0398 | -7.97% | -3.81 | -0.387 | +0.0248 |
| +phobert | -0.0365 | -7.31% | -3.55 | -0.223 | +0.0010 |

### Entity extraction agreement (train rows)

| structured | text entity | text coverage | col missing | overlap | agreement | recovered |
|---|---|---|---|---|---|---|
| area | text_area_m2 | 0.791 | 0.000 | 3,671 | 0.931 | 0 |
| frontage_width | text_frontage_m | 0.296 | 0.450 | 923 | 0.887 | 450 |
| house_depth | text_depth_m | 0.296 | 0.969 | 72 | 0.958 | 1,300 |
| floor_count | text_floor_count | 0.400 | 0.818 | 587 | 0.917 | 1,272 |
| bedroom_count | text_bedroom_count | 0.386 | 0.489 | 1,561 | 0.873 | 231 |
| bathroom_count | text_bathroom_count | 0.250 | 0.522 | 1,029 | 0.924 | 133 |
| road_width | text_road_width_m | 0.089 | 0.883 | 79 | 0.835 | 334 |
| price | text_price_vnd | 0.847 | 0.000 | 3,930 | 0.748 | 0 |

### Top-15 gain-importance features (richest arm)

- `area` — 22.14% of total gain
- `tgt_district_name` — 18.04% of total gain
- `tgt_ward_name` — 11.78% of total gain
- `freq_district_name` — 4.63% of total gain
- `bedroom_count` — 4.11% of total gain
- `tfidf_w_6` — 2.94% of total gain
- `floor_count` — 2.80% of total gain
- `tgt_province_name` — 2.61% of total gain
- `tfidf_w_5` — 2.49% of total gain
- `text_area_m2` — 2.20% of total gain
- `frontage_width` — 1.91% of total gain
- `freq_ward_name` — 1.67% of total gain
- `freq_property_type_name` — 1.28% of total gain
- `bathroom_count` — 1.04% of total gain
- `tfidf_w_1` — 1.00% of total gain
