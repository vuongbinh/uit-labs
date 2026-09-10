## Text-feature uplift benchmark

- Train: 55,713 clean rows of 60,000 (2025-06-01 → 2025-07-30)
- Test (out-of-time): 22,557 clean rows of 25,000 (2026-03-01 → 2026-03-30)
- Runtime: 454.6s

| arm | feats | RMSLE | MAPE % | MAE (tỷ) | MedAE (tỷ) | R² |
|---|---|---|---|---|---|---|
| tabular | 23 | 0.4144 | 30.34 | 4.196 | 1.504 | 0.7283 |
| +keywords | 70 | 0.3943 | 28.58 | 3.989 | 1.416 | 0.7439 |
| +entities | 85 | 0.3802 | 27.54 | 3.826 | 1.375 | 0.7609 |
| +tfidf | 341 | 0.3710 | 27.07 | 3.853 | 1.394 | 0.7522 |

### Marginal gain vs tabular-only

| arm | ΔRMSLE | ΔRMSLE % | ΔMAPE (pts) | ΔMAE (tỷ) | ΔR² |
|---|---|---|---|---|---|
| +keywords | -0.0201 | -4.85% | -1.76 | -0.207 | +0.0156 |
| +entities | -0.0342 | -8.24% | -2.80 | -0.370 | +0.0327 |
| +tfidf | -0.0434 | -10.47% | -3.27 | -0.343 | +0.0240 |

### Entity extraction agreement (train rows)

| structured | text entity | text coverage | col missing | overlap | agreement | recovered |
|---|---|---|---|---|---|---|
| area | text_area_m2 | 0.785 | 0.000 | 43,714 | 0.926 | 0 |
| frontage_width | text_frontage_m | 0.305 | 0.459 | 11,449 | 0.885 | 5,522 |
| house_depth | text_depth_m | 0.305 | 0.970 | 824 | 0.955 | 16,147 |
| floor_count | text_floor_count | 0.400 | 0.827 | 6,748 | 0.919 | 15,544 |
| bedroom_count | text_bedroom_count | 0.399 | 0.476 | 19,599 | 0.881 | 2,630 |
| bathroom_count | text_bathroom_count | 0.258 | 0.510 | 12,826 | 0.929 | 1,570 |
| road_width | text_road_width_m | 0.087 | 0.892 | 894 | 0.866 | 3,940 |
| price | text_price_vnd | 0.847 | 0.000 | 47,171 | 0.737 | 0 |

### Top-15 gain-importance features (richest arm)

- `tgt_district_name` — 21.57% of total gain
- `area` — 20.63% of total gain
- `tfidf_w_5` — 8.37% of total gain
- `tgt_ward_name` — 8.25% of total gain
- `bedroom_count` — 3.89% of total gain
- `freq_district_name` — 3.24% of total gain
- `frontage_width` — 2.91% of total gain
- `tgt_province_name` — 2.67% of total gain
- `floor_count` — 2.44% of total gain
- `freq_property_type_name` — 2.31% of total gain
- `text_area_m2` — 2.28% of total gain
- `tfidf_w_1` — 1.16% of total gain
- `freq_province_name` — 1.16% of total gain
- `bathroom_count` — 1.06% of total gain
- `freq_ward_name` — 0.98% of total gain
