## Text-feature uplift benchmark

- Train: 55,713 clean rows of 60,000 (2025-06-01 → 2025-07-30)
- Test (out-of-time): 22,557 clean rows of 25,000 (2026-03-01 → 2026-03-30)
- Runtime: 375.7s

| arm | feats | RMSLE | MAPE % | MAE (tỷ) | MedAE (tỷ) | R² |
|---|---|---|---|---|---|---|
| tabular | 23 | 0.4144 | 30.34 | 4.196 | 1.504 | 0.7283 |
| +keywords | 70 | 0.3943 | 28.58 | 3.989 | 1.416 | 0.7439 |
| +entities | 85 | 0.3802 | 27.54 | 3.826 | 1.375 | 0.7609 |
| +tfidf | 213 | 0.3774 | 27.51 | 3.889 | 1.385 | 0.7537 |

### Marginal gain vs tabular-only

| arm | ΔRMSLE | ΔRMSLE % | ΔMAPE (pts) | ΔMAE (tỷ) | ΔR² |
|---|---|---|---|---|---|
| +keywords | -0.0201 | -4.85% | -1.76 | -0.207 | +0.0156 |
| +entities | -0.0342 | -8.24% | -2.80 | -0.370 | +0.0327 |
| +tfidf | -0.0370 | -8.93% | -2.83 | -0.307 | +0.0254 |

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

- `area` — 23.42% of total gain
- `tgt_district_name` — 22.61% of total gain
- `tgt_ward_name` — 9.34% of total gain
- `bedroom_count` — 3.95% of total gain
- `floor_count` — 3.41% of total gain
- `freq_district_name` — 3.35% of total gain
- `tfidf_c_5` — 3.28% of total gain
- `tgt_province_name` — 3.03% of total gain
- `freq_property_type_name` — 2.97% of total gain
- `frontage_width` — 2.88% of total gain
- `freq_province_name` — 1.60% of total gain
- `text_area_m2` — 1.59% of total gain
- `bathroom_count` — 1.54% of total gain
- `freq_ward_name` — 1.31% of total gain
- `tfidf_c_1` — 0.62% of total gain
