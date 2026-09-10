## Text-feature uplift benchmark

- Train: 6,274 clean rows of 8,000 (2025-06-01 → 2025-07-30)
- Test (out-of-time): 2,975 clean rows of 4,000 (2026-03-01 → 2026-03-30)
- Runtime: 1073.2s

| arm | feats | RMSLE | MAPE % | MAE (tỷ) | MedAE (tỷ) | R² |
|---|---|---|---|---|---|---|
| tabular | 23 | 0.5539 | 42.52 | 9.556 | 1.711 | 0.0160 |
| +keywords | 70 | 0.5413 | 40.97 | 9.329 | 1.666 | 0.0165 |
| +entities | 85 | 0.5389 | 40.45 | 9.450 | 1.681 | 0.0153 |
| +tfidf | 213 | 0.5267 | 37.71 | 9.446 | 1.595 | 0.0168 |
| +phobert | 277 | 0.5246 | 38.18 | 9.402 | 1.590 | 0.0175 |

### Marginal gain vs tabular-only

| arm | ΔRMSLE | ΔRMSLE % | ΔMAPE (pts) | ΔMAE (tỷ) | ΔR² |
|---|---|---|---|---|---|
| +keywords | -0.0126 | -2.27% | -1.55 | -0.227 | +0.0005 |
| +entities | -0.0150 | -2.70% | -2.07 | -0.106 | -0.0008 |
| +tfidf | -0.0272 | -4.90% | -4.81 | -0.109 | +0.0008 |
| +phobert | -0.0293 | -5.29% | -4.34 | -0.154 | +0.0015 |

### Entity extraction agreement (train rows)

| structured | text entity | text coverage | col missing | overlap | agreement | recovered |
|---|---|---|---|---|---|---|
| area | text_area_m2 | 0.791 | 0.000 | 4,961 | 0.924 | 0 |
| frontage_width | text_frontage_m | 0.286 | 0.468 | 1,234 | 0.891 | 562 |
| house_depth | text_depth_m | 0.286 | 0.963 | 124 | 0.944 | 1,672 |
| floor_count | text_floor_count | 0.364 | 0.842 | 690 | 0.928 | 1,591 |
| bedroom_count | text_bedroom_count | 0.388 | 0.478 | 2,220 | 0.894 | 215 |
| bathroom_count | text_bathroom_count | 0.253 | 0.517 | 1,445 | 0.944 | 142 |
| road_width | text_road_width_m | 0.083 | 0.891 | 96 | 0.854 | 426 |
| price | text_price_vnd | 0.833 | 0.000 | 5,225 | 0.744 | 0 |

### Top-15 gain-importance features (richest arm)

- `area` — 20.46% of total gain
- `tgt_district_name` — 18.12% of total gain
- `tgt_ward_name` — 13.41% of total gain
- `freq_district_name` — 4.78% of total gain
- `tfidf_w_6` — 3.70% of total gain
- `bedroom_count` — 3.37% of total gain
- `tfidf_w_5` — 2.75% of total gain
- `tgt_province_name` — 2.71% of total gain
- `tfidf_w_10` — 2.36% of total gain
- `frontage_width` — 1.67% of total gain
- `text_area_m2` — 1.61% of total gain
- `freq_property_type_name` — 1.48% of total gain
- `freq_province_name` — 1.48% of total gain
- `freq_ward_name` — 1.33% of total gain
- `tfidf_w_2` — 1.08% of total gain
