## Text-feature uplift benchmark

- Train: 77,904 clean rows of 100,000 (2025-06-01 → 2025-07-30)
- Test (out-of-time): 29,794 clean rows of 40,000 (2026-03-01 → 2026-03-30)
- Runtime: 480.2s

| arm | feats | RMSLE | MAPE % | MAE (tỷ) | MedAE (tỷ) | R² |
|---|---|---|---|---|---|---|
| tabular | 23 | 0.4511 | 31.47 | 7.317 | 1.416 | 0.0453 |
| +keywords | 70 | 0.4345 | 30.10 | 7.108 | 1.353 | 0.0488 |
| +entities | 85 | 0.4287 | 29.66 | 6.969 | 1.331 | 0.0508 |
| +tfidf | 213 | 0.4119 | 28.73 | 7.013 | 1.344 | 0.0456 |

### Marginal gain vs tabular-only

| arm | ΔRMSLE | ΔRMSLE % | ΔMAPE (pts) | ΔMAE (tỷ) | ΔR² |
|---|---|---|---|---|---|
| +keywords | -0.0165 | -3.66% | -1.37 | -0.209 | +0.0035 |
| +entities | -0.0224 | -4.97% | -1.81 | -0.348 | +0.0055 |
| +tfidf | -0.0392 | -8.69% | -2.74 | -0.304 | +0.0003 |

### Entity extraction agreement (train rows)

| structured | text entity | text coverage | col missing | overlap | agreement | recovered |
|---|---|---|---|---|---|---|
| area | text_area_m2 | 0.789 | 0.000 | 61,462 | 0.924 | 0 |
| frontage_width | text_frontage_m | 0.296 | 0.457 | 15,894 | 0.883 | 7,169 |
| house_depth | text_depth_m | 0.296 | 0.964 | 1,461 | 0.952 | 21,608 |
| floor_count | text_floor_count | 0.372 | 0.838 | 8,776 | 0.919 | 20,234 |
| bedroom_count | text_bedroom_count | 0.388 | 0.491 | 27,121 | 0.887 | 3,114 |
| bathroom_count | text_bathroom_count | 0.255 | 0.526 | 17,964 | 0.935 | 1,936 |
| road_width | text_road_width_m | 0.085 | 0.890 | 1,209 | 0.845 | 5,424 |
| price | text_price_vnd | 0.836 | 0.000 | 65,113 | 0.749 | 0 |

### Top-15 gain-importance features (richest arm)

- `area` — 21.97% of total gain
- `tgt_district_name` — 21.77% of total gain
- `tfidf_w_5` — 9.62% of total gain
- `tgt_ward_name` — 8.08% of total gain
- `freq_district_name` — 4.90% of total gain
- `frontage_width` — 2.85% of total gain
- `bedroom_count` — 2.76% of total gain
- `freq_property_type_name` — 2.61% of total gain
- `freq_province_name` — 2.32% of total gain
- `tgt_province_name` — 1.59% of total gain
- `text_area_m2` — 1.34% of total gain
- `floor_count` — 1.31% of total gain
- `freq_ward_name` — 1.13% of total gain
- `tfidf_w_1` — 1.12% of total gain
- `bathroom_count` — 1.12% of total gain
