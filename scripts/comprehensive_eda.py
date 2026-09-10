import glob
import json
import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

def run_eda():
    shard_files = sorted(glob.glob("data/shard_*.parquet"))
    print(f"Found {len(shard_files)} shards: {shard_files}")
    if not shard_files:
        print("No shards found!")
        sys.exit(1)

    total_rows = 0
    null_counts = {}
    col_dtypes = {}
    
    # Anomaly counters
    price_null_count = 0
    price_non_numeric_count = 0
    price_le_zero_count = 0
    price_lt_10m_count = 0
    price_lt_100m_count = 0
    price_gt_100b_count = 0
    price_gt_500b_count = 0
    price_gt_1000b_count = 0
    
    area_null_count = 0
    area_le_zero_count = 0
    area_lt_10_count = 0
    area_lt_20_count = 0
    area_gt_1000_count = 0
    area_gt_5000_count = 0
    area_gt_10000_count = 0

    unit_price_lt_1m_count = 0
    unit_price_lt_5m_count = 0
    unit_price_gt_500m_count = 0
    unit_price_gt_1b_count = 0

    # Categorical counters
    province_counts = pd.Series(dtype=int)
    property_type_counts = pd.Series(dtype=int)
    hanoi_district_counts = pd.Series(dtype=int)
    hcm_district_counts = pd.Series(dtype=int)
    danang_district_counts = pd.Series(dtype=int)
    binhduong_district_counts = pd.Series(dtype=int)

    corr_samples = []

    for idx, shard_path in enumerate(shard_files):
        print(f"Processing shard {idx+1}/{len(shard_files)}: {shard_path}...")
        table = pq.read_table(shard_path)
        num_rows = table.num_rows
        total_rows += num_rows

        if idx == 0:
            for col in table.column_names:
                null_counts[col] = 0
                col_dtypes[col] = str(table.schema.field(col).type)

        # Null profiling
        for col in table.column_names:
            null_count = table.column(col).null_count
            null_counts[col] += null_count

        cols_to_read = [
            'property_type_name', 'province_name', 'district_name',
            'price', 'area', 'floor_count', 'frontage_width', 
            'house_depth', 'road_width', 'bedroom_count', 'bathroom_count'
        ]
        df_batch = table.select(cols_to_read).to_pandas()

        prov_c = df_batch['province_name'].value_counts()
        province_counts = province_counts.add(prov_c, fill_value=0)

        pt_c = df_batch['property_type_name'].value_counts()
        property_type_counts = property_type_counts.add(pt_c, fill_value=0)

        hn_mask = df_batch['province_name'] == 'Hà Nội'
        if hn_mask.any():
            hanoi_district_counts = hanoi_district_counts.add(
                df_batch.loc[hn_mask, 'district_name'].value_counts(), fill_value=0
            )

        hcm_mask = df_batch['province_name'] == 'Hồ Chí Minh'
        if hcm_mask.any():
            hcm_district_counts = hcm_district_counts.add(
                df_batch.loc[hcm_mask, 'district_name'].value_counts(), fill_value=0
            )

        dn_mask = df_batch['province_name'] == 'Đà Nẵng'
        if dn_mask.any():
            danang_district_counts = danang_district_counts.add(
                df_batch.loc[dn_mask, 'district_name'].value_counts(), fill_value=0
            )

        bd_mask = df_batch['province_name'] == 'Bình Dương'
        if bd_mask.any():
            binhduong_district_counts = binhduong_district_counts.add(
                df_batch.loc[bd_mask, 'district_name'].value_counts(), fill_value=0
            )

        price_s = df_batch['price']
        price_null_count += price_s.isna().sum()

        price_num = pd.to_numeric(price_s, errors='coerce')
        price_non_numeric_count += (price_s.notna() & price_num.isna()).sum()

        valid_price_mask = price_num.notna()
        price_le_zero_count += (price_num[valid_price_mask] <= 0).sum()
        price_lt_10m_count += (price_num[valid_price_mask] < 10_000_000).sum()
        price_lt_100m_count += (price_num[valid_price_mask] < 100_000_000).sum()
        price_gt_100b_count += (price_num[valid_price_mask] > 100_000_000_000).sum()
        price_gt_500b_count += (price_num[valid_price_mask] > 500_000_000_000).sum()
        price_gt_1000b_count += (price_num[valid_price_mask] > 1_000_000_000_000).sum()

        area_s = df_batch['area']
        area_null_count += area_s.isna().sum()
        valid_area_mask = area_s.notna()
        area_le_zero_count += (area_s[valid_area_mask] <= 0).sum()
        area_lt_10_count += (area_s[valid_area_mask] < 10.0).sum()
        area_lt_20_count += (area_s[valid_area_mask] < 20.0).sum()
        area_gt_1000_count += (area_s[valid_area_mask] > 1000.0).sum()
        area_gt_5000_count += (area_s[valid_area_mask] > 5000.0).sum()
        area_gt_10000_count += (area_s[valid_area_mask] > 10000.0).sum()

        both_valid_mask = valid_price_mask & valid_area_mask & (area_s > 0)
        unit_p = price_num[both_valid_mask] / area_s[both_valid_mask]
        unit_price_lt_1m_count += (unit_p < 1_000_000).sum()
        unit_price_lt_5m_count += (unit_p < 5_000_000).sum()
        unit_price_gt_500m_count += (unit_p > 500_000_000).sum()
        unit_price_gt_1b_count += (unit_p > 1_000_000_000).sum()

        step = 5  # ~700k records total across 10 shards
        sub_df = df_batch.iloc[::step].copy()
        sub_df['price_clean'] = pd.to_numeric(sub_df['price'], errors='coerce')
        corr_samples.append(sub_df)

    print("\nAll shards scanned. Combining samples for distribution and correlation...")
    sample_df = pd.concat(corr_samples, ignore_index=True)
    print(f"Sample size: {len(sample_df)} rows")

    missing_profile = []
    for col in null_counts:
        missing_profile.append({
            "column": col,
            "dtype": col_dtypes[col],
            "null_count": int(null_counts[col]),
            "null_pct": round(float(null_counts[col]) / total_rows * 100, 2)
        })

    anomalies = {
        "total_records": int(total_rows),
        "price": {
            "null_count": int(price_null_count),
            "null_pct": round(price_null_count / total_rows * 100, 2),
            "non_numeric": int(price_non_numeric_count),
            "le_zero": int(price_le_zero_count),
            "lt_10m": int(price_lt_10m_count),
            "lt_100m": int(price_lt_100m_count),
            "gt_100b": int(price_gt_100b_count),
            "gt_500b": int(price_gt_500b_count),
            "gt_1000b": int(price_gt_1000b_count)
        },
        "area": {
            "null_count": int(area_null_count),
            "null_pct": round(area_null_count / total_rows * 100, 2),
            "le_zero": int(area_le_zero_count),
            "lt_10m2": int(area_lt_10_count),
            "lt_20m2": int(area_lt_20_count),
            "gt_1000m2": int(area_gt_1000_count),
            "gt_5000m2": int(area_gt_5000_count),
            "gt_10000m2": int(area_gt_10000_count)
        },
        "unit_price": {
            "lt_1m": int(unit_price_lt_1m_count),
            "lt_5m": int(unit_price_lt_5m_count),
            "gt_500m": int(unit_price_gt_500m_count),
            "gt_1b": int(unit_price_gt_1b_count)
        }
    }

    valid_p = sample_df['price_clean'].dropna()
    valid_p_pos = valid_p[valid_p > 0]
    valid_a = sample_df['area'].dropna()
    valid_a_pos = valid_a[valid_a > 0]

    both_mask = (sample_df['price_clean'] > 0) & (sample_df['area'] > 0)
    valid_up = sample_df.loc[both_mask, 'price_clean'] / sample_df.loc[both_mask, 'area']

    quantiles_list = [0.0, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0]

    def get_stats(series, is_log=False):
        s = series.dropna()
        if is_log:
            s = np.log1p(s)
        return {
            "count": int(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "skewness": float(stats.skew(s)),
            "kurtosis": float(stats.kurtosis(s)),
            "quantiles": {str(q): float(s.quantile(q)) for q in quantiles_list}
        }

    distributions = {
        "raw_price": get_stats(valid_p_pos, is_log=False),
        "log_price": get_stats(valid_p_pos, is_log=True),
        "raw_unit_price": get_stats(valid_up, is_log=False),
        "log_unit_price": get_stats(valid_up, is_log=True),
        "raw_area": get_stats(valid_a_pos, is_log=False),
        "log_area": get_stats(valid_a_pos, is_log=True)
    }

    prov_df = province_counts.sort_values(ascending=False)
    top_provinces = []
    for prov, count in prov_df.head(20).items():
        sub = sample_df[sample_df['province_name'] == prov]
        sub_p = sub['price_clean'].dropna()
        sub_p = sub_p[sub_p > 0]
        sub_a = sub['area'].dropna()
        sub_a = sub_a[sub_a > 0]
        sub_up = (sub['price_clean'] / sub['area']).dropna()
        sub_up = sub_up[sub_up > 0]
        top_provinces.append({
            "province": prov,
            "listings": int(count),
            "share_pct": round(count / total_rows * 100, 2),
            "median_price_billion": round(float(sub_p.median()) / 1e9, 2) if len(sub_p) else None,
            "mean_price_billion": round(float(sub_p.mean()) / 1e9, 2) if len(sub_p) else None,
            "median_area": round(float(sub_a.median()), 1) if len(sub_a) else None,
            "median_unit_price_million": round(float(sub_up.median()) / 1e6, 2) if len(sub_up) else None,
        })

    def get_district_stats(prov_name, dist_counts, top_n=15):
        dist_res = []
        for dist, count in dist_counts.sort_values(ascending=False).head(top_n).items():
            sub = sample_df[(sample_df['province_name'] == prov_name) & (sample_df['district_name'] == dist)]
            sub_p = sub['price_clean'].dropna()
            sub_p = sub_p[sub_p > 0]
            sub_up = (sub['price_clean'] / sub['area']).dropna()
            sub_up = sub_up[sub_up > 0]
            sub_a = sub['area'].dropna()
            sub_a = sub_a[sub_a > 0]
            dist_res.append({
                "district": str(dist),
                "listings": int(count),
                "median_price_billion": round(float(sub_p.median()) / 1e9, 2) if len(sub_p) else None,
                "median_area": round(float(sub_a.median()), 1) if len(sub_a) else None,
                "median_unit_price_million": round(float(sub_up.median()) / 1e6, 2) if len(sub_up) else None,
            })
        return dist_res

    hanoi_districts = get_district_stats("Hà Nội", hanoi_district_counts, top_n=15)
    hcm_districts = get_district_stats("Hồ Chí Minh", hcm_district_counts, top_n=15)
    danang_districts = get_district_stats("Đà Nẵng", danang_district_counts, top_n=10)
    binhduong_districts = get_district_stats("Bình Dương", binhduong_district_counts, top_n=10)

    property_type_breakdown = []
    pt_series = property_type_counts.sort_values(ascending=False)
    for pt, count in pt_series.items():
        sub = sample_df[sample_df['property_type_name'] == pt]
        sub_p = sub['price_clean'].dropna()
        sub_p = sub_p[sub_p > 0]
        sub_a = sub['area'].dropna()
        sub_a = sub_a[sub_a > 0]
        sub_up = (sub['price_clean'] / sub['area']).dropna()
        sub_up = sub_up[sub_up > 0]
        property_type_breakdown.append({
            "property_type": str(pt),
            "listings": int(count),
            "share_pct": round(count / total_rows * 100, 2),
            "median_price_billion": round(float(sub_p.median()) / 1e9, 2) if len(sub_p) else None,
            "q25_price_billion": round(float(sub_p.quantile(0.25)) / 1e9, 2) if len(sub_p) else None,
            "q75_price_billion": round(float(sub_p.quantile(0.75)) / 1e9, 2) if len(sub_p) else None,
            "median_area": round(float(sub_a.median()), 1) if len(sub_a) else None,
            "median_unit_price_million": round(float(sub_up.median()) / 1e6, 2) if len(sub_up) else None,
            "q25_unit_price_million": round(float(sub_up.quantile(0.25)) / 1e6, 2) if len(sub_up) else None,
            "q75_unit_price_million": round(float(sub_up.quantile(0.75)) / 1e6, 2) if len(sub_up) else None,
        })

    corr_cols = [
        'area', 'floor_count', 'frontage_width', 'house_depth', 
        'road_width', 'bedroom_count', 'bathroom_count'
    ]
    sample_corr_df = sample_df[sample_df['price_clean'] > 0].copy()
    sample_corr_df['log_price'] = np.log1p(sample_corr_df['price_clean'])
    sample_corr_df['unit_price'] = sample_corr_df['price_clean'] / sample_corr_df['area']
    sample_corr_df['log_unit_price'] = np.log1p(sample_corr_df['unit_price'])
    sample_corr_df['log_area'] = np.log1p(sample_corr_df['area'])

    cols_for_pearson = ['log_price', 'log_unit_price', 'log_area'] + corr_cols
    pearson_corr = sample_corr_df[cols_for_pearson].corr(method='pearson').round(3).to_dict()
    spearman_corr = sample_corr_df[cols_for_pearson].corr(method='spearman').round(3).to_dict()

    f1_mask = (
        sample_df['price_clean'].between(100_000_000, 200_000_000_000) &
        sample_df['area'].between(15.0, 3000.0) &
        ((sample_df['price_clean'] / sample_df['area']).between(3_000_000, 800_000_000))
    )
    df_f1 = sample_df[f1_mask]
    log_p_f1 = np.log1p(df_f1['price_clean'])
    log_up_f1 = np.log1p(df_f1['price_clean'] / df_f1['area'])

    f2_mask = (
        sample_df['price_clean'].between(300_000_000, 100_000_000_000) &
        sample_df['area'].between(20.0, 1000.0) &
        ((sample_df['price_clean'] / sample_df['area']).between(10_000_000, 500_000_000))
    )
    df_f2 = sample_df[f2_mask]
    log_p_f2 = np.log1p(df_f2['price_clean'])

    p_p01 = valid_p_pos.quantile(0.01)
    p_p99 = valid_p_pos.quantile(0.99)
    a_p01 = valid_a_pos.quantile(0.01)
    a_p99 = valid_a_pos.quantile(0.99)
    f3_mask = (
        sample_df['price_clean'].between(p_p01, p_p99) &
        sample_df['area'].between(a_p01, a_p99)
    )
    df_f3 = sample_df[f3_mask]
    log_p_f3 = np.log1p(df_f3['price_clean'])

    cleaning_comparison = {
        "rule_baseline_recommended": {
            "description": "price: [100M, 200B], area: [15, 3000 m2], unit_price: [3M, 800M/m2]",
            "records_retained": int(f1_mask.sum()),
            "retention_pct": round(f1_mask.sum() / len(sample_df) * 100, 2),
            "log_price_skewness": round(float(stats.skew(log_p_f1)), 3),
            "log_price_kurtosis": round(float(stats.kurtosis(log_p_f1)), 3),
            "log_unit_price_skewness": round(float(stats.skew(log_up_f1)), 3),
            "log_unit_price_kurtosis": round(float(stats.kurtosis(log_up_f1)), 3),
        },
        "rule_strict_metro": {
            "description": "price: [300M, 100B], area: [20, 1000 m2], unit_price: [10M, 500M/m2]",
            "records_retained": int(f2_mask.sum()),
            "retention_pct": round(f2_mask.sum() / len(sample_df) * 100, 2),
            "log_price_skewness": round(float(stats.skew(log_p_f2)), 3),
            "log_price_kurtosis": round(float(stats.kurtosis(log_p_f2)), 3),
        },
        "rule_percentile_1_99": {
            "description": f"price: [{round(p_p01/1e6,1)}M, {round(p_p99/1e9,1)}B], area: [{round(a_p01,1)}, {round(a_p99,1)} m2]",
            "records_retained": int(f3_mask.sum()),
            "retention_pct": round(f3_mask.sum() / len(sample_df) * 100, 2),
            "log_price_skewness": round(float(stats.skew(log_p_f3)), 3),
            "log_price_kurtosis": round(float(stats.kurtosis(log_p_f3)), 3),
        }
    }

    results = {
        "total_records": int(total_rows),
        "missing_profile": missing_profile,
        "anomalies": anomalies,
        "distributions": distributions,
        "top_provinces": top_provinces,
        "hanoi_districts": hanoi_districts,
        "hcm_districts": hcm_districts,
        "danang_districts": danang_districts,
        "binhduong_districts": binhduong_districts,
        "property_type_breakdown": property_type_breakdown,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "cleaning_comparison": cleaning_comparison
    }

    with open("eda_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nEDA Profiling complete! Results written to eda_results.json")

if __name__ == "__main__":
    run_eda()
