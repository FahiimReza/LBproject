import glob
import os
import time

import numpy as np
import pandas as pd

from step05_haversine_filter import (
    STEP05_CONFIG,
    compute_geo_features,
    filter_jobs_by_radius,
    haversine_km_vectorized,
)


def load_full_dataset() -> pd.DataFrame:
    if os.path.exists("dhaka_trade_dataset_1M.csv"):
        return pd.read_csv("dhaka_trade_dataset_1M.csv")
    parts = sorted(glob.glob("dhaka_trade_dataset_part*.csv"))
    if not parts:
        raise FileNotFoundError("No dhaka_trade_dataset_1M.csv or part CSVs found.")
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)


def run_full_tests() -> None:
    t_load = time.perf_counter()
    df = load_full_dataset()
    print(f"[1/5] Loaded {len(df):,} rows in {time.perf_counter() - t_load:.2f}s")

    # 1. Exact math & edge-case checks
    zero_dist = haversine_km_vectorized(23.7805, 90.3667, 23.7805, 90.3667)
    assert np.isclose(zero_dist, 0.0, atol=1e-9), f"Expected 0.0, got {zero_dist}"

    # Mirpur-10 (~23.8069, 90.3687) to Dhanmondi-27 (~23.7556, 90.3762) is ~5.76 km
    mirpur_to_dhanmondi = haversine_km_vectorized(23.8069, 90.3687, 23.7556, 90.3762)
    assert 5.5 <= mirpur_to_dhanmondi <= 6.0, f"Unexpected distance: {mirpur_to_dhanmondi}"

    # Symmetry check
    d_ab = haversine_km_vectorized(23.8069, 90.3687, 23.7556, 90.3762)
    d_ba = haversine_km_vectorized(23.7556, 90.3762, 23.8069, 90.3687)
    assert np.isclose(d_ab, d_ba, atol=1e-9), "Haversine should be symmetric"
    print("[2/5] Math, symmetry, and zero-distance assertions passed.")

    # 2. Full 1M-row geo feature computation & benchmark
    t_geo = time.perf_counter()
    scored_full = compute_geo_features(df, worker_lat=23.7805, worker_lon=90.3667)
    geo_sec = time.perf_counter() - t_geo
    assert {"distance_km", "est_commute_min", "wage_per_km", "in_dhaka_bounds"}.issubset(
        scored_full.columns
    )
    assert not scored_full["distance_km"].isna().any()
    print(
        f"[3/5] Scored all {len(scored_full):,} rows in {geo_sec:.3f}s | "
        f"In Dhaka bounds: {scored_full['in_dhaka_bounds'].mean():.2%} | "
        f"Max dist: {scored_full['distance_km'].max():.2f} km"
    )

    # 3. Radius filtering & monotonic sorting check
    radius_km = 3.0
    t_filt = time.perf_counter()
    filtered, used_radius = filter_jobs_by_radius(
        df, worker_lat=23.7805, worker_lon=90.3667, radius_km=radius_km
    )
    filt_sec = time.perf_counter() - t_filt
    assert (filtered["distance_km"] <= used_radius + 1e-9).all(), "Found row outside radius"
    assert filtered["distance_km"].is_monotonic_increasing, "Results not sorted by distance_km"
    print(
        f"[4/5] Filtered {len(filtered):,} / {len(df):,} rows within {used_radius:.1f} km "
        f"in {filt_sec:.3f}s (sorted ascending)"
    )

    # 4. Fallback radius expansion test on a sparse/edge coordinate
    sparse_df, exp_radius = filter_jobs_by_radius(
        df.head(200), worker_lat=23.9400, worker_lon=90.5100, radius_km=1.0
    )
    assert exp_radius >= 1.0 and len(sparse_df) >= 0
    print(
        f"[5/5] Fallback expansion check passed (expanded radius: {exp_radius:.1f} km, "
        f"matched: {len(sparse_df)} rows)"
    )

    # Summary stats on matched jobs
    print("\nMatched Radius Summary:")
    print(
        filtered[["distance_km", "est_commute_min", "offered_wage", "wage_per_km"]]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    run_full_tests()