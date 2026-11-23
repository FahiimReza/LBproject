from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0

STEP05_CONFIG = {
    "default_radius_km": 5.0,
    "fallback_radii_km": [8.0, 12.0],
    "min_candidates": 5,
    "dhaka_avg_speed_kmh": 14.0,  # Conservative urban speed for Dhaka commute estimate
    "lat_bounds": (23.65, 23.95),
    "lon_bounds": (90.30, 90.52),
}


def haversine_km_vectorized(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
    radius_km: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    """Compute great-circle distance in kilometers for scalar or array coordinates."""
    lat1_r, lon1_r, lat2_r, lon2_r = map(
        np.radians,
        (
            np.asarray(lat1, dtype=np.float64),
            np.asarray(lon1, dtype=np.float64),
            np.asarray(lat2, dtype=np.float64),
            np.asarray(lon2, dtype=np.float64),
        ),
    )
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * (np.sin(dlon / 2.0) ** 2)
    )
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return radius_km * c


def compute_geo_features(
    jobs_df: pd.DataFrame,
    worker_lat: float,
    worker_lon: float,
    cfg: Dict = STEP05_CONFIG,
) -> pd.DataFrame:
    """Attach distance_km, commute estimate, and wage_per_km features to candidate jobs."""
    out = jobs_df.copy()
    job_lat = pd.to_numeric(out["lat"], errors="coerce").to_numpy(dtype=np.float64)
    job_lon = pd.to_numeric(out["lon"], errors="coerce").to_numpy(dtype=np.float64)

    distances = haversine_km_vectorized(
        np.full_like(job_lat, float(worker_lat), dtype=np.float64),
        np.full_like(job_lon, float(worker_lon), dtype=np.float64),
        job_lat,
        job_lon,
    )
    out["distance_km"] = np.round(distances, 4)
    out["est_commute_min"] = np.round(
        (out["distance_km"] / cfg["dhaka_avg_speed_kmh"]) * 60.0, 1
    )

    if "offered_wage" in out.columns:
        wages = pd.to_numeric(out["offered_wage"], errors="coerce").fillna(0.0)
        out["wage_per_km"] = np.round(wages / (out["distance_km"] + 0.5), 2)

    lat_min, lat_max = cfg["lat_bounds"]
    lon_min, lon_max = cfg["lon_bounds"]
    out["in_dhaka_bounds"] = (
        (job_lat >= lat_min)
        & (job_lat <= lat_max)
        & (job_lon >= lon_min)
        & (job_lon <= lon_max)
    )
    return out


def filter_jobs_by_radius(
    jobs_df: pd.DataFrame,
    worker_lat: float,
    worker_lon: float,
    radius_km: Optional[float] = None,
    cfg: Dict = STEP05_CONFIG,
) -> Tuple[pd.DataFrame, float]:
    """Filter candidate jobs within radius_km, expanding radius if too few matches appear."""
    target_radius = cfg["default_radius_km"] if radius_km is None else float(radius_km)
    scored = compute_geo_features(jobs_df, worker_lat, worker_lon, cfg=cfg)

    radii_to_try = [target_radius] + [
        r for r in cfg["fallback_radii_km"] if r > target_radius
    ]

    used_radius = target_radius
    filtered = scored[scored["distance_km"] <= target_radius].copy()

    for r in radii_to_try[1:]:
        if len(filtered) >= cfg["min_candidates"]:
            break
        candidate = scored[scored["distance_km"] <= r].copy()
        if len(candidate) > len(filtered):
            filtered = candidate
            used_radius = r

    filtered["applied_radius_km"] = used_radius
    filtered = filtered.sort_values("distance_km", ascending=True).reset_index(drop=True)
    return filtered, used_radius


if __name__ == "__main__":
    import glob
    import os

    if os.path.exists("dhaka_trade_dataset_1M.csv"):
        df = pd.read_csv("dhaka_trade_dataset_1M.csv", nrows=None)
    else:
        parts = sorted(glob.glob("dhaka_trade_dataset_part*.csv"))
        df = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True).head(50_000)

    # Example worker coordinate in central/west Dhaka (e.g., around Mirpur/Mohammadpur corridor)
    sample_worker_lat, sample_worker_lon = 23.7805, 90.3667
    matched, radius_used = filter_jobs_by_radius(
        df, worker_lat=sample_worker_lat, worker_lon=sample_worker_lon, radius_km=3.0
    )
    print(f"Matched {len(matched)} jobs within {radius_used:.1f} km")
    print(
        matched[["trade", "offered_wage", "distance_km", "est_commute_min", "wage_per_km"]]
        .head(8)
        .to_string(index=False)
    )