"""
Compute, for each point, the distance to its 10th nearest neighbor (not counting the
point itself), in kilometers.

Per the assignment's note: for this distance (10th neighbor, excluding the point itself),
the corresponding min_samples for DBSCAN would be 11, but throughout the project we use
min_samples=10 (this one-neighbor discrepancy is intentionally ignored).
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .geo import latlon_to_xyz, chordal_to_surface_km
from .config import K_NEIGHBOR


def kth_neighbor_distance_km(df, k: int = K_NEIGHBOR) -> np.ndarray:
    """For each row of df (Latitude/Longitude columns), compute the distance to the
    k-th nearest neighbor on the unit-sphere 3D coordinates and convert it to kilometers."""
    xyz = latlon_to_xyz(df["Latitude"].values, df["Longitude"].values)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")  # +1 because the point itself is returned too
    nn.fit(xyz)
    chordal_dists, _ = nn.kneighbors(xyz)
    kth_chordal = chordal_dists[:, -1]  # last column = distance to the k-th neighbor (column 0 is the point itself)
    return chordal_to_surface_km(kth_chordal)


def region_density_table(df, dist_km, regions_module):
    """Table of count, Q1, median, Q3 of the k-th neighbor distance for each region."""
    import pandas as pd
    rows = []
    for key in regions_module.REGIONS:
        mask = regions_module.region_mask(df, key).values
        d = dist_km[mask]
        if len(d) == 0:
            rows.append({"region": key, "count": 0, "Q1 (km)": None, "median (km)": None, "Q3 (km)": None})
            continue
        rows.append({
            "region": key,
            "count": int(len(d)),
            "Q1 (km)": round(float(np.percentile(d, 25)), 2),
            "median (km)": round(float(np.percentile(d, 50)), 2),
            "Q3 (km)": round(float(np.percentile(d, 75)), 2),
        })
    return pd.DataFrame(rows)
