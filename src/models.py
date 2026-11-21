"""
Thin wrappers around scikit-learn / scipy clustering algorithms, plus small helpers
for summarizing cluster sizes that Phases 2, 3 and 4 all need.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster

from .geo import latlon_to_xyz, eps_km_to_chordal


# ---------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------
def run_kmeans(X: np.ndarray, k: int, seed: int, n_init: int = 10) -> KMeans:
    """Fit K-Means with k clusters, n_init restarts, and a fixed random_state."""
    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    model.fit(X)
    return model


# ---------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------
def run_dbscan_km(xyz: np.ndarray, eps_km: float, min_samples: int = 10) -> DBSCAN:
    """Fit DBSCAN on 3D unit-sphere coordinates `xyz`, with eps given in kilometers
    (converted internally to the equivalent chordal radius in 3D space)."""
    eps_chord = eps_km_to_chordal(eps_km)
    model = DBSCAN(eps=eps_chord, min_samples=min_samples, metric="euclidean")
    model.fit(xyz)
    return model


# ---------------------------------------------------------------------
# Hierarchical clustering (used only on the shared sample S)
# ---------------------------------------------------------------------
def run_hierarchical(xyz: np.ndarray, method: str, n_clusters: int = 12):
    """method: 'single' or 'average'. Returns (linkage_matrix Z, flat labels [1..n_clusters])."""
    Z = linkage(xyz, method=method, metric="euclidean")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return Z, labels


# ---------------------------------------------------------------------
# Cluster-size / summary helpers (shared by Phase 2, 3 and 4)
# ---------------------------------------------------------------------
def cluster_size_table(labels: np.ndarray, noise_label=-1) -> pd.DataFrame:
    """One row per cluster label (noise excluded) with its size, sorted descending."""
    s = pd.Series(labels)
    s = s[s != noise_label]
    counts = s.value_counts().sort_values(ascending=False)
    return pd.DataFrame({"cluster": counts.index, "size": counts.values})


def summarize_labels(labels: np.ndarray, X_for_silhouette: np.ndarray = None,
                      noise_label=-1, small_cluster_threshold: int = 20) -> dict:
    """Generic summary used by Phase 3-A and Phase 4-A tables:
    n_clusters, noise_pct, largest cluster (count, share of ALL points),
    median cluster size, n_small_clusters (non-noise clusters with < threshold points),
    coverage_pct (share of points that are not noise), and silhouette (noise excluded,
    computed on X_for_silhouette if given and if >=2 clusters remain, else None)."""
    labels = np.asarray(labels)
    n_total = len(labels)
    is_noise = labels == noise_label
    n_noise = int(is_noise.sum())
    sizes = cluster_size_table(labels, noise_label)
    n_clusters = len(sizes)
    largest = int(sizes["size"].iloc[0]) if n_clusters > 0 else 0
    median_size = float(sizes["size"].median()) if n_clusters > 0 else 0.0
    n_small = int((sizes["size"] < small_cluster_threshold).sum()) if n_clusters > 0 else 0
    coverage_pct = 100.0 * (n_total - n_noise) / n_total if n_total else 0.0

    sil = None
    if X_for_silhouette is not None and n_clusters >= 2:
        mask = ~is_noise
        try:
            sil = float(silhouette_score(X_for_silhouette[mask], labels[mask]))
        except Exception:
            sil = None

    return {
        "n_clusters": n_clusters,
        "noise_pct": 100.0 * n_noise / n_total if n_total else 0.0,
        "largest_cluster_size": largest,
        "largest_cluster_share_pct": 100.0 * largest / n_total if n_total else 0.0,
        "median_cluster_size": median_size,
        "n_clusters_10_19": n_small,
        "coverage_pct": coverage_pct,
        "silhouette": sil,
    }


def main_cluster_for_mask(labels: np.ndarray, mask: np.ndarray, noise_label=-1):
    """Given a boolean mask selecting a region's points, return (main_cluster_id,
    pct_noise_in_region, pct_in_main_cluster) where main cluster = the non-noise
    cluster holding the most points of this region."""
    region_labels = np.asarray(labels)[mask]
    n_region = len(region_labels)
    if n_region == 0:
        return None, 0.0, 0.0
    pct_noise = 100.0 * np.sum(region_labels == noise_label) / n_region
    non_noise = region_labels[region_labels != noise_label]
    if len(non_noise) == 0:
        return None, pct_noise, 0.0
    vals, counts = np.unique(non_noise, return_counts=True)
    main_cluster = vals[np.argmax(counts)]
    pct_main = 100.0 * counts.max() / n_region
    return main_cluster, pct_noise, pct_main


def ari_on_common_ids(df_a: pd.DataFrame, labels_a: np.ndarray,
                       df_b: pd.DataFrame, labels_b: np.ndarray) -> tuple:
    """ARI between two labelings that come from two different (overlapping) subsets of
    records, matched by `record_id`. Returns (ari, n_common). Used by Phase 4-A's
    resampling-stability experiment, where each pair of 80%-of-S subsets shares only
    part of their points and cluster numbering is not meant to be aligned."""
    ids_a = df_a["record_id"].values
    ids_b = df_b["record_id"].values
    common = np.intersect1d(ids_a, ids_b)
    la = labels_for_ids_local(ids_a, labels_a, common)
    lb = labels_for_ids_local(ids_b, labels_b, common)
    return float(adjusted_rand_score(la, lb)), int(len(common))


def labels_for_ids_local(ids: np.ndarray, labels: np.ndarray, wanted_ids: np.ndarray) -> np.ndarray:
    """Like sampling.labels_for_ids, but takes the id array directly instead of a
    full dataframe (small helper to avoid a circular import with sampling.py)."""
    pos_of_id = pd.Series(np.arange(len(ids)), index=ids)
    positions = pos_of_id.loc[np.asarray(wanted_ids)].values
    return np.asarray(labels)[positions]


def xyz_of(df) -> np.ndarray:
    """Convenience: 3D unit-sphere coordinates for a dataframe with Latitude/Longitude."""
    return latlon_to_xyz(df["Latitude"].values, df["Longitude"].values)
