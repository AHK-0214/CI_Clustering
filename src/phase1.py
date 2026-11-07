"""
Full Phase 1 run: data understanding and predictions.

This module only looks at the raw data and does not run any clustering algorithm,
per the assignment's requirement ("do not look at clustering output yet").
"""
import os
import pandas as pd

from . import data as data_mod
from . import regions as regions_mod
from . import viz as viz_mod
from . import density as density_mod
from .config import OUTPUT_DIR, S_BASE


# ---------------------------------------------------------------------
# Question D - predictions P1 through P4 (must be recorded before seeing any
# clustering output). The "prediction", "reason" and "rejection_criterion" values
# are the actual answer text submitted for the assignment's report (English).
# ---------------------------------------------------------------------
PREDICTIONS = {
    "P1": {
        "prediction": (
            "The elongated structure most likely to be split across several K-Means clusters "
            "(k=12) is the Mid-Atlantic Ridge; also parts of the Ring of Fire along the west "
            "coast of South America and East Asia, which are long and elongated."
        ),
        "reason": (
            "K-Means builds clusters with a Voronoi decision boundary around k centers, which "
            "suits compact/spherical clusters. A narrow, elongated structure (such as a fault "
            "line) whose length exceeds the typical diameter of a Voronoi cell will be covered "
            "by several neighboring centers and fragmented across multiple clusters."
        ),
        "rejection_criterion": (
            "If, after running the algorithm, the entire length of the Mid-Atlantic Ridge is "
            "found within one or two clusters (rather than three or more), this prediction is "
            "rejected."
        ),
    },
    "P2": {
        "prediction": (
            "As eps increases from a small to a large value, the percentage of noise points "
            "decreases monotonically and the size of the largest cluster increases "
            "monotonically. At large eps, because of the chaining phenomenon between regions "
            "of different density, one giant cluster absorbs most of the dense regions. We do "
            "not expect a single eps to cluster both dense regions (e.g. R1) and sparse "
            "regions (e.g. R3) well simultaneously, because by definition DBSCAN uses one "
            "global eps, and the order-of-magnitude difference in density between regions "
            "(observed in Phase 1-B) makes this trade-off difficult."
        ),
        "reason": (
            "DBSCAN defines core points based on counting neighbors within a fixed radius eps; "
            "increasing eps enlarges each point's neighborhood radius and therefore increases "
            "the chance of reaching the min_samples threshold, and also connects neighboring "
            "clusters together."
        ),
        "rejection_criterion": (
            "If, in the Phase 3-A table, a range of eps is found in which the noise percentage "
            "in R1 (dense) stays low AND R3 (sparse) remains reasonably separated from noise, "
            "the second part of this prediction is rejected."
        ),
    },
    "P3": {
        "prediction": (
            "Single Link is more prone to forming one very large cluster and several very "
            "small clusters (a 'rich-get-richer' structure), whereas Average Link produces "
            "more balanced clusters."
        ),
        "reason": (
            "In Single Link, the distance between two clusters equals the minimum distance "
            "between any member of the two clusters; this criterion is very sensitive to the "
            "chaining phenomenon — a single thin chain of nearby points is enough to connect "
            "two large masses. Average Link uses the average of all pairwise distances, which "
            "is more robust to such thin bridges and produces more compact, balanced clusters."
        ),
        "rejection_criterion": (
            "If, in Phase 3-C, the cluster-size distribution of Average Link turns out to be "
            "more unbalanced than that of Single Link (e.g. higher variance in cluster size), "
            "this prediction is rejected."
        ),
    },
    "P4": {
        "prediction": (
            "For scenario A (zoning/regionalization), we consider K-Means more suitable "
            "because it assigns every point to exactly one cluster (100% coverage) and the "
            "number of regions (k) can be set directly in the range 10 to 20. For scenario B "
            "(structure discovery), we consider DBSCAN more suitable because it does not "
            "impose the number of clusters in advance and allows outlier points to be labeled "
            "as noise."
        ),
        "reason": (
            "K-Means's assumptions (spherical clusters around a center) are more consistent "
            "with scenario A's goal (describable regions with full coverage); DBSCAN's "
            "assumption (density connectivity) is more consistent with scenario B's goal "
            "(discovering belts without fixing k)."
        ),
        "rejection_criterion": (
            "If, in Phase 4, it turns out that DBSCAN with a suitable eps produces "
            "near-complete coverage and regions that are just as describable for scenario A, "
            "or that K-Means performs acceptably at discovering the non-spherical belts of "
            "scenario B, the corresponding part of this prediction is rejected."
        ),
    },
}


# ---------------------------------------------------------------------
# Question C - explanation of why plain Euclidean distance on lat/lon is misleading.
# This is the actual answer text for the assignment's report (English).
# ---------------------------------------------------------------------
DISTANCE_EXPLANATION = """
Computing the plain Euclidean distance directly on the raw Latitude and Longitude
columns is misleading, because these two quantities are angles, not Cartesian
coordinates on a flat plane:

1) Discontinuity at 180 degrees longitude:
   Two points at longitude 179 and -179 degrees (e.g. near the International Date
   Line in the Pacific Ocean) are only about 2 degrees (~222 km) apart in reality,
   but their raw Euclidean distance along the longitude axis computes to 358
   degrees -- nearly the maximum possible value. These two points, which are in
   fact neighbors, appear to be very far apart under this incorrect metric.

2) Distortion near the poles:
   The length of one degree of longitude shrinks with cos(latitude). Near the
   poles, a large difference in longitude can correspond to a very small real
   distance, while the raw Euclidean distance computes it as large.

Representing points in 3D on the unit sphere (x = cos(phi) cos(lambda),
y = cos(phi) sin(lambda), z = sin(phi)) resolves both problems: this
transformation is a continuous, seamless mapping from the surface of the sphere
to 3D Euclidean space, so geographically close points (including points on
either side of the 180-degree meridian) are also close together in this space,
and the Euclidean distance in this space (after converting it to a surface
distance with the arc formula) is consistent with the true distance on the
sphere.
""".strip()


def run_phase1(save_outputs: bool = True) -> dict:
    """Run all of Phase 1 and return a results dict (for use in the notebook)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load and clean ----
    df = data_mod.load_clean_earthquakes()
    n_total = len(df)

    # ---- Question A: map and three regions ----
    fig_map, _ = viz_mod.plot_world_map(
        df, save_path=os.path.join(OUTPUT_DIR, "01_world_map.png") if save_outputs else None
    )
    fig_map_geo = viz_mod.plot_world_map_geo(
        df, save_path=os.path.join(OUTPUT_DIR, "01b_world_map_geo.html") if save_outputs else None
    )
    fig_regions, _ = viz_mod.plot_world_map_with_regions(
        df, save_path=os.path.join(OUTPUT_DIR, "02_world_map_regions.png") if save_outputs else None
    )
    region_table = regions_mod.region_summary_table(df)

    # ---- Question D / P1 — where the elongated structures named in the prediction
    # actually are (still raw data, no clustering run yet) ----
    fig_p1_structures, _ = viz_mod.plot_p1_predicted_structures(
        df, save_path=os.path.join(OUTPUT_DIR, "04_p1_predicted_structures.png") if save_outputs else None
    )

    # ---- Question B: density (distance to 10th neighbor) ----
    dist_km = density_mod.kth_neighbor_distance_km(df)
    fig_hist, _ = viz_mod.plot_density_histogram(
        dist_km, save_path=os.path.join(OUTPUT_DIR, "03_density_histogram.png") if save_outputs else None
    )
    density_table = density_mod.region_density_table(df, dist_km, regions_mod)

    if save_outputs:
        region_table.to_csv(os.path.join(OUTPUT_DIR, "region_table.csv"), index=False)
        density_table.to_csv(os.path.join(OUTPUT_DIR, "density_table.csv"), index=False)
        df.assign(dist_to_10th_neighbor_km=dist_km).to_csv(
            os.path.join(OUTPUT_DIR, "cleaned_earthquakes_with_density.csv"), index=False
        )
        with open(os.path.join(OUTPUT_DIR, "distance_explanation.txt"), "w", encoding="utf-8") as f:
            f.write(DISTANCE_EXPLANATION)
        predictions_df = pd.DataFrame(PREDICTIONS).T
        predictions_df.to_csv(os.path.join(OUTPUT_DIR, "predictions_P1_P4.csv"), encoding="utf-8-sig")

    return {
        "df": df,
        "n_total": n_total,
        "seed_base": S_BASE,
        "region_table": region_table,
        "density_table": density_table,
        "dist_km": dist_km,
        "figures": {
            "map": fig_map, "map_geo": fig_map_geo, "map_regions": fig_regions,
            "hist": fig_hist, "p1_structures": fig_p1_structures,
        },
        "distance_explanation": DISTANCE_EXPLANATION,
        "predictions": PREDICTIONS,
    }


if __name__ == "__main__":
    results = run_phase1()
    print(f"Total number of earthquakes after cleaning: {results['n_total']}")
    print(results["region_table"])
    print(results["density_table"])
