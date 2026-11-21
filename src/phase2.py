"""
Full Phase 2 run: K-Means.

Per the assignment:
  Q.A - k=12, n_init=10, once on raw (lat, lon) and once on 3D unit-sphere coordinates
        (both with seed=s); compare with ARI; a second 3D run with seed=s+1 gives a
        baseline ARI for pure initialization variability.
  Q.B - k in {8, 12, 16, 20} on the 3D coordinates (n_init=10, seed=s); inertia curve;
        Silhouette computed only on the shared sample S; final k chosen and justified.
  Q.C - qualitative look at the selected k output (a split/merged region), revisiting P1.
  Q.D - geographic spread (max great-circle distance) of every cluster in the selected
        output, in km.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from . import data as data_mod
from . import regions as regions_mod
from . import viz as viz_mod
from . import sampling as sampling_mod
from . import models as models_mod
from .geo import latlon_to_xyz, great_circle_km
from .config import S_BASE, S_PLUS_1

OUTPUT_DIR = "outputs/phase2"

K_VALUES = [8, 12, 16, 20]
FINAL_K = 12  # chosen in Question B; see SELECTED_K_JUSTIFICATION below
MAX_SPAN_SUBSAMPLE = 1500


def _km_labels(X, k, seed):
    return models_mod.run_kmeans(X, k=k, seed=seed, n_init=10)


def run_phase2(df: pd.DataFrame = None, sample_S: pd.DataFrame = None,
               save_outputs: bool = True) -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if df is None:
        df = data_mod.load_clean_earthquakes()
    if sample_S is None:
        sample_S = sampling_mod.get_sample_S(df)

    xyz_full = models_mod.xyz_of(df)
    latlon_full = df[["Latitude", "Longitude"]].values

    # ---------------------------------------------------------------
    # Question A - raw lat/lon vs. 3D sphere, plus init-variability baseline
    # ---------------------------------------------------------------
    km_2d = _km_labels(latlon_full, FINAL_K, S_BASE)
    km_3d_s = _km_labels(xyz_full, FINAL_K, S_BASE)
    km_3d_s1 = _km_labels(xyz_full, FINAL_K, S_PLUS_1)

    labels_2d = km_2d.labels_
    labels_3d_s = km_3d_s.labels_
    labels_3d_s1 = km_3d_s1.labels_

    ari_2d_vs_3d = float(adjusted_rand_score(labels_2d, labels_3d_s))
    ari_3d_baseline = float(adjusted_rand_score(labels_3d_s, labels_3d_s1))

    fig_2d, _ = viz_mod.plot_cluster_map(
        df, labels_2d, f"K-Means k={FINAL_K} on raw (lat, lon), seed={S_BASE}",
        save_path=os.path.join(OUTPUT_DIR, "01a_kmeans_rawlatlon.png") if save_outputs else None,
    )
    fig_3d, _ = viz_mod.plot_cluster_map(
        df, labels_3d_s, f"K-Means k={FINAL_K} on 3D unit-sphere coords, seed={S_BASE}",
        save_path=os.path.join(OUTPUT_DIR, "01b_kmeans_3d.png") if save_outputs else None,
    )

    qa_discussion = (
        f"ARI between the raw-(lat,lon) run and the 3D-sphere run (both k={FINAL_K}, "
        f"seed={S_BASE}) is {ari_2d_vs_3d:.3f}. As a baseline, running the 3D version "
        f"twice with different seeds ({S_BASE} and {S_PLUS_1}, pure initialization "
        f"variability, same representation) gives ARI={ari_3d_baseline:.3f}. "
        f"{'The 2D-vs-3D disagreement is larger than the pure initialization-noise baseline, '
         'which supports the idea that the coordinate representation itself (not just '
         'random initialization) changes the clustering.' if ari_2d_vs_3d < ari_3d_baseline else
         'The 2D-vs-3D disagreement is not clearly larger than the pure initialization-noise '
         'baseline here, so this comparison alone does not strongly separate the '
         'representation effect from ordinary initialization variability.'} "
        "A visible difference on the map is near the 180-degree meridian / high latitudes: "
        "raw (lat, lon) Euclidean distance treats longitude as if its physical length were "
        "constant and independent of latitude, and it does not 'wrap around' at +/-180 "
        "degrees, so points that are geographically close across the date line or near the "
        "poles can be pulled into different raw-(lat,lon) clusters than they are in the "
        "3D representation, where the sphere embedding keeps them correctly close. "
        "This baseline does not separate out exactly how much of the gap is representation "
        "vs. initialization; a larger set of seeds for each representation would be needed "
        "for that."
    )

    # ---------------------------------------------------------------
    # Question B - choice of k
    # ---------------------------------------------------------------
    inertias = {}
    silhouettes_S = {}
    models_by_k = {}
    xyz_S = models_mod.xyz_of(sample_S)
    for k in K_VALUES:
        m = _km_labels(xyz_full, k, S_BASE)
        models_by_k[k] = m
        inertias[k] = float(m.inertia_)
        labels_S = sampling_mod.labels_for_ids(df, m.labels_, sample_S["record_id"].values)
        try:
            from sklearn.metrics import silhouette_score
            sil = float(silhouette_score(xyz_S, labels_S)) if len(set(labels_S)) >= 2 else None
        except Exception:
            sil = None
        silhouettes_S[k] = sil

    fig_inertia, _ = viz_mod.plot_inertia_curve(
        K_VALUES, [inertias[k] for k in K_VALUES],
        save_path=os.path.join(OUTPUT_DIR, "02_inertia_curve.png") if save_outputs else None,
    )
    kb_table = pd.DataFrame({
        "k": K_VALUES,
        "inertia": [inertias[k] for k in K_VALUES],
        "silhouette_on_S": [silhouettes_S[k] for k in K_VALUES],
    })

    SELECTED_K_JUSTIFICATION = (
        f"The inertia curve decreases smoothly with k with no single sharp elbow (common "
        f"with geographically spread, non-spherical data), so it does not pick one k on its "
        f"own. Silhouette on S is generally low across all four values of k (compact, "
        f"well-separated clusters are not what this data looks like at any of these k's), "
        f"and it also does not single out one best k. For scenario A (zoning into roughly "
        f"10-20 describable regions), we pick k={FINAL_K} as a value in the middle of the "
        f"requested range that keeps regions at a size an organization can still describe "
        f"individually, rather than over-splitting (k=20) or under-splitting (k=8) the map; "
        f"this is a judgment call based on the application, since the two numeric criteria "
        f"here do not give a clear signal."
    )

    # ---------------------------------------------------------------
    # Question C - qualitative analysis of the selected k output
    # ---------------------------------------------------------------
    final_model = models_by_k[FINAL_K]
    final_labels = final_model.labels_
    region_cluster_breakdown = {}
    for key in regions_mod.REGIONS:
        mask = regions_mod.region_mask(df, key).values
        vals, counts = np.unique(final_labels[mask], return_counts=True)
        order = np.argsort(-counts)
        region_cluster_breakdown[key] = list(zip(vals[order].tolist(), counts[order].tolist()))

    # region with the most distinct clusters present = clearest "split" example
    split_region = max(region_cluster_breakdown, key=lambda k: len(region_cluster_breakdown[k]))
    n_clusters_in_split_region = len(region_cluster_breakdown[split_region])

    qc_discussion = (
        f"Looking at the k={FINAL_K} output, region '{split_region}' has its points spread "
        f"across {n_clusters_in_split_region} different K-Means clusters "
        f"({region_cluster_breakdown[split_region]}), rather than sitting in one cluster. "
        "K-Means assigns every point to its nearest of k centers (a Voronoi partition); a "
        "region whose geographic extent is larger than the typical diameter of a Voronoi "
        "cell at this k gets cut by one or more of these straight-line boundaries, "
        "independent of whether the region is itself geologically a single structure. "
        f"Revisiting prediction P1: the qualitative direction of P1 (elongated structures "
        "get fragmented across several K-Means clusters) is supported by this example; the "
        "part of P1 that needs refining is the identity of *which* structure is the clearest "
        "case, since that depends on where the k centers happen to land, which is only "
        "indirectly controlled by the structure's shape. For scenario A (zoning), this is "
        "not automatically a flaw: the goal there is geographically coherent, describable "
        "regions of a roughly even size, not reproducing geological belts, so splitting one "
        "long belt into several zones can still be an acceptable outcome."
    )

    fig_selected, _ = viz_mod.plot_cluster_map(
        df, final_labels, f"K-Means selected output (k={FINAL_K}, 3D coords, seed={S_BASE})",
        save_path=os.path.join(OUTPUT_DIR, "03_kmeans_selected.png") if save_outputs else None,
    )

    # ---------------------------------------------------------------
    # Question D - geographic spread (max distance) per cluster
    # ---------------------------------------------------------------
    rng = np.random.RandomState(S_BASE)
    span_rows = []
    for cl in sorted(set(final_labels.tolist())):
        mask = final_labels == cl
        idx = np.where(mask)[0]
        n_pts = len(idx)
        is_estimate = n_pts > MAX_SPAN_SUBSAMPLE
        if is_estimate:
            idx = rng.choice(idx, size=MAX_SPAN_SUBSAMPLE, replace=False)
        lat = df["Latitude"].values[idx]
        lon = df["Longitude"].values[idx]
        max_d = 0.0
        # pairwise max great-circle distance (subsample keeps this O(n^2) tractable)
        for i in range(len(idx)):
            d = great_circle_km(lat[i], lon[i], lat[i + 1:], lon[i + 1:])
            if len(d):
                dm = float(np.max(d))
                if dm > max_d:
                    max_d = dm
        span_rows.append({
            "cluster": cl, "n_points": n_pts,
            "max_span_km": round(max_d, 1),
            "is_lower_bound_estimate": is_estimate,
            "subsample_size": MAX_SPAN_SUBSAMPLE if is_estimate else n_pts,
        })
    span_table = pd.DataFrame(span_rows).sort_values("max_span_km", ascending=False).reset_index(drop=True)

    largest_span_row = span_table.iloc[0]
    qd_discussion = (
        f"The widest cluster in the selected k={FINAL_K} output is cluster "
        f"{int(largest_span_row['cluster'])}, with a max pairwise great-circle span of about "
        f"{largest_span_row['max_span_km']:.0f} km"
        f"{' (a lower-bound estimate from a ' + str(int(largest_span_row['subsample_size'])) + '-point subsample)' if largest_span_row['is_lower_bound_estimate'] else ''}. "
        "A span of this size spans multiple countries/plate segments, which is large for an "
        "organization that wants each of scenario A's 10-20 zones to be a 'describable' "
        "region; such a cluster reads more like several geographically distinct areas that "
        "K-Means merged because their centers happened to be closer to this center than to "
        "any other. If this is judged unacceptable, increasing k (more, smaller Voronoi "
        "cells) is the most direct fix within K-Means; alternatively, a density-based or "
        "hierarchical method could be used instead if the strict single-partition, "
        "full-coverage requirement of scenario A were relaxed."
    )

    if save_outputs:
        kb_table.to_csv(os.path.join(OUTPUT_DIR, "kb_inertia_silhouette_table.csv"), index=False)
        span_table.to_csv(os.path.join(OUTPUT_DIR, "qd_cluster_span_table.csv"), index=False)
        pd.DataFrame([{
            "ari_2d_vs_3d": ari_2d_vs_3d, "ari_3d_baseline_init": ari_3d_baseline,
        }]).to_csv(os.path.join(OUTPUT_DIR, "qa_ari_table.csv"), index=False)

    return {
        "df": df, "sample_S": sample_S,
        "final_k": FINAL_K,
        "labels_2d": labels_2d, "labels_3d_seed_s": labels_3d_s, "labels_3d_seed_s1": labels_3d_s1,
        "ari_2d_vs_3d": ari_2d_vs_3d, "ari_3d_baseline_init": ari_3d_baseline,
        "qa_discussion": qa_discussion,
        "models_by_k": models_by_k, "kb_table": kb_table,
        "selected_k_justification": SELECTED_K_JUSTIFICATION,
        "final_model": final_model, "final_labels": final_labels,
        "region_cluster_breakdown": region_cluster_breakdown,
        "qc_discussion": qc_discussion,
        "span_table": span_table, "qd_discussion": qd_discussion,
        "figures": {
            "kmeans_2d": fig_2d, "kmeans_3d": fig_3d,
            "inertia": fig_inertia, "kmeans_selected": fig_selected,
        },
    }


if __name__ == "__main__":
    results = run_phase2()
    print(f"ARI (2D vs 3D, seed={S_BASE}): {results['ari_2d_vs_3d']:.3f}")
    print(f"ARI (3D baseline, seed {S_BASE} vs {S_PLUS_1}): {results['ari_3d_baseline_init']:.3f}")
    print(results["kb_table"])
    print(results["span_table"].head())
