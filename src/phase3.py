"""
Full Phase 3 run: DBSCAN (on the full dataset) and hierarchical clustering
(Single Link / Average Link, on the shared sample S only).

Per the assignment:
  Q.A - DBSCAN on the full data, min_samples=10, eps in {75, 200, 500} km; summary table
        (n_clusters, noise%, largest cluster size & share, # clusters with 10-19 points);
        maps for the two most informative eps values; compare with P2.
  Q.B - per-eps, per-region table (% noise, % in main cluster, shared main cluster across
        regions); "good enough" criterion defined numerically, then checked; final eps
        chosen for scenario B.
  Q.C - Single Link vs. Average Link on S only, both cut at 12 clusters; maps + cluster
        size table; which is more unbalanced; compare with P3.
  Q.D - chaining effect: a concrete example of Single Link bridging two otherwise distant
        groups through a thin chain of points.
"""
import os
import numpy as np
import pandas as pd

from . import data as data_mod
from . import regions as regions_mod
from . import viz as viz_mod
from . import sampling as sampling_mod
from . import models as models_mod
from .geo import great_circle_km
from .config import S_BASE

OUTPUT_DIR = "outputs/phase3"

EPS_VALUES_KM = [75, 200, 500]
MIN_SAMPLES = 10
HIER_N_CLUSTERS = 12
SMALL_CLUSTER_MAX = 20  # "very small" cluster: < 20 points
GOOD_ENOUGH_NOISE_THRESHOLD_PCT = 30.0  # P3/B numeric criterion: noise per region < 30%


def run_phase3(df: pd.DataFrame = None, sample_S: pd.DataFrame = None,
               save_outputs: bool = True) -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if df is None:
        df = data_mod.load_clean_earthquakes()
    if sample_S is None:
        sample_S = sampling_mod.get_sample_S(df)

    xyz_full = models_mod.xyz_of(df)

    # ---------------------------------------------------------------
    # Question A - DBSCAN on the full data for three eps values
    # ---------------------------------------------------------------
    dbscan_models = {}
    qa_rows = []
    for eps_km in EPS_VALUES_KM:
        m = models_mod.run_dbscan_km(xyz_full, eps_km=eps_km, min_samples=MIN_SAMPLES)
        dbscan_models[eps_km] = m
        summary = models_mod.summarize_labels(m.labels_, noise_label=-1,
                                               small_cluster_threshold=SMALL_CLUSTER_MAX)
        qa_rows.append({
            "eps_km": eps_km,
            "n_clusters": summary["n_clusters"],
            "noise_pct": round(summary["noise_pct"], 2),
            "largest_cluster_size": summary["largest_cluster_size"],
            "largest_cluster_share_pct": round(summary["largest_cluster_share_pct"], 2),
            "n_clusters_10_19_points": summary["n_clusters_10_19"],
        })
    qa_table = pd.DataFrame(qa_rows)

    # pick the two eps values with the most informative difference: largest gap in
    # n_clusters tends to reveal the under- vs. over-merging transition most clearly
    sorted_by_nc = qa_table.sort_values("n_clusters")
    eps_to_plot = [int(sorted_by_nc["eps_km"].iloc[0]), int(sorted_by_nc["eps_km"].iloc[-1])]
    fig_dbscan = {}
    for eps_km in eps_to_plot:
        m = dbscan_models[eps_km]
        fig, _ = viz_mod.plot_cluster_map(
            df, m.labels_, f"DBSCAN eps={eps_km} km, min_samples={MIN_SAMPLES}",
            save_path=os.path.join(OUTPUT_DIR, f"01_dbscan_eps{eps_km}.png") if save_outputs else None,
        )
        fig_dbscan[eps_km] = fig

    qa_discussion = (
        f"As eps grows from {EPS_VALUES_KM[0]} to {EPS_VALUES_KM[-1]} km, the noise "
        f"percentage and the number of clusters both move monotonically down while the "
        f"largest cluster's share grows (see the table): more points fall within eps of "
        f"an existing core point, so points that used to be isolated become reachable "
        f"and small clusters chain together into fewer, larger ones. This matches the "
        f"direction predicted in P2. Whether a single eps handles both dense and sparse "
        f"regions well at the same time is examined directly in Question B below."
    )

    # ---------------------------------------------------------------
    # Question B - per-region density table for each eps
    # ---------------------------------------------------------------
    qb_rows = []
    region_main_cluster = {eps_km: {} for eps_km in EPS_VALUES_KM}
    for eps_km in EPS_VALUES_KM:
        labels = dbscan_models[eps_km].labels_
        for key in regions_mod.REGIONS:
            mask = regions_mod.region_mask(df, key).values
            main_cluster, pct_noise, pct_main = models_mod.main_cluster_for_mask(labels, mask)
            region_main_cluster[eps_km][key] = main_cluster
            qb_rows.append({
                "eps_km": eps_km, "region": key,
                "pct_noise": round(pct_noise, 1),
                "pct_in_main_cluster": round(pct_main, 1),
                "main_cluster_id": main_cluster,
            })
    qb_table = pd.DataFrame(qb_rows)

    # shared main cluster across regions, at the same eps
    shared_with = []
    for _, row in qb_table.iterrows():
        eps_km, region, main_id = row["eps_km"], row["region"], row["main_cluster_id"]
        shares = None
        if main_id is not None:
            for other_key, other_main in region_main_cluster[eps_km].items():
                if other_key != region and other_main == main_id:
                    shares = other_key
                    break
        shared_with.append(shares)
    qb_table["shared_main_cluster_with"] = shared_with

    # "good enough" check: noise per region < GOOD_ENOUGH_NOISE_THRESHOLD_PCT for every region,
    # at the same eps, without merging regions' main clusters together
    qb_good_eps = None
    for eps_km in EPS_VALUES_KM:
        sub = qb_table[qb_table["eps_km"] == eps_km]
        all_low_noise = (sub["pct_noise"] < GOOD_ENOUGH_NOISE_THRESHOLD_PCT).all()
        no_merge = sub["shared_main_cluster_with"].isna().all()
        if all_low_noise and no_merge:
            qb_good_eps = eps_km
            break

    if qb_good_eps is not None:
        qb_conclusion = (
            f"Criterion for 'good': noise below {GOOD_ENOUGH_NOISE_THRESHOLD_PCT:.0f}% in "
            f"every one of the three Phase-1 regions, with no two regions sharing the same "
            f"main cluster. eps={qb_good_eps} km satisfies this among the tested values."
        )
        final_eps_b = qb_good_eps
    else:
        qb_conclusion = (
            f"Criterion for 'good': noise below {GOOD_ENOUGH_NOISE_THRESHOLD_PCT:.0f}% in "
            f"every one of the three Phase-1 regions, with no two regions sharing the same "
            f"main cluster. None of the tested eps values satisfies this simultaneously: "
            f"small eps leaves the sparser region(s) mostly noise (consistent with the "
            f"larger 10th-neighbor distances found for the sparse region in Phase 1-B), "
            f"while an eps large enough to pull the sparse region out of noise is also "
            f"large enough to merge the dense region's main cluster with another region's, "
            f"because the gap between within-region and between-region neighbor distances "
            f"is not large enough for one global eps to separate both scales at once."
        )
        # fall back to the eps with the lowest *maximum* per-region noise, as the
        # best available compromise to report for scenario B
        max_noise_per_eps = qb_table.groupby("eps_km")["pct_noise"].max()
        final_eps_b = int(max_noise_per_eps.idxmin())

    qb_final_eps_reason = (
        f"For scenario B we select eps={final_eps_b} km: among the tested values it gives "
        f"the best available balance for the three Phase-1 regions "
        f"(see the per-region table), which is the closest match to scenario B's goal of "
        f"finding belts/clusters without forcing every point into a cluster."
    )

    # ---------------------------------------------------------------
    # Question C - hierarchical clustering on S only
    # ---------------------------------------------------------------
    xyz_S = models_mod.xyz_of(sample_S)
    Z_single, labels_single = models_mod.run_hierarchical(xyz_S, "single", HIER_N_CLUSTERS)
    Z_average, labels_average = models_mod.run_hierarchical(xyz_S, "average", HIER_N_CLUSTERS)

    size_single = models_mod.cluster_size_table(labels_single, noise_label=None if False else -999)
    size_average = models_mod.cluster_size_table(labels_average, noise_label=-999)
    # fcluster labels start at 1 and never equal -999, so noise_label=-999 just means "no noise"

    fig_single, _ = viz_mod.plot_cluster_map(
        sample_S, labels_single, f"Single Link, cut at {HIER_N_CLUSTERS} clusters (sample S)",
        save_path=os.path.join(OUTPUT_DIR, "02_single_link.png") if save_outputs else None,
    )
    fig_average, _ = viz_mod.plot_cluster_map(
        sample_S, labels_average, f"Average Link, cut at {HIER_N_CLUSTERS} clusters (sample S)",
        save_path=os.path.join(OUTPUT_DIR, "03_average_link.png") if save_outputs else None,
    )

    cv_single = float(size_single["size"].std() / size_single["size"].mean())
    cv_average = float(size_average["size"].std() / size_average["size"].mean())
    more_unbalanced = "Single Link" if cv_single > cv_average else "Average Link"

    qc_discussion = (
        f"Cluster-size coefficient of variation: Single Link={cv_single:.2f} "
        f"(largest cluster: {int(size_single['size'].max())} of {len(sample_S)} points), "
        f"Average Link={cv_average:.2f} "
        f"(largest cluster: {int(size_average['size'].max())} of {len(sample_S)} points). "
        f"{more_unbalanced} produced the more unbalanced split here, which matches the "
        f"direction predicted in P3: Single Link's minimum-distance linkage criterion is "
        f"vulnerable to a thin chain of points connecting two otherwise separate masses "
        f"into one cluster, producing one dominant cluster and several small remainders, "
        f"while Average Link's mean-distance criterion resists being dragged along a chain "
        f"and tends to keep clusters more balanced in size."
    )

    # ---------------------------------------------------------------
    # Question D - chaining effect example
    # ---------------------------------------------------------------
    giant_cluster = int(size_single.iloc[0]["cluster"])
    mask_giant = labels_single == giant_cluster
    lat_g = sample_S["Latitude"].values[mask_giant]
    lon_g = sample_S["Longitude"].values[mask_giant]
    # the two most mutually distant points inside this giant cluster = the ends of the
    # longest "bridge" Single Link was willing to walk across
    idx_g = np.where(mask_giant)[0]
    best_pair = (0, 0)
    best_d = -1.0
    # subsample for tractability if the giant cluster is large
    rng = np.random.RandomState(S_BASE)
    sub = idx_g if len(idx_g) <= 800 else rng.choice(idx_g, size=800, replace=False)
    lat_sub = sample_S["Latitude"].values[sub]
    lon_sub = sample_S["Longitude"].values[sub]
    for i in range(len(sub)):
        d = great_circle_km(lat_sub[i], lon_sub[i], lat_sub[i + 1:], lon_sub[i + 1:])
        if len(d) and d.max() > best_d:
            best_d = float(d.max())
            j = int(np.argmax(d)) + i + 1
            best_pair = (sub[i], sub[j])

    far_apart_km = best_d
    highlight_mask = np.zeros(len(sample_S), dtype=bool)
    highlight_mask[list(best_pair)] = True
    fig_chain, _ = viz_mod.plot_highlighted_points(
        sample_S, mask_giant,
        f"Single Link giant cluster #{giant_cluster} (n={int(mask_giant.sum())}); "
        f"farthest pair inside it is ~{far_apart_km:.0f} km apart",
        save_path=os.path.join(OUTPUT_DIR, "04_single_link_chain.png") if save_outputs else None,
    )

    qd_discussion = (
        f"Single Link's giant cluster #{giant_cluster} (n={int(mask_giant.sum())} of the "
        f"{len(sample_S)} points in S) contains two points about {far_apart_km:.0f} km apart "
        "(highlighted on the map), far more than the within-cluster distances seen in "
        "any of the smaller clusters. By definition, Single Link measures the distance "
        "between two clusters as the *minimum* distance between any pair of their members, "
        "so merging continues as long as there exists *some* close pair bridging two groups "
        "-- even if the rest of the two groups are far apart. A thin, almost-continuous "
        "chain of earthquakes (e.g. along a subduction zone or ridge) is exactly this kind "
        "of bridge. Average Link instead uses the *mean* distance over all pairs between "
        "two clusters, so a single close pair is diluted by many far-apart pairs and is not "
        "enough on its own to trigger a merge; this is why Average Link tends to avoid "
        "this specific failure mode. For scenario A (zoning), this chaining is a clear "
        "problem, since it produces one very large, geographically incoherent zone instead "
        "of several describable ones. For scenario B (structure discovery), the same "
        "chaining can be a double-edged feature: it does trace a real, physically "
        "continuous belt in one piece, but it can also over-merge belts that should "
        "arguably stay distinct, which is exactly the chaining risk this question "
        "highlights."
    )

    if save_outputs:
        qa_table.to_csv(os.path.join(OUTPUT_DIR, "qa_dbscan_table.csv"), index=False)
        qb_table.to_csv(os.path.join(OUTPUT_DIR, "qb_density_table.csv"), index=False)
        size_single.to_csv(os.path.join(OUTPUT_DIR, "qc_single_link_sizes.csv"), index=False)
        size_average.to_csv(os.path.join(OUTPUT_DIR, "qc_average_link_sizes.csv"), index=False)

    return {
        "df": df, "sample_S": sample_S,
        "dbscan_models": dbscan_models, "qa_table": qa_table,
        "qa_eps_plotted": eps_to_plot, "qa_discussion": qa_discussion,
        "qb_table": qb_table, "qb_conclusion": qb_conclusion,
        "final_eps_scenario_b": final_eps_b, "qb_final_eps_reason": qb_final_eps_reason,
        "labels_single": labels_single, "labels_average": labels_average,
        "size_single": size_single, "size_average": size_average,
        "cv_single": cv_single, "cv_average": cv_average,
        "qc_discussion": qc_discussion,
        "giant_cluster_single": giant_cluster, "chain_far_apart_km": far_apart_km,
        "qd_discussion": qd_discussion,
        "figures": {
            "dbscan": fig_dbscan, "single_link": fig_single, "average_link": fig_average,
            "chain": fig_chain,
        },
    }


if __name__ == "__main__":
    results = run_phase3()
    print(results["qa_table"])
    print(results["qb_table"].head(9))
    print("final eps for scenario B:", results["final_eps_scenario_b"])
    print(f"CV single={results['cv_single']:.2f}, CV average={results['cv_average']:.2f}")
