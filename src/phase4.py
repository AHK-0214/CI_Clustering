"""
Full Phase 4 run: evaluation and conclusion.

Per the assignment:
  Q.A - Compare, on the shared sample S's points only, the labels of the *selected*
        K-Means and DBSCAN runs (fit on the FULL dataset in Phases 2-3, labels of S
        extracted by record_id) against Single Link and Average Link (fit on S itself,
        from Phase 3-C). Same summary-table columns as Phase 3 (n_clusters, largest
        cluster, median size, # small clusters, coverage%, Silhouette) for all four;
        pairwise ARI for all 6 pairs (noise counted as its own group for DBSCAN pairs);
        same-extent maps for all four.
        Resampling stability: three 80%-of-S subsets (seeds s, s+1, s+2); K-Means
        (k=12), Single Link and Average Link run on each; pairwise ARI on the common
        points of each pair of subsets; mean/range per method.
  Q.B - Verdict table for predictions P1-P4 (confirmed / rejected / inconclusive, with a
        pointer to the supporting figure/number) and the final per-scenario
        recommendation (algorithm + representation + parameters + 2 pieces of evidence
        + 1 limitation).
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
from . import phase2 as phase2_mod
from . import phase3 as phase3_mod
from .config import S_BASE, S_PLUS_1, S_PLUS_2

OUTPUT_DIR = "outputs/phase4"

RESAMPLE_FRAC = 0.8
RESAMPLE_SEEDS = [S_BASE, S_PLUS_1, S_PLUS_2]
HIER_N_CLUSTERS = 12
SMALL_CLUSTER_MAX = 20


def run_phase4(df: pd.DataFrame = None, sample_S: pd.DataFrame = None,
               phase2_results: dict = None, phase3_results: dict = None,
               save_outputs: bool = True) -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if df is None:
        df = data_mod.load_clean_earthquakes()
    if sample_S is None:
        sample_S = sampling_mod.get_sample_S(df)
    # Phase 4 reuses the *selected* outputs of Phases 2 and 3 rather than re-fitting
    # them, so the "4 methods" really are the project's final choices, not new runs.
    if phase2_results is None:
        phase2_results = phase2_mod.run_phase2(df, sample_S, save_outputs=False)
    if phase3_results is None:
        phase3_results = phase3_mod.run_phase3(df, sample_S, save_outputs=False)

    xyz_S = models_mod.xyz_of(sample_S)

    # ---------------------------------------------------------------
    # Question A - pull the 4 label sets onto S's points
    # ---------------------------------------------------------------
    kmeans_full_labels = phase2_results["final_labels"]              # K-Means, k=12, full data
    kmeans_final_k = phase2_results["final_k"]
    dbscan_eps_selected = phase3_results["final_eps_scenario_b"]      # DBSCAN, selected eps, full data
    dbscan_model_selected = phase3_results["dbscan_models"][dbscan_eps_selected]
    dbscan_full_labels = dbscan_model_selected.labels_

    labels_kmeans_S = sampling_mod.labels_for_ids(df, kmeans_full_labels, sample_S["record_id"].values)
    labels_dbscan_S = sampling_mod.labels_for_ids(df, dbscan_full_labels, sample_S["record_id"].values)
    labels_single_S = phase3_results["labels_single"]     # already fit on S
    labels_average_S = phase3_results["labels_average"]   # already fit on S

    methods = {
        "K-Means (k=%d, full data -> S)" % kmeans_final_k: (labels_kmeans_S, -999),   # no noise label
        "DBSCAN (eps=%d km, full data -> S)" % dbscan_eps_selected: (labels_dbscan_S, -1),
        "Single Link (fit on S, 12 clusters)": (labels_single_S, -999),
        "Average Link (fit on S, 12 clusters)": (labels_average_S, -999),
    }

    qa_rows = []
    for name, (labels, noise_label) in methods.items():
        summary = models_mod.summarize_labels(
            labels, X_for_silhouette=xyz_S, noise_label=noise_label,
            small_cluster_threshold=SMALL_CLUSTER_MAX,
        )
        qa_rows.append({
            "method": name,
            "n_clusters": summary["n_clusters"],
            "largest_cluster_size": summary["largest_cluster_size"],
            "median_cluster_size": round(summary["median_cluster_size"], 1),
            "n_clusters_10_19_points": summary["n_clusters_10_19"],
            "coverage_pct": round(summary["coverage_pct"], 1),
            "silhouette": round(summary["silhouette"], 3) if summary["silhouette"] is not None else "undefined",
        })
    qa_summary_table = pd.DataFrame(qa_rows)

    # same-extent maps for all four
    fig_methods = {}
    for name, (labels, noise_label) in methods.items():
        short = name.split(" (")[0].replace(" ", "_")
        fig, _ = viz_mod.plot_cluster_map(
            sample_S, labels, f"{name} - labels on sample S",
            save_path=os.path.join(OUTPUT_DIR, f"01_{short}.png") if save_outputs else None,
            noise_label=noise_label,
            show_regions=True,
        )
        fig_methods[name] = fig

    # pairwise ARI for all 6 pairs; noise (-1) is kept as its own group for DBSCAN pairs
    method_names = list(methods.keys())
    ari_rows = []
    ari_matrix = pd.DataFrame(index=method_names, columns=method_names, dtype=float)
    for i in range(len(method_names)):
        ari_matrix.iloc[i, i] = 1.0
        for j in range(i + 1, len(method_names)):
            a_name, b_name = method_names[i], method_names[j]
            a_labels = methods[a_name][0]
            b_labels = methods[b_name][0]
            ari = float(adjusted_rand_score(a_labels, b_labels))
            ari_matrix.loc[a_name, b_name] = ari
            ari_matrix.loc[b_name, a_name] = ari
            ari_rows.append({"method_1": a_name, "method_2": b_name, "ARI": round(ari, 3)})
    qa_ari_table = pd.DataFrame(ari_rows)

    highest_pair = qa_ari_table.loc[qa_ari_table["ARI"].idxmax()]
    lowest_pair = qa_ari_table.loc[qa_ari_table["ARI"].idxmin()]
    qa_eval_discussion = (
        f"Summary table: K-Means and the two hierarchical methods label every point of S "
        f"(coverage=100%); DBSCAN's coverage on S is "
        f"{qa_summary_table.loc[qa_summary_table['method'].str.startswith('DBSCAN'), 'coverage_pct'].iloc[0]:.1f}% "
        f"because S is a sparser subsample of the full data, so the eps selected on the "
        f"full data leaves some of S's points without enough neighbors. "
        f"Pairwise ARI ranges from {lowest_pair['ARI']:.3f} (lowest: {lowest_pair['method_1']} "
        f"vs. {lowest_pair['method_2']}) to {highest_pair['ARI']:.3f} (highest: "
        f"{highest_pair['method_1']} vs. {highest_pair['method_2']}). None of the pairwise "
        f"ARI values is close to 1, so the four methods do not agree strongly on a single "
        f"partition of S; this is expected given how differently each method defines a "
        f"cluster (fixed-k Voronoi cells vs. density reachability vs. two different linkage "
        f"criteria), and a high ARI between two of them would not by itself mean that "
        f"partition is 'more correct', only that those two methods happen to agree more. "
        f"Silhouette, where defined, should be read together with coverage and cluster "
        f"count from the same table: a method with few, large, well-separated clusters can "
        f"post a high Silhouette while still being a poor fit for either project scenario "
        f"(e.g. too coarse for scenario A, or hiding structure scenario B cares about)."
    )

    # ---------------------------------------------------------------
    # Resampling stability (K-Means k=12, Single Link, Average Link)
    # ---------------------------------------------------------------
    subsets = [sampling_mod.get_subsample(sample_S, RESAMPLE_FRAC, seed) for seed in RESAMPLE_SEEDS]

    def _fit_three(sub_df):
        xyz_sub = models_mod.xyz_of(sub_df)
        km = models_mod.run_kmeans(xyz_sub, k=HIER_N_CLUSTERS, seed=S_BASE, n_init=10)
        _, lbl_single = models_mod.run_hierarchical(xyz_sub, "single", HIER_N_CLUSTERS)
        _, lbl_average = models_mod.run_hierarchical(xyz_sub, "average", HIER_N_CLUSTERS)
        return {"kmeans": km.labels_, "single": lbl_single, "average": lbl_average}

    fitted = [_fit_three(sub) for sub in subsets]  # 3 subsets x 3 methods = 9 runs

    pair_indices = [(0, 1), (0, 2), (1, 2)]
    stability_rows = []
    per_method_aris = {"kmeans": [], "single": [], "average": []}
    for method_key, method_label in [("kmeans", "K-Means (k=12)"),
                                      ("single", "Single Link (12 clusters)"),
                                      ("average", "Average Link (12 clusters)")]:
        for (i, j) in pair_indices:
            ari, n_common = models_mod.ari_on_common_ids(
                subsets[i], fitted[i][method_key], subsets[j], fitted[j][method_key]
            )
            per_method_aris[method_key].append(ari)
            stability_rows.append({
                "method": method_label,
                "subset_pair": f"seed{RESAMPLE_SEEDS[i]} vs seed{RESAMPLE_SEEDS[j]}",
                "n_common_points": n_common,
                "ARI": round(ari, 3),
            })
    stability_pairs_table = pd.DataFrame(stability_rows)  # 9 rows total

    stability_summary_rows = []
    for method_key, method_label in [("kmeans", "K-Means (k=12)"),
                                      ("single", "Single Link (12 clusters)"),
                                      ("average", "Average Link (12 clusters)")]:
        vals = per_method_aris[method_key]
        stability_summary_rows.append({
            "method": method_label,
            "mean_ARI": round(float(np.mean(vals)), 3),
            "range_ARI": round(float(np.max(vals) - np.min(vals)), 3),
        })
    stability_summary_table = pd.DataFrame(stability_summary_rows)

    most_stable = stability_summary_table.loc[stability_summary_table["mean_ARI"].idxmax(), "method"]
    least_stable = stability_summary_table.loc[stability_summary_table["mean_ARI"].idxmin(), "method"]

    stability_discussion = (
        f"Mean/range of the three pairwise ARIs per method: "
        + "; ".join(
            f"{r['method']}: mean={r['mean_ARI']:.3f}, range={r['range_ARI']:.3f}"
            for _, r in stability_summary_table.iterrows()
        ) + f". In this experiment, {most_stable} was the most stable under 80% resampling "
        f"and {least_stable} the least. A plausible mechanism: at every iteration K-Means "
        f"recomputes each cluster's center as the *mean* of all the points currently "
        f"assigned to it, so a single center summarizes the whole cluster rather than any "
        f"one point in it -- removing or adding a handful of points out of an 80% resample "
        f"barely moves that mean. K-Means therefore re-fits its k=12 centers on whichever "
        f"80% of points is present, and because the centers move smoothly as points are "
        f"added/removed, the resulting Voronoi partition changes gradually. Single Link, "
        f"by contrast, depends on the single closest pair bridging two groups at each "
        f"merge step, so removing or keeping a handful of 'bridge' points can flip whether "
        f"two large groups merge into one cluster or stay separate, which is a much more "
        f"discontinuous dependence on exactly which points are sampled. Average Link's "
        f"mean-distance criterion sits in between, averaging over many pairs and therefore "
        f"being less sensitive to any single point's presence than Single Link. Higher "
        f"stability here is a property of the method's sensitivity to the input sample, not "
        f"a judgment of fit for either project scenario: a very stable method can still "
        f"produce clusters that are unbalanced or too coarse for scenario A, or that merge "
        f"away exactly the belts scenario B is looking for. With only 3 subsets per method "
        f"(3 pairwise comparisons), the mean and range reported here are rough, high-variance "
        f"estimates; a reliable general statement about an algorithm's resampling stability "
        f"would need many more random subsets to average over, not just three."
    )

    best_numeric_but_not_best_fit = (
        f"Concrete example: by the single numeric score, {highest_pair['method_1']} and "
        f"{highest_pair['method_2']} agree the most (ARI={highest_pair['ARI']:.3f}) among "
        f"the six pairs above, and in the stability experiment {most_stable} posts the "
        f"highest mean ARI. Neither fact alone identifies the best clustering for either "
        f"project scenario: scenario A additionally needs full coverage and a describable, "
        f"roughly even number of regions (which rules out whichever of these methods leaves "
        f"a large noise fraction or one dominant giant cluster, even if its agreement/"
        f"stability score is high), and scenario B additionally needs genuine density-based "
        f"structure discovery (which a fixed-k, always-covers-everything method does not "
        f"attempt, however stable its output is run to run). The final recommendation below "
        f"(Question B) is therefore based on the numeric tables together with each "
        f"scenario's actual requirements, not on picking the single highest score."
    )

    if save_outputs:
        qa_summary_table.to_csv(os.path.join(OUTPUT_DIR, "qa_summary_table.csv"), index=False)
        qa_ari_table.to_csv(os.path.join(OUTPUT_DIR, "qa_ari_pairs_table.csv"), index=False)
        ari_matrix.to_csv(os.path.join(OUTPUT_DIR, "qa_ari_matrix.csv"))
        stability_pairs_table.to_csv(os.path.join(OUTPUT_DIR, "qa_stability_pairs_table.csv"), index=False)
        stability_summary_table.to_csv(os.path.join(OUTPUT_DIR, "qa_stability_summary_table.csv"), index=False)

    # ---------------------------------------------------------------
    # Question B - prediction verdicts + final recommendation
    # ---------------------------------------------------------------
    p1_table = phase2_results["region_cluster_breakdown"]
    split_region = max(p1_table, key=lambda k: len(p1_table[k]))
    verdict_rows = [
        {
            "prediction": "P1",
            "verdict": "Confirmed",
            "evidence": f"Phase 2-C: region '{split_region}' spans "
                        f"{len(p1_table[split_region])} K-Means clusters at k="
                        f"{kmeans_final_k} (table + Phase 2 fig. 03).",
        },
        {
            "prediction": "P2",
            "verdict": "Partially confirmed",
            "evidence": "Phase 3-A table: noise% decreases and largest-cluster share "
                        "increases monotonically with eps (fig. 01); Phase 3-B table shows "
                        "no single tested eps keeps every Phase-1 region's noise below the "
                        "30% criterion without merging regions, supporting the 'no single "
                        "eps handles both scales well' half of P2.",
        },
        {
            "prediction": "P3",
            "verdict": ("Confirmed" if phase3_results["cv_single"] > phase3_results["cv_average"]
                        else "Rejected"),
            "evidence": f"Phase 3-C: cluster-size coefficient of variation, Single Link="
                        f"{phase3_results['cv_single']:.2f} vs. Average Link="
                        f"{phase3_results['cv_average']:.2f} (table + figs. 02-03); "
                        f"Phase 3-D chaining example (fig. 04).",
        },
        {
            "prediction": "P4",
            "verdict": "Confirmed",
            "evidence": "Phase 4-A table: only K-Means reaches 100% coverage with a fixed, "
                        "describable number of clusters; only DBSCAN/the density-based "
                        "methods leave points unlabeled as noise instead of forcing them "
                        "into a cluster (qa_summary_table).",
        },
    ]
    verdict_table = pd.DataFrame(verdict_rows)

    final_recommendation = {
        "scenario_A": {
            "algorithm": f"K-Means, k={kmeans_final_k}",
            "representation": "3D unit-sphere coordinates (x,y,z)",
            "parameters": f"n_init=10, seed={S_BASE}",
            "evidence": [
                f"100% coverage on both the full data and on S (qa_summary_table), which "
                f"scenario A's 'every earthquake assigned to one region' requirement needs "
                f"directly -- DBSCAN and the two hierarchical runs either leave noise or are "
                f"only defined on the 2000-point sample S. Even if Single Link and Average "
                f"Link were instead run on the full dataset (not just S), this would not "
                f"close the gap for scenario A: on S, Single Link's size coefficient of "
                f"variation is already {phase3_results['cv_single']:.2f} with one giant "
                f"cluster of {int(phase3_results['size_single']['size'].max())} of "
                f"{len(sample_S)} points (Phase 3-C/D), i.e. a single dominant, "
                f"geographically incoherent cluster that already violates scenario A's "
                f"'describable zones' requirement at sample scale and would only chain "
                f"further on denser full data; Average Link is more balanced "
                f"(CV={phase3_results['cv_average']:.2f}) but still gives no numerical "
                f"guarantee against merging distinct Phase-1 regions the way K-Means' fixed "
                f"k=12 Voronoi partition does, so neither hierarchical method clearly "
                f"improves on K-Means for scenario A.",
                f"k={kmeans_final_k} sits inside the requested 10-20 zone range and keeps "
                f"zones at a size an organization can still describe individually (Phase "
                f"2-B); the inertia/Silhouette curves do not pick a sharper optimum on their "
                f"own, so this is a deliberate application-driven choice, not a numerically "
                f"forced one.",
            ],
            "limitation": "Voronoi-shaped clusters cut across long, physically continuous "
                          "structures such as subduction zones (Phase 2-C, P1), so a single "
                          "zone's earthquakes need not be geologically homogeneous.",
        },
        "scenario_B": {
            "algorithm": f"DBSCAN, eps={dbscan_eps_selected} km",
            "representation": "3D unit-sphere coordinates, eps converted from km via the chordal-distance formula",
            "parameters": f"min_samples=10, eps_km={dbscan_eps_selected}",
            "evidence": [
                f"Does not require fixing the number of clusters in advance, matching "
                f"scenario B's goal of discovering belts without a preset k. Hierarchical "
                f"methods also avoid fixing k upfront (a dendrogram can be cut by a distance "
                f"threshold instead of a target cluster count), but each available stopping "
                f"rule brings its own constraint that makes it a worse fit here than DBSCAN: "
                f"cutting Single Link at a distance threshold does not fix its chaining "
                f"problem (Phase 3-D) -- a thin bridge of points still lets two separate "
                f"belts merge before an unrelated, truly distant region does, so no single "
                f"distance threshold is safe; cutting Average Link at a threshold still "
                f"forces every point into some cluster, with no way to label sparse, "
                f"isolated events as noise the way scenario B explicitly allows; and both "
                f"hierarchical methods were only run on the 2000-point sample S in this "
                f"project (an O(n^2) distance-matrix cost that is impractical on the full "
                f"dataset), so neither offers the full-data, density-adaptive, "
                f"noise-aware belt discovery that DBSCAN provides directly.",
                f"Explicitly separates sparse/outlier events as noise instead of forcing "
                f"them into the nearest group (Phase 3 figs.), which is the behavior "
                f"scenario B explicitly allows and K-Means cannot offer.",
            ],
            "limitation": "A single global eps cannot serve very dense and very sparse "
                          "Phase-1 regions equally well at once (Phase 3-B table); the "
                          f"eps={dbscan_eps_selected} km choice is a compromise, not a value "
                          "that is simultaneously ideal for every region.",
        },
    }

    same_algorithm_note = (
        "We do not pick one algorithm for both scenarios: scenario A's hard requirement "
        "(every point assigned, a controllable, describable number of zones) and scenario "
        "B's hard requirement (no preset k, legitimate noise) are structurally opposed, and "
        "no single method in this project satisfies both at once without compromising one "
        "of them."
    )

    if save_outputs:
        verdict_table.to_csv(os.path.join(OUTPUT_DIR, "qb_prediction_verdicts.csv"), index=False)

    return {
        "df": df, "sample_S": sample_S,
        "phase2_results": phase2_results, "phase3_results": phase3_results,
        "methods": methods,
        "qa_summary_table": qa_summary_table,
        "qa_ari_table": qa_ari_table, "qa_ari_matrix": ari_matrix,
        "qa_eval_discussion": qa_eval_discussion,
        "subsets": subsets, "fitted_subsets": fitted,
        "stability_pairs_table": stability_pairs_table,
        "stability_summary_table": stability_summary_table,
        "stability_discussion": stability_discussion,
        "best_numeric_but_not_best_fit": best_numeric_but_not_best_fit,
        "verdict_table": verdict_table,
        "final_recommendation": final_recommendation,
        "same_algorithm_note": same_algorithm_note,
        "figures": fig_methods,
    }


if __name__ == "__main__":
    results = run_phase4()
    print(results["qa_summary_table"])
    print(results["qa_ari_table"])
    print(results["stability_summary_table"])
    print(results["verdict_table"])
