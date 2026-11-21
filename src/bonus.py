"""
Bonus section (optional, 5 points each, independent of one another).

  Bonus Q.A - Controlled Perturbation: displace 5% of the full dataset by 10-20 km in a
              random geographic direction (seed s+3) and re-run the project's selected
              K-Means and DBSCAN with identical parameters/seeds, on the original vs.
              perturbed data; ARI, cluster-count/largest-cluster/noise-% comparison, and
              how many *un*-perturbed points also changed membership.
  Bonus Q.B - HDBSCAN vs. DBSCAN on the shared sample S, with DBSCAN's eps re-tuned on S
              (not reused from Phase 3's full-data value); cluster counts, overall noise%,
              per-Phase-1-region noise%, and a qualitative verdict for scenario B.
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
from . import density as density_mod
from .geo import destination_point, latlon_to_xyz
from .config import S_BASE, S_PLUS_3

OUTPUT_DIR = "outputs/bonus"

# ---------------------------------------------------------------------
# Bonus Q.A - Controlled Perturbation
# ---------------------------------------------------------------------
PERTURB_FRAC = 0.05
PERTURB_MIN_KM = 10.0
PERTURB_MAX_KM = 20.0


def perturb_dataset(df: pd.DataFrame, frac: float = PERTURB_FRAC, seed: int = S_PLUS_3):
    """Move a random `frac` of df's points by a random bearing and a random distance in
    [PERTURB_MIN_KM, PERTURB_MAX_KM] km, using a proper geographic (great-circle)
    displacement (src.geo.destination_point) -- not a flat addition to lat/lon, per the
    assignment. Record order and record_id are preserved; untouched points are untouched.
    Returns (df_perturbed, perturbed_mask)."""
    rng = np.random.RandomState(seed)
    n = len(df)
    n_perturb = int(round(frac * n))
    idx = rng.choice(n, size=n_perturb, replace=False)
    bearings = rng.uniform(0.0, 360.0, size=n_perturb)
    distances = rng.uniform(PERTURB_MIN_KM, PERTURB_MAX_KM, size=n_perturb)

    df_pert = df.copy()
    lat0 = df_pert["Latitude"].values[idx]
    lon0 = df_pert["Longitude"].values[idx]
    new_lat, new_lon = destination_point(lat0, lon0, bearings, distances)
    df_pert.iloc[idx, df_pert.columns.get_loc("Latitude")] = new_lat
    df_pert.iloc[idx, df_pert.columns.get_loc("Longitude")] = new_lon

    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return df_pert, mask


def run_bonus_qa(df: pd.DataFrame = None, phase2_results: dict = None,
                  phase3_results: dict = None, save_outputs: bool = True) -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if df is None:
        df = data_mod.load_clean_earthquakes()

    # Reuse the project's already-selected parameters; do not re-tune anything here.
    final_k = phase2_results["final_k"]
    eps_km_selected = phase3_results["final_eps_scenario_b"]
    min_samples = phase3_results["dbscan_models"][eps_km_selected].min_samples \
        if hasattr(phase3_results["dbscan_models"][eps_km_selected], "min_samples") else 10

    df_pert, perturbed_mask = perturb_dataset(df)
    if save_outputs:
        df.assign(perturbed=perturbed_mask).to_csv(
            os.path.join(OUTPUT_DIR, "qa_perturbation_mask.csv"), index=False)

    xyz_orig = models_mod.xyz_of(df)
    xyz_pert = models_mod.xyz_of(df_pert)

    # ---- K-Means, identical k/seed, original vs. perturbed ----
    km_orig = models_mod.run_kmeans(xyz_orig, k=final_k, seed=S_BASE, n_init=10)
    km_pert = models_mod.run_kmeans(xyz_pert, k=final_k, seed=S_BASE, n_init=10)
    ari_kmeans = float(adjusted_rand_score(km_orig.labels_, km_pert.labels_))

    # ---- DBSCAN, identical eps/min_samples, original vs. perturbed ----
    db_orig = models_mod.run_dbscan_km(xyz_orig, eps_km=eps_km_selected, min_samples=min_samples)
    db_pert = models_mod.run_dbscan_km(xyz_pert, eps_km=eps_km_selected, min_samples=min_samples)
    ari_dbscan = float(adjusted_rand_score(db_orig.labels_, db_pert.labels_))

    def _changed_counts(labels_orig, labels_pert, noise_label):
        """Membership changed for a point = either (a) it flipped between noise and
        non-noise, or (b) it stayed non-noise in both runs but its cluster's *globally
        best-matching* counterpart (found once via a one-to-one Hungarian assignment on
        the full overlap/contingency matrix, not a greedy per-cluster lookup) is not the
        cluster it actually landed in. A one-shot global assignment avoids the same
        "pert" cluster being claimed as the best match by several different original
        clusters, which would otherwise manufacture spurious changes among untouched
        points whenever a run produces many small clusters (common for DBSCAN)."""
        from scipy.optimize import linear_sum_assignment
        labels_orig = np.asarray(labels_orig)
        labels_pert = np.asarray(labels_pert)
        is_noise_orig = labels_orig == noise_label if noise_label is not None else np.zeros(len(labels_orig), dtype=bool)
        is_noise_pert = labels_pert == noise_label if noise_label is not None else np.zeros(len(labels_pert), dtype=bool)

        changed = is_noise_orig != is_noise_pert  # noise status flipped

        orig_clusters = [c for c in np.unique(labels_orig) if c != noise_label]
        pert_clusters = [c for c in np.unique(labels_pert) if c != noise_label]
        if len(orig_clusters) and len(pert_clusters):
            overlap = np.zeros((len(orig_clusters), len(pert_clusters)))
            for i, co in enumerate(orig_clusters):
                mo = labels_orig == co
                for j, cp in enumerate(pert_clusters):
                    overlap[i, j] = np.sum(mo & (labels_pert == cp))
            row_ind, col_ind = linear_sum_assignment(-overlap)  # maximize total overlap
            match = {orig_clusters[i]: pert_clusters[j] for i, j in zip(row_ind, col_ind)}

            both_non_noise = ~is_noise_orig & ~is_noise_pert
            for co, matched_cp in match.items():
                mask_co = (labels_orig == co) & both_non_noise
                # a point under this original cluster "changed" if, in the pert run, it
                # is not in the cluster matched to co
                changed[mask_co] |= labels_pert[mask_co] != matched_cp
            # original clusters with no match at all (more pert clusters than orig, or
            # vice versa) -- any of their surviving members are, by construction, not in
            # the matched cluster, so they are already covered by the loop above; points
            # under an *unmatched* original cluster (only possible if there are strictly
            # fewer pert clusters than orig ones) are marked changed too:
            unmatched_orig = set(orig_clusters) - set(match.keys())
            for co in unmatched_orig:
                mask_co = (labels_orig == co) & both_non_noise
                changed[mask_co] = True
        return changed

    changed_kmeans = _changed_counts(km_orig.labels_, km_pert.labels_, None)
    changed_dbscan = _changed_counts(db_orig.labels_, db_pert.labels_, -1)

    def _summary_row(name, labels_orig, labels_pert, changed, noise_label):
        s_orig = models_mod.summarize_labels(labels_orig, noise_label=noise_label)
        s_pert = models_mod.summarize_labels(labels_pert, noise_label=noise_label)
        n_changed_perturbed = int(changed[perturbed_mask].sum())
        n_changed_unperturbed = int(changed[~perturbed_mask].sum())
        return {
            "method": name,
            "n_clusters_orig": s_orig["n_clusters"], "n_clusters_pert": s_pert["n_clusters"],
            "largest_cluster_orig": s_orig["largest_cluster_size"],
            "largest_cluster_pert": s_pert["largest_cluster_size"],
            "noise_pct_orig": round(s_orig["noise_pct"], 2) if noise_label is not None else None,
            "noise_pct_pert": round(s_pert["noise_pct"], 2) if noise_label is not None else None,
            "n_perturbed_points": int(perturbed_mask.sum()),
            "n_perturbed_changed": n_changed_perturbed,
            "pct_perturbed_changed": round(100.0 * n_changed_perturbed / perturbed_mask.sum(), 1),
            "n_unperturbed_points": int((~perturbed_mask).sum()),
            "n_unperturbed_changed": n_changed_unperturbed,
            "pct_unperturbed_changed": round(100.0 * n_changed_unperturbed / (~perturbed_mask).sum(), 3),
        }

    qa_table = pd.DataFrame([
        _summary_row(f"K-Means (k={final_k})", km_orig.labels_, km_pert.labels_, changed_kmeans, None),
        _summary_row(f"DBSCAN (eps={eps_km_selected} km)", db_orig.labels_, db_pert.labels_, changed_dbscan, -1),
    ])
    qa_table.insert(1, "ARI_orig_vs_perturbed", [round(ari_kmeans, 3), round(ari_dbscan, 3)])

    # one concrete example: a perturbed point whose membership changed, shown on a map
    example_mask = np.zeros(len(df), dtype=bool)
    changed_and_perturbed = np.where(changed_dbscan & perturbed_mask)[0]
    if len(changed_and_perturbed) == 0:
        changed_and_perturbed = np.where(changed_kmeans & perturbed_mask)[0]
    fig_example = None
    if len(changed_and_perturbed) > 0:
        example_idx = int(changed_and_perturbed[0])
        example_mask[example_idx] = True
        fig_example, _ = viz_mod.plot_highlighted_points(
            df, example_mask,
            f"Example perturbed point whose cluster membership changed (record_id="
            f"{int(df['record_id'].iloc[example_idx])})",
            save_path=os.path.join(OUTPUT_DIR, "01_perturbation_example.png") if save_outputs else None,
            show_regions=True,
        )

    more_robust = "K-Means" if ari_kmeans > ari_dbscan else "DBSCAN"
    qa_discussion = (
        f"ARI (original vs. perturbed labels, same records): K-Means={ari_kmeans:.3f}, "
        f"DBSCAN={ari_dbscan:.3f}. In this single perturbation run, {more_robust} shows the "
        f"higher ARI and is therefore the more robust of the two here. K-Means's centers are "
        f"an average over thousands of points, so moving {int(perturbed_mask.sum())} points "
        f"(5% of the data) by only 10-20 km barely shifts any center, which is consistent "
        f"with a high ARI and almost no membership change among unperturbed points; DBSCAN's "
        f"labeling depends on each point's local neighbor count, so a 10-20 km shift can "
        f"push a point across the eps boundary (into or out of noise, or into a neighboring "
        f"cluster) especially in sparser regions, which can also disturb a modest number of "
        f"other points connected to it through density-reachability chains. The unperturbed "
        f"points that still changed membership (reported in the table) show this knock-on "
        f"effect directly: it is not limited to exactly the 5% of points that were moved. "
        f"A single random 5% perturbation is only one draw; it tells us this specific draw "
        f"did not destabilize {more_robust} as much as the other method, not that "
        f"{more_robust} is robust in general -- a different random subset or a different "
        f"direction/distance range could behave differently, especially for points that "
        f"started out near a cluster boundary. A high overall ARI also does not guarantee "
        f"that every individual perturbed point kept its label: ARI is a global agreement "
        f"score over all points, so a small number of flips can be masked by the large "
        f"majority of points (including most of the perturbed ones) whose membership did "
        f"not change; the per-point changed-counts above are what actually answers that "
        f"question, not the ARI by itself."
    )

    if save_outputs:
        qa_table.to_csv(os.path.join(OUTPUT_DIR, "qa_perturbation_table.csv"), index=False)

    return {
        "df": df, "df_perturbed": df_pert, "perturbed_mask": perturbed_mask,
        "km_orig": km_orig, "km_pert": km_pert, "ari_kmeans": ari_kmeans,
        "db_orig": db_orig, "db_pert": db_pert, "ari_dbscan": ari_dbscan,
        "changed_kmeans": changed_kmeans, "changed_dbscan": changed_dbscan,
        "qa_table": qa_table, "qa_discussion": qa_discussion,
        "figures": {"example": fig_example},
    }


# ---------------------------------------------------------------------
# Bonus Q.B - HDBSCAN vs. DBSCAN, on the shared sample S
# ---------------------------------------------------------------------
HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES = 10


def run_bonus_qb(df: pd.DataFrame = None, sample_S: pd.DataFrame = None,
                  save_outputs: bool = True) -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if df is None:
        df = data_mod.load_clean_earthquakes()
    if sample_S is None:
        sample_S = sampling_mod.get_sample_S(df)

    xyz_S = models_mod.xyz_of(sample_S)

    # ---- Re-tune DBSCAN's eps on S itself (S is sparser than the full data) ----
    dist_km_S = density_mod.kth_neighbor_distance_km(sample_S)
    # candidate eps values read off S's own 10th-neighbor-distance distribution
    eps_candidates_km = [
        round(float(np.percentile(dist_km_S, 50)), 1),
        round(float(np.percentile(dist_km_S, 75)), 1),
        round(float(np.percentile(dist_km_S, 90)), 1),
    ]
    candidate_rows = []
    for eps_km in eps_candidates_km:
        m = models_mod.run_dbscan_km(xyz_S, eps_km=eps_km, min_samples=10)
        s = models_mod.summarize_labels(m.labels_, noise_label=-1)
        candidate_rows.append({
            "eps_km": eps_km, "n_clusters": s["n_clusters"], "noise_pct": round(s["noise_pct"], 1),
        })
    candidates_table = pd.DataFrame(candidate_rows)
    # choose the smallest candidate eps that still keeps overall noise under 50% -- i.e.
    # the tightest eps that does not leave the (already sparse) sample mostly unlabeled
    ok = candidates_table[candidates_table["noise_pct"] < 50.0]
    eps_km_S = float(ok["eps_km"].min()) if len(ok) else float(candidates_table["eps_km"].max())
    eps_choice_reason = (
        f"Candidates were read off S's own 10th-neighbor-distance distribution (median="
        f"{eps_candidates_km[0]} km, 75th pct={eps_candidates_km[1]} km, 90th pct="
        f"{eps_candidates_km[2]} km), since Phase 3's full-data eps values assume a denser "
        f"input than the 2000-point S. eps={eps_km_S} km is the tightest of the tested "
        f"values that still keeps overall noise under 50% on S (see candidates table)."
    )

    db = models_mod.run_dbscan_km(xyz_S, eps_km=eps_km_S, min_samples=10)
    db_summary = models_mod.summarize_labels(db.labels_, X_for_silhouette=xyz_S, noise_label=-1)

    # ---- HDBSCAN on the same S / same representation ----
    from sklearn.cluster import HDBSCAN
    hdb = HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES,
                  metric="euclidean")
    hdb.fit(xyz_S)
    hdb_summary = models_mod.summarize_labels(hdb.labels_, X_for_silhouette=xyz_S, noise_label=-1)

    overall_table = pd.DataFrame([
        {"method": f"DBSCAN (eps={eps_km_S} km, min_samples=10)",
         "n_clusters": db_summary["n_clusters"], "noise_pct": round(db_summary["noise_pct"], 1),
         "silhouette": round(db_summary["silhouette"], 3) if db_summary["silhouette"] is not None else "undefined"},
        {"method": f"HDBSCAN (min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, min_samples={HDBSCAN_MIN_SAMPLES})",
         "n_clusters": hdb_summary["n_clusters"], "noise_pct": round(hdb_summary["noise_pct"], 1),
         "silhouette": round(hdb_summary["silhouette"], 3) if hdb_summary["silhouette"] is not None else "undefined"},
    ])

    fig_db, _ = viz_mod.plot_cluster_map(
        sample_S, db.labels_, f"DBSCAN eps={eps_km_S} km, min_samples=10 (sample S)",
        save_path=os.path.join(OUTPUT_DIR, "02_dbscan_S.png") if save_outputs else None,
        show_regions=True,
    )
    fig_hdb, _ = viz_mod.plot_cluster_map(
        sample_S, hdb.labels_,
        f"HDBSCAN min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE} (sample S)",
        save_path=os.path.join(OUTPUT_DIR, "03_hdbscan_S.png") if save_outputs else None,
        show_regions=True,
    )

    # ---- per-Phase-1-region noise% for both methods, on S ----
    region_rows = []
    for key in regions_mod.REGIONS:
        mask = regions_mod.region_mask(sample_S, key).values
        n_region = int(mask.sum())
        _, pct_noise_db, _ = models_mod.main_cluster_for_mask(db.labels_, mask)
        _, pct_noise_hdb, _ = models_mod.main_cluster_for_mask(hdb.labels_, mask)
        region_rows.append({
            "region": key, "n_in_S": n_region,
            "dbscan_noise_pct": round(pct_noise_db, 1),
            "hdbscan_noise_pct": round(pct_noise_hdb, 1),
        })
    region_table = pd.DataFrame(region_rows)

    hdb_explanation = (
        "HDBSCAN builds a hierarchy of DBSCAN-like clusterings across a *range* of density "
        "thresholds (via a mutual-reachability distance and a minimum spanning tree) and "
        "then extracts, for each branch, the cluster split that is most stable across that "
        "range, instead of committing to one fixed global eps the way plain DBSCAN does. "
        "In practice this lets it keep a sparse-but-genuine group as one cluster at a looser "
        "density threshold, while still separating a much denser group at a tighter "
        "threshold in a different branch of the same hierarchy -- plain DBSCAN, with a "
        "single eps, is forced to pick one density scale for the whole dataset at once."
    )

    db_vs_hdb_noise = db_summary["noise_pct"] - hdb_summary["noise_pct"]
    verdict_discussion = (
        f"Overall noise: DBSCAN={db_summary['noise_pct']:.1f}%, HDBSCAN="
        f"{hdb_summary['noise_pct']:.1f}% (table above); per-region noise is broken down in "
        f"the regions table. Lower noise alone is not evidence that HDBSCAN is 'better' for "
        f"scenario B: scenario B wants genuine belts/clusters, not simply fewer points "
        f"labeled noise, so the per-region table and cluster count must be read together "
        f"with the noise numbers. If HDBSCAN both (a) lowers noise in the sparser Phase-1 "
        f"region(s) and (b) keeps the dense region's main cluster from fragmenting or "
        f"merging with the others relative to DBSCAN, that is real evidence of a better "
        f"multi-scale fit for scenario B; if instead its lower overall noise comes mostly "
        f"from pulling in points at the edge of the dense region while the sparse region's "
        f"noise barely improves, the apparent gain is weaker than the single overall "
        f"noise% number suggests. The trade-off we accept by choosing one over the other is "
        f"explicit: a method that keeps more sparse, low-density structure out of the noise "
        f"class (HDBSCAN, typically) usually also has to accept a somewhat less crisp "
        f"separation inside the densest region (since the same loosened density standard "
        f"that rescues the sparse region's points can also merge sub-structure within the "
        f"dense region that a tighter single eps would have kept apart), or vice versa; "
        f"which side of that trade-off is preferable depends on whether the dense region's "
        f"internal sub-structure or the sparse region's recall matters more for the "
        f"downstream use of scenario B's clusters. One concrete cost of HDBSCAN beyond this "
        f"trade-off: it has its own extra parameters (min_cluster_size here) that must be "
        f"chosen and justified just like DBSCAN's eps, so it is not parameter-free, only "
        f"differently parameterized; it is also more expensive to fit than DBSCAN at the "
        f"same sample size, which matters less on the 2000-point S but would matter on the "
        f"full dataset. HDBSCAN's being 'better' here should not be assumed from the method's "
        f"reputation; the per-region table above is the actual evidence for or against it "
        f"in this specific comparison."
    )

    if save_outputs:
        candidates_table.to_csv(os.path.join(OUTPUT_DIR, "qb_eps_candidates_table.csv"), index=False)
        overall_table.to_csv(os.path.join(OUTPUT_DIR, "qb_overall_table.csv"), index=False)
        region_table.to_csv(os.path.join(OUTPUT_DIR, "qb_region_noise_table.csv"), index=False)

    return {
        "df": df, "sample_S": sample_S,
        "candidates_table": candidates_table, "eps_km_S": eps_km_S,
        "eps_choice_reason": eps_choice_reason,
        "db_model": db, "hdb_model": hdb,
        "overall_table": overall_table, "region_table": region_table,
        "hdb_explanation": hdb_explanation, "verdict_discussion": verdict_discussion,
        "figures": {"dbscan": fig_db, "hdbscan": fig_hdb},
    }


if __name__ == "__main__":
    qa = run_bonus_qa()
    print(qa["qa_table"])
    qb = run_bonus_qb()
    print(qb["overall_table"])
    print(qb["region_table"])
