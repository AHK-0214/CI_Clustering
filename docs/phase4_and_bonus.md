# Phase 4 and the Bonus section — what the code computes

This note documents `src/phase4.py` and `src/bonus.py`: what each function computes, how
it maps onto the assignment's questions, and how to read the tables/figures they produce.
It assumes Phases 1-3 (`src/phase1.py`-`src/phase3.py`) have already been read, since
Phase 4 and the bonus section reuse their outputs rather than re-deriving them.

---

## 1. `src/phase4.py` — Evaluation and conclusion (25 points)

Entry point: `run_phase4(df=None, sample_S=None, phase2_results=None, phase3_results=None,
save_outputs=True)`. If `phase2_results` / `phase3_results` are not supplied, it re-runs
`run_phase2` / `run_phase3` internally (with `save_outputs=False`) so the function is
callable on its own, but in the notebook it is always given the already-computed
`results2` / `results3` so the *selected* parameters match exactly what Phases 2-3 chose.

### Question A — four-method evaluation on the shared sample S

**Which four label sets are compared, and where each comes from:**

| Method | Fit on | How its S-labels are obtained |
|---|---|---|
| K-Means (`final_k`, Phase 2's selected k) | full dataset | `sampling.labels_for_ids(df, kmeans_full_labels, sample_S.record_id)` |
| DBSCAN (Phase 3's `final_eps_scenario_b`) | full dataset | same `labels_for_ids` lookup, by `record_id` |
| Single Link (cut at 12 clusters) | sample S directly | taken as-is from `phase3_results["labels_single"]` |
| Average Link (cut at 12 clusters) | sample S directly | taken as-is from `phase3_results["labels_average"]` |

The assignment's "equal footing" caveat is kept explicit in code and in
`qa_eval_discussion`: DBSCAN is **not** re-fit on S with the same eps (its coverage on S
is simply whatever the full-data eps happens to produce on this subsample, which is
visibly lower than on the full data because S is sparser).

**Summary table** (`qa_summary_table`, via `models.summarize_labels`): for each method —
`n_clusters`, `largest_cluster_size`, `median_cluster_size`, `n_clusters_10_19_points`
(clusters under 20 points), `coverage_pct` (100% for the three methods that label every
point; non-noise share for DBSCAN), and `silhouette` (computed on the 3D sphere
coordinates of S, noise excluded for DBSCAN, reported as `"undefined"` if fewer than 2
non-noise clusters remain — never a numeric 0 or `None` silently).

**Maps** (`figures`): all four use `viz.plot_cluster_map` on the same `sample_S`
dataframe, so axis ranges and the plotted universe of points are identical across panels.

**Pairwise ARI** (`qa_ari_table`, `qa_ari_matrix`): all 6 pairs via
`sklearn.metrics.adjusted_rand_score` on the raw label arrays — DBSCAN's `-1` noise label
is **not** filtered out before computing ARI, so for the two pairs involving DBSCAN,
"noise" is implicitly treated as its own group, exactly as the assignment specifies.
`qa_eval_discussion` names the lowest/highest-agreement pair and states plainly that a
high pairwise ARI is evidence of *agreement between two methods*, not evidence that
either partition is "correct" for a scenario.

### Resampling stability (sub-part of Question A)

Implemented by `sampling.get_subsample` (added for this phase) + `models.run_hierarchical`
/ `models.run_kmeans` + `models.ari_on_common_ids` (also added for this phase):

1. Three 80%-of-S subsets are drawn with `sampling.get_subsample(sample_S, 0.8, seed)` for
   `seed in [S_BASE, S_PLUS_1, S_PLUS_2]` (i.e. `s, s+1, s+2`), **without replacement**.
2. On each subset: K-Means (`k=12`, `n_init=10`, **algorithm seed fixed at `S_BASE` in all
   three runs**, so only the sampled points change, not the initialization), Single Link
   and Average Link (both cut at 12 clusters) — 3 subsets × 3 methods = **9 runs**.
3. For each method, the three subsets give three pairs ((0,1), (0,2), (1,2)). For each
   pair, `models.ari_on_common_ids` restricts both label arrays to the `record_id`s present
   in *both* subsets (via `models.labels_for_ids_local`, a small helper that avoids a
   circular import with `sampling.py`) and computes ARI only on that common set — 3
   methods × 3 pairs = **9 pairwise ARI values**, matching the assignment's "9 clustering
   runs and 9 pairwise ARI computations" count exactly.
4. `stability_summary_table` reports, per method, the mean and the range (max − min) of
   its three ARI values. `stability_discussion` interprets the result mechanistically
   (why K-Means tends to be smoother under resampling than Single Link, why Average Link
   sits in between) and explicitly cautions that 3 subsets is too few to generalize from.
5. `best_numeric_but_not_best_fit` gives the required concrete example: it names whichever
   pair/method tops the ARI tables and explains in one paragraph why topping a numeric
   score does not by itself satisfy either scenario's actual requirements (full coverage
   + describable zone count for A; genuine density-based structure discovery for B).

### Question B — conclusion

`verdict_table`: one row per prediction (`P1`-`P4`) with a `verdict`
(`Confirmed` / `Rejected` / `Partially confirmed`) and an `evidence` string that names the
specific phase, table, or figure backing the verdict — never a bare assertion. The verdicts
are derived from quantities already computed earlier in the pipeline (e.g. P3's verdict is
literally `"Confirmed" if cv_single > cv_average else "Rejected"`, not a hard-coded string),
so re-running with a different seed or dataset snapshot updates the verdict automatically
instead of silently going stale.

`final_recommendation`: a nested dict with one block per scenario
(`scenario_A`, `scenario_B`), each giving `algorithm`, `representation`, `parameters`, a
list of exactly **2 evidence** strings, and **1 limitation** string, per the assignment's
format. `same_algorithm_note` states explicitly why the project does *not* recommend one
algorithm for both scenarios (their hard requirements are structurally opposed).

---

## 2. `src/bonus.py` — optional section (5 + 5 points)

Both halves are independent; either can be run without the other.

### Bonus Q.A — Controlled Perturbation (`run_bonus_qa`)

**Geographic displacement**, not a flat lat/lon offset:

- `geo.destination_point(lat, lon, bearing_deg, distance_km)` (added to `geo.py`) implements
  the standard spherical "destination point given start, bearing and distance" formula —
  the same sphere/great-circle model used throughout the project (`EARTH_RADIUS_KM`) — and
  is fully vectorized.
- `bonus.perturb_dataset(df, frac=0.05, seed=S_PLUS_3)` selects 5% of the full (cleaned)
  dataset with `seed=s+3`, draws a uniformly random bearing (0-360°) and a uniformly random
  distance in **[10, 20] km** per selected point, and calls `destination_point` on exactly
  those rows. Row order and `record_id` are untouched; only `Latitude`/`Longitude` of the
  selected rows change. The boolean `perturbed_mask` it returns is reused throughout the
  rest of the function (and saved to `qa_perturbation_mask.csv`) so later steps can always
  tell perturbed from untouched points.

**Re-running with identical parameters/seeds:** K-Means uses `final_k` (Phase 2's selected
k) and `random_state=S_BASE` in *both* the original and the perturbed run; DBSCAN uses
`final_eps_scenario_b`/`min_samples` from Phase 3's selected run, unchanged, on both.

**Membership-change detection (`_changed_counts`)** — this is the part worth reading
closely, since a naive approach is wrong here: cluster ids are not aligned between the
"original" and "perturbed" runs (DBSCAN in particular can produce a different *number* of
clusters each time), so comparing raw cluster-id numbers would misreport almost every
point as "changed". Instead:

1. A point's noise status flipping (`noise -> cluster` or `cluster -> noise`) always counts
   as changed.
2. For the non-noise, non-noise case, a **one-shot global optimal assignment**
   (`scipy.optimize.linear_sum_assignment` on the full overlap/contingency matrix between
   original and perturbed clusters, maximizing total overlap) finds the best 1-to-1 cluster
   correspondence *once*, instead of letting each original cluster greedily pick its own
   "best match" independently (which was tried first and produces spurious changes: several
   original clusters can end up claiming the same small perturbed cluster as their
   "closest match"). A point is "changed" iff, after perturbation, it is not inside the
   cluster matched to its original cluster.

This function was validated against the naive greedy version during development
specifically because the greedy version inflated unperturbed-point "changes" to ~18% for
DBSCAN; the Hungarian-matching version brings that down to a figure consistent with the
actual ARI (sub-1% to low single digits), which is what gets reported in `qa_table` and
discussed in `qa_discussion`.

**Output:** `qa_table` has one row per method with the ARI(original, perturbed), cluster
counts, largest-cluster sizes, noise% (DBSCAN only), and — the question's specific ask —
`n_perturbed_changed` / `n_unperturbed_changed` (and their percentages) so the report can
state plainly whether the disturbance stayed confined to the moved points or spread to
others via chaining/boundary effects. `figures["example"]` highlights one concrete
perturbed-and-changed point on the world map (falls back to the K-Means change set if no
DBSCAN example exists).

### Bonus Q.B — HDBSCAN vs. DBSCAN (`run_bonus_qb`)

- **DBSCAN's eps is re-tuned on S**, not reused from Phase 3's full-data values, per the
  assignment's explicit instruction. `density.kth_neighbor_distance_km(sample_S)` gives
  S's own 10th-neighbor-distance distribution; three candidates (its 50th/75th/90th
  percentile) are tried, reported in `candidates_table`, and the tightest one that still
  keeps **overall** noise on S under 50% is selected as `eps_km_S` (`eps_choice_reason`
  states the rule used). This mirrors Phase 3's own eps-selection style (pick from a small
  set of density-grounded candidates, state the numeric rule explicitly) rather than
  reusing Phase 3's code outright, since S's density scale genuinely differs from the full
  data's.
- **HDBSCAN** uses `sklearn.cluster.HDBSCAN` (available since scikit-learn 1.3, already in
  `requirements.txt`) with `min_cluster_size=10, min_samples=10` on the same 3D
  unit-sphere coordinates of S used everywhere else in the project — same representation,
  so the two methods are only being compared on their clustering rule, not on a
  feature-space difference.
- `overall_table`: n_clusters, overall noise%, Silhouette (noise excluded) for both.
- `region_table`: per Phase-1 region (`R1`/`R2`/`R3`), the number of that region's points
  present in S plus the noise% each method gives that region — computed with
  `models.main_cluster_for_mask`, the same helper Phase 3-B already uses, applied to a
  mask restricted to `sample_S` instead of the full dataframe.
- `hdb_explanation` is the plain-language explanation of HDBSCAN vs. DBSCAN the question
  asks for (mutual-reachability hierarchy + per-branch stability extraction, vs. one fixed
  global eps).
- `verdict_discussion` explicitly avoids assuming HDBSCAN is better: it states the
  specific trade-off being accepted (recall on sparse structure vs. crispness inside dense
  structure) and names a concrete cost (an extra parameter to justify, higher fit cost)
  rather than declaring a winner from the noise-percentage number alone.

---

## 3. Shared helpers added to existing modules

Phase 4 and the bonus section reuse Phases 2-3's modules as much as possible; the only
additions made to shared files (as opposed to new, phase-specific code) are:

- `geo.py`: `destination_point(...)` — spherical bearing+distance displacement (bonus Q.A).
- `sampling.py`: `get_subsample(...)` — 80%-of-S draws (Phase 4-A resampling stability).
- `models.py`: `adjusted_rand_score` import, `ari_on_common_ids(...)` and its small helper
  `labels_for_ids_local(...)` — ARI between two overlapping, record_id-matched label sets
  (Phase 4-A resampling stability).

No existing function's signature or behavior was changed; everything above is additive.

---

## 4. Outputs on disk

When `save_outputs=True` (the default), results are written under:

```
outputs/phase4/
  qa_summary_table.csv, qa_ari_pairs_table.csv, qa_ari_matrix.csv,
  qa_stability_pairs_table.csv, qa_stability_summary_table.csv,
  qb_prediction_verdicts.csv,
  01_<method>.png            # one map per of the 4 methods
outputs/bonus/
  qa_perturbation_mask.csv, qa_perturbation_table.csv, 01_perturbation_example.png,
  qb_eps_candidates_table.csv, qb_overall_table.csv, qb_region_noise_table.csv,
  02_dbscan_S.png, 03_hdbscan_S.png
```

The notebook's final cell zips `outputs/phase1` through `outputs/phase4` plus
`outputs/bonus` together for the report attachment.
