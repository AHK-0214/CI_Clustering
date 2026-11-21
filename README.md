# CI_Clustering — Global Earthquake Clustering Project

Implementation for the Computational Intelligence course (Ferdowsi University of Mashhad).
Project logic is kept in `src/` as Python modules; the notebooks in `notebooks/` only
clone/pull and run these modules, so the `.ipynb` file itself doesn't need frequent edits
on Colab.

## Structure

```
src/
  config.py    # base seed (s) and constants
  geo.py       # lat/lon <-> xyz on the unit sphere, chordal <-> surface distance,
               # great-circle distance, and the destination-point (bearing+distance) formula
  data.py      # load/clean the Significant Earthquakes 1965-2016 dataset
  regions.py   # the three sample regions for Phase 1
  density.py   # distance to the k-th neighbor (density metric)
  sampling.py  # shared sample S, 80%-of-S subsamples, record_id-matched label lookup/ARI
  models.py    # thin sklearn/scipy wrappers (K-Means, DBSCAN, hierarchical) + shared
               # cluster-summary / ARI-on-common-ids helpers used by Phases 2-4 and bonus
  viz.py       # map and chart plotting (incl. a real-world map with country borders)
  phase1.py    # Phase 1 — data understanding & predictions P1-P4
  phase2.py    # Phase 2 — K-Means
  phase3.py    # Phase 3 — DBSCAN + hierarchical clustering
  phase4.py    # Phase 4 — evaluation (4-method comparison on S, resampling stability)
               # and conclusion (P1-P4 verdicts, final per-scenario recommendation)
  bonus.py     # Bonus Q.A (Controlled Perturbation) and Q.B (HDBSCAN vs. DBSCAN)
notebooks/
  project_colab.ipynb
data/          # dataset is cached here (not in git)
outputs/       # each phase's output is written here (not in git): outputs/phase1 .. phase4, outputs/bonus
```

## Seed

`S_BASE` in `src/config.py` is the last three digits of the student ID (429).
`s+1 = 430`, `s+2 = 431`, `s+3 = 432`, computed per the assignment.

## Dataset

The dataset is from Kaggle: `usgs/earthquake-database`. Two ways to make it available:

1. Download it manually from Kaggle and place it at `data/database.csv`
2. Use `kagglehub` with `kaggle.json` (the notebook tries this path too)

## Running

```bash
pip install -r requirements.txt
python -m src.phase1
python -m src.phase2
python -m src.phase3
python -m src.phase4   # needs the Phase 2/3 "selected" results; re-runs them if not passed in
python -m src.bonus    # optional: Bonus Q.A (perturbation) and Q.B (HDBSCAN), independent of each other
```

Or on Colab: see `notebooks/project_colab.ipynb`, which runs all four phases plus the bonus
section in order and lets you download everything under `outputs/` as one zip at the end.

See `docs/phase4_and_bonus.md` for what each of Phase 4's and the bonus section's functions
compute and how their numbers map onto the assignment's questions.

## Notes on language

Code, comments, docstrings, and chart titles throughout this project are in English.
The few text blocks that are the literal answers to the assignment's written questions
(the P1-P4 predictions in `src/phase1.py`, the distance explanation, and the region
selection rationale in `src/regions.py`) are kept in Persian on purpose, since they are
submitted as-is as part of the report.
