# CI_Clustering — Global Earthquake Clustering Project

Implementation for the Computational Intelligence course (Ferdowsi University of Mashhad).
Project logic is kept in `src/` as Python modules; the notebooks in `notebooks/` only
clone/pull and run these modules, so the `.ipynb` file itself doesn't need frequent edits
on Colab.

## Structure

```
src/
  config.py    # base seed (s) and constants
  geo.py       # lat/lon <-> xyz on the unit sphere, chordal <-> surface distance
  data.py      # load/clean the Significant Earthquakes 1965-2016 dataset
  regions.py   # the three sample regions for Phase 1
  density.py   # distance to the k-th neighbor (density metric)
  viz.py       # map and chart plotting (incl. a real-world map with country borders)
  phase1.py    # full Phase 1 orchestration
notebooks/
  phase1_colab.ipynb
data/          # dataset is cached here (not in git)
outputs/       # each phase's output is written here (not in git)
```

## Seed

`S_BASE` in `src/config.py` is the last three digits of the student ID (629).
`s+1 = 630`, `s+2 = 631`, `s+3 = 632`, computed per the assignment.

## Dataset

The dataset is from Kaggle: `usgs/earthquake-database`. Two ways to make it available:

1. Download it manually from Kaggle and place it at `data/database.csv`
2. Use `kagglehub` with `kaggle.json` (the notebook tries this path too)

## Running

```bash
pip install -r requirements.txt
python -m src.phase1
```

Or on Colab: see `notebooks/phase1_colab.ipynb`.

## Notes on language

Code, comments, docstrings, and chart titles throughout this project are in English.
The few text blocks that are the literal answers to the assignment's written questions
(the P1-P4 predictions in `src/phase1.py`, the distance explanation, and the region
selection rationale in `src/regions.py`) are kept in Persian on purpose, since they are
submitted as-is as part of the report.
