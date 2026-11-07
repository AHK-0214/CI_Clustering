"""
Global project settings.

Note: the base seed (S_BASE) comes from the last three digits of the student ID.
Wherever the assignment says to use seed+1, seed+2, or seed+3, use the values defined
in this file so they stay consistent across the whole project.
"""

# ---- Seed ----
S_BASE = 629          # s         : last three digits of the student ID
S_PLUS_1 = S_BASE + 1  # s+1
S_PLUS_2 = S_BASE + 2  # s+2
S_PLUS_3 = S_BASE + 3  # s+3

# ---- Geographic constants ----
EARTH_RADIUS_KM = 6371.0

# ---- Sampling ----
SAMPLE_SIZE_S = 2000   # size of the shared sample S

# ---- Paths ----
DATA_DIR = "data"
RAW_CSV_CANDIDATES = [
    f"{DATA_DIR}/database.csv",
    f"{DATA_DIR}/earthquake-database/database.csv",
]
OUTPUT_DIR = "outputs/phase1"

# ---- Phase 1: density parameter ----
K_NEIGHBOR = 10  # 10th neighbor (not counting the point itself)
