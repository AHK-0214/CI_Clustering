"""
The shared sample S used across phases 3 and 4.

Per the assignment: draw 2000 points once, with the base seed s, from the cleaned
dataset. Hierarchical clustering runs only on S. In later comparisons, labels for the
points of S are extracted from methods that were run on the full dataset -- this works
because every record keeps its original `record_id`.
"""
import numpy as np
import pandas as pd
from .config import SAMPLE_SIZE_S, S_BASE


def get_sample_S(df: pd.DataFrame, n: int = SAMPLE_SIZE_S, seed: int = S_BASE) -> pd.DataFrame:
    """Randomly draw `n` rows (without replacement) from the cleaned dataframe `df`
    using `seed`, and return them as a fresh dataframe (original `record_id` preserved,
    sorted by record_id for reproducible ordering)."""
    rng = np.random.RandomState(seed)
    n = min(n, len(df))
    chosen_pos = rng.choice(len(df), size=n, replace=False)
    sample = df.iloc[chosen_pos].sort_values("record_id").reset_index(drop=True)
    return sample


def labels_for_ids(full_df: pd.DataFrame, full_labels: np.ndarray, record_ids) -> np.ndarray:
    """Given labels produced by a model run on `full_df` (aligned by position with
    full_df's rows), extract the labels corresponding to a list/array of `record_id`
    values, in the order the ids are given."""
    pos_of_id = pd.Series(np.arange(len(full_df)), index=full_df["record_id"].values)
    positions = pos_of_id.loc[np.asarray(record_ids)].values
    return np.asarray(full_labels)[positions]
