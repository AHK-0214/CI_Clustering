"""
Load and clean the "Significant Earthquakes, 1965-2016" dataset.

The dataset comes from Kaggle and requires authentication, so this module tries three
options in order:
  1) If the file already exists under data/, use it as-is.
  2) If kagglehub is installed and credentials are available, download it automatically.
  3) Otherwise, raise a clear error explaining where to put the file.
"""
import os
import pandas as pd
from .config import RAW_CSV_CANDIDATES


def _find_local_csv():
    for path in RAW_CSV_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def download_via_kagglehub(dest_dir="data"):
    """Try to download the dataset automatically with kagglehub (requires kaggle.json)."""
    import kagglehub
    path = kagglehub.dataset_download("usgs/earthquake-database")
    # kagglehub downloads into its own cache dir; copy the file into data/
    os.makedirs(dest_dir, exist_ok=True)
    for fname in os.listdir(path):
        if fname.lower().endswith(".csv"):
            src = os.path.join(path, fname)
            dst = os.path.join(dest_dir, "database.csv")
            if not os.path.exists(dst):
                import shutil
                shutil.copy(src, dst)
            return dst
    raise FileNotFoundError("Could not find a CSV file in kagglehub's output.")


def load_raw_dataframe() -> pd.DataFrame:
    """Load the raw dataset CSV (no filtering)."""
    local = _find_local_csv()
    if local is None:
        try:
            local = download_via_kagglehub()
        except Exception as e:
            raise FileNotFoundError(
                "Dataset not found. Do one of the following:\n"
                "  (a) Download database.csv from Kaggle and place it at data/database.csv, or\n"
                "  (b) Set up kaggle.json under ~/.kaggle/ or call kagglehub.login().\n"
                f"Underlying error: {e}"
            )
    df = pd.read_csv(local)
    return df


def load_clean_earthquakes() -> pd.DataFrame:
    """Filter records where Type == 'Earthquake' and return the required columns with a
    stable index preserved."""
    df = load_raw_dataframe()
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Type"].astype(str).str.strip() == "Earthquake"].copy()
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    df["Latitude"] = df["Latitude"].astype(float)
    df["Longitude"] = df["Longitude"].astype(float)
    df = df.reset_index(drop=True)
    df["record_id"] = df.index  # stable identifier for records across the whole project
    keep_cols = ["record_id", "Date", "Latitude", "Longitude", "Depth", "Magnitude", "Type"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]
