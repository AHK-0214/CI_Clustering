"""
لود و پاک‌سازی دیتاست Significant Earthquakes, 1965-2016.

دیتاست از کگل می‌آید و نیاز به احراز هویت دارد، پس این ماژول سه راه را امتحان می‌کند:
  1) اگر فایل از قبل در data/ موجود باشد، همان استفاده می‌شود.
  2) اگر kagglehub نصب و credentials موجود باشد، دانلود خودکار انجام می‌شود.
  3) در غیر این صورت خطای روشن می‌دهد و می‌گوید فایل را کجا بگذارید.
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
    """تلاش برای دانلود خودکار دیتاست با kagglehub (نیازمند kaggle.json)."""
    import kagglehub
    path = kagglehub.dataset_download("usgs/earthquake-database")
    # kagglehub در مسیر cache خودش دانلود می‌کند؛ فایل را به data/ کپی می‌کنیم
    os.makedirs(dest_dir, exist_ok=True)
    for fname in os.listdir(path):
        if fname.lower().endswith(".csv"):
            src = os.path.join(path, fname)
            dst = os.path.join(dest_dir, "database.csv")
            if not os.path.exists(dst):
                import shutil
                shutil.copy(src, dst)
            return dst
    raise FileNotFoundError("فایل CSV در خروجی kagglehub پیدا نشد.")


def load_raw_dataframe() -> pd.DataFrame:
    """بارگذاری CSV خام دیتاست (بدون فیلتر)."""
    local = _find_local_csv()
    if local is None:
        try:
            local = download_via_kagglehub()
        except Exception as e:
            raise FileNotFoundError(
                "دیتاست پیدا نشد. یکی از این دو کار را انجام دهید:\n"
                "  (الف) فایل database.csv را از کگل دانلود و در data/database.csv قرار دهید، یا\n"
                "  (ب) kaggle.json را در ~/.kaggle/ یا با kagglehub.login() تنظیم کنید.\n"
                f"خطای اصلی: {e}"
            )
    df = pd.read_csv(local)
    return df


def load_clean_earthquakes() -> pd.DataFrame:
    """فیلتر رکوردهای Type == 'Earthquake' و بازگرداندن ستون‌های موردنیاز با ایندکس اصلی حفظ‌شده."""
    df = load_raw_dataframe()
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Type"].astype(str).str.strip() == "Earthquake"].copy()
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    df["Latitude"] = df["Latitude"].astype(float)
    df["Longitude"] = df["Longitude"].astype(float)
    df = df.reset_index(drop=True)
    df["record_id"] = df.index  # شناسه پایدار برای رکوردها در کل پروژه
    keep_cols = ["record_id", "Date", "Latitude", "Longitude", "Depth", "Magnitude", "Type"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]
