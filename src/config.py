"""
تنظیمات سراسری پروژه.

نکته: seed پایه (S_BASE) از سه رقم آخر شماره دانشجویی گرفته می‌شود.
هر جا در دستورالعمل گفته شده seed+1 یا seed+2 یا seed+3 استفاده شود،
از مقادیر تعریف‌شده در همین فایل استفاده کنید تا در کل پروژه یکدست بماند.
"""

# ---- Seed ----
S_BASE = 629          # s         : سه رقم آخر شماره دانشجویی
S_PLUS_1 = S_BASE + 1  # s+1
S_PLUS_2 = S_BASE + 2  # s+2
S_PLUS_3 = S_BASE + 3  # s+3

# ---- ثابت‌های جغرافیایی ----
EARTH_RADIUS_KM = 6371.0

# ---- نمونه‌گیری ----
SAMPLE_SIZE_S = 2000   # اندازه نمونه مشترک S

# ---- مسیرها ----
DATA_DIR = "data"
RAW_CSV_CANDIDATES = [
    f"{DATA_DIR}/database.csv",
    f"{DATA_DIR}/earthquake-database/database.csv",
]
OUTPUT_DIR = "outputs/phase1"

# ---- فاز ۱: پارامتر چگالی ----
K_NEIGHBOR = 10  # دهمین همسایه (بدون شمارش خود نقطه)
