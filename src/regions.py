"""
سه ناحیه نمونه فاز ۱ (پرسش آ).

این سه ناحیه بر اساس مشاهده نقشه خام (قبل از هر خوشه‌بندی) انتخاب شده‌اند:
  R1 - اندونزی/فیلیپین (بخشی از کمربند آتش): بسیار متراکم، چندبخشی/کشیده در امتداد گسل‌ها.
  R2 - میان‌اقیانوس اطلس (Mid-Atlantic Ridge): کشیده و باریک، تراکم متوسط به پایین.
  R3 - مرکز استرالیا/اقیانوس آرام جنوبی: کم‌تراکم و پراکنده (نمونه درون‌صفحه‌ای / intraplate).

محدوده‌ها به صورت باکس مستطیلی lat/lon تعریف شده‌اند.
"""
import pandas as pd

REGIONS = {
    "R1_Indonesia_Philippines": {
        "lat_min": -10.0, "lat_max": 20.0,
        "lon_min": 95.0, "lon_max": 130.0,
        "reason": "بخشی از کمربند آتش اقیانوس آرام؛ انتظار تراکم بسیار بالا و الگوی چندبخشی/کشیده در امتداد گسل‌ها.",
    },
    "R2_MidAtlanticRidge": {
        "lat_min": -40.0, "lat_max": 40.0,
        "lon_min": -40.0, "lon_max": -10.0,
        "reason": "پشته میان‌اقیانوسی؛ انتظار الگوی کشیده و باریک با تراکم متوسط به پایین در امتداد یک خط مشخص.",
    },
    "R3_CentralAustralia": {
        "lat_min": -30.0, "lat_max": -15.0,
        "lon_min": 120.0, "lon_max": 145.0,
        "reason": "ناحیه درون‌صفحه‌ای دور از مرز صفحات؛ انتظار تراکم بسیار پایین و پراکندگی نامنظم (نمونه کم‌تراکم).",
    },
}


def region_mask(df: pd.DataFrame, region_key: str) -> pd.Series:
    r = REGIONS[region_key]
    return (
        (df["Latitude"] >= r["lat_min"]) & (df["Latitude"] <= r["lat_max"]) &
        (df["Longitude"] >= r["lon_min"]) & (df["Longitude"] <= r["lon_max"])
    )


def region_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, r in REGIONS.items():
        mask = region_mask(df, key)
        rows.append({
            "ناحیه": key,
            "بازه عرض جغرافیایی": f"[{r['lat_min']}, {r['lat_max']}]",
            "بازه طول جغرافیایی": f"[{r['lon_min']}, {r['lon_max']}]",
            "تعداد زلزله": int(mask.sum()),
            "دلیل انتخاب": r["reason"],
        })
    return pd.DataFrame(rows)
