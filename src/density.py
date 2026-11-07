"""
محاسبه فاصله هر نقطه تا دهمین همسایه نزدیک (بدون شمارش خود نقطه)، برحسب کیلومتر.

طبق یادداشت دستورالعمل: برای این فاصله (دهمین همسایه بدون خود نقطه)
min_samples معادل آن در DBSCAN برابر ۱۱ است، اما در کل پروژه از min_samples=10
استفاده می‌کنیم (این اختلاف یک‌همسایه‌ای عمداً نادیده گرفته می‌شود).
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .geo import latlon_to_xyz, chordal_to_surface_km
from .config import K_NEIGHBOR


def kth_neighbor_distance_km(df, k: int = K_NEIGHBOR) -> np.ndarray:
    """برای هر سطر df (ستون‌های Latitude/Longitude)، فاصله تا k-امین همسایه نزدیک را
    روی مختصات سه‌بعدی کره واحد محاسبه و به کیلومتر تبدیل می‌کند."""
    xyz = latlon_to_xyz(df["Latitude"].values, df["Longitude"].values)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")  # +1 چون خود نقطه هم برمی‌گردد
    nn.fit(xyz)
    chordal_dists, _ = nn.kneighbors(xyz)
    kth_chordal = chordal_dists[:, -1]  # ستون آخر = فاصله تا k-امین همسایه (خود نقطه ستون ۰ است)
    return chordal_to_surface_km(kth_chordal)


def region_density_table(df, dist_km, regions_module):
    """جدول تعداد، چارک اول، میانه، چارک سوم فاصله k-امین همسایه برای هر ناحیه."""
    import pandas as pd
    rows = []
    for key in regions_module.REGIONS:
        mask = regions_module.region_mask(df, key).values
        d = dist_km[mask]
        if len(d) == 0:
            rows.append({"ناحیه": key, "تعداد": 0, "Q1 (km)": None, "میانه (km)": None, "Q3 (km)": None})
            continue
        rows.append({
            "ناحیه": key,
            "تعداد": int(len(d)),
            "Q1 (km)": round(float(np.percentile(d, 25)), 2),
            "میانه (km)": round(float(np.percentile(d, 50)), 2),
            "Q3 (km)": round(float(np.percentile(d, 75)), 2),
        })
    return pd.DataFrame(rows)
