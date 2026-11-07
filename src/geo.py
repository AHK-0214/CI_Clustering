"""
توابع جغرافیایی: تبدیل lat/lon به مختصات سه‌بعدی روی کره واحد،
و تبدیل بین فاصله وتری (chordal) و فاصله روی سطح زمین (great-circle).

فرمول‌های دستورالعمل:
    x = cos(phi) cos(lambda), y = cos(phi) sin(lambda), z = sin(phi)
    d = 2R * arcsin(c / 2)         (فاصله روی سطح زمین از فاصله وتری)
    c = 2 * sin(d / (2R))          (فاصله وتری از فاصله روی سطح زمین؛ برای تبدیل eps کیلومتری DBSCAN)
"""
import numpy as np
from .config import EARTH_RADIUS_KM


def latlon_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """تبدیل آرایه‌های lat/lon (درجه) به مختصات سه‌بعدی روی کره واحد. خروجی: (N,3)."""
    phi = np.radians(np.asarray(lat_deg, dtype=float))
    lam = np.radians(np.asarray(lon_deg, dtype=float))
    x = np.cos(phi) * np.cos(lam)
    y = np.cos(phi) * np.sin(lam)
    z = np.sin(phi)
    return np.column_stack([x, y, z])


def chordal_to_surface_km(c: np.ndarray, R: float = EARTH_RADIUS_KM) -> np.ndarray:
    """فاصله وتری روی کره واحد -> فاصله روی سطح زمین بر حسب کیلومتر."""
    c = np.clip(c, 0.0, 2.0)  # ایمنی عددی
    return 2.0 * R * np.arcsin(c / 2.0)


def surface_km_to_chordal(d_km: np.ndarray, R: float = EARTH_RADIUS_KM) -> np.ndarray:
    """فاصله روی سطح زمین (km) -> فاصله وتری روی کره واحد.
    برای تبدیل eps کیلومتری DBSCAN به شعاع متریک اقلیدسی در فضای سه‌بعدی استفاده می‌شود."""
    d_km = np.asarray(d_km, dtype=float)
    return 2.0 * np.sin(d_km / (2.0 * R))


def eps_km_to_chordal(eps_km: float, R: float = EARTH_RADIUS_KM) -> float:
    """نسخه اسکالر مناسب برای پارامتر eps در sklearn.cluster.DBSCAN."""
    return float(surface_km_to_chordal(np.array([eps_km]), R)[0])


def great_circle_km(lat1, lon1, lat2, lon2, R: float = EARTH_RADIUS_KM):
    """فاصله روی سطح زمین بین دو نقطه (یا آرایه نقاط) با فرمول هاورساین، برحسب کیلومتر.
    برای محاسبه گستره خوشه‌ها (فاز ۲) و بررسی‌های نقطه‌به‌نقطه مفید است."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlam = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * R * np.arcsin(np.sqrt(a))
