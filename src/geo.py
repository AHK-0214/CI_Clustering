"""
Geographic helper functions: convert lat/lon to 3D coordinates on the unit sphere,
and convert between chordal (straight-line) distance and great-circle (Earth-surface) distance.

Formulas from the assignment:
    x = cos(phi) cos(lambda), y = cos(phi) sin(lambda), z = sin(phi)
    d = 2R * arcsin(c / 2)         (surface distance from chordal distance)
    c = 2 * sin(d / (2R))          (chordal distance from surface distance; used to convert
                                     DBSCAN's eps from kilometers)
"""
import numpy as np
from .config import EARTH_RADIUS_KM


def latlon_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Convert lat/lon arrays (degrees) to 3D coordinates on the unit sphere. Output: (N,3)."""
    phi = np.radians(np.asarray(lat_deg, dtype=float))
    lam = np.radians(np.asarray(lon_deg, dtype=float))
    x = np.cos(phi) * np.cos(lam)
    y = np.cos(phi) * np.sin(lam)
    z = np.sin(phi)
    return np.column_stack([x, y, z])


def chordal_to_surface_km(c: np.ndarray, R: float = EARTH_RADIUS_KM) -> np.ndarray:
    """Chordal distance on the unit sphere -> Earth-surface distance in kilometers."""
    c = np.clip(c, 0.0, 2.0)  # numerical safety
    return 2.0 * R * np.arcsin(c / 2.0)


def surface_km_to_chordal(d_km: np.ndarray, R: float = EARTH_RADIUS_KM) -> np.ndarray:
    """Earth-surface distance (km) -> chordal distance on the unit sphere.
    Used to convert DBSCAN's eps (given in km) into a Euclidean radius in 3D space."""
    d_km = np.asarray(d_km, dtype=float)
    return 2.0 * np.sin(d_km / (2.0 * R))


def eps_km_to_chordal(eps_km: float, R: float = EARTH_RADIUS_KM) -> float:
    """Scalar version suitable for the eps parameter of sklearn.cluster.DBSCAN."""
    return float(surface_km_to_chordal(np.array([eps_km]), R)[0])


def great_circle_km(lat1, lon1, lat2, lon2, R: float = EARTH_RADIUS_KM):
    """Great-circle (Earth-surface) distance between two points (or arrays of points)
    using the haversine formula, in kilometers.
    Useful for measuring cluster spread (Phase 2) and point-to-point checks."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlam = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * R * np.arcsin(np.sqrt(a))
