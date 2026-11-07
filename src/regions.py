"""
The three sample regions for Phase 1 (Question A).

These three regions were chosen by looking at the raw map (before any clustering):
  R1 - Indonesia/Philippines (part of the Ring of Fire): very dense, multi-part/elongated
       along fault lines.
  R2 - Mid-Atlantic Ridge: elongated and narrow, low-to-medium density.
  R3 - Central Australia/southern Pacific: low density and scattered (an intraplate sample).

Boundaries are defined as rectangular lat/lon boxes.

Note: the "reason" field is the actual analysis/justification text that answers the
assignment's question and is meant to go into the report as-is (English).
"""
import pandas as pd

REGIONS = {
    "R1_Indonesia_Philippines": {
        "lat_min": -10.0, "lat_max": 20.0,
        "lon_min": 95.0, "lon_max": 130.0,
        "reason": "Part of the Pacific Ring of Fire; expected to show very high density and a "
                  "multi-part / elongated pattern along fault lines.",
    },
    "R2_MidAtlanticRidge": {
        "lat_min": -40.0, "lat_max": 40.0,
        "lon_min": -40.0, "lon_max": -10.0,
        "reason": "A mid-ocean ridge; expected to show an elongated, narrow pattern with "
                  "medium-to-low density along a well-defined line.",
    },
    "R3_CentralAustralia": {
        "lat_min": -30.0, "lat_max": -15.0,
        "lon_min": 120.0, "lon_max": 145.0,
        "reason": "An intraplate region far from plate boundaries; expected to show very low "
                  "density and irregular scattering (the low-density sample).",
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
            "region": key,
            "lat_range": f"[{r['lat_min']}, {r['lat_max']}]",
            "lon_range": f"[{r['lon_min']}, {r['lon_max']}]",
            "count": int(mask.sum()),
            "reason": r["reason"],
        })
    return pd.DataFrame(rows)
