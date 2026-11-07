"""Map and chart plotting functions for Phase 1."""
import matplotlib.pyplot as plt
from .regions import REGIONS


def plot_world_map_geo(df, title="Earthquake locations on a real-world map", save_path=None):
    """Interactive world map (Plotly Scattergeo) with real country borders and coastlines,
    so individual earthquakes can be visually tied to actual countries/regions
    (e.g. Iran, the US, China) instead of a bare lat/lon scatter plot.
    Works out of the box on Colab (no extra geodata packages needed).
    Falls back silently (returns None) if plotly is not available.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly is not installed; skipping plot_world_map_geo. "
              "Install it with `pip install plotly` to enable this map.")
        return None

    fig = go.Figure(
        go.Scattergeo(
            lon=df["Longitude"],
            lat=df["Latitude"],
            mode="markers",
            marker=dict(size=3, color="crimson", opacity=0.45, line_width=0),
            text=df.get("Magnitude"),
            hovertemplate="lat=%{lat:.2f}, lon=%{lon:.2f}<extra></extra>",
        )
    )
    fig.update_geos(
        showcountries=True, countrycolor="rgba(120,120,120,0.6)",
        showcoastlines=True, coastlinecolor="rgba(120,120,120,0.6)",
        showland=True, landcolor="rgb(235,235,235)",
        showocean=True, oceancolor="rgb(250,250,252)",
        projection_type="natural earth",
    )
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0), height=500)
    if save_path:
        fig.write_html(save_path.rsplit(".", 1)[0] + ".html")
    return fig


def plot_world_map(df, title="Earthquake locations (non-earthquake events removed)", save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(df["Longitude"], df["Latitude"], s=2, alpha=0.4, color="tab:blue")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_world_map_with_regions(df, save_path=None):
    fig, ax = plot_world_map(df, title="World earthquake map with the three selected regions")
    colors = ["red", "green", "purple"]
    for (key, r), c in zip(REGIONS.items(), colors):
        width = r["lon_max"] - r["lon_min"]
        height = r["lat_max"] - r["lat_min"]
        rect = plt.Rectangle((r["lon_min"], r["lat_min"]), width, height,
                              fill=False, edgecolor=c, linewidth=2, label=key)
        ax.add_patch(rect)
    ax.legend(loc="lower left", fontsize=8)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_density_histogram(dist_km, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dist_km, bins=80, color="tab:orange", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Distance to 10th nearest neighbor (km)")
    ax.set_ylabel("Number of earthquakes")
    ax.set_title("Distribution of distance to the 10th nearest neighbor (density metric)")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax
