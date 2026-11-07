"""توابع رسم نقشه و نمودار برای فاز ۱."""
import matplotlib.pyplot as plt
from .regions import REGIONS


def plot_world_map(df, title="موقعیت زلزله‌ها (پس از حذف رویدادهای غیرزلزله)", save_path=None):
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
    fig, ax = plot_world_map(df, title="نقشه جهانی زلزله‌ها با سه ناحیه انتخابی")
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
    ax.set_xlabel("فاصله تا دهمین همسایه (km)")
    ax.set_ylabel("تعداد زلزله")
    ax.set_title("توزیع فاصله تا دهمین همسایه نزدیک (معیار چگالی)")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax
