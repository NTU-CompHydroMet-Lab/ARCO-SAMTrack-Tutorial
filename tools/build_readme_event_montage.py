"""Render docs/event_montage.png for the README hero (lower).

This script is **not part of the tutorial**. It exists only because the
GIFs that notebook 01 saves under `outputs/` are sized for a paper
panel and don't read well as a hero on a GitHub README.

This builds a paper-grade 3-panel static figure (pre-event / climax /
post-event) using exactly the same visual recipe as
`docs/hero.png` — same colormap, same panel layout, same shared
horizontal colorbar at the bottom — so the two heroes look like a pair.

It will be removed before v1.0.0 ships; the produced
`docs/event_montage.png` is what the README references.

Usage:
    uv run python tools/build_readme_hero_gif.py
"""
from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cmcrameri.cm as cmc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr


# Match notebook 01's configuration. Keep these in sync if the receptor
# box / event window ever changes there.
HF_REVISION = "main"
DAILY_URL = (
    "hf://datasets/AguaTrack/AguaTrack-ARCO-SA"
    "/AguaTrack_ARCO_SA_daily/2011.zarr"
)
BOX_LAT_S, BOX_LAT_N = -23.5, -21.5
BOX_LON_W, BOX_LON_E = -44.0, -41.0

# Three representative days: pre-event build-up, climax, decay.
PANEL_DAYS = [
    ("2011-01-08", "Pre-event"),
    ("2011-01-12", "Climax (Nova Friburgo landslides)"),
    ("2011-01-18", "Decay"),
]

OUT_PATH = Path("docs/event_montage.png")


def cos_lat_weights(lats_deg: np.ndarray) -> xr.DataArray:
    return xr.DataArray(np.cos(np.deg2rad(lats_deg)), dims="tagging_mask")


def precip_weighted_source(
    e_region: xr.DataArray,
    tp_region: xr.DataArray,
    tag_area_w: xr.DataArray,
) -> xr.DataArray:
    """Reproduce notebook 01's precipitation-and-area-weighted source field."""
    rain_weights = tp_region * tag_area_w
    rain_total = rain_weights.sum("tagging_mask")
    safe_denom = rain_total.where(rain_total > 0)
    return (e_region * rain_weights).sum("tagging_mask") / safe_denom


def load_panel_field() -> xr.DataArray:
    """Open the 2011 daily zarr and return the source field on PANEL_DAYS only."""
    ds = xr.open_zarr(DAILY_URL, storage_options={"revision": HF_REVISION})
    in_box = (
        (ds.tag_lat >= BOX_LAT_S) & (ds.tag_lat <= BOX_LAT_N)
        & (ds.tag_lon >= BOX_LON_W) & (ds.tag_lon <= BOX_LON_E)
    )
    box_tag_idx = np.flatnonzero(in_box.values)
    box_lat = ds.tag_lat.isel(tagging_mask=box_tag_idx)

    days = [d for d, _ in PANEL_DAYS]
    print(f"loading {len(days)} days x {len(box_tag_idx)} tags from HF…")
    e_region = (
        ds.e_track.isel(tagging_mask=box_tag_idx).sel(time=days)
        .chunk({"tagging_mask": 1, "time": -1, "latitude": -1, "longitude": -1})
        .load()
    )
    tp_region = ds.tagged_precip.isel(tagging_mask=box_tag_idx).sel(time=days).load()

    tag_area_w = cos_lat_weights(box_lat.values)
    return precip_weighted_source(e_region, tp_region, tag_area_w)


def style_ax(ax, title: str, draw_y: bool) -> None:
    """Decorate one panel — same recipe as notebook 02's style_ax."""
    ax.coastlines(resolution="50m", color="black", linewidth=0.8)
    ax.set_extent([-95, -30, -55, 15], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.1)
    gl.top_labels = gl.right_labels = False
    if not draw_y:
        gl.left_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 10))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 10))
    ax.set_title(title)


def build_montage(field: xr.DataArray) -> None:
    # Shared colour scale: 99th percentile of non-zero pixels across all
    # three panels — same trick notebook 01 uses for its GIFs.
    field_pos = field.where(field > 0)
    vmax = float(field_pos.quantile(0.99))
    levels = np.linspace(0, vmax, 11)

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5.3),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    cf = None
    for col_idx, ((day, label), ax) in enumerate(zip(PANEL_DAYS, axes)):
        f = field.sel(time=day)
        cf = f.plot.contourf(
            ax=ax, levels=levels, cmap=cmc.batlowW_r,
            transform=ccrs.PlateCarree(), add_colorbar=False,
        )
        # Red box at the receptor.
        ax.add_patch(mpatches.Rectangle(
            (BOX_LON_W, BOX_LAT_S),
            BOX_LON_E - BOX_LON_W, BOX_LAT_N - BOX_LAT_S,
            linewidth=1.8, edgecolor="red", facecolor="none",
            transform=ccrs.PlateCarree(), zorder=5,
        ))
        date_pretty = pd.Timestamp(day).strftime("%d %b %Y")
        title = f"{label}\n{date_pretty}"
        style_ax(ax, title, draw_y=col_idx == 0)

    fig.subplots_adjust(bottom=0.14, wspace=0.05)
    cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.025])
    fig.colorbar(cf, cax=cbar_ax, orientation="horizontal",
                 label="Tracked moisture source (kg m⁻² day⁻¹)")
    fig.suptitle("Daily evolution — Serra do Mar 2011 flash-flood event",
                 fontsize=14)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=150)
    plt.close(fig)
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"\nwrote {OUT_PATH}: {size_kb:.0f} KB")


def main() -> None:
    field = load_panel_field()
    build_montage(field)


if __name__ == "__main__":
    main()
