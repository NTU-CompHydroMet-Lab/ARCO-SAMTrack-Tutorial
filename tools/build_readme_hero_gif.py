"""Render docs/event_animation.gif for the README hero.

This script is **not part of the tutorial**. It exists only because the
GIFs that notebook 01 saves under `outputs/` are sized for a paper
panel — square, dense, with paper-sized fonts — and don't read well as
a wide hero on a GitHub README. This script re-renders the same Serra
do Mar 2011 event with a wide-aspect figure, larger fonts, and a
single-panel layout that scales nicely from desktop to mobile.

It will be removed before v1.0.0 ships; the produced
`docs/event_animation.gif` is what the README references.

Usage:
    uv run python tools/build_readme_hero_gif.py
"""
from __future__ import annotations

import io
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image


# Match notebook 01's configuration. Keep these in sync if the receptor
# box / event window ever changes there.
HF_REVISION = "main"
DAILY_URL = (
    "hf://datasets/AguaTrack/AguaTrack-ARCO-SA"
    "/AguaTrack_ARCO_SA_daily/2011.zarr"
)
BOX_LAT_S, BOX_LAT_N = -23.5, -21.5
BOX_LON_W, BOX_LON_E = -44.0, -41.0
DATE_START = "2011-01-05"
DATE_END = "2011-01-22"
EVENT_ONSET = "2011-01-11"  # night of 11->12 Jan: landslides begin

OUT_PATH = Path("docs/event_animation.gif")

# README-friendly canvas. The tracking domain is roughly 65° lon × 75°
# lat (almost square), so figsize ≈ 9×5 keeps the map filling its axes
# while leaving a comfortable margin for the day counter and date.
FIG_W_IN, FIG_H_IN = 9.0, 5.0
DPI = 110  # → 990x550 px raster per frame
FRAME_DURATION_MS = 250  # ~4 fps; pause longer on the climax frame below
CLIMAX_DURATION_MS = 600


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


def load_event_field() -> xr.DataArray:
    """Open the 2011 daily zarr and return the 18-day source field."""
    ds = xr.open_zarr(DAILY_URL, storage_options={"revision": HF_REVISION})
    in_box = (
        (ds.tag_lat >= BOX_LAT_S) & (ds.tag_lat <= BOX_LAT_N)
        & (ds.tag_lon >= BOX_LON_W) & (ds.tag_lon <= BOX_LON_E)
    )
    box_tag_idx = np.flatnonzero(in_box.values)
    box_lat = ds.tag_lat.isel(tagging_mask=box_tag_idx)

    days = pd.date_range(DATE_START, DATE_END).strftime("%Y-%m-%d").tolist()
    e_lazy = (
        ds.e_track.isel(tagging_mask=box_tag_idx).sel(time=days)
        .chunk({"tagging_mask": 1, "time": -1, "latitude": -1, "longitude": -1})
    )
    tp_lazy = ds.tagged_precip.isel(tagging_mask=box_tag_idx).sel(time=days)

    print(f"loading {len(days)} days x {len(box_tag_idx)} tags from HF…")
    e_region = e_lazy.load()
    tp_region = tp_lazy.load()

    tag_area_w = cos_lat_weights(box_lat.values)
    e_abs = precip_weighted_source(e_region, tp_region, tag_area_w)
    e_abs.attrs["domain_lon"] = (float(ds.longitude.min()), float(ds.longitude.max()))
    e_abs.attrs["domain_lat"] = (float(ds.latitude.min()), float(ds.latitude.max()))
    return e_abs


def render_frame(field_2d: xr.DataArray, day_idx: int, n_days: int,
                 vmax: float, domain_lon, domain_lat) -> Image.Image:
    """Render a single frame and return it as a PIL.Image (RGB)."""
    date_str = pd.Timestamp(field_2d.time.values).strftime("%Y-%m-%d")

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    ax = fig.add_axes(
        [0.07, 0.09, 0.78, 0.80],
        projection=ccrs.PlateCarree(),
    )

    cf = field_2d.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="cmc.batlowW_r",  # white -> warm, leaves the "no source" sea white
        vmin=0,
        vmax=vmax,
        add_colorbar=False,
    )

    ax.coastlines(resolution="50m", color="black", linewidth=0.9)
    ax.set_xlim(domain_lon)
    ax.set_ylim(domain_lat)
    ax.add_patch(mpatches.Rectangle(
        (BOX_LON_W, BOX_LAT_S),
        BOX_LON_E - BOX_LON_W, BOX_LAT_N - BOX_LAT_S,
        linewidth=2.2, edgecolor="red", facecolor="none",
        transform=ccrs.PlateCarree(), zorder=5,
    ))
    gl = ax.gridlines(draw_labels=True, linewidth=0.0)
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 10))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 10))
    gl.xlabel_style = {"size": 13}
    gl.ylabel_style = {"size": 13}

    ax.set_title("")

    # Lean colorbar on the right.
    cax = fig.add_axes([0.87, 0.16, 0.020, 0.66])
    cbar = fig.colorbar(cf, cax=cax)
    cbar.set_label("Tracked evaporation (kg m⁻² day⁻¹)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Top-left day counter (big), top-right date (smaller monospace).
    fig.text(0.07, 0.93, f"Day {day_idx + 1} / {n_days}",
             fontsize=19, fontweight="bold", color="#222222")
    is_climax = date_str == "2011-01-12"
    date_color = "#cc0000" if is_climax else "#444444"
    fig.text(0.85, 0.93, date_str, fontsize=17, color=date_color,
             ha="right", fontfamily="monospace",
             fontweight="bold" if is_climax else "normal")
    if is_climax:
        fig.text(0.85, 0.885, "Nova Friburgo landslides",
                 fontsize=10, color="#cc0000", ha="right", style="italic")

    fig.text(0.07, 0.03,
             "Precipitation-weighted moisture source for the Serra do Mar receptor box (red).",
             fontsize=9, color="#666666")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def build_gif(field: xr.DataArray) -> None:
    # Shared colour scale across all 18 frames.
    field_pos = field.where(field > 0)
    vmax = float(field_pos.quantile(0.99))
    print(f"vmax (99th percentile of non-zero pixels): {vmax:.4f}")

    domain_lon = field.attrs["domain_lon"]
    domain_lat = field.attrs["domain_lat"]
    n_days = field.sizes["time"]

    # Need cmcrameri registered before pcolormesh resolves "cmc.batlowK_r".
    import cmcrameri.cm  # noqa: F401  (registers the colormaps)

    frames: list[Image.Image] = []
    durations: list[int] = []
    for i in range(n_days):
        f = field.isel(time=i)
        date_str = pd.Timestamp(f.time.values).strftime("%Y-%m-%d")
        print(f"  frame {i + 1}/{n_days}  {date_str}")
        frames.append(render_frame(f, i, n_days, vmax, domain_lon, domain_lat))
        durations.append(
            CLIMAX_DURATION_MS if date_str == EVENT_ONSET[:10] or date_str == "2011-01-12"
            else FRAME_DURATION_MS
        )

    # Quantize to a palette to keep file size sane.
    palette_frames = [f.quantize(method=Image.MEDIANCUT, colors=128) for f in frames]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    palette_frames[0].save(
        OUT_PATH,
        save_all=True,
        append_images=palette_frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"\nwrote {OUT_PATH}: {size_kb:.0f} KB ({n_days} frames, "
          f"{frames[0].size[0]}x{frames[0].size[1]} px)")


def main() -> None:
    field = load_event_field()
    build_gif(field)


if __name__ == "__main__":
    main()
