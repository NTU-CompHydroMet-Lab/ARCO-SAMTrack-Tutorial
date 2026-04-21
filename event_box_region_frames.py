"""
================================================================================
  TUTORIAL: Daily moisture-source maps for the 2011-01 Região Serrana disaster
  Region:  Box around Serra do Mar (Nova Friburgo / Teresópolis / RJ metro)
           lat ∈ [-23.5, -21.5]  lon ∈ [-44.0, -41.0]  (≈ 220 × 330 km)
  Dataset: WAM2Layers backward-tracking output, SA 0.25° daily, 1990–2019
           /cache/.../zarr_SA_0_25_deg_daily_1990_2020/2011.zarr
  Period:  2011-01-05 to 2011-01-22 (pre-event, event, recovery)

  Outputs:  outputs/box_abs/<YYYY-MM-DD>.png   — absolute source strength
            outputs/box_pdf/<YYYY-MM-DD>.png   — area-weighted PDF
            outputs/box_cdf/<YYYY-MM-DD>.png   — cumulative-rank % contours

  Version: v2 (2026-04-21)
================================================================================

Design choices (please read before editing)
-------------------------------------------
  * Box mask, not polygon: a rectangular region matches the ~500 km synoptic
    scale of this Serra do Mar SACZ event. A HYBAS level-2 basin here is
    40× too large and averages in unrelated NE Brazil rainfall (see the
    comparison notebook if you want to see the effect).
  * Tag inclusion: tag_cell centre ∈ box. Edge fractional coverage is ignored;
    on a 220×330 km box with 28×28 km cells the error is well under 1%.
  * Aggregation: area-weighted (cos lat) AND precip-weighted mean over the
    tags in the box. Physical meaning: "where did today's box-total tagged
    rainfall come from?" On dry days (all tags have zero tagged_precip) the
    source map is NaN, not zero.
  * Colour scale: 99th percentile of non-zero pixels across the 18-day window,
    shared across frames so intensity is directly comparable.
  * .where(data > 0) on the map is a visualisation choice; WAM2Layers e_track
    is non-negative by construction and zeros mean "no traced evaporation".

Units
-----
WAM2Layers stores fluxes as kg m⁻² accumulated per output time step. Output
frequency is 1 day, so numerical values equal kg m⁻² day⁻¹ (= mm/day of
water-equivalent). We display them with that label throughout.
"""

from pathlib import Path

import cartopy.crs as ccrs
import cmcrameri.cm as cmc
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr

# ---------- plotting rcParams (full-page 16 inch wide, academic style) ----------
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.titlesize": 18,
})

# ---------- config ----------
ZARR = "/cache/isaacgbhk/Work/i_wam/output/zarr_SA_0_25_deg_daily_1990_2020/2011.zarr"
SHP = "data/hybas_sa_lev02_v1c/hybas_sa_lev02_v1c.shp"
OUT_ABS = Path("outputs/box_abs"); OUT_ABS.mkdir(parents=True, exist_ok=True)
OUT_PDF = Path("outputs/box_pdf"); OUT_PDF.mkdir(parents=True, exist_ok=True)
OUT_CDF = Path("outputs/box_cdf"); OUT_CDF.mkdir(parents=True, exist_ok=True)

# Serra do Mar box.
BOX_LAT_S, BOX_LAT_N = -23.5, -21.5
BOX_LON_W, BOX_LON_E = -44.0, -41.0

DATE_START = "2011-01-05"
DATE_END = "2011-01-22"
JAN_START, JAN_END = "2011-01-01", "2011-01-31"

# CDF contour levels (percent of cumulative source captured).
CDF_LEVELS = [10, 30, 50, 70, 90]

# ---------- open zarr + shapefile (shp is for map context only) ----------
ds = xr.open_zarr(ZARR)
gdf = gpd.read_file(SHP)

# ---------- pick tag cells inside the box ----------
tag_lat = ds.tag_lat.compute().values
tag_lon = ds.tag_lon.compute().values
in_box = (
    (tag_lat >= BOX_LAT_S) & (tag_lat <= BOX_LAT_N)
    & (tag_lon >= BOX_LON_W) & (tag_lon <= BOX_LON_E)
)
box_tag_idx = np.where(in_box)[0]
print(
    f"tags inside box: {len(box_tag_idx)} "
    f"(lat {tag_lat[box_tag_idx].min():.2f}..{tag_lat[box_tag_idx].max():.2f}, "
    f"lon {tag_lon[box_tag_idx].min():.2f}..{tag_lon[box_tag_idx].max():.2f})"
)

# ---------- load e_track + tagged_precip (single chunk worth of data) ----------
days = pd.date_range(DATE_START, DATE_END).strftime("%Y-%m-%d").tolist()
jan_days = pd.date_range(JAN_START, JAN_END).strftime("%Y-%m-%d").tolist()

print(f"loading e_track for {len(box_tag_idx)} tags × {len(days)} days ...")
e_region = (
    ds.e_track.isel(tagging_mask=box_tag_idx).sel(time=days).load()
)  # (time, tag, lat, lon)
tp_region = (
    ds.tagged_precip.isel(tagging_mask=box_tag_idx).sel(time=jan_days).load()
)  # (time, tag)
print(f"e_region: shape={e_region.shape}  size={e_region.nbytes/1e9:.2f} GB")

# ---------- area weights: cos(lat) proxy for cell area ----------
tag_area_w = xr.DataArray(
    np.cos(np.deg2rad(tag_lat[box_tag_idx])),
    dims="tagging_mask",
)
# For PDF normalisation on the 2-D source grid we also need cos(lat) weights.
grid_area_w = xr.DataArray(
    np.cos(np.deg2rad(ds.latitude.values))[:, None]
    * np.ones(len(ds.longitude))[None, :],
    dims=("latitude", "longitude"),
    coords={"latitude": ds.latitude, "longitude": ds.longitude},
)

# ---------- aggregation (precip- and area-weighted) ----------
tp_window = tp_region.sel(time=days)  # (time, tag)
rain_weights = tp_window * tag_area_w  # (time, tag)
rain_total = rain_weights.sum("tagging_mask")  # (time,)
safe_denom = rain_total.where(rain_total > 0)

e_abs = (e_region * rain_weights).sum("tagging_mask") / safe_denom
# (time, lat, lon)   — absolute, kg m⁻² day⁻¹, NaN on dry days

# Basin-mean tagged precipitation for the bar chart context.
tp_basin_mean = (
    (tp_region * tag_area_w).sum("tagging_mask") / tag_area_w.sum()
)  # (time,)  — January full month

# ---------- PDF normalisation (area-weighted, per day) ----------
# Integral over space ∫∫ e(x,y) cos(lat) dlat dlon. dlat=dlon=const so they
# cancel; only cos(lat) weight matters for the normalisation.
daily_integral = (e_abs * grid_area_w).sum(("latitude", "longitude"))
e_pdf = e_abs / daily_integral.where(daily_integral > 0)
# (time, lat, lon) — dimensionless density, NaN on dry days

# ---------- CDF per day (spatial rank-cumulative %) ----------
e_cdf = xr.full_like(e_abs, np.nan, dtype=float)
for t in range(e_abs.sizes["time"]):
    vals = e_abs.isel(time=t).values
    flat = vals.ravel()
    mask = np.isfinite(flat) & (flat > 0)
    if not mask.any():
        continue
    valid = flat[mask]
    order_desc = np.argsort(valid)[::-1]
    cum_pct = 100.0 * np.cumsum(valid[order_desc]) / valid.sum()
    flat_cdf = np.full_like(flat, np.nan, dtype=float)
    valid_positions = np.where(mask)[0]
    flat_cdf[valid_positions[order_desc]] = cum_pct
    e_cdf.values[t] = flat_cdf.reshape(vals.shape)

# ---------- shared colour scales (so frames are comparable) ----------
abs_pos = e_abs.values[(e_abs.values > 0) & np.isfinite(e_abs.values)]
VMAX_ABS = float(np.quantile(abs_pos, 0.99))
pdf_pos = e_pdf.values[(e_pdf.values > 0) & np.isfinite(e_pdf.values)]
VMAX_PDF = float(np.quantile(pdf_pos, 0.99))
TP_YMAX = float(tp_basin_mean.max()) * 1.1
print(f"vmax abs (99 pct): {VMAX_ABS:.4f} kg m-2 day-1")
print(f"vmax pdf (99 pct): {VMAX_PDF:.3e}  (dimensionless density)")


# ============================================================================
#  Plotting
# ============================================================================

import matplotlib.patches as mpatches


def _base_map(ax):
    """Project plotting conventions applied to a cartopy axis."""
    gdf.boundary.plot(
        ax=ax, linewidth=0.4, edgecolor="grey",
        transform=ccrs.PlateCarree(), zorder=3,
    )
    ax.coastlines(resolution="50m", color="black", linewidth=1)
    # Target box in red (so readers see *where* they're looking at rain from).
    ax.add_patch(mpatches.Rectangle(
        (BOX_LON_W, BOX_LAT_S),
        BOX_LON_E - BOX_LON_W, BOX_LAT_N - BOX_LAT_S,
        linewidth=2, edgecolor="red", facecolor="none",
        transform=ccrs.PlateCarree(), zorder=5,
    ))
    ax.set_xlim(ds.longitude.min(), ds.longitude.max())
    ax.set_ylim(ds.latitude.min(), ds.latitude.max())
    gl = ax.gridlines(draw_labels=True, linewidth=0.0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 10))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 10))
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}


def _tp_bar(ax, day):
    """Bottom strip: January tagged-precip bar chart, current day highlighted."""
    bar_colors = ["steelblue"] * len(tp_basin_mean)
    idx = int(np.where(tp_basin_mean.time.values == np.datetime64(day))[0][0])
    bar_colors[idx] = "crimson"
    ax.bar(tp_basin_mean.time.values, tp_basin_mean.values,
           width=0.8, color=bar_colors, edgecolor="black", linewidth=0.3)
    ax.set_ylim(0, TP_YMAX)
    ax.set_ylabel("box mean\n[kg m⁻² d⁻¹]", fontsize=14)
    ax.tick_params(axis="x", rotation=30, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.3)


def draw_abs(day, out_png):
    tp_day = float(tp_basin_mean.sel(time=day))
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.02)
    ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    da = e_abs.sel(time=day)
    mesh = ax.pcolormesh(
        da.longitude, da.latitude, da.where(da > 0),
        cmap=cmc.batlowW_r, vmin=0, vmax=VMAX_ABS,
        transform=ccrs.PlateCarree(), shading="auto",
    )
    _base_map(ax)
    ax.set_title(
        f"{day}  |  box mean tagged precip = {tp_day:.1f} kg m⁻² day⁻¹  |  "
        f"absolute source"
    )
    cb = fig.colorbar(mesh, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Tracked evaporation (precip-weighted)  [kg m⁻² day⁻¹]")
    cb.ax.tick_params(labelsize=14)
    _tp_bar(fig.add_subplot(gs[1]), day)
    fig.savefig(out_png, dpi=300, pad_inches=0.1, facecolor="white")
    plt.close(fig)


def draw_pdf(day, out_png):
    tp_day = float(tp_basin_mean.sel(time=day))
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.02)
    ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    da = e_pdf.sel(time=day)
    mesh = ax.pcolormesh(
        da.longitude, da.latitude, da.where(da > 0),
        cmap=cmc.batlowW_r, vmin=0, vmax=VMAX_PDF,
        transform=ccrs.PlateCarree(), shading="auto",
    )
    _base_map(ax)
    ax.set_title(
        f"{day}  |  box mean tagged precip = {tp_day:.1f} kg m⁻² day⁻¹  |  "
        f"source PDF (area-weighted, ∫=1)"
    )
    cb = fig.colorbar(mesh, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Source probability density  [dimensionless]")
    cb.ax.tick_params(labelsize=14)
    _tp_bar(fig.add_subplot(gs[1]), day)
    fig.savefig(out_png, dpi=300, pad_inches=0.1, facecolor="white")
    plt.close(fig)


def draw_cdf(day, out_png):
    tp_day = float(tp_basin_mean.sel(time=day))
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.02)
    ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    da = e_cdf.sel(time=day)
    # Filled contours show the cumulative percentage "source shed" layers.
    cf = ax.contourf(
        da.longitude, da.latitude, da,
        levels=CDF_LEVELS + [100.0],
        cmap=cmc.batlowW, extend="neither",
        transform=ccrs.PlateCarree(),
    )
    ax.contour(
        da.longitude, da.latitude, da,
        levels=CDF_LEVELS, colors="black", linewidths=0.5,
        transform=ccrs.PlateCarree(),
    )
    _base_map(ax)
    ax.set_title(
        f"{day}  |  box mean tagged precip = {tp_day:.1f} kg m⁻² day⁻¹  |  "
        f"cumulative source percentile"
    )
    cb = fig.colorbar(cf, ax=ax, shrink=0.75, pad=0.02,
                      ticks=CDF_LEVELS + [100])
    cb.set_label("Cumulative share of total tracked evaporation  [%]")
    cb.ax.tick_params(labelsize=14)
    _tp_bar(fig.add_subplot(gs[1]), day)
    fig.savefig(out_png, dpi=300, pad_inches=0.1, facecolor="white")
    plt.close(fig)


# ---------- render all frames ----------
for d in days:
    draw_abs(d, OUT_ABS / f"{d}.png")
    draw_pdf(d, OUT_PDF / f"{d}.png")
    draw_cdf(d, OUT_CDF / f"{d}.png")
    print(f"  {d}")

print(f"\ndone. {len(days)} frames × 3 views in outputs/box_{{abs,pdf,cdf}}/")
