"""
Minimal shared helpers for the WAM2Layers tutorial scripts.
Only functions that are used by more than one script live here.

Units & conventions
-------------------
WAM2Layers stores flux variables as kg m⁻² accumulated per output time step.
Output_frequency is 1 day, so numerical values equal kg m⁻² day⁻¹ (= mm/day).
"""

from pathlib import Path

import cartopy.crs as ccrs
import cmcrameri.cm as cmc
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr

PLT_RC = {
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 18,
}


def cos_lat_weights(lats_deg: np.ndarray) -> xr.DataArray:
    """cos(lat) weights along the tagging_mask axis (cell-area proxy at 0.25°)."""
    return xr.DataArray(
        np.cos(np.deg2rad(lats_deg)),
        dims="tagging_mask",
    )


def area_weighted_basin_mean(
    tp_region: xr.DataArray,
    tag_area_w: xr.DataArray,
) -> xr.DataArray:
    """
    Area-weighted basin-mean tagged precipitation.

    tp_region: (time, tagging_mask)   kg m⁻² day⁻¹ per tag cell
    tag_area_w: (tagging_mask,)       cos(lat) weights
    returns:    (time,)               kg m⁻² day⁻¹  basin mean
    """
    return (tp_region * tag_area_w).sum("tagging_mask") / tag_area_w.sum()


def precip_weighted_source(
    e_region: xr.DataArray,
    tp_window: xr.DataArray,
    tag_area_w: xr.DataArray,
) -> xr.DataArray:
    """
    Precipitation- and area-weighted moisture-source field per day.

    e_region:  (time, tagging_mask, latitude, longitude)
    tp_window: (time, tagging_mask)
    tag_area_w:(tagging_mask,)

    On dry days (total weighted rain = 0) the result is NaN — there is
    nothing to attribute.
    """
    rain_weights = tp_window * tag_area_w
    rain_total = rain_weights.sum("tagging_mask")  # (time,)
    safe_denom = rain_total.where(rain_total > 0)
    return (e_region * rain_weights).sum("tagging_mask") / safe_denom


def source_pdf(field: xr.DataArray, lat_weights_2d: xr.DataArray) -> xr.DataArray:
    """
    Normalise a daily source field so each day integrates to 1 (area-weighted).

    field:          (time, latitude, longitude)
    lat_weights_2d: (latitude, longitude)  — cos(lat) broadcast to 2-D

    Returns: (time, latitude, longitude) with units of m⁻² (probability density
    per unit area). Dry days stay NaN.
    """
    # Integral over 2-D surface: sum_{y,x} field(y,x) * dA(y,x)
    integral = (field * lat_weights_2d).sum(("latitude", "longitude"))
    safe = integral.where(integral > 0)
    return field / safe


def source_cdf(field: xr.DataArray) -> xr.DataArray:
    """
    Per-day spatial CDF of a source field.

    For each day, rank all positive pixels from largest to smallest, cumulatively
    sum, and normalise to [0, 100]. Each pixel's value is then "percentage of
    total source accounted for by this pixel and all larger-contributing pixels".

    The 10% contour = "pixels making up the top 10% of sources"
    The 90% contour = "the smallest set covering 90% of the source"

    Dry/zero pixels become NaN so they don't occlude the contour map.
    """
    out = xr.full_like(field, np.nan, dtype=float)
    for t in range(field.sizes["time"]):
        vals = field.isel(time=t).values
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
        out.values[t] = flat_cdf.reshape(vals.shape)
    return out


def decorate_map(
    ax,
    ds: xr.Dataset,
    gdf: gpd.GeoDataFrame | None = None,
    highlight_geom=None,
):
    """Apply project plotting conventions to a cartopy axis."""
    ax.coastlines(resolution="50m", color="black", linewidth=1)
    ax.set_xlim(ds.longitude.min(), ds.longitude.max())
    ax.set_ylim(ds.latitude.min(), ds.latitude.max())

    if gdf is not None:
        gdf.boundary.plot(
            ax=ax, linewidth=0.4, edgecolor="grey",
            transform=ccrs.PlateCarree(), zorder=3,
        )
    if highlight_geom is not None:
        highlight_geom.boundary.plot(
            ax=ax, linewidth=2.0, edgecolor="red",
            transform=ccrs.PlateCarree(), zorder=5,
        )

    gl = ax.gridlines(draw_labels=True, linewidth=0.0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 10))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 10))
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
