# Dataset: `zarr_SA_0_25_deg_daily_1990_2020`

WAM2Layers (v3.3.1) backward moisture-tracking output over South America,
0.25° daily resolution, 1990–2019.

- **Path on hinton**: `/cache/isaacgbhk/Work/i_wam/output/zarr_SA_0_25_deg_daily_1990_2020/`
- **Layout**: 30 per-year zarr stores (`1990.zarr` … `2019.zarr`), Zarr v3, blosc/zstd + bitshuffle
- **Total size on disk**: ~7.6 TB compressed (uncompressed float32 would be ~260 TB for the three 4-D variables)

## Opening

```python
import xarray as xr

# one year
ds = xr.open_zarr(
    "/cache/isaacgbhk/Work/i_wam/output/zarr_SA_0_25_deg_daily_1990_2020/2011.zarr"
)

# all 30 years
from glob import glob
paths = sorted(glob("/cache/isaacgbhk/Work/i_wam/output/zarr_SA_0_25_deg_daily_1990_2020/*.zarr"))
ds_all = xr.open_mfdataset(paths, engine="zarr", parallel=True)
```

## Schema (per year)

```
Dimensions:        (time: 365, tagging_mask: 25186, latitude: 301, longitude: 261)
Coordinates:
  * time           (time)         datetime64[ns]  1990-01-01 … 1990-12-31
  * tagging_mask   (tagging_mask) int32           0 … 25185
    tag_lat        (tagging_mask) float32         lat of each tag cell
    tag_lon        (tagging_mask) float32         lon of each tag cell
  * latitude       (latitude)     float32         +15.0 → -60.0   [DESCENDING]
  * longitude      (longitude)    float32         -90.0 → -25.0   [ascending]
Data variables:
    e_track        (time, tagging_mask, latitude, longitude)  float32
    gains          (time, tagging_mask, latitude, longitude)  float32
    losses         (time, tagging_mask, latitude, longitude)  float32
    tagged_precip  (time, tagging_mask)                       float32
```

### Variable semantics

All data variables have units `kg m-2 accumulated per output time step`. Output
frequency is 1 day, so each pixel is effectively **mm day⁻¹** of water.

| Variable | Meaning |
|---|---|
| `e_track` | tracked evaporation — moisture evaporated at (lat, lon) that ends up as rain at the tag cell on day `time`. **Primary variable** for "where did this rain come from?" |
| `gains` | moisture gained by the tracked parcel along the trajectory |
| `losses` | moisture lost by the tracked parcel along the trajectory |
| `tagged_precip` | precipitation at each tag cell per day — useful for time-series and for picking event days |

### Chunk layout

```
shape  : (time=365, tagging_mask=25186, latitude=301, longitude=261)
chunks : (       365,            10,           301,           261)
```

- 2519 chunks per year for each 4-D variable, each ~1.15 GB decompressed
- **Fast axis: `tagging_mask`** — reading 1–10 tag indices pulls a single chunk
- **Slow axis: `time` / `latitude` / `longitude`** — slicing these without an explicit `tagging_mask` selection touches every chunk (2.9 TB/yr)

## Run config (from `ds.attrs["WAM2Layers_config"]`)

```
tracking_direction : backward
tagging_region     : [-60.0, -15.0, -59.0, -14.0]   # lon_W, lat_S, lon_E, lat_N
tracking_domain    : [-90.0, -60.0, -25.0, 15.0]
input_frequency    : 1h
output_frequency   : 1D
timestep           : 600                              # seconds
kvf                : 3.0                              # vertical flux tuning
periodic_boundary  : False
level_type         : model_levels
```

## Non-obvious insights

### 1. `tagging_mask` ≠ the `tagging_region` in the config

The config's `tagging_region = [-60, -15, -59, -14]` is a tiny 1°×1° box, but in
the zarr store `tag_lat/tag_lon` span essentially the **entire tracking
domain** (-90 … -26 lon, -58 … +14.75 lat), with 25186 tag cells ≈ 32% of the
78561 domain cells.

**Practical meaning:** each `tagging_mask` index corresponds to one specific
tagged grid cell somewhere in the domain — not to the config's initial box.
To find the tag for a real-world location, search `tag_lat / tag_lon` for the
nearest cell:

```python
tag_lat = ds.tag_lat.compute().values
tag_lon = ds.tag_lon.compute().values
i = int(np.argmin((tag_lat - TARGET_LAT)**2 + (tag_lon - TARGET_LON)**2))
```

### 2. Analysis should be tag-first, time/space-second

Because chunks bundle `(full year × 10 tags × full map)`, an analysis pattern
of "pick a tag, look at its full time/space source field" is effectively free,
while "give me 2011-01-12 across all tags" is the worst case.

- ✅ `ds.e_track.isel(tagging_mask=i)` → one chunk
- ❌ `ds.e_track.sel(time="2011-01-12")` → touches all 2519 chunks

### 3. Lazy by default — never `.load()` before selecting

A full 4-D variable is ~3 TB/year uncompressed float32. The 30-year `open_mfdataset`
Dataset reports as ~260 TB. Any `.compute()` / `.values` / `.load()` without
a preceding `isel(tagging_mask=...)` will OOM.

### 4. `latitude` is descending

`ds.latitude[0] = +15.0`, `ds.latitude[-1] = -60.0`. Any positional slice must
respect this:

```python
# right
ds.sel(latitude=slice(0, -30))          # 0°N down to 30°S

# wrong — returns empty
ds.sel(latitude=slice(-30, 0))
```

### 5. `tracking_direction: backward` only

`e_track[t, tag, y, x]` = "the rain that fell at `tag` on day `t` was sourced
from evaporation at `(y, x)`." This answers **"where did this rain come from?"**
It does *not* answer **"where did this evaporation end up?"** — that requires a
separate forward-tracking run.

### 6. `WAM2Layers_config` reflects one batch, not the whole archive

The 1990 zarr's config shows `tracking_start/end: 1990-01-01 … 1994-01-31`,
yet we have 30 annual stores (1990–2019). The dataset was produced in multiple
batches and concatenated. **Before doing multi-year work, spot-check that
`attrs["WAM2Layers_config"]` is consistent across years** (e.g., same
`tracking_domain`, `kvf`, `level_type`, `input_frequency`).

### 7. Units caveat for aggregations

`e_track` is accumulated per output time step (= 1 day). When summing over a
window of *N* days, the result is "total tracked evaporation over those N days
contributing to the tag cell." For a fair **before vs during vs after**
comparison with different window lengths, divide by *N* to get a daily mean.

### 8. The time series `tagged_precip[:, tag_idx]` is the cheap diagnostic

To pick event days or sanity-check, read the 1-D `tagged_precip` for your tag
— it's ~1.5 kB per tag per year and tells you which days actually had tagged
rain. Use it to:

- Confirm the event date window is right (peak day may shift from reports)
- Detect secondary events in your "before" / "after" buffer
- Normalize source maps by tagged rain amount if you want unitless maps

## Companion data on hinton

- `data/hybas_sa_lev02_v1c/` — HydroBASINS level-2 polygons for South America
  (synced from lorenz). Useful for overlay / regionmask-ing moisture source
  maps by basin.
- Aggregated stores also present locally but not yet inspected:
  - `zarr_SA_0_25_deg_monthly_1990_2020/`
  - `zarr_SA_0_25_deg_yearly_1990_2020/`
