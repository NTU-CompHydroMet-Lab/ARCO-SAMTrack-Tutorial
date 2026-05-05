# Dataset: AguaTrack-ARCO-SA (monthly + yearly aggregates)

Time-aggregated companions to the daily store
([`dataset_daily.md`](dataset_daily.md)). Both stores are produced by
summing the daily AguaTrack-ARCO-SA fields over fixed time windows
(calendar months / calendar years). Spatial schema, units, and the
`tagging_mask` / `tag_lat` / `tag_lon` semantics are identical to the
daily store — only the `time` axis changes.

- **Distribution**:
  [`AguaTrackSA/AguaTrack-ARCO-SA-Aggregated`](https://huggingface.co/datasets/AguaTrackSA/AguaTrack-ARCO-SA-Aggregated)
  on HuggingFace.
- **Layout**: two consolidated zarr stores at the dataset root.
  - `AguaTrack_ARCO_SA_monthly.zarr` — `time=360` (1990-01-01 … 2019-12-01)
  - `AguaTrack_ARCO_SA_yearly.zarr` — `time=30` (1990-01-01 … 2019-01-01)
- **DOI**: shares the daily-store DOI
  [`10.57967/hf/8650`](https://doi.org/10.57967/hf/8650) — the
  aggregates are derived data, not an independent dataset.

## Opening

```python
import xarray as xr

ds_monthly = xr.open_zarr(
    "hf://datasets/AguaTrackSA/AguaTrack-ARCO-SA-Aggregated"
    "/AguaTrack_ARCO_SA_monthly.zarr",
    storage_options={"revision": "main"},
)

ds_yearly = xr.open_zarr(
    "hf://datasets/AguaTrackSA/AguaTrack-ARCO-SA-Aggregated"
    "/AguaTrack_ARCO_SA_yearly.zarr",
    storage_options={"revision": "main"},
)
```

For local users with the archive on disk, swap the URLs for filesystem
paths and drop `storage_options`.

## Schema

```
Dimensions:        (time: T, tagging_mask: 25186, latitude: 301, longitude: 261)
Coordinates:
  * time           (time)         datetime64[ns]    monthly: 1990-01-01 … 2019-12-01 (T=360)
                                                    yearly:  1990-01-01 … 2019-01-01 (T=30)
  * tagging_mask   (tagging_mask) int32             0 … 25185
    tag_lat        (tagging_mask) float32           lat of each tag cell
    tag_lon        (tagging_mask) float32           lon of each tag cell
  * latitude       (latitude)     float32           +15.0 -> -60.0   [DESCENDING]
  * longitude      (longitude)    float32           -90.0 -> -25.0   [ascending]
Data variables:
    e_track        (time, tagging_mask, latitude, longitude)  float32
    lsm            (latitude, longitude)                      bool
    tagged_precip  (time, tagging_mask)                       float32
```

### Differences from the daily store

| | Daily | Aggregated |
|---|---|---|
| Number of stores | 30 (one per year) | 2 (one per resolution, 30 years each) |
| `time` length | 365 / 366 per store | 360 (monthly) or 30 (yearly) |
| `gains`, `losses` | included | **omitted** — only `e_track` + `tagged_precip` are aggregated |
| `lsm` | implied by `latitude`/`longitude` | shipped explicitly as a `(lat, lon)` bool variable |

`e_track` and `tagged_precip` are summed over each window. Units stay
"mm of water-equivalent accumulated over the window" — divide by the
number of days in the window if you want a daily mean.

### Chunk layout

```
yearly:  shape (30,  25186, 301, 261)   chunks (30,  100, 301, 261)
monthly: shape (360, 25186, 301, 261)   chunks (360,  10, 301, 261)
```

- **Fast axis: `tagging_mask`** — same story as the daily store. Read
  one tag, one chunk; read all tags, all chunks.
- **Whole-record stripe**: each chunk spans the full `time` axis. So
  `ds.sel(time="2011")` is *not* a chunk-reducing slice — it still
  pulls the same chunks as `ds`. The cheap thing is `isel(tagging_mask=…)`.

## Why these exist (vs. the daily store)

The daily store is the ground truth and the only place to do
event-scale work. But three patterns become *much* cheaper on the
aggregates:

1. **Multi-year climatologies** — averaging or summing over decades is
   30× cheaper on the yearly store.
2. **Seasonal / ENSO composites** — picking a phase's worth of months
   is 12× cheaper on the monthly store than re-aggregating from daily.
3. **Receptor-box analyses summing over many tags** — the smaller chunk
   size lets you fan out across tags without OOMing.

Use the daily store only when you need sub-monthly resolution
(individual events, spell statistics) — for everything else, prefer
the aggregate that matches your analysis cadence.

## Caveats

- **No `gains` / `losses`.** Diagnostic variables that close the
  WAM2Layers budget aren't carried forward. If you need them, go to the
  daily store.
- **Calendar-window sums, not running means.** The "monthly" axis is
  Jan, Feb, …, Dec — not 30-day rolling sums. Compositing across years
  is therefore a clean multi-year mean of like calendar months.
- **`time` indexing.** `ds.time.dt.year` / `ds.time.dt.month` are the
  natural way to slice; the existing tutorial notebooks (02, 03, 04)
  show the standard `swap_dims({"time": "year"})` pattern when
  downstream code wants an integer year axis.
