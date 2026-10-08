# Committed test data

Small real-data subsets so the test suite renders reports without downloading from the sample
registry. Built by `make_subset.py` (reproducible: `python tests/data/make_subset.py`).

| File | From | Profiles | Size | OG1 | Renders |
|------|------|----------|------|-----|---------|
| `sea045_subset.nc` | `sea045_20230604T1253_delayed.nc` (VOTO) | first 12 | ~780 KB | 16/16 mandatory | CTD, Oxygen, Optics, Inventory, landing |
| `sg014_subset.nc` | `sg014_20040924T182454_delayed_subset.nc` (Seaglider) | first 6 | ~280 KB | 15/16 (no `id`) | Flight page |

Each is the first N `PROFILE_NUMBER`s of the full mission, reduced to a **keep-list** of variables
(the ones the report and interactive layers use — including `DIVE_NUM` for `interactive.mission_map`
— plus a few the report does not plot — `DENSITY`, `BBP700`, `VOLTAGE` — so the inventory page has
extras to list), with geophysical data downcast to **float32** and only time and position (`TIME`,
`TIME_GPS`, `LATITUDE`/`LONGITUDE` and their `_GPS`) kept **float64**. Written with h5netcdf gzip.
The keep-list and profile count are tuned against a panel-parity check.

## Why these two

- **sea045** is the clean, complete file — `load_sample_dataset()`'s default, 16/16 mandatory OG1
  globals, CTD + oxygen + optics. It is what most report tests render.
- **sg014** is a realistic *imperfect* file: a Seaglider→OG1 mission missing the mandatory `id`
  global (15/16) and the `SENSOR_*` catalog. It is kept on purpose — it is the only flight-capable
  sample, and its gap exercises the missing-mandatory → amber conformance path and the
  `mission_id` → source-stem fallback on real data.

## What the subsets cannot exercise (these keep a `slow` full-sample test)

The profile count keeps every diagnostic panel non-stub **except**:

- **Day/night offset** (`ctd_daynight_psal`, `opt_daynight_chla`) — sea045 is a Baltic-June mission
  that barely has night; catching night profiles needs the multi-day full mission (its own `slow`
  test on the full sample).
- **Flight page** — needs `GLIDER_VERT_VELO_MODEL`, which sea045 lacks; `sg014_subset.nc` covers it.

## Provenance and credit

Real glider data. `sea045` is Voice of the Ocean (VOTO) sample data, the default of
`fetchers.load_sample_dataset()`; `sg014` is a Seaglider mission in the sample registry. Sample-data
credit is in the package `README.md`. Redistributed here for testing only.
