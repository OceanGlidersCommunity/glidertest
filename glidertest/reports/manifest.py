"""The per-mission manifest: a ``report.json`` written beside each mission's ``index.html``.

:func:`mission_manifest` serialises the facts the navigator needs — identity, extent, counts, sensor
presence, OG1 conformance, worst QC, a decimated track, and the page list — into a plain dict. Every
value is one glidertest already computes for the masthead, metadata and QC sections; the manifest is
the machine-readable form, read back by :func:`glidertest.reports.navigator.build_navigator` (and by
downstream tools) without reopening the NetCDF file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .. import og1_attrs, qc, tools

if TYPE_CHECKING:
    from collections.abc import Sequence

    import xarray as xr

#: Track points kept in the manifest (decimated); enough to draw a recognisable track on the map.
_TRACK_POINTS = 200


def _platform_serial(ds: xr.Dataset) -> str | None:
    """Return the platform serial (first ``PLATFORM_SERIAL_NUMBER`` value), or None."""
    if "PLATFORM_SERIAL_NUMBER" not in ds:
        return None
    sv = np.atleast_1d(ds["PLATFORM_SERIAL_NUMBER"].values).ravel()
    if not sv.size:
        return None
    first = sv[0]
    if isinstance(first, (float, np.floating)) and not np.isfinite(first):
        return None
    return str(first)


def _profile_counts(ds: xr.Dataset) -> tuple[int, int, int]:
    """Return ``(n_profiles, n_dive, n_climb)`` from PROFILE_NUMBER and PROFILE_DIRECTION."""
    if "PROFILE_NUMBER" not in ds:
        return (0, 0, 0)
    pn = np.asarray(ds["PROFILE_NUMBER"].values)
    finite = np.isfinite(pn)
    n_prof = int(np.unique(pn[finite]).size)
    if "PROFILE_DIRECTION" not in ds:
        return (n_prof, 0, 0)
    pdir = np.asarray(ds["PROFILE_DIRECTION"].values)
    m = finite & np.isfinite(pdir)
    _uniq, idx = np.unique(pn[m], return_index=True)
    prof_dir = pdir[m][idx]
    return (n_prof, int((prof_dir == -1).sum()), int((prof_dir == 1).sum()))


def _extent(ds: xr.Dataset) -> dict[str, float | None]:
    """Return the lat/lon bounding box from the data, or Nones when positions are absent."""
    out: dict[str, float | None] = {k: None for k in ("lat_min", "lat_max", "lon_min", "lon_max")}
    if "LATITUDE" in ds and "LONGITUDE" in ds:
        la = np.asarray(ds["LATITUDE"].values)
        lo = np.asarray(ds["LONGITUDE"].values)
        la, lo = la[np.isfinite(la)], lo[np.isfinite(lo)]
        if la.size and lo.size:
            out = {
                "lat_min": round(float(la.min()), 4),
                "lat_max": round(float(la.max()), 4),
                "lon_min": round(float(lo.min()), 4),
                "lon_max": round(float(lo.max()), 4),
            }
    return out


def _track(ds: xr.Dataset) -> list[list[float]]:
    """Return the track as ``[[lon, lat], ...]`` decimated to at most :data:`_TRACK_POINTS` points."""
    if "LATITUDE" not in ds or "LONGITUDE" not in ds:
        return []
    lon = np.asarray(ds["LONGITUDE"].values)
    lat = np.asarray(ds["LATITUDE"].values)
    m = np.isfinite(lon) & np.isfinite(lat)
    lon, lat = lon[m], lat[m]
    if lon.size > _TRACK_POINTS:
        idx = np.linspace(0, lon.size - 1, _TRACK_POINTS).astype(int)
        lon, lat = lon[idx], lat[idx]
    return [[round(float(x), 4), round(float(y), 4)] for x, y in zip(lon, lat)]


def _worst_qc(ds: xr.Dataset) -> dict[str, Any]:
    """Return ``{delivered, worst_var, worst_bad_pct}`` across the file's ``*_QC`` variables.

    "Bad" is the suspect + fail fraction of a flag array (missing/not-evaluated are not bad). The
    worst variable is the one with the highest bad fraction; ``delivered`` is whether the file
    carries any ``*_QC`` variable at all.
    """
    qc_vars = [n for n in ds.data_vars if n.endswith("_QC")]
    worst_var: str | None = None
    worst_pct = -1.0
    for name in qc_vars:
        counts = qc.flag_counts(ds[name].values)
        n = sum(counts.values())
        if not n:
            continue
        pct = 100.0 * (counts["suspect"] + counts["fail"]) / n
        if pct > worst_pct:
            worst_pct, worst_var = pct, name[: -len("_QC")]
    return {
        "delivered": bool(qc_vars),
        "worst_var": worst_var,
        "worst_bad_pct": round(worst_pct, 2) if worst_pct >= 0 else None,
    }


def _iso(value: np.datetime64) -> str | None:
    """Return an ISO minute-resolution string for a datetime64, or None for NaT."""
    return None if np.isnat(value) else str(value.astype("datetime64[m]"))


def mission_manifest(
    ds: xr.Dataset,
    *,
    mission_id: str,
    source_name: str,
    source_size_bytes: int | None,
    pages: Sequence[tuple[str, str]],
    version: str,
    generated_at: str,
) -> dict[str, Any]:
    """Return the mission manifest dict, serialised to ``report.json`` beside ``index.html``.

    Parameters
    ----------
    ds : xarray.Dataset
        An OG1 glider dataset.
    mission_id : str
        The mission identifier (the subdirectory name the report is written into).
    source_name : str
        The source NetCDF file name (or ``<id>.nc`` when not from a file).
    source_size_bytes : int or None
        The source file size, or None when the dataset did not come from a file.
    pages : sequence of (str, str)
        ``(filename, label)`` for each page written for this mission.
    version : str
        The glidertest version that produced the report.
    generated_at : str
        The generation timestamp (same string shown in the masthead).

    Returns
    -------
    dict
        The manifest, JSON-serialisable.
    """
    n_prof, n_dive, n_climb = _profile_counts(ds)
    t0 = ds["TIME"].min().values if "TIME" in ds else np.datetime64("NaT")
    t1 = ds["TIME"].max().values if "TIME" in ds else np.datetime64("NaT")
    duration_s = None
    if "TIME" in ds and not (np.isnat(t0) or np.isnat(t1)):
        duration_s = int((t1 - t0) / np.timedelta64(1, "s"))
    max_depth_m = None
    if "DEPTH" in ds and "PROFILE_NUMBER" in ds:
        hi = float(tools.max_depth_per_profile(ds).max())
        max_depth_m = round(hi, 1) if np.isfinite(hi) else None

    missing = [n for n in og1_attrs.MANDATORY_GLOBALS if not str(ds.attrs.get(n) or "").strip()]
    summary = og1_attrs.conformance_summary(ds.attrs)

    return {
        "manifest_version": 1,
        "id": mission_id,
        "platform": str(ds.attrs.get("platform", "")) or None,
        "platform_serial": _platform_serial(ds),
        "start": _iso(t0),
        "end": _iso(t1),
        "duration_s": duration_s,
        "n_profiles": n_prof,
        "n_dive": n_dive,
        "n_climb": n_climb,
        **_extent(ds),
        "max_depth_m": max_depth_m,
        "n_records": int(ds.sizes.get("N_MEASUREMENTS", 0)),
        "source_file": source_name,
        "source_size_bytes": source_size_bytes,
        "sensors": {
            "ctd": "TEMP" in ds or "PSAL" in ds,
            "oxygen": "DOXY" in ds,
            "optics": "CHLA" in ds or "BBP700" in ds,
            "flight": "GLIDER_VERT_VELO_MODEL" in ds,
        },
        "og1": {
            "mandatory_present": summary["mandatory_present"],
            "mandatory_total": summary["mandatory_total"],
            "missing": missing,
        },
        "qc": _worst_qc(ds),
        "pages": [{"file": f, "label": lbl} for f, lbl in pages],
        "track": _track(ds),
        "glidertest_version": version,
        "generated_at": generated_at,
    }
