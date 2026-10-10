"""The per-mission manifest: a ``report.json`` written beside each mission's ``index.html``.

:func:`mission_manifest` serialises the facts the navigator needs — identity, extent, counts, sensor
presence, OG1 conformance, worst QC, a decimated track, and the page list — into a plain dict. Every
value is one glidertest already computes for the masthead, metadata and QC sections; the manifest is
the machine-readable form, read back by :func:`glidertest.reports._navigator.build_navigator` (and by
downstream tools) without reopening the NetCDF file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .. import og1_attrs, qc

if TYPE_CHECKING:
    from collections.abc import Sequence

    import xarray as xr

#: Track points kept in the manifest (decimated); enough to draw a recognisable track on the map.
_TRACK_POINTS = 200


def _track(ds: xr.Dataset) -> list[list[float]]:
    """Return the track as ``[[lon, lat], ...]`` decimated to at most :data:`_TRACK_POINTS` points."""
    if "LATITUDE" not in ds or "LONGITUDE" not in ds:
        return []
    lon = np.asarray(ds["LONGITUDE"].values)
    lat = np.asarray(ds["LATITUDE"].values)
    m = np.isfinite(lon) & np.isfinite(lat)
    # Drop only positions the file flags bad (OG1 QC flag 3 bad / 4 probably-bad), so a spurious fix
    # does not stretch the track or the fleet-map extent. Flag 0 ("no QC applied") is the common
    # delivered state and is kept — dropping it would empty the track of every file whose provider
    # ran no position QC. Missing positions are already removed by the finite mask. No QC var -> keep.
    for var in ("LONGITUDE", "LATITUDE"):
        qc = f"{var}_QC"
        if qc in ds:
            m &= ~np.isin(np.asarray(ds[qc].values), (3, 4))
    lon, lat = lon[m], lat[m]
    if lon.size > _TRACK_POINTS:
        idx = np.linspace(0, lon.size - 1, _TRACK_POINTS).astype(int)
        lon, lat = lon[idx], lat[idx]
    return [[round(float(x), 4), round(float(y), 4)] for x, y in zip(lon, lat)]


def _worst_qc(ds: xr.Dataset) -> dict[str, Any]:
    """Return ``{delivered, worst_var, worst_bad_pct}`` across the file's ``*_QC`` variables.

    Three distinct states, so an all-missing / "no QC applied" file never reads as clean:

    - no ``*_QC`` variable at all → ``delivered=False``, ``worst_bad_pct=None``;
    - ``*_QC`` present but nothing evaluated (all flags missing/not-evaluated/other, e.g. OG1 flag 0)
      → ``delivered=True``, ``worst_bad_pct=None``;
    - at least one evaluated flag → ``worst_bad_pct`` is the highest suspect+fail fraction *of the
      evaluated flags only* (good + suspect + fail), ``worst_var`` the variable carrying it.
    """
    qc_vars = [n for n in ds.data_vars if n.endswith("_QC")]
    if not qc_vars:
        return {"delivered": False, "worst_var": None, "worst_bad_pct": None}
    worst_var: str | None = None
    worst_pct = -1.0
    for name in qc_vars:
        counts = qc.flag_counts(ds[name].values)
        evaluated = counts["good"] + counts["suspect"] + counts["fail"]
        if evaluated == 0:
            continue
        pct = 100.0 * (counts["suspect"] + counts["fail"]) / evaluated
        if pct > worst_pct:
            worst_pct, worst_var = pct, name[: -len("_QC")]
    return {
        "delivered": True,
        "worst_var": worst_var,
        "worst_bad_pct": round(worst_pct, 2) if worst_pct >= 0 else None,
    }


def mission_manifest(
    ds: xr.Dataset,
    *,
    mission_id: str,
    source_name: str,
    source_size_bytes: int | None,
    pages: Sequence[tuple[str, str]],
    version: str,
    generated_at: str,
    facts: dict[str, Any] | None = None,
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
    facts : dict, optional
        A prebuilt :func:`glidertest.reports.metadata.mission_facts` result; computed from *ds*
        when None. The report passes one build so the masthead and manifest share it.

    Returns
    -------
    dict
        The manifest, JSON-serialisable.
    """
    from . import metadata

    # The same facts the masthead reads, so the two cannot disagree; the report passes one build in.
    f = metadata.mission_facts(ds) if facts is None else facts
    missing = [n for n in og1_attrs.MANDATORY_GLOBALS if not str(ds.attrs.get(n) or "").strip()]
    summary = og1_attrs.conformance_summary(ds.attrs)

    def _r(x: float | None, nd: int) -> float | None:
        return round(x, nd) if x is not None else None

    return {
        "manifest_version": 1,
        "id": mission_id,
        "platform": str(ds.attrs.get("platform", "")) or None,
        "platform_serial": f["platform_serial"],
        "start": metadata.iso_minute(f["t0"]),
        "end": metadata.iso_minute(f["t1"]),
        "duration_s": f["duration_s"],
        "n_profiles": f["n_profiles"],
        "n_dive": f["n_dive"],
        "n_climb": f["n_climb"],
        "lat_min": _r(f["lat_min"], 4),
        "lat_max": _r(f["lat_max"], 4),
        "lon_min": _r(f["lon_min"], 4),
        "lon_max": _r(f["lon_max"], 4),
        "max_depth_m": _r(f["max_depth_m"], 1),
        "n_records": f["n_records"],
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
