"""Metadata and OG1-conformance sections as data for their templates.

:func:`metadata_data` returns the landing page's Metadata section — the one-line OG1 conformance
verdict and the payload-presence table. :func:`conformance_data` returns the inventory page's Global
attributes section — the categorised attribute tables (value *and* conformance status in one table,
via :func:`glidertest.og1_attrs.group_globals`) and the geospatial-extent comparison. Amber marks a
missing *mandatory* attribute only (values are not format-checked; see :mod:`glidertest.og1_attrs`).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import og1_attrs, tools
from . import inventory

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)


def fmt_duration(seconds: int | None) -> str:
    """Return a ``Nd Nh`` duration from *seconds*, or ``UNK`` when None."""
    if seconds is None:
        return "UNK"
    days, rem = divmod(int(seconds), 86_400)
    return f"{days}d {rem // 3600}h"


def iso_minute(value: object | None) -> str | None:
    """Return minute-resolution ISO ``YYYY-MM-DDTHH:MM`` for a ``datetime64``, or None for NaT/None.

    Canonical ISO (``T`` separator) for the manifest; the masthead replaces ``T`` with a space.
    """
    if value is None or np.isnat(value):
        return None
    return str(np.asarray(value).astype("datetime64[m]"))


def mission_facts(ds: xr.Dataset) -> dict[str, Any]:
    """Return the shared mission facts read by both the masthead and the manifest.

    Raw values (unformatted) so :func:`glidertest.reports._mission.header_card` and
    :func:`glidertest.reports.manifest.mission_manifest` cannot disagree. Profile counts are None
    when ``PROFILE_NUMBER`` is all-NaN (a warning is logged naming the file); the callers then show
    ``—`` / ``null`` rather than a substituted ``0``.

    Returns
    -------
    dict
        ``platform_serial``, ``n_profiles``/``n_dive``/``n_climb`` (int or None), ``t0``/``t1``
        (datetime64 or None), ``duration_s``, ``lat_min``/``lat_max``/``lon_min``/``lon_max``,
        ``depth_min``/``depth_max``/``max_depth_m``, and ``n_records``.
    """
    f: dict[str, Any] = dict.fromkeys(
        (
            "platform_serial", "n_profiles", "n_dive", "n_climb", "t0", "t1", "duration_s",
            "lat_min", "lat_max", "lon_min", "lon_max", "depth_min", "depth_max", "max_depth_m",
        )
    )
    f["n_records"] = int(ds.sizes.get("N_MEASUREMENTS", 0))

    if "PLATFORM_SERIAL_NUMBER" in ds:
        sv = np.atleast_1d(ds["PLATFORM_SERIAL_NUMBER"].values).ravel()
        if sv.size:
            first = sv[0]
            if not (isinstance(first, (float, np.floating)) and not np.isfinite(first)):
                f["platform_serial"] = str(first)

    if "PROFILE_NUMBER" in ds:
        pn = np.asarray(ds["PROFILE_NUMBER"].values)
        finite = np.isfinite(pn)
        if not finite.any():
            logger.warning(
                "PROFILE_NUMBER is all-NaN in %s; profile counts unavailable",
                ds.encoding.get("source", "<in-memory dataset>"),
            )
        else:
            f["n_profiles"] = int(np.unique(pn[finite]).size)
            if "PROFILE_DIRECTION" in ds:
                pdir = np.asarray(ds["PROFILE_DIRECTION"].values)
                m = finite & np.isfinite(pdir)
                _uniq, idx = np.unique(pn[m], return_index=True)
                d = pdir[m][idx]
                f["n_dive"], f["n_climb"] = int((d == -1).sum()), int((d == 1).sum())

    if "TIME" in ds:
        t0, t1 = ds["TIME"].min().values, ds["TIME"].max().values
        if not (np.isnat(t0) or np.isnat(t1)):
            f["t0"], f["t1"] = t0, t1
            f["duration_s"] = int((t1 - t0) / np.timedelta64(1, "s"))

    if "LATITUDE" in ds and "LONGITUDE" in ds:
        la = np.asarray(ds["LATITUDE"].values)
        lo = np.asarray(ds["LONGITUDE"].values)
        la, lo = la[np.isfinite(la)], lo[np.isfinite(lo)]
        if la.size and lo.size:
            f.update(
                lat_min=float(la.min()), lat_max=float(la.max()),
                lon_min=float(lo.min()), lon_max=float(lo.max()),
            )

    if "DEPTH" in ds and "PROFILE_NUMBER" in ds:
        try:
            md = tools.max_depth_per_profile(ds)
        except ValueError:  # all-NaN PROFILE_NUMBER -> groupby cannot form groups
            md = None
        if md is not None:
            lo_d, hi_d = float(md.min()), float(md.max())
            if np.isfinite(lo_d) and np.isfinite(hi_d):
                f["depth_min"], f["depth_max"], f["max_depth_m"] = lo_d, hi_d, hi_d
    return f

#: Payload sensors and the OG1 variable whose presence marks the sensor (from create_docfile).
_PAYLOAD = (
    ("Temperature", "TEMP"),
    ("Salinity", "PSAL"),
    ("Oxygen", "DOXY"),
    ("Chlorophyll", "CHLA"),
    ("Backscatter", "BBP700"),
    ("Altimeter", "ALTITUDE"),
    ("ADCP", "PRES_ADCP"),
)

#: Suggested OG1 geospatial-extent attributes and the (variable, reduction) that computes each.
_GEOSPATIAL = (
    ("geospatial_lat_min", "LATITUDE", "min"),
    ("geospatial_lat_max", "LATITUDE", "max"),
    ("geospatial_lon_min", "LONGITUDE", "min"),
    ("geospatial_lon_max", "LONGITUDE", "max"),
)


def _summary(ds: xr.Dataset) -> str:
    """Return the one-line OG1 conformance verdict (mandatory present, highly-desirable missing)."""
    c = og1_attrs.conformance_summary(ds.attrs)
    line = f"OG1: {c['mandatory_present']} of {c['mandatory_total']} mandatory global attributes present"
    if c["highly_desirable_missing"]:
        line += f" · {c['highly_desirable_missing']} highly-desirable missing"
    return line


def metadata_data(ds: xr.Dataset) -> dict[str, Any]:
    """Return the landing page's Metadata section as data for ``_metadata.html``.

    Parameters
    ----------
    ds : xarray.Dataset
        An OG1 glider dataset.

    Returns
    -------
    dict
        ``summary`` (the one-line OG1 verdict, linking the reader to the inventory for detail) and
        ``payload`` (``{label, var, present, source}`` per expected sensor). The full conformance
        table lives on the inventory page (see :func:`conformance_data`).
    """
    payload = []
    for label, var in _PAYLOAD:
        present = var in ds.variables
        source = str(ds[var].attrs.get("sensor", "")) if present else ""
        # Surface the sensor model and its attrs dropdown from the SENSOR_* catalog entry, the same
        # as the inventory sensor catalog, so the payload table answers "which instrument" on its own.
        meta = inventory._sensor_meta(ds, source) if source and source in ds.variables else None
        payload.append(
            {
                "label": label,
                "var": var,
                "present": present,
                "source": source,
                "model": meta["model"] if meta else "",
                "attrs": meta["attrs"] if meta else {},
            }
        )
    return {"summary": _summary(ds), "payload": payload}


def conformance_data(ds: xr.Dataset) -> dict[str, Any]:
    """Return the inventory page's Global-attributes section as data for ``_og1_conformance.html``.

    Parameters
    ----------
    ds : xarray.Dataset
        An OG1 glider dataset.

    Returns
    -------
    dict
        ``summary`` (the one-line verdict), ``groups`` (categorised attribute tables from
        :func:`glidertest.og1_attrs.group_globals` — each row carries ``name``/``value``/``tier``/
        ``present`` so the table shows value and conformance together), and ``geospatial``
        (``{attr, file_val, computed, missing}`` comparing the file's suggested geospatial bounds
        against the extent computed from the data).
    """
    geospatial = []
    for attr, var, op in _GEOSPATIAL:
        raw = ds.attrs.get(attr)
        file_val = "" if raw is None else str(raw)
        val = float(getattr(ds[var], op)()) if var in ds else float("nan")
        geospatial.append(
            {
                "attr": attr,
                "file_val": file_val,
                "computed": f"{val:.4f}" if np.isfinite(val) else "—",
                "missing": not file_val.strip(),
            }
        )
    return {
        "summary": _summary(ds),
        "groups": og1_attrs.group_globals(ds.attrs),
        "geospatial": geospatial,
    }
