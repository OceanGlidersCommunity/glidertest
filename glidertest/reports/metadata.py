"""Metadata and OG1-conformance sections as data for their templates.

:func:`metadata_data` returns the landing page's Metadata section — the one-line OG1 conformance
verdict and the payload-presence table. :func:`attr_category_data` returns one inventory-page
attribute category — its categorised attribute table (value *and* conformance status, via
:func:`glidertest.og1_attrs.group_globals`), and for the Spatiotemporal category the combined
file-vs-computed extent table (:func:`spatiotemporal_rows`). Amber marks a missing *mandatory*
attribute only (values are not format-checked; see :mod:`glidertest.og1_attrs`).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

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
        table lives on the inventory page (see :func:`attr_category_data`).
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


def verdict_line(ds: xr.Dataset) -> str:
    """Return the one-line OG1 conformance verdict (public wrapper over the internal summary)."""
    return _summary(ds)


def attr_category_data(ds: xr.Dataset, title: str, facts: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one OG1 attribute category's data for ``_og1_conformance.html``.

    *title* is an :data:`glidertest.og1_attrs.ATTR_GROUPS` category title (or
    :data:`glidertest.og1_attrs.OTHER_GROUP`). Returns ``group`` — the category's
    ``name``/``value``/``tier``/``present`` rows from :func:`glidertest.og1_attrs.group_globals`, so
    the table shows value and conformance together — or ``None`` when the file has no rows for it
    (only possible for ``OTHER_GROUP``; the four OG1 categories always carry their mandatory and
    highly-desirable rows). For the Spatiotemporal category it also returns ``spatiotemporal`` (the
    combined file-vs-computed table) and ``spatiotemporal_caption``, which the template renders
    instead of the plain attribute table; *facts* (a prebuilt :func:`mission_facts`) is passed
    through to it, computed from *ds* when None.
    """
    group = next((g for g in og1_attrs.group_globals(ds.attrs) if g["title"] == title), None)
    data: dict[str, Any] = {"group": group}
    if title == "Spatiotemporal coverage":
        data["spatiotemporal"] = spatiotemporal_rows(ds, facts)
        data["spatiotemporal_caption"] = _spatiotemporal_caption(ds)
    return data


def has_other_attrs(ds: xr.Dataset) -> bool:
    """Return True when the file carries attributes not in the OG1 registry (the Other category)."""
    return any(g["title"] == og1_attrs.OTHER_GROUP for g in og1_attrs.group_globals(ds.attrs))


#: Amber when ``|file − computed|`` exceeds these in the Spatiotemporal table. Fixed per-axis
#: thresholds (not derived from the file value's precision): 5e-4° of latitude is ~50 m at
#: mid-latitude, 1 m of depth, 60 s of time. The diff is **always shown**; only the amber flag is
#: gated. We may later suppress sub-threshold diffs entirely to hide the tiny numbers — if so, drop
#: the diff string when ``over`` is False; for now they are shown so a reader sees the magnitude.
_DIFF_THRESHOLDS: dict[str, float] = {"lat": 5e-4, "lon": 5e-4, "vertical": 1.0, "time": 60.0}
_THRESHOLD_TEXT: dict[str, str] = {
    "lat": "5e-4° (~50 m)", "lon": "5e-4° (~50 m)", "vertical": "1 m", "time": "60 s",
}
#: One row per OG1 spatiotemporal attribute: (attribute, axis). The axis selects the amber
#: threshold and the formatting (time vs numeric; vertical honours geospatial_vertical_positive).
_SPATIOTEMPORAL: tuple[tuple[str, str], ...] = (
    ("time_coverage_start", "time"),
    ("time_coverage_end", "time"),
    ("geospatial_lat_min", "lat"),
    ("geospatial_lat_max", "lat"),
    ("geospatial_lon_min", "lon"),
    ("geospatial_lon_max", "lon"),
    ("geospatial_vertical_min", "vertical"),
    ("geospatial_vertical_max", "vertical"),
)


def _vertical_extent(ds: xr.Dataset) -> tuple[float | None, float | None]:
    """Return the data's (shallowest, deepest) vertical extent from DEPTH (else PRES), or (None, None).

    The min and max over *all* samples — the actual vertical coverage — not ``mission_facts``'s
    per-profile maximum-depth range (which answers "how deep did each dive go", a different question).
    """
    for var in ("DEPTH", "PRES"):
        if var in ds:
            a = np.asarray(ds[var].values)
            a = a[np.isfinite(a)]
            if a.size:
                return float(a.min()), float(a.max())
    return None, None


def _coerce_float(text: str) -> float | None:
    """Return *text* as a float, or None when it is empty or not numeric."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _fmt_diff(d: float) -> str:
    """Format a numeric diff at the data's effective (float32) precision, signed.

    Glider position/depth data is float32-valued even when stored in a float64 container, so a
    float64 ``file − computed`` shows ~1e-7 subtraction noise. The float32 shortest round-trip
    collapses that while keeping any real difference intact (a 0.01° diff stays ``+0.01``). The amber
    flag is still gated on the raw (float64) difference, so only the *display* is rounded, not the
    judgement.
    """
    v = np.float32(d)
    if v == 0:
        return "0"
    return ("+" if v > 0 else "-") + np.format_float_positional(np.abs(v), unique=True, trim="-")


#: The OG1-prescribed compact spelling of ``time_coverage_start`` / ``_end``: ``YYYYmmddTHHMMSS``.
#: A readable value not matching this is noted (not amber) — a deviation, not a data problem.
_OG1_TIME_RE = re.compile(r"\d{8}T\d{6}$")


def _parse_time(text: str) -> np.datetime64 | None:
    """Return *text* as a ``datetime64``, or None when empty / unparseable.

    Parsed with :func:`pandas.to_datetime`, so the OG1-prescribed compact ``YYYYmmddTHHMMSS`` form
    (e.g. ``20040924T180910``) parses as well as the extended ISO spellings. A timezone-aware value
    (trailing ``Z``) is reduced to its naive instant.
    """
    if not text:
        return None
    try:
        ts = pd.to_datetime(text)
    except (ValueError, TypeError):
        return None
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return np.datetime64(ts)


def _iso_seconds(value: object | None) -> str:
    """Return second-resolution ISO for a ``datetime64``, or '' for NaT/None."""
    if value is None or np.isnat(np.asarray(value)):
        return ""
    return str(np.asarray(value).astype("datetime64[s]"))


def spatiotemporal_rows(ds: xr.Dataset, facts: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return the combined spatiotemporal table: file attribute vs value computed from the data.

    One row per OG1 spatiotemporal attribute (time coverage, lat/lon and vertical extent). Each
    carries the file's attribute value, the value computed from the data at shortest round-trip
    precision (no rounding — glidertest never writes conservative extents), their difference
    ``file − computed`` (always shown when both exist), an ``over`` flag set when ``|diff|`` exceeds
    the fixed per-axis threshold (:data:`_DIFF_THRESHOLDS`), the hover sentence for a flagged cell in
    ``flag``, and a ``note``. A time value is flagged (``diff="unreadable"``) when it cannot be
    parsed, and noted (not flagged) when it is readable but not the OG1-prescribed compact
    ``YYYYmmddTHHMMSS`` form. Vertical honours the file's ``geospatial_vertical_positive`` (compares
    against ``−depth`` when ``"up"``, notes when the convention is undeclared). ``—`` where the file
    attribute or the computed value is absent.

    *facts* is a prebuilt :func:`mission_facts` result; when None it is computed from *ds*.
    """
    f = mission_facts(ds) if facts is None else facts
    vmin, vmax = _vertical_extent(ds)
    computed_vals: dict[str, Any] = {
        "time_coverage_start": f["t0"], "time_coverage_end": f["t1"],
        "geospatial_lat_min": f["lat_min"], "geospatial_lat_max": f["lat_max"],
        "geospatial_lon_min": f["lon_min"], "geospatial_lon_max": f["lon_max"],
        "geospatial_vertical_min": vmin, "geospatial_vertical_max": vmax,
    }
    vert_up = str(ds.attrs.get("geospatial_vertical_positive", "")).lower() == "up"
    vert_declared = "geospatial_vertical_positive" in ds.attrs
    rows: list[dict[str, Any]] = []
    for attr, axis in _SPATIOTEMPORAL:
        raw = ds.attrs.get(attr)
        file_val = "" if raw is None else str(raw)
        comp = computed_vals[attr]
        note = ""
        diff, over = "", False
        flag = f"flagged: |diff| exceeds {_THRESHOLD_TEXT[axis]}"
        if axis == "time":
            computed = _iso_seconds(comp)
            fv = _parse_time(file_val)
            if file_val and fv is None:
                # A declared timestamp we cannot read is itself a finding — flag it, not a dash.
                diff, over = "unreadable", True
                flag = "flagged: unreadable timestamp (OG1 expects YYYYmmddTHHMMSS)"
            else:
                if file_val and not _OG1_TIME_RE.match(file_val):
                    # Readable but not the OG1-prescribed compact form: note the deviation (not
                    # amber — we surface it, we do not format-check the value).
                    note = "(not OG1 YYYYmmddTHHMMSS)"
                if fv is not None and comp is not None and not np.isnat(np.asarray(comp)):
                    secs = float((fv - comp) / np.timedelta64(1, "s"))
                    diff, over = f"{secs:+.0f} s", abs(secs) > _DIFF_THRESHOLDS[axis]
        else:
            cval = comp
            if axis == "vertical" and vert_up and cval is not None:
                # positive="up": the file bounds are heights, so negate the depth extent — and the
                # extremes swap, because the deepest sample (largest depth) is the most-negative
                # height, i.e. the file's *min*. Negate-and-swap, not negate-in-place.
                cval = -vmax if attr.endswith("_min") else -vmin
            elif axis == "vertical" and not vert_declared:
                note = "(sign convention not declared)"
            computed = "" if cval is None or not np.isfinite(cval) else str(float(cval))
            fv = _coerce_float(file_val)
            if fv is not None and np.isfinite(fv) and cval is not None and np.isfinite(cval):
                d = fv - float(cval)
                diff, over = _fmt_diff(d), abs(d) > _DIFF_THRESHOLDS[axis]
        rows.append(
            {
                "attr": attr, "file_val": file_val, "computed": computed, "diff": diff,
                "over": over, "flag": flag, "note": note,
            }
        )
    return rows


def _spatiotemporal_caption(ds: xr.Dataset) -> str:
    """Return the caption under the Spatiotemporal table.

    The Computed column is the min/max over the records *in this file*. When the first profile
    number is not 1 the file is a subset of a longer mission, so the computed ranges need not match
    the full dataset — the caption says so and names the first profile.
    """
    base = "Computed values are the min/max over the records in this file."
    if "PROFILE_NUMBER" in ds:
        pn = np.asarray(ds["PROFILE_NUMBER"].values)
        pn = pn[np.isfinite(pn)]
        if pn.size and int(pn.min()) != 1:
            return (
                f"{base} The first profile is number {int(pn.min())}, so this file is a subset — "
                "these ranges need not match the full mission."
            )
    return base
