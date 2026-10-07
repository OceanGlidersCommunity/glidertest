"""Metadata and OG1-conformance sections as data for their templates.

:func:`metadata_data` returns the landing page's Metadata section — the one-line OG1 conformance
verdict and the payload-presence table. :func:`conformance_data` returns the inventory page's Global
attributes section — the categorised attribute tables (value *and* conformance status in one table,
via :func:`glidertest.og1_attrs.group_globals`) and the geospatial-extent comparison. Amber marks a
missing *mandatory* attribute only (values are not format-checked; see :mod:`glidertest.og1_attrs`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .. import og1_attrs
from . import inventory

if TYPE_CHECKING:
    import xarray as xr

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
