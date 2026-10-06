"""Metadata section as data for ``templates/_metadata.html``.

:func:`metadata_data` returns the OG1 mandatory-attribute presence table, the payload-presence
table, and the geospatial-extent comparison — plain dicts the template renders. Amber marks a
missing mandatory attribute only (values are not format-checked; see :mod:`glidertest.og1_attrs`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .. import og1_attrs

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


def metadata_data(ds: xr.Dataset) -> dict[str, Any]:
    """Return the Metadata section as data for ``_metadata.html``.

    Parameters
    ----------
    ds : xarray.Dataset
        An OG1 glider dataset.

    Returns
    -------
    dict
        ``summary`` (the "N of 16 present" line), ``conformance`` (``{attr, value, missing}`` per
        mandatory attribute), ``payload`` (``{label, var, present, source}``), and ``geospatial``
        (``{attr, file_val, computed, missing}``).
    """
    rows = og1_attrs.check_globals(ds)
    present = sum(1 for _, status, _ in rows if status != "none")
    missing = len(rows) - present
    summary = f"{present} of {len(rows)} mandatory global attributes present"
    if missing:
        summary += f"; {missing} missing"

    conformance = [
        {"attr": attr, "value": value, "missing": status != "match"}
        for attr, status, value in rows
    ]
    payload = [
        {
            "label": label,
            "var": var,
            "present": var in ds.variables,
            "source": str(ds[var].attrs.get("sensor", "")) if var in ds.variables else "",
        }
        for label, var in _PAYLOAD
    ]
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
        "summary": summary,
        "conformance": conformance,
        "payload": payload,
        "geospatial": geospatial,
    }
