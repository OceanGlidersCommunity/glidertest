"""Sensor-row data for a sensor page's ``templates/_sensor_row.html``.

A sensor page (CTD, oxygen, optics, …) opens with the ``SENSOR_*`` catalog entries its variables
come from — model, serial, calibration date, and the rest of the sensor attributes. The linkage is
each variable's ``sensor`` attribute, pointing at a ``SENSOR_*`` variable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import inventory

if TYPE_CHECKING:
    import xarray as xr


def sensor_row_data(ds: xr.Dataset, var_names: tuple[str, ...]) -> dict[str, Any]:
    """Return the ``SENSOR_*`` catalog entries behind *var_names*, deduplicated, in first-seen order.

    Parameters
    ----------
    ds : xarray.Dataset
        An OG1 glider dataset.
    var_names : tuple of str
        The page's science variables (e.g. ``("TEMP", "PSAL")`` for CTD). Each variable's ``sensor``
        attribute names the ``SENSOR_*`` catalog entry.

    Returns
    -------
    dict
        ``{"sensors": [<sensor meta>, ...]}`` for ``_sensor_row.html``; empty when none resolve.
    """
    sensors = []
    seen: set[str] = set()
    for var in var_names:
        if var not in ds.variables:
            continue
        name = ds[var].attrs.get("sensor")
        if name and name in ds.variables and name not in seen:
            seen.add(name)
            sensors.append(inventory._sensor_meta(ds, name))
    return {"sensors": sensors}
