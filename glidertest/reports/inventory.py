"""File-contents inventory as data for ``templates/_inventory.html``.

:func:`inventory_data` returns the dataset's variables grouped by dimension signature and the
``SENSOR_*`` catalog — plain dicts the template renders. No HTML is built here (the template owns
markup and escaping). The global attributes live on the inventory page's Global-attributes section
(:func:`glidertest.reports.metadata.conformance_data`), not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import xarray as xr


def _fmt_scalar(x: float | None) -> str:
    """Format a scalar min/max compactly: ``—`` for None/non-finite, 4 significant figures else."""
    if x is None:
        return "—"
    if isinstance(x, (float, np.floating)):
        return f"{x:.4g}" if np.isfinite(x) else "—"
    return str(x)[:40]


def _var_meta(ds: xr.Dataset, name: str) -> dict[str, Any]:
    """Return the inventory row for one variable or coordinate.

    The dimension is omitted — every variable in a table shares the dimension named in the group
    heading. ``n_cell`` is ``N`` when every point is finite, or ``N (valid)`` when some are not;
    ``rng`` is a single ``min / max`` string (``—`` when there is no numeric range). ``attrs`` holds
    the attributes not already shown as their own columns, for the dropdown.
    """
    v = ds[name]
    n = int(np.prod(v.shape)) if v.shape else 1
    v_min = v_max = None
    n_valid = n
    if v.dtype.kind in "fiu" and n:
        vals = np.asarray(v.values)
        finite = np.isfinite(vals)
        n_valid = int(finite.sum())
        if n_valid:
            v_min = vals[finite].min().item()
            v_max = vals[finite].max().item()
    rng = "—" if v_min is None and v_max is None else f"{_fmt_scalar(v_min)} / {_fmt_scalar(v_max)}"
    return {
        "name": name,
        "has_qc": f"{name}_QC" in ds.variables,
        "dtype": str(v.dtype),
        "n_cell": f"{n:,}" if n_valid == n else f"{n:,} ({n_valid:,})",
        "rng": rng,
        "units": v.attrs.get("units", ""),
        "long_name": v.attrs.get("long_name", ""),
        "standard_name": v.attrs.get("standard_name", ""),
        "attrs": {
            str(k): str(val)
            for k, val in v.attrs.items()
            if k not in ("units", "long_name", "standard_name")
        },
    }


def _sensor_meta(ds: xr.Dataset, name: str) -> dict[str, Any]:
    """Return the sensor-catalog row for one ``SENSOR_*`` variable (model, serial, calibration, attrs)."""
    a = ds[name].attrs
    return {
        "name": name,
        "model": str(a.get("sensor_model", "")),
        "serial": str(a.get("serial_number", "")),
        "calibration": str(a.get("calibration_date", "")),
        "attrs": {
            str(k): str(val)
            for k, val in a.items()
            if k not in ("sensor_model", "serial_number", "calibration_date")
        },
    }


def inventory_data(ds: xr.Dataset) -> dict[str, Any]:
    """Return the file-contents inventory as data for ``_inventory.html``.

    Parameters
    ----------
    ds : xarray.Dataset
        An OG1 glider dataset.

    Returns
    -------
    dict
        ``groups`` (list of ``{title, variables}`` grouped by dimension signature), ``sensors``, and
        the counts ``n_vars``/``n_sensors``/``n_qc``/``n_coords``/``n_with_qc`` for the caption and
        the QC-coverage line (``n_with_qc`` of ``n_vars`` data variables carry a ``_QC`` companion).
    """
    qc_vars = sorted(n for n in ds.data_vars if n.endswith("_QC"))
    sensors = sorted(n for n in ds.data_vars if n.startswith("SENSOR_"))
    science = sorted(
        n for n in ds.data_vars if not n.startswith("SENSOR_") and not n.endswith("_QC")
    )
    by_dims: dict[tuple[str, ...], list[str]] = {}
    for n in science:
        by_dims.setdefault(tuple(str(d) for d in ds[n].dims), []).append(n)

    def rows(names: list[str]) -> list[dict[str, Any]]:
        return [_var_meta(ds, n) for n in names]

    coords = sorted(ds.coords)
    coord_dims = {tuple(str(d) for d in ds[n].dims) for n in coords}
    coord_title = "Coordinates"
    if len(coord_dims) == 1 and (only := next(iter(coord_dims))):
        coord_title = f"Coordinates on {', '.join(only)}"
    groups = [{"title": coord_title, "variables": rows(coords)}]
    measurement = by_dims.pop(("N_MEASUREMENTS",), None)
    if measurement:
        groups.append({"title": "Variables on N_MEASUREMENTS", "variables": rows(measurement)})
    scalars = by_dims.pop((), None)
    for dims in sorted(by_dims, key=lambda d: (len(d), d)):
        groups.append({"title": f"Variables on {', '.join(dims)}", "variables": rows(by_dims[dims])})
    if scalars:
        groups.append({"title": "Scalar variables", "variables": rows(scalars)})

    n_with_qc = sum(1 for n in science if f"{n}_QC" in ds.variables)
    return {
        "groups": groups,
        "sensors": [_sensor_meta(ds, n) for n in sensors],
        "n_vars": len(science),
        "n_sensors": len(sensors),
        "n_qc": len(qc_vars),
        "n_coords": len(ds.coords),
        "n_with_qc": n_with_qc,
    }
