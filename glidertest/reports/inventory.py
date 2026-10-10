"""File-contents inventory as data for ``templates/_inventory.html``.

:func:`inventory_data` returns the dataset's variables grouped by dimension signature and the
``SENSOR_*`` catalog — plain dicts the template renders. No HTML is built here (the template owns
markup and escaping). The global attributes live on the inventory page's attribute-category sections
(:func:`glidertest.reports.metadata.attr_category_data`), not here.
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


def _fmt_dt64(value: object) -> str:
    """Return ``YYYY-MM-DD HH:MM`` for a datetime64 value, or ``—`` for NaT."""
    return "—" if np.isnat(value) else str(np.asarray(value).astype("datetime64[m]")).replace("T", " ")


def _decode_cf_time(item: float, units: str, calendar: str = "standard") -> str | None:
    """Return a date string for a numeric CF time (``"... since ..."`` units), else None.

    Decoded with :func:`xarray.coding.times.decode_cf_datetime` so the stored epoch and calendar are
    honoured. Without CF units there is nothing to decode, so None is returned rather than guessing a
    date from the raw number — a guess could be off by orders of magnitude.
    """
    if not (isinstance(units, str) and "since" in units and np.isfinite(item)):
        return None
    from xarray.coding.times import decode_cf_datetime

    try:
        dt = decode_cf_datetime(np.asarray(item), units, calendar)
    except (ValueError, KeyError):  # malformed units/calendar
        return None
    return str(np.asarray(dt).astype("datetime64[m]")).replace("T", " ")


def _scalar_value(v: xr.DataArray) -> str:
    """Format a 0-d variable's single value, decoding CF time through its stored units (never guessed).

    A numeric scalar that merely *looks* like a time (``standard_name == "time"``) but has no units to
    decode it is shown as the raw number annotated with that attribute, not a date.
    """
    if v.dtype.kind == "M":
        return _fmt_dt64(v.values)
    item = v.values.item()
    if isinstance(item, (int, float)) and np.isfinite(item):
        decoded = _decode_cf_time(item, v.attrs.get("units", ""), v.attrs.get("calendar", "standard"))
        if decoded is not None:
            return decoded
        if v.attrs.get("standard_name") == "time":
            return f"{_fmt_scalar(item)} (standard_name time; no units)"
    return _fmt_scalar(item)


def _var_meta(ds: xr.Dataset, name: str) -> dict[str, Any]:
    """Return the inventory row for one variable or coordinate.

    The dimension is omitted — every variable in a table shares the dimension named in the group
    heading. ``n_cell`` is ``N`` when every point is finite, or ``N (valid)`` when some are not;
    ``rng`` is a single ``min / max`` string (``—`` when there is no numeric range). ``attrs`` holds
    the attributes not already shown as their own columns, for the dropdown.
    """
    v = ds[name]
    n = int(np.prod(v.shape)) if v.shape else 1
    is_time = v.attrs.get("standard_name") == "time" or v.dtype.kind == "M"
    rng = "—"
    n_valid = n
    if v.dtype.kind == "M" and n:  # datetime64: show the date range
        vals = np.asarray(v.values)
        finite = ~np.isnat(vals)
        n_valid = int(finite.sum())
        if n_valid:
            rng = f"{_fmt_dt64(vals[finite].min())} / {_fmt_dt64(vals[finite].max())}"
    elif v.dtype.kind in "fiu" and n:
        vals = np.asarray(v.values)
        finite = np.isfinite(vals)
        n_valid = int(finite.sum())
        if n_valid:
            lo, hi = vals[finite].min().item(), vals[finite].max().item()
            if is_time:
                # Numeric time: decode the range with the stored units, or "—" when there are none —
                # a raw epoch min/max (1.69e+09) is meaningless to show.
                units, cal = v.attrs.get("units", ""), v.attrs.get("calendar", "standard")
                dlo, dhi = _decode_cf_time(lo, units, cal), _decode_cf_time(hi, units, cal)
                rng = f"{dlo} / {dhi}" if dlo and dhi else "—"
            else:
                rng = f"{_fmt_scalar(lo)} / {_fmt_scalar(hi)}"
    # A scalar variable (0-d) has one value, not a range: show the value, drop min/max and N.
    value = _scalar_value(v) if v.ndim == 0 else None
    return {
        "name": name,
        "has_qc": f"{name}_QC" in ds.variables,
        "dtype": str(v.dtype),
        "n_cell": f"{n:,}" if n_valid == n else f"{n:,} ({n_valid:,})",
        "rng": rng,
        "value": value,
        "units": v.attrs.get("units", ""),
        "long_name": v.attrs.get("long_name", ""),
        "standard_name": v.attrs.get("standard_name", ""),
        "attrs": {
            str(k): str(val)
            for k, val in v.attrs.items()
            if k not in ("units", "long_name", "standard_name")
        },
    }


def _sensor_field(attrs: dict[str, Any], og1_name: str, legacy_name: str) -> tuple[str, bool]:
    """Return ``(value, legacy)`` for a sensor attribute, preferring the OG1 name.

    OG1 names a sensor's serial and calibration ``sensor_serial_number`` /
    ``sensor_calibration_date``; older files use ``serial_number`` / ``calibration_date``. The value
    is read from the OG1 name when present, else the legacy name; *legacy* is ``True`` when it came
    from the old name only, so the report can flag it (amber) as non-OG1 attribute naming.
    """
    if str(attrs.get(og1_name, "")):
        return str(attrs[og1_name]), False
    if str(attrs.get(legacy_name, "")):
        return str(attrs[legacy_name]), True
    return "", False


def _sensor_meta(ds: xr.Dataset, name: str) -> dict[str, Any]:
    """Return the sensor-catalog row for one ``SENSOR_*`` variable (model, serial, calibration, attrs)."""
    a = ds[name].attrs
    serial, serial_legacy = _sensor_field(a, "sensor_serial_number", "serial_number")
    calibration, calibration_legacy = _sensor_field(a, "sensor_calibration_date", "calibration_date")
    shown = {
        "sensor_model",
        "sensor_serial_number",
        "serial_number",
        "sensor_calibration_date",
        "calibration_date",
    }
    return {
        "name": name,
        "model": str(a.get("sensor_model", "")),
        "serial": serial,
        "serial_legacy": serial_legacy,
        "calibration": calibration,
        "calibration_legacy": calibration_legacy,
        "attrs": {str(k): str(val) for k, val in a.items() if k not in shown},
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

    # Each group carries a `header` (the h3 subheader) and a `label` (a sub-caption within it). The
    # coordinates and the variables on the same dimension share one header ("On N_MEASUREMENTS") so
    # the template renders them under a single subheader, with "Coordinates"/"Variables" sub-labels.
    coords = sorted(ds.coords)
    coord_dims = {tuple(str(d) for d in ds[n].dims) for n in coords}
    coord_header = "Coordinates"
    if len(coord_dims) == 1 and (only := next(iter(coord_dims))):
        coord_header = f"On {', '.join(only)}"
    groups = [{"header": coord_header, "label": "Coordinates", "variables": rows(coords)}]
    measurement = by_dims.pop(("N_MEASUREMENTS",), None)
    if measurement:
        groups.append({"header": "On N_MEASUREMENTS", "label": "Variables", "variables": rows(measurement)})
    scalars = by_dims.pop((), None)
    for dims in sorted(by_dims, key=lambda d: (len(d), d)):
        groups.append({"header": f"On {', '.join(dims)}", "label": "Variables", "variables": rows(by_dims[dims])})
    if scalars:
        groups.append(
            {"header": "Scalar variables", "label": None, "variables": rows(scalars), "scalar": True}
        )

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


def inventory_slice(ds: xr.Dataset, which: str, *, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one inventory subsection's data for ``_inventory.html``.

    Filters the groups :func:`inventory_data` already builds, so the grouping rule lives in one
    place. *which* is one of ``"coords"`` (the coordinate group), ``"measurements"`` (variables on
    ``N_MEASUREMENTS``), ``"other_dims"`` (variables on any other dimension), ``"scalars"`` (the
    scalar group), or ``"sensors"`` (the ``SENSOR_*`` catalog). Returns ``{"groups", "sensors"}``.

    *data* is a prebuilt :func:`inventory_data` result to slice; when None it is built from *ds*.
    The report passes one build (cached on :class:`glidertest.reports._mission.Ctx`) to every
    subsection rather than rebuilding it per slice.
    """
    if data is None:
        data = inventory_data(ds)
    groups = data["groups"]
    if which == "coords":
        sel = [g for g in groups if g.get("label") == "Coordinates"]
    elif which == "measurements":
        sel = [g for g in groups if g.get("label") == "Variables" and g.get("header") == "On N_MEASUREMENTS"]
    elif which == "other_dims":
        sel = [g for g in groups if g.get("label") == "Variables" and g.get("header") != "On N_MEASUREMENTS"]
    elif which == "scalars":
        sel = [g for g in groups if g.get("scalar")]
    elif which == "sensors":
        return {"groups": [], "sensors": data["sensors"]}
    else:
        msg = f"unknown inventory slice {which!r}"
        raise ValueError(msg)
    return {"groups": sel, "sensors": []}
