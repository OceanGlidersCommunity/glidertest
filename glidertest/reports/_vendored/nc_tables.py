"""Vendored netCDF inventory tables — pure builders over an open ``xr.Dataset``.

Each builder takes an **open** dataset (the caller opens and closes; never a path) and
returns ``(rows, source)``, where *source* is the ``"<module>.py:<function>"`` caption
string from :func:`figure_png.source_for` — so a report can show which function built a
table under the same toggle the per-figure caption uses.

Package-neutral by construction: this module names no package and knows nothing about
casts, dives or moorings.  It is vendored byte-identical across ctdcast, oceanarray and
glidertest (ctdcast ``main`` the source of truth; see ``MANIFEST.txt``), which the
import-purity test guards — it may import only the standard library, numpy, xarray and its
vendored siblings.

The per-package rules live in each package's owned ``reports/design.py`` and are passed in:
``is_qc(name, ds)`` (the QC-companion rule), ``is_sensor(name)``, the ``extras`` that add a
package's own globals to the canonical :data:`ATTR_GROUPS`, and (for sensors) a ``read_record``
reader.  Nothing package-specific is hard-coded here.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from .design_tokens import QC_FLAG_COLORS, QC_FLAG_DEFAULT_COLOR, QC_FLAG_LABELS
from .figure_png import source_for

if TYPE_CHECKING:
    import xarray as xr

#: Attribute names shown as their own table columns, so omitted from the per-variable
#: attribute disclosure list.
_COLUMNED = frozenset({"units", "standard_name", "long_name"})

#: Canonical global-attribute groups, holding **only** names a named convention defines
#: (ACDD 1.3, CF, and the mandatory globals OceanSITES / OG1 / CCHDO share) — each annotated
#: with its convention so the file is checkable against the published lists and never carries
#: an invented name.  A package adds its own vocabulary to these same titles through the
#: ``extras`` argument of :func:`globals_rows` (from its owned ``reports/design.py``); a
#: package may add a group but not rename one.  Whatever is named by neither falls into
#: :data:`OTHER_GROUP`.
ATTR_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Identity & discovery",
        (
            "title",  # ACDD
            "summary",  # ACDD
            "id",  # ACDD
            "naming_authority",  # ACDD
            "project",  # ACDD
            "program",  # ACDD
            "platform",  # OG1
            "platform_vocabulary",  # OG1
            "platform_code",  # OceanSITES
            "site_code",  # OceanSITES
            "references",  # CF / ACDD
            "standard_name_vocabulary",  # ACDD
        ),
    ),
    (
        "Spatiotemporal coverage",
        (
            "geospatial_lat_min",  # ACDD
            "geospatial_lat_max",  # ACDD
            "geospatial_lat_units",  # ACDD
            "geospatial_lon_min",  # ACDD
            "geospatial_lon_max",  # ACDD
            "geospatial_lon_units",  # ACDD
            "geospatial_vertical_min",  # ACDD
            "geospatial_vertical_max",  # ACDD
            "geospatial_vertical_units",  # ACDD
            "geospatial_vertical_positive",  # ACDD
            "time_coverage_start",  # ACDD
            "time_coverage_end",  # ACDD
            "time_coverage_duration",  # ACDD
            "time_coverage_resolution",  # ACDD
        ),
    ),
    (
        "People & institutions",
        (
            "creator_name",  # ACDD
            "creator_email",  # ACDD
            "creator_type",  # ACDD
            "creator_url",  # ACDD
            "creator_id",  # ACDD
            "contributor_name",  # ACDD
            "contributor_role",  # ACDD
            "contributing_institutions",  # OG1
            "institution",  # CF / ACDD
            "publisher_name",  # ACDD
            "publisher_email",  # ACDD
            "publisher_url",  # ACDD
        ),
    ),
    (
        "Rights & access",
        (
            "acknowledgement",  # ACDD
            "license",  # ACDD
            "citation",  # ACDD
        ),
    ),
    (
        "Provenance & processing",
        (
            "date_created",  # ACDD
            "date_modified",  # ACDD
            "source",  # CF / ACDD
            "history",  # CF / ACDD
            "comment",  # CF / ACDD
            "processing_level",  # ACDD / OceanSITES
            "data_mode",  # OceanSITES
            "QC_indicator",  # OceanSITES
            "Conventions",  # CF / ACDD
            "featureType",  # CF
            "cdm_data_type",  # ACDD
            "tracking_id",  # OceanSITES / ACDD
        ),
    ),
)

#: Title for attributes named by no group (never dropped).
OTHER_GROUP = "Other"


def _readable_dtype(dtype: np.dtype) -> str:
    """Return a readable dtype label: the precision for numbers, a word for the rest."""
    kind = dtype.kind
    if kind in "US":
        return "string"
    if kind == "O":
        return "object"
    if kind == "M":
        return "datetime"
    if kind == "m":
        return "timedelta"
    return str(dtype)  # float64 / float32 / int32 — keep the precision visible


def _fmt_minmax(v: float) -> str:
    """Format a numeric min/max compactly: ``0``, scientific for extremes, else 4 s.f."""
    if v == 0.0:
        return "0"
    if abs(v) < 1e-3 or abs(v) >= 1e5:
        return f"{v:.3g}"
    return f"{v:.4g}"


def _attr_type_name(value: Any) -> str:  # noqa: ANN401  # any global-attribute value
    """Return a readable type name for a global-attribute value.

    Read straight off the stored value's dtype through :func:`_readable_dtype`, the one
    type vocabulary shared with the variable tables — so text reads ``string`` (never
    ``str`` or a raw numpy dtype), the signal a reviewer uses to catch a position written
    ``"65 43.913"`` where a decimal degree was meant, or a count stored as a string.
    """
    arr = np.asarray(value)
    if arr.ndim:
        return f"{_readable_dtype(arr.dtype)}[{arr.size}]"
    return _readable_dtype(arr.dtype)


def _fmt_scalar(x: Any) -> str:  # noqa: ANN401  # any 0-d value the caller decided to show
    """Format a scalar value for the Value column, truncating long strings."""
    if isinstance(x, bytes):
        # netCDF char attributes come back as bytes; decode so the cell shows text, not b'…'.
        return x.decode("utf-8", "replace").strip()[:120] or "—"
    if isinstance(x, (float, np.floating)):
        return "—" if not np.isfinite(x) else _fmt_minmax(float(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return str(x)[:120]


def fmt_datetime(value: Any) -> str:  # noqa: ANN401  # a datetime64 / ISO string / epoch value
    """Return an ISO-8601 datetime string to the second, with milliseconds only when non-zero.

    Replaces numpy's nanosecond ``repr`` (``2020-01-19T20:30:00.000000000``) and avoids silently
    dropping a real sub-second part: a whole-second time renders ``2020-01-19 20:30:00``, a
    fractional one keeps milliseconds.  ``T`` is shown as a space.

    Parameters
    ----------
    value : numpy.datetime64 or str
        A datetime value (or an ISO string numpy can parse).

    Returns
    -------
    str
        The formatted datetime.
    """
    dt = np.datetime64(value, "ns")
    if np.isnat(dt):
        return "—"
    whole = dt.astype("datetime64[s]")
    remainder = (dt - whole.astype("datetime64[ns]")) // np.timedelta64(1, "ns")
    out = str(whole) if remainder == 0 else str(dt.astype("datetime64[ms]"))
    return out.replace("T", " ")


def _numeric_minmax(v: xr.DataArray) -> tuple[float | None, float | None, int]:
    """Return ``(min, max, n_valid)`` over the finite values, or ``(None, None, 0)``."""
    vals = np.asarray(v.values).ravel()
    finite = np.isfinite(vals)
    n_valid = int(finite.sum())
    if not n_valid:
        return None, None, 0
    good = vals[finite]
    return float(good.min()), float(good.max()), n_valid


def _datetime_minmax(v: xr.DataArray) -> tuple[str | None, str | None, int]:
    """Return ISO ``(min, max, n_valid)`` for a datetime array, NaT excluded."""
    vals = np.asarray(v.values).ravel()
    good = vals[~np.isnat(vals)]
    if not good.size:
        return None, None, 0
    return fmt_datetime(good.min()), fmt_datetime(good.max()), int(good.size)


def _scalar_value(v: xr.DataArray) -> str:
    """Return a 0-d value as a display string, CF-decoding an encoded time.

    A numeric scalar carrying ``units`` like ``"seconds since ..."`` is decoded through
    xarray so an epoch reads as a date, never a raw count.  Everything else is formatted
    plainly.
    """
    if v.dtype.kind == "M":
        lo, _hi, n = _datetime_minmax(v)
        return lo if n else "—"
    item = v.values.item() if v.size else None
    if item is None:
        return "—"
    units = str(v.attrs.get("units", ""))
    if (
        isinstance(item, (int, float, np.number))
        and np.isfinite(item)
        and "since" in units
    ):
        try:
            import xarray.coding.times as _xtimes

            decoded = _xtimes.decode_cf_datetime(
                np.asarray(item), units, v.attrs.get("calendar", "standard")
            )
            return fmt_datetime(decoded.ravel()[0])
        except Exception:  # noqa: BLE001  # undecodable → show the raw value, never crash
            pass
    return _fmt_scalar(item)


def _attr_disclosure(v: xr.DataArray) -> dict[str, str]:
    """Return the variable's attributes minus the ones shown as columns, stringified."""
    return {
        str(k): _fmt_scalar(val)
        for k, val in v.attrs.items()
        if str(k) not in _COLUMNED
    }


def _original_name(v: xr.DataArray, name: str, attrs: tuple[str, ...]) -> str | None:
    """Return the variable's source name for the ``original → name`` cell, or ``None``.

    The first attribute present from *attrs* (e.g. ``cnv_original_name``, ``source_variable``)
    gives the name the variable had before canonicalisation; ``None`` when none is present or
    the recorded source equals the current name (a pass-through, so no arrow is shown).
    """
    for attr in attrs:
        if attr in v.attrs and str(v.attrs[attr]):
            src = str(v.attrs[attr])
            return src if src != name else None
    return None


def _row_base(
    ds: xr.Dataset, name: str, original_name_attrs: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Return the fields shared by coordinate, variable and scalar rows."""
    v = ds[name]
    return {
        "name": name,
        "dtype": _readable_dtype(v.dtype),
        "units": str(v.attrs.get("units", "")),
        "standard_name": str(v.attrs.get("standard_name", "")),
        "long_name": str(v.attrs.get("long_name", "")),
        # Float is the kind CF expects a unit for; an integer flag/index/channel or a string
        # legitimately has none, so only a float missing its units is a defect (units_cell).
        "is_numeric": v.dtype.kind == "f",
        "original_name": _original_name(v, name, original_name_attrs),
        "attrs": _attr_disclosure(v),
    }


def _value_row(
    ds: xr.Dataset, name: str, original_name_attrs: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Return a row with the full min/max/valid block for a dimensioned variable/coord."""
    v = ds[name]
    row = _row_base(ds, name, original_name_attrs)
    n = int(np.prod(v.shape)) if v.shape else 1
    v_min_raw: float | None = None
    v_min = v_max = "—"
    n_valid = n
    if v.dtype.kind == "M":
        lo, hi, n_valid = _datetime_minmax(v)
        v_min, v_max = (lo or "—"), (hi or "—")
    elif v.dtype.kind in "fiu":
        lo_f, hi_f, n_valid = _numeric_minmax(v)
        if lo_f is not None:
            v_min_raw = lo_f
            v_min, v_max = _fmt_minmax(lo_f), _fmt_minmax(hi_f)  # type: ignore[arg-type]
    row.update(
        n=n,
        n_valid=n_valid,
        valid_pct=round(100.0 * n_valid / n, 1) if n else 0.0,
        v_min=v_min,
        v_max=v_max,
        v_min_raw=v_min_raw,
    )
    return row


def _qc_parent(name: str) -> str:
    """Return a QC companion's parent variable name (a trailing ``_qc`` removed).

    Case-insensitive so it serves both conventions in the family — ``_qc`` (ctdcast) and
    ``_QC`` (OceanSITES / OG1).
    """
    return re.sub(r"_qc$", "", name, flags=re.IGNORECASE)


def coords_rows(
    ds: xr.Dataset, *, original_name_attrs: tuple[str, ...] = ()
) -> tuple[list[dict[str, Any]], str]:
    """Return ``(rows, source)`` for the coordinate variables.

    Parameters
    ----------
    ds : xarray.Dataset
        An open dataset.
    original_name_attrs : tuple of str, optional
        Attribute names read, in order, for each row's ``original_name`` (the source name
        shown as ``original → name``); the first present and non-empty wins.

    Returns
    -------
    rows : list of dict
        One :func:`_value_row` per coordinate (sorted by name), each carrying an extra
        ``length`` field (the coordinate's size).
    source : str
        The caption string for this builder.
    """
    rows = []
    for name in sorted(ds.coords):
        row = _value_row(ds, name, original_name_attrs)
        row["length"] = int(ds[name].size)
        row["dims"] = " × ".join(str(d) for d in ds[name].dims)
        rows.append(row)
    return rows, source_for(coords_rows)


def variables_rows(
    ds: xr.Dataset,
    *,
    is_qc: Callable[[str, xr.Dataset], bool],
    is_sensor: Callable[[str], bool] | None = None,
    include: Callable[[str], bool] | None = None,
    group_by_dims: bool = True,
    original_name_attrs: tuple[str, ...] = (),
) -> tuple[list[tuple[str, list[dict[str, Any]]]], str]:
    """Return ``([(group_title, rows)], source)`` for the dimensioned data variables.

    QC companion variables (``is_qc`` true) are **not** listed; instead their parent row
    carries ``has_qc=True`` so the template can show an inline ``(_QC)`` marker.  Sensor
    variables (``is_sensor`` true) are excluded — the sensor catalog is its own table.

    Parameters
    ----------
    ds : xarray.Dataset
        An open dataset.
    is_qc : callable
        ``is_qc(name, ds) -> bool`` — the package's QC-companion rule.
    is_sensor : callable, optional
        ``is_sensor(name) -> bool`` — excludes sensor-catalog variables.
    include : callable, optional
        ``include(name) -> bool`` — when given, keep **only** names it accepts (a first-class
        positive predicate, so a caller selecting a subset passes this rather than an inverted
        ``is_sensor``).  Composes with ``is_qc``/``is_sensor``, which still exclude.
    group_by_dims : bool, default True
        Group variables by their dimension tuple, one group per tuple; the per-group sample
        count goes in the title.  When ``False`` a single ``""``-titled group holds all rows.
    original_name_attrs : tuple of str, optional
        Attribute names read, in order, for each row's ``original_name``; the first present
        and non-empty wins.

    Returns
    -------
    groups : list of (str, list of dict)
        ``(group_title, rows)`` in order of first appearance; ``group_title`` reads e.g.
        ``"N_PROF × pressure (236 × 2,001 = 472,236 samples)"``.
    source : str
        The caption string for this builder.
    """
    sensor = is_sensor or (lambda _n: False)
    keep = include or (lambda _n: True)
    qc_names = {n for n in ds.data_vars if is_qc(n, ds)}
    # Parent → its QC companion's *actual* name, so the marker shows the file's own suffix
    # (ctdcast ``{var}_qc`` → ``(_qc)``; OG1 ``{var}_QC`` → ``(_QC)``), never a hard-coded spelling.
    qc_by_parent = {_qc_parent(n): n for n in qc_names}

    candidates = [
        n
        for n in sorted(ds.data_vars)
        if ds[n].ndim >= 1 and n not in qc_names and not sensor(n) and keep(n)
    ]

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    order: list[tuple[str, ...]] = []
    for name in candidates:
        dims = tuple(str(d) for d in ds[name].dims) if group_by_dims else ()
        if dims not in groups:
            groups[dims] = []
            order.append(dims)
        row = _value_row(ds, name, original_name_attrs)
        row["has_qc"] = name in qc_by_parent
        row["qc_name"] = qc_by_parent.get(name)
        groups[dims].append(row)

    out = [(_dims_heading(ds, dims), groups[dims]) for dims in order]
    return out, source_for(variables_rows)


def _dims_heading(ds: xr.Dataset, dims: tuple[str, ...]) -> str:
    """Return a group title with the per-group sample count, thousands-separated."""
    if not dims:
        return ""
    sizes = [int(ds.sizes[d]) for d in dims]
    n = int(np.prod(sizes)) if sizes else 0
    label = " × ".join(dims)
    if len(dims) > 1:
        factors = " × ".join(f"{s:,}" for s in sizes)
        return f"{label} ({factors} = {n:,} samples)"
    return f"{label} ({n:,} samples)"


def scalars_rows(
    ds: xr.Dataset,
    *,
    exclude: Callable[[str], bool] | None = None,
    original_name_attrs: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], str]:
    """Return ``(rows, source)`` for the 0-d (scalar) variables.

    Parameters
    ----------
    ds : xarray.Dataset
        An open dataset.
    exclude : callable, optional
        ``exclude(name) -> bool`` — drop a scalar from the table (ctdcast excludes
        ``SENSOR_*``, which has its own catalog).
    original_name_attrs : tuple of str, optional
        Attribute names read, in order, for each row's ``original_name``; the first present
        and non-empty wins.

    Returns
    -------
    rows : list of dict
        One row per scalar (sorted by name), each a :func:`_row_base` plus a ``value``
        field (CF-decoded when it is an encoded time).
    source : str
        The caption string for this builder.
    """
    drop = exclude or (lambda _n: False)
    rows = []
    for name in sorted(ds.data_vars):
        if ds[name].ndim != 0 or drop(name):
            continue
        row = _row_base(ds, name, original_name_attrs)
        row["value"] = _scalar_value(ds[name])
        rows.append(row)
    return rows, source_for(scalars_rows)


#: QC threshold tests read off each ``{var}_qc`` companion, in display order.  Each entry is
#: ``(label, suspect_attrs, fail_attrs, kind)``: ``kind="range"`` reads a ``(min, max)`` attr
#: pair and renders ``[lo, hi]``; ``kind="threshold"`` reads one attr and renders ``|Δ| > x``.
#: A package extends this tuple in its ``design.py`` for a test it writes; a ``qc_*`` attribute
#: matching no entry is still shown (an "other" row), never dropped.
QC_TESTS: tuple[tuple[str, tuple[str, ...], tuple[str, ...], str], ...] = (
    (
        "gross-range",
        ("qc_gross_range_suspect_min", "qc_gross_range_suspect_max"),
        ("qc_gross_range_fail_min", "qc_gross_range_fail_max"),
        "range",
    ),
    (
        "spike",
        ("qc_spike_suspect_threshold",),
        ("qc_spike_fail_threshold",),
        "threshold",
    ),
)


def _threshold_tier(
    attrs: dict[str, Any], keys: tuple[str, ...], kind: str
) -> str | None:
    """Render one tier (suspect or fail) of a QC test, or ``None`` when it has no value."""
    values = [attrs.get(k) for k in keys]
    if all(v is None for v in values):
        return None
    if kind == "range":
        lo, hi = values
        return f"[{lo}, {hi}]" if lo is not None and hi is not None else None
    return f"|Δ| > {values[0]}" if values[0] is not None else None


def qc_threshold_rows(
    ds: xr.Dataset,
    *,
    is_qc: Callable[[str, xr.Dataset], bool],
    tests: tuple[tuple[str, tuple[str, ...], tuple[str, ...], str], ...] = QC_TESTS,
) -> tuple[list[dict[str, Any]], str]:
    """Return ``(rows, source)`` for the QC thresholds recorded on each ``{var}_qc``.

    One row per (variable, test) from the companion's ``qc_*`` attributes — the values the
    pipeline actually applied (defaults, config, or per-cast overrides), read off the file
    rather than parsed from ``history``.  A ``qc_*`` attribute matched by no entry in *tests*
    is listed as an "other" row with its name and value, never dropped.

    Parameters
    ----------
    ds : xarray.Dataset
        An open dataset carrying the ``{var}_qc`` companions.
    is_qc : callable
        ``is_qc(name, ds) -> bool`` — the package's QC-companion rule.
    tests : tuple, optional
        The :data:`QC_TESTS` entries; a package passes a widened tuple for a test it writes.

    Returns
    -------
    rows : list of dict
        ``{var, test, suspect, fail}`` rows (a tier with no value renders as an en-dash).
    source : str
        The caption string for this builder.
    """
    consumed = {k for _label, suspect, fail, _kind in tests for k in (*suspect, *fail)}
    rows: list[dict[str, Any]] = []
    for name in sorted(ds.data_vars):
        if not is_qc(name, ds):
            continue
        base = _qc_parent(name)
        if base not in ds.data_vars:
            continue
        a = dict(ds[name].attrs)
        for label, suspect, fail, kind in tests:
            s = _threshold_tier(a, suspect, kind)
            f = _threshold_tier(a, fail, kind)
            if s or f:
                rows.append(
                    {"var": base, "test": label, "suspect": s or "–", "fail": f or "–"}
                )
        for key in sorted(a):
            if key.startswith("qc_") and key not in consumed:
                # Never dropped: an unrecognised qc_* attribute is shown under its raw name
                # (not prettified — a reader greps the file for exactly this string).
                rows.append(
                    {
                        "var": base,
                        "test": key,
                        "suspect": str(a[key]),
                        "fail": "–",
                        "kind": "other",
                    }
                )
    return rows, source_for(qc_threshold_rows)


def qc_flag_rows(
    ds: xr.Dataset,
    *,
    is_qc: Callable[[str, xr.Dataset], bool],
    labels: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Return ``(rows, source)`` for the per-variable QC flag-count breakdown.

    One row per variable that has a QC companion, with one entry per flag the companion
    *declares* (its ``flag_values`` / ``flag_meanings``, in declared order — a zero-count
    declared flag still gets a column, so a clean cast reads as a fixed table).  A companion
    that declares no ``flag_values`` falls back to the integers actually present and the row is
    marked ``undeclared`` (the table flags it, since the labels are then just the raw codes).

    Parameters
    ----------
    ds : xarray.Dataset
        An open dataset carrying the ``{var}_qc`` companions.
    is_qc : callable
        ``is_qc(name, ds) -> bool`` — the package's QC-companion rule.
    labels : dict of str to str, optional
        Meaning-token → short label; defaults to the vendored
        :data:`QC_FLAG_LABELS`.  A package passes its
        own wording to override.
    colors : dict of str to str, optional
        Meaning-token → colour; defaults to the vendored
        :data:`QC_FLAG_COLORS`.

    Returns
    -------
    rows : list of dict
        ``{var, total, undeclared, flags: [{flag, token, label, color, n, pct}, …]}`` per
        variable.
    source : str
        The caption string for this builder.
    """
    labels = labels if labels is not None else QC_FLAG_LABELS
    colors = colors if colors is not None else QC_FLAG_COLORS
    rows: list[dict[str, Any]] = []
    for name in sorted(ds.data_vars):
        if not is_qc(name, ds):
            continue
        base = _qc_parent(name)
        if base not in ds.data_vars:
            continue
        flags = np.asarray(ds[name].values).astype(int).ravel()
        total = int(flags.size)
        if total == 0:
            continue
        values = ds[name].attrs.get("flag_values")
        meanings = str(ds[name].attrs.get("flag_meanings", "")).split()
        value_list = np.asarray(values).ravel().tolist() if values is not None else []
        # A well-formed declaration needs flag_values AND a flag_meanings token per value (CF
        # pairs them positionally), and the flag_values must be integer codes.  No flag_values, no
        # flag_meanings, a length mismatch, or a non-integer flag value is a malformed declaration:
        # fall back to the observed integer codes and flag the row amber, rather than silently
        # truncating a float (int(1.5)=1) or pairing some values with a raw-integer "label".
        try:
            declared = (
                bool(value_list)
                and len(meanings) == len(value_list)
                and all(float(v).is_integer() for v in value_list)
            )
        except (TypeError, ValueError, OverflowError):
            # A non-numeric flag_values entry is malformed too — fall back, do not crash.
            declared = False
        if declared:
            vals = [int(v) for v in value_list]
            tokens = list(meanings)
        else:
            vals = sorted({int(v) for v in flags.tolist()})
            tokens = [str(v) for v in vals]
        flag_entries = []
        for value, token in zip(vals, tokens, strict=True):
            n = int(np.sum(flags == value))
            flag_entries.append(
                {
                    "flag": value,
                    "token": token,
                    "label": labels.get(token, token.replace("_", " ")),
                    "color": colors.get(token, QC_FLAG_DEFAULT_COLOR),
                    "n": n,
                    "pct": round(100.0 * n / total, 1),
                }
            )
        rows.append(
            {
                "var": base,
                "total": total,
                "undeclared": not declared,
                "flags": flag_entries,
            }
        )
    return rows, source_for(qc_flag_rows)


def _merge_groups(
    groups: tuple[tuple[str, tuple[str, ...]], ...],
    extras: dict[str, Iterable[str]] | None,
) -> list[tuple[str, tuple[str, ...]]]:
    """Return *groups* with each title's ``extras`` names appended (order preserved)."""
    extra = {k: tuple(v) for k, v in (extras or {}).items()}
    merged = [(title, names + extra.get(title, ())) for title, names in groups]
    # A title that appears only in extras (a package-added group) is kept, trailing.
    seen = {title for title, _ in groups}
    merged += [(t, tuple(v)) for t, v in extra.items() if t not in seen]
    return merged


def _fmt_attr_value(value: Any) -> str:  # noqa: ANN401  # any global-attribute value
    """Format a global-attribute value: a datetime through :func:`fmt_datetime`, bytes decoded as
    UTF-8 (a netCDF char attribute, else a ``b'…'`` repr), else ``str``."""
    if isinstance(value, np.datetime64):
        return fmt_datetime(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace").strip() or "—"
    return str(value)


def globals_rows(
    ds: xr.Dataset,
    *,
    groups: tuple[tuple[str, tuple[str, ...]], ...] = ATTR_GROUPS,
    extras: dict[str, Iterable[str]] | None = None,
    exclude: Iterable[str] = (),
    requirement: Mapping[str, str] | None = None,
) -> tuple[list[tuple[str, list[tuple[str, ...]]]], str]:
    """Return ``([(group_title, [(attr, type, value), ...])], source)`` for the global attributes.

    Groups appear in *groups* order with the package's *extras* merged into each title;
    within a group the attributes keep the file's own order so the page can be checked
    against the spec.  Attributes named by no group fall into a trailing
    :data:`OTHER_GROUP`.  Empty groups are omitted.

    Parameters
    ----------
    ds : xarray.Dataset
        An open dataset.
    groups : tuple, default :data:`ATTR_GROUPS`
        The canonical group definitions.
    extras : dict, optional
        ``{group_title: (attr_name, ...)}`` — a package's own global-attribute names added
        to the canonical groups (from its owned ``reports/design.py``).
    exclude : iterable of str, optional
        Attribute names to omit from the table — for a consumer that renders one in its own
        panel (e.g. ``history``) and would otherwise show it twice.
    requirement : mapping of str to str, optional
        ``{attr_name: requirement_label}`` (e.g. a CF/ACDD "required"/"recommended" level).
        When given, each row gains a fourth element — the attribute's label, or ``""`` when
        the mapping does not name it — and :func:`globals_tables` renders a "Requirement"
        column after Attribute.  When ``None`` (the default) rows stay ``(attr, type, value)``
        triples and no column is added.

    Returns
    -------
    grouped : list of (str, list of tuple)
        ``(group_title, rows)`` with each row an ``(attribute, type, value)`` triple — or an
        ``(attribute, type, value, requirement)`` quad when *requirement* is given; the
        *type* is the stored value's type name (:func:`_attr_type_name`), the *value* a
        datetime formatted through :func:`fmt_datetime`.
    source : str
        The caption string for this builder.
    """
    merged = _merge_groups(groups, extras)
    group_of = {name: title for title, names in merged for name in names}
    dropped = set(exclude)
    buckets: dict[str, list[tuple[str, ...]]] = {title: [] for title, _ in merged}
    buckets[OTHER_GROUP] = []
    for key, value in ds.attrs.items():
        if str(key) in dropped:
            continue
        row: tuple[str, ...] = (
            str(key),
            _attr_type_name(value),
            _fmt_attr_value(value),
        )
        if requirement is not None:
            row = (*row, requirement.get(str(key), ""))
        buckets.setdefault(group_of.get(str(key), OTHER_GROUP), []).append(row)
    titles = [title for title, _ in merged] + [OTHER_GROUP]
    return [(t, buckets[t]) for t in titles if buckets[t]], source_for(globals_rows)


def summary_counts(
    ds: xr.Dataset,
    *,
    is_qc: Callable[[str, xr.Dataset], bool],
    is_sensor: Callable[[str], bool] | None = None,
    exclude: Callable[[str], bool] | None = None,
) -> dict[str, int]:
    """Return the one-line inventory summary counts.

    Parameters
    ----------
    ds : xarray.Dataset
        An open dataset.
    is_qc : callable
        ``is_qc(name, ds) -> bool``.
    is_sensor : callable, optional
        ``is_sensor(name) -> bool`` — the sensor-catalog predicate, counted as ``sensors``.
    exclude : callable, optional
        ``exclude(name) -> bool`` — a package's data variables shown in their own tables, not
        in the main variable table (e.g. ctdcast's per-profile linkage/channel).  Counted as
        ``excluded`` and removed from ``data_variables`` so that count stays honest.

    Returns
    -------
    dict
        ``{"data_variables", "sensors", "qc_companions", "coordinates", "excluded"}``.
    """
    sensor = is_sensor or (lambda _n: False)
    excl = exclude or (lambda _n: False)
    sensors = [n for n in ds.data_vars if sensor(n)]
    qc = [n for n in ds.data_vars if is_qc(n, ds)]
    accounted = set(sensors) | set(qc)
    # Excluded: named by the package's own predicate but not already a sensor or QC companion.
    excluded = [n for n in ds.data_vars if excl(n) and n not in accounted]
    data = [n for n in ds.data_vars if n not in accounted | set(excluded)]
    return {
        "data_variables": len(data),
        "sensors": len(sensors),
        "qc_companions": len(qc),
        "coordinates": len(ds.coords),
        "excluded": len(excluded),
    }


#: Attribute names the default OG1 sensor reader consumes into columns.
_SENSOR_COLUMNED = frozenset(
    {
        "sensor_model",
        "sensor_serial_number",
        "serial_number",
        "sensor_calibration_date",
        "calibration_date",
        "model_source",
        "standard_name",
        "long_name",
    }
)


def _sensor_field(attrs: dict[str, Any], primary: str, legacy: str) -> tuple[str, bool]:
    """Return ``(value, used_legacy)`` preferring *primary*, falling back to *legacy*."""
    if attrs.get(primary) not in (None, ""):
        return str(attrs[primary]), False
    if attrs.get(legacy) not in (None, ""):
        return str(attrs[legacy]), True
    return "", False


def _og1_sensor_record(ds: xr.Dataset, name: str) -> dict[str, Any]:
    """Read one ``SENSOR_*`` variable into a catalog row (OG1 attribute names)."""
    a = dict(ds[name].attrs)
    serial, serial_legacy = _sensor_field(a, "sensor_serial_number", "serial_number")
    cal, cal_legacy = _sensor_field(a, "sensor_calibration_date", "calibration_date")
    return {
        "name": name,
        "model": str(a.get("sensor_model", "")),
        "model_assumed": str(a.get("model_source", "")) == "assumed",
        "serial": serial,
        "serial_legacy": serial_legacy,
        "calibration": cal,
        "calibration_legacy": cal_legacy,
        "standard_name": str(a.get("standard_name", "")),
        "long_name": str(a.get("long_name", "")),
        "attrs": {
            str(k): _fmt_scalar(v)
            for k, v in a.items()
            if str(k) not in _SENSOR_COLUMNED
        },
    }


def sensor_rows(
    ds: xr.Dataset,
    *,
    names: Iterable[str] | None = None,
    read_record: Callable[[xr.Dataset, str], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Return ``(rows, source)`` for the sensor catalog.

    Parameters
    ----------
    ds : xarray.Dataset
        An open dataset.
    names : iterable of str, optional
        The sensor variable names; defaults to the ``SENSOR_*`` variables (the OG1 / ctdcast
        convention), sorted.
    read_record : callable, optional
        ``read_record(ds, name) -> dict`` building one catalog row; defaults to the OG1
        ``SENSOR_*`` reader (model, serial, calibration date, with legacy fallbacks, an
        ``model_assumed`` flag, and the remaining attributes).  oceanarray passes its own.

    Returns
    -------
    rows : list of dict
        One catalog row per sensor.
    source : str
        The caption string for this builder.
    """
    reader = read_record or _og1_sensor_record
    if names is None:
        names = sorted(n for n in ds.data_vars if str(n).startswith("SENSOR_"))
    return [reader(ds, n) for n in names], source_for(sensor_rows)
