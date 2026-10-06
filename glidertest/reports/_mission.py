"""The mission report page: render context, panel registry, and section profile.

Defines what a glidertest mission report contains — a masthead meta-grid plus Metadata, Track,
Hydrography, Sampling and QC sections — by binding the plot adapters to panels and ordering them in
a profile. The page is resolved against a dataset by :func:`build`, then rendered by
:func:`glidertest.reports.report`.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .. import og1_attrs, qc, tools
from . import _plots
from ._manifest import Panel, Profile, ResolvedReport, Section, resolve

if TYPE_CHECKING:
    from collections.abc import Callable

    import xarray as xr


@dataclass(frozen=True)
class Ctx:
    """Render context for the mission page: the dataset every panel reads."""

    ds: xr.Dataset


def _duration(t0: np.datetime64, t1: np.datetime64) -> str:
    """Return a ``Nd Nh`` duration string between two datetimes (hours carry into days)."""
    total_hours = int(round(float((t1 - t0) / np.timedelta64(1, "h"))))
    days, hours = divmod(total_hours, 24)
    return f"{days}d {hours}h"


def header_card(ds: xr.Dataset) -> list[tuple[str, str]]:
    """Return (label, value) pairs for the masthead meta-grid: platform serial plus a mission summary.

    The report title is the OG1 ``id`` (set separately); this grid carries the serial and the
    overview statistics, derived from the data where possible. A field whose source is absent in the
    file is skipped; a field that cannot be computed (all-NaN) shows ``UNK``.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams.
    """
    fields: list[tuple[str, str]] = []
    if "PLATFORM_SERIAL_NUMBER" in ds:
        sv = np.atleast_1d(ds["PLATFORM_SERIAL_NUMBER"].values)
        fields.append(("Platform serial", str(sv.ravel()[0]) if sv.size else "UNK"))
    if "PROFILE_NUMBER" in ds:
        pn = np.asarray(ds["PROFILE_NUMBER"].values)
        fields.append(("Profiles", str(int(np.unique(pn[np.isfinite(pn)]).size))))
    if "TIME" in ds:
        t0 = ds["TIME"].min().values
        t1 = ds["TIME"].max().values
        fields.append(("Start", str(t0.astype("datetime64[m]")).replace("T", " ")))
        fields.append(("End", str(t1.astype("datetime64[m]")).replace("T", " ")))
        fields.append(("Duration", _duration(t0, t1)))
        diffs = np.diff(np.asarray(ds["TIME"].values)).astype("timedelta64[s]").astype(float)
        fields.append(("Sampling", f"{np.median(diffs):.0f} s" if diffs.size else "UNK"))
    if "DEPTH" in ds and "PROFILE_NUMBER" in ds:
        md = tools.max_depth_per_profile(ds)
        lo, hi = float(md.min()), float(md.max())
        depth = f"{int(lo)}–{int(hi)} m" if np.isfinite(lo) and np.isfinite(hi) else "UNK"
        fields.append(("Dive depth", depth))
    if "N_MEASUREMENTS" in ds.sizes:
        fields.append(("Records", f"{ds.sizes['N_MEASUREMENTS']:,}"))
    return fields


_URL_RE = re.compile(r"(https?://[^\s,]+)")


def _linkify(escaped: str) -> str:
    """Wrap bare http(s) URLs in *escaped* (already HTML-escaped text) in anchor tags."""
    return _URL_RE.sub(r'<a href="\1">\1</a>', escaped)


def _metadata_table(ds: xr.Dataset) -> str:
    """Return an HTML conformance table for the 16 mandatory OG1 global attributes.

    Each row shows the attribute name and its value; a row that does not conform — missing,
    present-but-empty, or failing the format/value check — has its value cell highlighted in amber.
    A present-but-empty value counts as missing.
    """
    rows = og1_attrs.check_globals(ds)
    present = sum(1 for _, status, _ in rows if status != "none")
    nonconform = sum(1 for _, status, _ in rows if status != "match")
    cells = []
    for attr, status, value in rows:
        cls = ' class="nonconform"' if status != "match" else ""
        val = _linkify(html.escape(value)) if value else "—"
        cells.append(f"<tr><td>{html.escape(attr)}</td><td{cls}>{val}</td></tr>")
    head = f"{present} of {len(rows)} mandatory global attributes present"
    if nonconform:
        head += f"; {nonconform} not conforming"
    return (
        f"<p class='caption'>{html.escape(head)}</p><table class='meta'>{''.join(cells)}</table>"
        + _payload_table(ds)
        + _geospatial_table(ds)
    )


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


def _payload_table(ds: xr.Dataset) -> str:
    """Return the payload table: each sensor's presence by whether its OG1 variable is in the file.

    Mirrors ``summary_sheet.create_docfile``'s "Basic payload configuration": presence is whether
    the variable exists, not whether it carries valid data. Absence is not flagged — a glider need
    not carry every sensor.
    """
    rows = []
    for label, var in _PAYLOAD:
        present = var in ds.variables
        mark = "<span class='qc-good'>✓</span>" if present else "✗"
        source = html.escape(str(ds[var].attrs.get("sensor", ""))) if present else ""
        rows.append(
            f"<tr><td>{label}</td><td>{mark}</td><td class='mono'>{var}</td>"
            f"<td class='mono'>{source or '—'}</td></tr>"
        )
    return (
        "<p class='caption'>Payload — a sensor counts as present when its OG1 variable exists in the "
        "file; the source is the <code>SENSOR_*</code> catalog entry named by the variable's "
        "<code>sensor</code> attribute (see the File-contents sensor catalog).</p>"
        + _table(
            ["<th>Sensor</th>", "<th>Present</th>", "<th>Variable</th>", "<th>Source</th>"], rows
        )
    )


def _table(headers: list[str], rows: list[str]) -> str:
    """Return a column-header table: *headers* (``<th>`` cells) over *rows* (each a full ``<tr>``)."""
    return "<table><thead><tr>" + "".join(headers) + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


_GEOSPATIAL = (
    ("geospatial_lat_min", "LATITUDE", "min"),
    ("geospatial_lat_max", "LATITUDE", "max"),
    ("geospatial_lon_min", "LONGITUDE", "min"),
    ("geospatial_lon_max", "LONGITUDE", "max"),
)


def _geospatial_table(ds: xr.Dataset) -> str:
    """Return the suggested OG1 geospatial-extent attributes beside the extent computed from the data.

    ``geospatial_lat/lon_min/max`` are OG1-suggested attributes; each file value sits next to the
    value computed from LATITUDE/LONGITUDE, so the attribute and its check read as one line. A row
    whose file value is absent is highlighted.
    """
    rows = []
    for attr, var, op in _GEOSPATIAL:
        raw = ds.attrs.get(attr)
        file_val = "" if raw is None else str(raw)
        computed = f"{float(getattr(ds[var], op)()):.4f}" if var in ds else "—"
        cls = "" if file_val.strip() else ' class="nonconform"'
        rows.append(f"<tr><td>{attr}</td><td{cls}>{html.escape(file_val) or '—'}</td><td>{computed}</td></tr>")
    return (
        "<p class='caption'>Geospatial extent — suggested attributes versus the extent computed from "
        "the data</p>"
        + _table(["<th>Attribute</th>", "<th>File value</th>", "<th>Computed</th>"], rows)
    )


_QC_VARS = ("TEMP", "PSAL", "DOXY", "CHLA")


def _pct(count: int, n: int) -> str:
    """Format *count* of *n* as a percent: ``–`` when zero, ``<0.1% (count)`` when tiny, else ``12.3``."""
    if count == 0 or n == 0:
        return "–"
    p = 100 * count / n
    return f"&lt;0.1% ({count})" if p < 0.1 else f"{p:.1f}"


def _dist_bar(
    counts: dict[str, int], n: int, *, small: bool = False, labels: dict[int, str] | None = None
) -> str:
    """Return a stacked distribution bar over all flag categories; segments sum to *n*.

    *counts* is a ``{category: count}`` dict from :func:`glidertest.qc.flag_counts`. Segment colours
    come from the ``qc-seg-*`` CSS classes; *small* selects the dashed glidertest-diagnostics style.
    *labels* overrides the segment tooltip text with the file's own flag meanings (keyed by flag
    value); when ``None`` the default :data:`glidertest.qc.QC_FLAG_CATEGORIES` labels are used.
    """
    if n == 0:
        return ""
    divs = ""
    for value, key, default_label in qc.QC_FLAG_CATEGORIES:
        count = counts.get(key, 0)
        if not count:
            continue
        label = labels.get(value, default_label) if labels else default_label
        pct = 100 * count / n
        divs += (
            f"<div class='qc-seg-{key}' style='width:{pct:.1f}%' "
            f"title='{html.escape(label)}: {pct:.1f}%'></div>"
        )
    cls = "qc-bar qc-bar-sm" if small else "qc-bar"
    return f"<div class='{cls}'>{divs}</div>"


def _qc_sample_cell(flags: np.ndarray) -> str:
    """Matrix cell for a per-sample QARTOD test (spike, flat): flagged counts + a small bar."""
    c = qc.flag_counts(flags)
    n = int(np.asarray(flags).size)
    if c["suspect"] == 0 and c["fail"] == 0:
        return "<td class='qc-good'>clean</td>"
    label = []
    if c["suspect"]:
        label.append(f"<span class='qc-susp'>{c['suspect']:,} susp</span>")
    if c["fail"]:
        label.append(f"<span class='qc-fail'>{c['fail']:,} fail</span>")
    return f"<td>{' '.join(label)}<br>{_dist_bar(c, n, small=True)}</td>"


def _gross_cell(r: tuple, ds: xr.Dataset, v: str) -> str:
    """Matrix cell for the gross-range test (suspect span only): out-of-range count + a small bar."""
    n = ds.sizes["N_MEASUREMENTS"] if "N_MEASUREMENTS" in ds.sizes else int(np.asarray(ds[v].values).size)
    nv = int(len(r[0]))
    if nv == 0:
        return "<td class='qc-good'>clean</td>"
    bar = _dist_bar({"good": n - nv, "suspect": nv}, n, small=True)
    return f"<td><span class='qc-susp'>{nv:,} out of range</span><br>{bar}</td>"


def _hyst_cell(err: np.ndarray) -> str:
    """Matrix cell for a hysteresis test: depth bins over threshold, graded by the shared verdict."""
    arr = np.asarray(err)
    n_over, flagged = qc.hysteresis_verdict(arr)
    cls = "qc-fail" if flagged else ("qc-susp" if n_over else "qc-good")
    return f"<td class='{cls}'>{n_over}/{arr.size} bins</td>"


#: Diagnostics matrix rows: (row label, cell kind, index into the qc_checks tuple).
_MATRIX_ROWS = (
    ("Gross range", "gross", 0),
    ("Spike", "sample", 1),
    ("Flat line", "sample", 2),
    ("Hysteresis (mean)", "hyst", 3),
    ("Hysteresis (range)", "hyst", 4),
)


#: Fixed colour class per flag category (by numeric value), independent of the file's labels.
_DELIVERED_CELL_CLASS = {"good": "qc-good", "suspect": "qc-susp", "fail": "qc-fail"}


def _qc_delivered(ds: xr.Dataset) -> str:
    """Return the 'QC as delivered' block: a per-variable census of the file's own ``*_QC`` flags.

    These are the flags the provider's pipeline wrote. Column labels come from each variable's
    ``flag_meanings`` (read via :func:`glidertest.qc.flag_labels`), so the page reports the file's
    own scale rather than an assumed one; the colour of a category stays fixed by flag value. The
    file does not record the thresholds used, so only the distribution and ``rtqc_method`` are shown.
    """
    rtqc = str(ds.attrs.get("rtqc_method", "") or "—")
    qc_vars = sorted(n for n in ds.data_vars if n.endswith("_QC"))
    if not qc_vars:
        return (
            f"<p class='caption'>No QC flags delivered in the file (rtqc_method: {html.escape(rtqc)}) "
            "— itself a finding.</p>"
        )
    labels = qc.flag_labels(ds[qc_vars[0]])
    cats = [(key, labels[value], _DELIVERED_CELL_CLASS.get(key, "")) for value, key, _ in qc.QC_FLAG_CATEGORIES]
    headers = [
        "<th>Variable</th>", "<th class='num'>N</th>",
        *(f"<th class='num'>{html.escape(label)} %</th>" for _key, label, _cls in cats),
        "<th>Distribution</th>",
    ]
    rows = []
    for qcv in qc_vars:
        var = qcv[:-3]
        f = np.asarray(ds[qcv].values)
        n = int(f.size)
        c = qc.flag_counts(f)
        cells = "".join(
            f"<td class='{f'num {cls}'.strip()}'>{_pct(c[key], n)}</td>" for key, _label, cls in cats
        )
        rows.append(
            f"<tr><td>{var}</td><td class='num'>{n:,}</td>{cells}"
            f"<td>{_dist_bar(c, n, labels=labels)}</td></tr>"
        )
    return (
        f"<p class='caption'>As delivered — rtqc_method: {html.escape(rtqc)}. Flag labels are read "
        "from each variable's flag_meanings; the file does not record the thresholds used.</p>"
        + _table(headers, rows)
    )


def _qc_diagnostics(ds: xr.Dataset) -> str | None:
    """Return the glidertest-diagnostics block: thresholds plus a graded test×variable matrix.

    glidertest runs the checks on the fly and writes no flags; thresholds come from
    :data:`glidertest.qc.configs` (the single config source — hardcoded for the Baltic today). Each
    cell is a count, with a small dashed bar for the per-sample tests, marking it as computed here.
    """
    config = qc.configs
    present = [v for v in _QC_VARS if v in ds]
    if not present:
        return None

    thr_rows = []
    for v in present:
        g = config[v]["gross_range_test"]
        s = config[v]["spike_test"]
        thr_rows.append(
            f"<tr><td>{v}</td><td>gross-range</td>"
            f"<td class='mono qc-susp'>[{g['suspect_span'][0]}, {g['suspect_span'][1]}]</td>"
            f"<td class='mono qc-fail'>[{g['fail_span'][0]}, {g['fail_span'][1]}] (not applied)</td></tr>"
        )
        thr_rows.append(
            f"<tr><td>{v}</td><td>spike</td>"
            f"<td class='mono qc-susp'>|Δ| &gt; {s['suspect_threshold']}</td>"
            f"<td class='mono qc-fail'>|Δ| &gt; {s['fail_threshold']}</td></tr>"
        )
    thr_html = _table(
        ["<th>Variable</th>", "<th>Test</th>", "<th>Suspect range / threshold</th>", "<th>Fail range / threshold</th>"],
        thr_rows,
    )

    results: dict[str, tuple | None] = {}
    for v in present:
        try:
            results[v] = qc.qc_checks(ds, var=v)
        except Exception:  # noqa: BLE001  # QC runs QARTOD/hysteresis on real data; a failure blanks the column
            results[v] = None

    matrix_rows = []
    for label, kind, idx in _MATRIX_ROWS:
        cells = []
        for v in present:
            r = results[v]
            if r is None:
                cells.append("<td>–</td>")
            elif kind == "gross":
                cells.append(_gross_cell(r, ds, v))
            elif kind == "sample":
                cells.append(_qc_sample_cell(np.asarray(r[idx])))
            else:
                cells.append(_hyst_cell(r[idx]))
        matrix_rows.append(f"<tr><td>{label}</td>{''.join(cells)}</tr>")
    matrix_html = _table(["<th>Test</th>", *(f"<th>{v}</th>" for v in present)], matrix_rows)

    return (
        "<p class='caption'>glidertest diagnostics — run on the fly, not written to the file. "
        "Thresholds are glidertest's own (hardcoded for the Baltic), shown so the verdict can be "
        "reproduced.</p>" + thr_html
        + "<p class='caption'>Test × variable: flagged-sample counts (per-sample tests show a small "
        "dashed bar). Gross range applies the suspect span only — the fail span above is not yet "
        "evaluated. Hysteresis counts depth bins with &gt;5% dive–climb error; a variable is flagged "
        "when more than 5 bins exceed it.</p>" + matrix_html
    )


def _qc_section(ds: xr.Dataset) -> str:
    """Return the QC section: 'QC as delivered' (the file's own flags) then glidertest diagnostics.

    The two are never merged: delivered flags are what the provider's pipeline wrote; the
    diagnostics are what glidertest computed on the fly.
    """
    parts = ["<h3>As delivered</h3>", _qc_delivered(ds)]
    diagnostics = _qc_diagnostics(ds)
    if diagnostics is not None:
        parts += ["<h3>glidertest diagnostics</h3>", diagnostics]
    return "".join(parts)


def _fmt_scalar(x: float | None) -> str:
    """Format a scalar min/max compactly: ``—`` for None/non-finite, 4 significant figures else."""
    if x is None:
        return "—"
    if isinstance(x, (float, np.floating)):
        return f"{x:.4g}" if np.isfinite(x) else "—"
    return str(x)[:40]


def _attrs_details(attrs: dict, exclude: tuple[str, ...] = ()) -> str:
    """Return a ``<details>`` dropdown listing *attrs* (minus *exclude*), or ``—`` when none remain.

    Attributes already shown as their own columns are passed in *exclude* so the dropdown holds only
    what the columns do not — matching ctdcast's inventory cells.
    """
    rows = [
        f"<tr><td class='mono'>{html.escape(str(k))}</td><td>{_linkify(html.escape(str(val)))}</td></tr>"
        for k, val in attrs.items()
        if k not in exclude
    ]
    if not rows:
        return "—"
    return f"<details class='attrs'><summary>{len(rows)} attrs</summary><table>{''.join(rows)}</table></details>"


def _var_row(name: str, v: xr.DataArray, *, has_qc: bool) -> str:
    """Return one file-contents table row inventorying a variable: dims, dtype, N, valid, range, CF names."""
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
    dims = ", ".join(str(d) for d in v.dims) or "()"
    qc_mark = "<span class='qc-good'>✓</span>" if has_qc else "—"
    return (
        f"<tr><td class='mono'>{html.escape(name)}</td>"
        f"<td class='mono'>{html.escape(dims)}</td>"
        f"<td class='mono'>{html.escape(str(v.dtype))}</td>"
        f"<td class='num'>{n:,}</td><td class='num'>{n_valid:,}</td>"
        f"<td class='num'>{html.escape(_fmt_scalar(v_min))}</td>"
        f"<td class='num'>{html.escape(_fmt_scalar(v_max))}</td>"
        f"<td>{html.escape(v.attrs.get('units', ''))}</td>"
        f"<td>{html.escape(v.attrs.get('long_name', ''))}</td>"
        f"<td class='mono'>{html.escape(v.attrs.get('standard_name', ''))}</td>"
        f"<td>{qc_mark}</td>"
        f"<td>{_attrs_details(v.attrs, exclude=('units', 'long_name', 'standard_name'))}</td></tr>"
    )


_VAR_HEADERS = [
    "<th>Variable</th>", "<th>Dims</th>", "<th>Dtype</th>", "<th class='num'>N</th>",
    "<th class='num'>Valid</th>", "<th class='num'>Min</th>", "<th class='num'>Max</th>",
    "<th>Units</th>", "<th>Long name</th>", "<th>Standard name</th>", "<th>QC</th>",
    "<th>Attributes</th>",
]

#: Columns for the sensor-catalog table: OG1 ``SENSOR_*`` variables are NaN scalars whose content
#: is in their attributes, so min/max/valid are meaningless — show the sensor attributes instead.
_SENSOR_HEADERS = [
    "<th>Variable</th>", "<th>Model</th>", "<th>Serial</th>", "<th>Calibration</th>",
    "<th>Attributes</th>",
]


def _var_table(ds: xr.Dataset, names: list[str]) -> str:
    """Return a standard inventory table over *names*, or ``""`` if *names* is empty."""
    if not names:
        return ""
    rows = [_var_row(n, ds[n], has_qc=f"{n}_QC" in ds.variables) for n in names]
    return _table(_VAR_HEADERS, rows)


def _sensor_row(name: str, v: xr.DataArray) -> str:
    """Return one sensor-catalog row: model, serial, calibration date and the rest of the attrs."""
    a = v.attrs
    return (
        f"<tr><td class='mono'>{html.escape(name)}</td>"
        f"<td>{html.escape(str(a.get('sensor_model', '')))}</td>"
        f"<td class='mono'>{html.escape(str(a.get('serial_number', '')))}</td>"
        f"<td>{html.escape(str(a.get('calibration_date', '')))}</td>"
        f"<td>{_attrs_details(a, exclude=('sensor_model', 'serial_number', 'calibration_date'))}</td></tr>"
    )


def _file_contents(ds: xr.Dataset) -> str:
    """Return the file-contents inventory: variable tables grouped by dimension, then the attr dump.

    Variables are split into tables by their dimension signature — coordinates, the
    ``N_MEASUREMENTS`` science variables, any other dimension group, and dimensionless scalars.
    OG1 ``SENSOR_*`` variables (dimensionless, NaN-valued, carrying their content in attributes) get
    a separate catalog table. The attribute dump then shows every global attribute in file order.
    """
    qc_vars = sorted(n for n in ds.data_vars if n.endswith("_QC"))
    sensors = sorted(n for n in ds.data_vars if n.startswith("SENSOR_"))
    science = sorted(
        n for n in ds.data_vars if not n.startswith("SENSOR_") and not n.endswith("_QC")
    )
    by_dims: dict[tuple[str, ...], list[str]] = {}
    for n in science:
        by_dims.setdefault(tuple(str(d) for d in ds[n].dims), []).append(n)

    blocks = [f"<h3>Coordinates</h3>{_var_table(ds, sorted(ds.coords))}"]
    measurement = by_dims.pop(("N_MEASUREMENTS",), None)
    if measurement:
        blocks.append(f"<h3>Variables on N_MEASUREMENTS</h3>{_var_table(ds, measurement)}")
    scalars = by_dims.pop((), None)
    for dims in sorted(by_dims, key=lambda d: (len(d), d)):
        blocks.append(f"<h3>Variables on {', '.join(dims)}</h3>{_var_table(ds, by_dims[dims])}")
    if scalars:
        blocks.append(f"<h3>Scalar variables</h3>{_var_table(ds, scalars)}")
    if sensors:
        sensor_rows = [_sensor_row(n, ds[n]) for n in sensors]
        blocks.append(f"<h3>Sensor catalog</h3>{_table(_SENSOR_HEADERS, sensor_rows)}")

    attr_rows = [
        f"<tr><td>{html.escape(str(k))}</td><td>{_linkify(html.escape(str(val)))}</td></tr>"
        for k, val in ds.attrs.items()
    ]
    return (
        f"<p class='caption'>{len(science)} data variables, {len(sensors)} sensors, "
        f"{len(qc_vars)} QC-flag variables (shown in the QC section, not listed here), "
        f"{len(ds.coords)} coordinates, {len(ds.attrs)} global attributes.</p>" + "".join(blocks)
        + "<h3>Global attributes</h3><p class='caption'>In file order.</p>"
        + _table(["<th>Attribute</th>", "<th>Value</th>"], attr_rows)
    )


def _guarded(fn: Callable[[Ctx], str]) -> Callable[[Ctx], str]:
    """Wrap an html-panel render so a failure is a visible block on the page, not a dead report.

    The vendored encoder guards figure panels; html panels (which call live-data numpy/xarray ops)
    get the same resilience here — on exception they render a ``.none-note`` block naming the error,
    and the page still writes.
    """

    def render(ctx: Ctx) -> str:
        """Render *fn*, returning a ``.none-note`` failure block instead of raising."""
        try:
            return fn(ctx)
        except Exception as exc:  # noqa: BLE001  # surface any html-panel failure on the page; never abort the report
            return f"<p class='none-note'>{html.escape(type(exc).__name__)}: {html.escape(str(exc))}</p>"

    return render


def _qc_applies(c: Ctx) -> bool:
    """QC section applies when a QC variable or any delivered ``*_QC`` flag variable is present."""
    return any(v in c.ds for v in _QC_VARS) or any(n.endswith("_QC") for n in c.ds.data_vars)


def _has(var: str) -> Callable[[Ctx], bool]:
    """Return an ``applies_to`` predicate: the panel applies only when *var* is in the dataset."""
    return lambda c: var in c.ds


PANELS: dict[str, Panel] = {
    "metadata": Panel(id="metadata", kind="html", render=_guarded(lambda c: _metadata_table(c.ds))),
    "track": Panel(id="track", render=lambda c: _plots.track(c.ds), caption="Glider track"),
    "basic_vars": Panel(
        id="basic_vars",
        render=lambda c: _plots.basic_vars(c.ds),
        caption="Core variables versus depth",
    ),
    "ts": Panel(id="ts", render=lambda c: _plots.ts(c.ds), caption="Temperature–salinity diagram"),
    "section_temp": Panel(
        id="section_temp",
        render=lambda c: _plots.section(c.ds, "TEMP"),
        caption="Temperature section",
    ),
    "section_psal": Panel(
        id="section_psal",
        render=lambda c: _plots.section(c.ds, "PSAL"),
        caption="Salinity section",
    ),
    "section_doxy": Panel(
        id="section_doxy",
        render=lambda c: _plots.section(c.ds, "DOXY"),
        caption="Dissolved oxygen section",
        applies_to=_has("DOXY"),
    ),
    "section_chla": Panel(
        id="section_chla",
        render=lambda c: _plots.section(c.ds, "CHLA"),
        caption="Chlorophyll section",
        applies_to=_has("CHLA"),
    ),
    "grid_spacing": Panel(
        id="grid_spacing", render=lambda c: _plots.grid_spacing(c.ds), caption="Grid spacing"
    ),
    "sampling_period": Panel(
        id="sampling_period",
        render=lambda c: _plots.sampling_period(c.ds),
        caption="Sampling period",
    ),
    "max_depth": Panel(
        id="max_depth",
        render=lambda c: _plots.max_depth(c.ds),
        caption="Maximum depth per profile",
    ),
    "prof_monotony": Panel(
        id="prof_monotony",
        render=lambda c: _plots.prof_monotony(c.ds),
        caption="Profile-number monotonicity",
    ),
    "qc": Panel(id="qc", kind="html", render=_guarded(lambda c: _qc_section(c.ds))),
    "file_contents": Panel(
        id="file_contents", kind="html", render=_guarded(lambda c: _file_contents(c.ds))
    ),
}

PROFILE = Profile(
    entries=(
        Section(id="metadata", title="Metadata", panels=("metadata",)),
        Section(id="track", title="Track", panels=("track",)),
        Section(
            id="hydrography",
            title="Hydrography",
            panels=("basic_vars", "ts", "section_temp", "section_psal", "section_doxy", "section_chla"),
        ),
        Section(
            id="sampling",
            title="Sampling",
            panels=("grid_spacing", "sampling_period", "max_depth", "prof_monotony"),
        ),
        Section(
            id="qc",
            title="QC",
            panels=("qc",),
            applies_to=_qc_applies,
        ),
        Section(id="file_contents", title="File contents", panels=("file_contents",)),
    ),
)


def build(ds: xr.Dataset) -> ResolvedReport:
    """Resolve the mission profile against *ds* into a numbered report."""
    return resolve(PROFILE, Ctx(ds=ds), PANELS)
