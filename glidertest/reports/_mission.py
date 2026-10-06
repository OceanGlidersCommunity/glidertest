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
        cells.append(f"<tr{cls}><th>{html.escape(attr)}</th><td>{val}</td></tr>")
    head = f"{present} of {len(rows)} mandatory global attributes present"
    if nonconform:
        head += f"; {nonconform} not conforming"
    return (
        f"<p class='caption'>{html.escape(head)}</p><table class='meta'>{''.join(cells)}</table>"
        + _geospatial_table(ds)
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


def _dist_bar(counts: dict[str, int], n: int, *, small: bool = False) -> str:
    """Return a stacked distribution bar over all QARTOD categories; segments sum to *n*.

    *counts* is a ``{category: count}`` dict from :func:`glidertest.qc.flag_counts`. Segment colours
    come from the ``qc-seg-*`` CSS classes; *small* selects the dashed glidertest-diagnostics style.
    """
    if n == 0:
        return ""
    divs = ""
    for _value, key, label in qc.QC_FLAG_CATEGORIES:
        count = counts.get(key, 0)
        if not count:
            continue
        pct = 100 * count / n
        divs += f"<div class='qc-seg-{key}' style='width:{pct:.1f}%' title='{label}: {pct:.1f}%'></div>"
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


def _qc_delivered(ds: xr.Dataset) -> str:
    """Return the 'QC as delivered' block: a per-variable census of the file's own ``*_QC`` flags.

    These are the flags the provider's pipeline wrote (QARTOD 1/2/3/4/9). The file does not record
    the thresholds used, so only the distribution and the ``rtqc_method`` are shown.
    """
    rtqc = str(ds.attrs.get("rtqc_method", "") or "—")
    pairs = [(v, f"{v}_QC") for v in _QC_VARS if f"{v}_QC" in ds]
    if not pairs:
        return (
            f"<p class='caption'>No QC flags delivered in the file (rtqc_method: {html.escape(rtqc)}) "
            "— itself a finding.</p>"
        )
    rows = []
    for var, qcv in pairs:
        f = np.asarray(ds[qcv].values)
        n = int(f.size)
        c = qc.flag_counts(f)
        rows.append(
            f"<tr><td>{var}</td><td class='num'>{n:,}</td>"
            f"<td class='num qc-good'>{_pct(c['good'], n)}</td>"
            f"<td class='num qc-susp'>{_pct(c['suspect'], n)}</td>"
            f"<td class='num qc-fail'>{_pct(c['fail'], n)}</td>"
            f"<td class='num'>{_pct(c['not_eval'], n)}</td>"
            f"<td class='num'>{_pct(c['missing'], n)}</td>"
            f"<td>{_dist_bar(c, n)}</td></tr>"
        )
    headers = [
        "<th>Variable</th>", "<th class='num'>N</th>", "<th class='num'>Good %</th>",
        "<th class='num'>Suspect %</th>", "<th class='num'>Fail %</th>",
        "<th class='num'>Not eval %</th>", "<th class='num'>Missing %</th>", "<th>Distribution</th>",
    ]
    return (
        f"<p class='caption'>As delivered — rtqc_method: {html.escape(rtqc)}. "
        "The file does not record the thresholds used.</p>" + _table(headers, rows)
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


def _guarded(fn: Callable[[Ctx], str]) -> Callable[[Ctx], str]:
    """Wrap an html-panel render so a failure is a visible block on the page, not a dead report.

    The vendored encoder guards figure panels; html panels (which call live-data numpy/xarray ops)
    get the same resilience here — on exception they render a ``.none-note`` block naming the error,
    and the page still writes.
    """

    def render(ctx: Ctx) -> str:
        try:
            return fn(ctx)
        except Exception as exc:  # noqa: BLE001  # surface any html-panel failure on the page; never abort the report
            return f"<p class='none-note'>{html.escape(type(exc).__name__)}: {html.escape(str(exc))}</p>"

    return render


def _qc_applies(c: Ctx) -> bool:
    """QC section applies when a QC variable or its delivered ``*_QC`` flags are present."""
    return any(v in c.ds for v in _QC_VARS) or any(f"{v}_QC" in c.ds for v in _QC_VARS)


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
    ),
)


def build(ds: xr.Dataset) -> ResolvedReport:
    """Resolve the mission profile against *ds* into a numbered report."""
    return resolve(PROFILE, Ctx(ds=ds), PANELS)
