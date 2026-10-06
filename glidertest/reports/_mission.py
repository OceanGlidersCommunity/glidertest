"""The mission report page: render context, panel registry, and section profile.

Defines what a glidertest mission report contains — a masthead header card plus Track, Hydrography
and Sampling sections — by binding the plot adapters to panels and ordering them in a profile. The
page is resolved against a dataset by :func:`build`, then rendered by
:func:`glidertest.reports.report`.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .. import og1_attrs, tools
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
    """Return a ``Nd Nh`` duration string between two datetimes."""
    days = float((t1 - t0) / np.timedelta64(1, "D"))
    whole = int(days)
    hours = int(round((days - whole) * 24))
    return f"{whole}d {hours}h"


def header_card(ds: xr.Dataset) -> list[tuple[str, str]]:
    """Return (label, value) pairs for the masthead meta-grid: platform serial plus a mission summary.

    The report title is the OG1 ``id`` (set separately); this grid carries the serial and the
    overview statistics, derived from the data where possible. A field whose source is absent in the
    file is skipped.
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
        dt = np.median(np.diff(np.asarray(ds["TIME"].values)).astype("timedelta64[s]").astype(float))
        fields.append(("Sampling", f"{dt:.0f} s"))
    if "DEPTH" in ds and "PROFILE_NUMBER" in ds:
        md = tools.max_depth_per_profile(ds)
        fields.append(("Dive depth", f"{int(md.min())}–{int(md.max())} m"))
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
    return f"<p class='caption'>{html.escape(head)}</p><table class='meta'>{''.join(cells)}</table>"


_QC_VARS = ("TEMP", "PSAL", "DOXY", "CHLA")


def _pct(count: int, n: int) -> str:
    """Format *count* of *n* as a percent: ``–`` when zero, ``<0.1% (count)`` when tiny, else ``12.3``."""
    if count == 0 or n == 0:
        return "–"
    p = 100 * count / n
    return f"&lt;0.1% ({count})" if p < 0.1 else f"{p:.1f}"


def _dist_bar(good: int, suspect: int, bad: int, n: int) -> str:
    """Return a stacked distribution bar (oceanarray style): percent-width segments with tooltips."""
    if n == 0:
        return ""
    segs = (
        ("good", good, "#27ae60"),
        ("suspect", suspect, "#f39c12"),
        ("bad", bad, "#e74c3c"),
    )
    divs = "".join(
        f"<div style='width:{100 * c / n:.1f}%;background:{col}' title='{label}: {100 * c / n:.1f}%'></div>"
        for label, c, col in segs
        if c
    )
    return f"<div class='qc-bar'>{divs}</div>"


def _dist_bar_sm(good: int, suspect: int, bad: int, n: int) -> str:
    """A small, dashed-outline distribution bar for the glidertest-diagnostics cells (Option 2)."""
    if n == 0:
        return ""
    segs = (("good", good, "#27ae60"), ("suspect", suspect, "#f39c12"), ("bad", bad, "#e74c3c"))
    divs = "".join(
        f"<div style='width:{100 * c / n:.1f}%;background:{col}' title='{label}: {100 * c / n:.1f}%'></div>"
        for label, c, col in segs
        if c
    )
    return f"<div class='qc-bar qc-bar-sm'>{divs}</div>"


def _qc_sample_cell(flags: np.ndarray) -> str:
    """Matrix cell for a per-sample QARTOD test (spike, flat): flagged counts + a small bar."""
    n = int(flags.size)
    ns = int((flags == 3).sum())
    nb = int((flags == 4).sum())
    if ns == 0 and nb == 0:
        return "<td class='qc-good'>clean</td>"
    label = []
    if ns:
        label.append(f"<span class='qc-susp'>{ns:,} susp</span>")
    if nb:
        label.append(f"<span class='qc-fail'>{nb:,} fail</span>")
    return f"<td>{' '.join(label)}<br>{_dist_bar_sm(n - ns - nb, ns, nb, n)}</td>"


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
    rows = [
        "<table><thead><tr><th>Variable</th><th class='num'>N</th><th class='num'>Good %</th>"
        "<th class='num'>Suspect %</th><th class='num'>Bad %</th><th>Distribution</th>"
        "</tr></thead><tbody>"
    ]
    for var, qcv in pairs:
        f = np.asarray(ds[qcv].values)
        n = int(f.size)
        ng, ns, nb = int((f == 1).sum()), int((f == 3).sum()), int((f == 4).sum())
        rows.append(
            f"<tr><td>{var}</td><td class='num'>{n:,}</td>"
            f"<td class='num qc-good'>{_pct(ng, n)}</td>"
            f"<td class='num qc-susp'>{_pct(ns, n)}</td>"
            f"<td class='num qc-fail'>{_pct(nb, n)}</td>"
            f"<td>{_dist_bar(ng, ns, nb, n)}</td></tr>"
        )
    rows.append("</tbody></table>")
    return (
        f"<p class='caption'>As delivered — rtqc_method: {html.escape(rtqc)}. "
        "The file does not record the thresholds used.</p>" + "".join(rows)
    )


def _qc_diagnostics(ds: xr.Dataset) -> str | None:
    """Return the glidertest-diagnostics block: thresholds plus a graded test×variable matrix.

    glidertest runs the checks on the fly and writes no flags; thresholds come from
    ``summary_sheet.configs`` (the single config source — hardcoded for the Baltic today). Each cell
    is a count, with a small dashed bar for the per-sample tests, marking it as computed here.
    """
    from .. import summary_sheet

    config = summary_sheet.configs
    present = [v for v in _QC_VARS if v in ds]
    if not present:
        return None

    thr = [
        "<table><thead><tr><th>Variable</th><th>Test</th>"
        "<th>Suspect range / threshold</th><th>Fail range / threshold</th></tr></thead><tbody>"
    ]
    for v in present:
        g = config[v]["gross_range_test"]
        s = config[v]["spike_test"]
        thr.append(
            f"<tr><td>{v}</td><td>gross-range</td>"
            f"<td class='mono qc-susp'>[{g['suspect_span'][0]}, {g['suspect_span'][1]}]</td>"
            f"<td class='mono qc-fail'>[{g['fail_span'][0]}, {g['fail_span'][1]}]</td></tr>"
        )
        thr.append(
            f"<tr><td>{v}</td><td>spike</td>"
            f"<td class='mono qc-susp'>|Δ| &gt; {s['suspect_threshold']}</td>"
            f"<td class='mono qc-fail'>|Δ| &gt; {s['fail_threshold']}</td></tr>"
        )
    thr.append("</tbody></table>")

    results: dict[str, tuple | None] = {}
    for v in present:
        try:
            results[v] = summary_sheet.qc_checks(ds, var=v)
        except Exception:  # noqa: BLE001  # QC runs QARTOD/hysteresis on real data; a failure blanks the column
            results[v] = None

    matrix = ["<table><thead><tr><th>Test</th>" + "".join(f"<th>{v}</th>" for v in present) + "</tr></thead><tbody>"]
    matrix.append("<tr><td>Gross range</td>")
    for v in present:
        r = results[v]
        if r is None:
            matrix.append("<td>–</td>")
            continue
        n = int(np.asarray(ds[v].values).size)
        nv = int(len(r[0]))
        if nv == 0:
            matrix.append("<td class='qc-good'>clean</td>")
        else:
            matrix.append(
                f"<td><span class='qc-susp'>{nv:,} out of range</span><br>{_dist_bar_sm(n - nv, nv, 0, n)}</td>"
            )
    matrix.append("</tr>")
    for label, idx in (("Spike", 1), ("Flat line", 2)):
        matrix.append(f"<tr><td>{label}</td>")
        for v in present:
            r = results[v]
            matrix.append(_qc_sample_cell(np.asarray(r[idx])) if r is not None else "<td>–</td>")
        matrix.append("</tr>")
    for label, idx in (("Hysteresis (mean)", 3), ("Drift (range)", 4)):
        matrix.append(f"<tr><td>{label}</td>")
        for v in present:
            r = results[v]
            if r is None:
                matrix.append("<td>–</td>")
                continue
            arr = np.asarray(r[idx])
            flagged = int((arr > 5).sum())
            cls = "qc-good" if flagged == 0 else "qc-susp"
            matrix.append(f"<td class='{cls}'>{flagged}/{arr.size} profiles</td>")
        matrix.append("</tr>")
    matrix.append("</tbody></table>")

    return (
        "<p class='caption'>glidertest diagnostics — run on the fly, not written to the file. "
        "Thresholds are glidertest's own (hardcoded for the Baltic), shown so the verdict can be "
        "reproduced.</p>" + "".join(thr)
        + "<p class='caption'>Test × variable: flagged-sample counts (per-sample tests show a small "
        "dashed bar); Hysteresis/Drift are profiles with &gt;5% dive–climb error.</p>" + "".join(matrix)
    )


def _qc_section(ds: xr.Dataset) -> str:
    """Return the QC section: 'QC as delivered' (the file's own flags) then glidertest diagnostics.

    The two are never merged (plan §10.2): delivered flags are what the provider wrote; the
    diagnostics are what glidertest found on the fly.
    """
    parts = ["<h3>As delivered</h3>", _qc_delivered(ds)]
    diagnostics = _qc_diagnostics(ds)
    if diagnostics is not None:
        parts += ["<h3>glidertest diagnostics</h3>", diagnostics]
    return "".join(parts)


def _has(var: str) -> Callable[[Ctx], bool]:
    """Return an ``applies_to`` predicate: the panel applies only when *var* is in the dataset."""
    return lambda c: var in c.ds


PANELS: dict[str, Panel] = {
    "metadata": Panel(id="metadata", kind="html", render=lambda c: _metadata_table(c.ds)),
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
    "qc": Panel(id="qc", kind="html", render=lambda c: _qc_section(c.ds)),
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
            applies_to=lambda c: any(v in c.ds for v in _QC_VARS),
        ),
    ),
)


def build(ds: xr.Dataset) -> ResolvedReport:
    """Resolve the mission profile against *ds* into a numbered report."""
    return resolve(PROFILE, Ctx(ds=ds), PANELS)
