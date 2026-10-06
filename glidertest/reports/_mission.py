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


def header_card(ds: xr.Dataset) -> list[tuple[str, str]]:
    """Return (label, value) pairs for the masthead header card.

    Labels follow the OG1 names: ``id`` (the formatted mission name ``<serial>_<start>_<mode>``), the
    platform serial from the ``PLATFORM_SERIAL_NUMBER`` variable, and the contributor (with
    contributing institutions). ``platform`` is not repeated here — it appears in the Metadata
    section. Values absent from the file show as ``UNK``.
    """
    a = ds.attrs
    if "PLATFORM_SERIAL_NUMBER" in ds:
        sv = np.atleast_1d(ds["PLATFORM_SERIAL_NUMBER"].values)
        serial = str(sv.ravel()[0]) if sv.size else "UNK"
    else:
        serial = "UNK"
    creator = str(a.get("contributor_name", "UNK"))
    institutions = a.get("contributing_institutions")
    if institutions:
        creator = f"{creator} ({institutions})"
    return [
        ("ID", str(a.get("id", "UNK"))),
        ("Platform serial", serial),
        ("Contributor", creator),
    ]


def _mission_summary(ds: xr.Dataset) -> str:
    """Return an HTML table of mission overview statistics, derived from the data (not attributes).

    Rows are included only for the variables present; a variable absent from the file is skipped.
    """
    rows: list[tuple[str, str]] = []
    if "PROFILE_NUMBER" in ds:
        pn = np.asarray(ds["PROFILE_NUMBER"].values)
        rows.append(("Profiles", str(int(np.unique(pn[np.isfinite(pn)]).size))))
    if "TIME" in ds:
        t0 = ds["TIME"].min().values
        t1 = ds["TIME"].max().values
        rows.append(("Deployment date", str(t0.astype("datetime64[D]"))))
        rows.append(("Recovery date", str(t1.astype("datetime64[D]"))))
        rows.append(("Duration", f"{(t1 - t0) / np.timedelta64(1, 'D'):.1f} days"))
    if "LATITUDE" in ds:
        rows.append(("Latitude", f"{float(ds['LATITUDE'].min()):.3f} to {float(ds['LATITUDE'].max()):.3f} °N"))
    if "LONGITUDE" in ds:
        rows.append(("Longitude", f"{float(ds['LONGITUDE'].min()):.3f} to {float(ds['LONGITUDE'].max()):.3f} °E"))
    if "DEPTH" in ds and "PROFILE_NUMBER" in ds:
        md = tools.max_depth_per_profile(ds)
        rows.append(("Diving depth range", f"{int(md.min())} to {int(md.max())} m"))
    rows.append(("Variables", str(len(ds.data_vars))))
    body = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in rows)
    return f"<table class='meta'>{body}</table>"


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


def _has(var: str) -> Callable[[Ctx], bool]:
    """Return an ``applies_to`` predicate: the panel applies only when *var* is in the dataset."""
    return lambda c: var in c.ds


PANELS: dict[str, Panel] = {
    "summary": Panel(id="summary", kind="html", render=lambda c: _mission_summary(c.ds)),
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
}

PROFILE = Profile(
    entries=(
        Section(id="summary", title="Summary", panels=("summary",)),
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
    ),
)


def build(ds: xr.Dataset) -> ResolvedReport:
    """Resolve the mission profile against *ds* into a numbered report."""
    return resolve(PROFILE, Ctx(ds=ds), PANELS)
