"""The mission report page: render context, panel registry, and section profile.

Defines what a glidertest mission report contains — a masthead header card plus Track, Hydrography
and Sampling sections — by binding the plot adapters to panels and ordering them in a profile. The
page is resolved against a dataset by :func:`build`, then rendered by
:func:`glidertest.reports.report`.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .. import og1_attrs
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
    """Return (label, value) pairs for the masthead header card from the dataset's global attributes.

    Values absent from the file show as ``UNK`` rather than being guessed.
    """
    a = ds.attrs
    return [
        ("ID", str(a.get("id", "UNK"))),
        ("Platform", str(a.get("platform", "UNK"))),
        ("Serial", str(a.get("glider_serial", "UNK"))),
        ("Start", str(a.get("start_date", "UNK"))),
        ("Contributor", str(a.get("contributor_name", "UNK"))),
    ]


def _metadata_table(ds: xr.Dataset) -> str:
    """Return an HTML conformance table for the 16 mandatory OG1 global attributes.

    Each row is a status pip (present / present-but-wrong-format / missing), the attribute name and
    its value. A present-but-empty value counts as missing.
    """
    rows = og1_attrs.check_globals(ds)
    present = sum(1 for _, status, _ in rows if status != "none")
    glyph = {"match": "✓", "differ": "!", "none": "–"}
    body = "".join(
        f"<tr><td><span class='conf conf-{status}'>{glyph[status]}</span></td>"
        f"<th>{html.escape(attr)}</th><td>{html.escape(value) if value else '—'}</td></tr>"
        for attr, status, value in rows
    )
    head = f"<p class='caption'>{present} of {len(rows)} mandatory global attributes present</p>"
    return f"{head}<table class='meta'>{body}</table>"


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
    ),
)


def build(ds: xr.Dataset) -> ResolvedReport:
    """Resolve the mission profile against *ds* into a numbered report."""
    return resolve(PROFILE, Ctx(ds=ds), PANELS)
