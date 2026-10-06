"""The mission report page: render context, panel registry, and section profile.

Defines what a glidertest mission report contains — one page with Mission, Track, Hydrography and
Sampling sections — by binding the plot adapters to panels and ordering them in a profile. The page
is resolved against a dataset by :func:`build`, then rendered by :func:`glidertest.reports.report`.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import _plots
from ._manifest import Panel, Profile, ResolvedReport, Section, resolve

if TYPE_CHECKING:
    import xarray as xr


@dataclass(frozen=True)
class Ctx:
    """Render context for the mission page: the dataset every panel reads."""

    ds: xr.Dataset


def _mission_table(ds: xr.Dataset) -> str:
    """Return an HTML table of mission metadata from the dataset's global attributes.

    Values absent from the file show as ``UNK`` rather than being guessed. Profile count and the
    variable list are derived from the data, not the attributes.
    """
    a = ds.attrs
    if "PROFILE_NUMBER" in ds:
        pn = np.asarray(ds["PROFILE_NUMBER"].values)
        n_prof = int(np.unique(pn[np.isfinite(pn)]).size)
    else:
        n_prof = "UNK"
    rows = [
        ("Deployment ID", a.get("id", "UNK")),
        ("Title", a.get("title", "UNK")),
        ("Platform", a.get("platform", "UNK")),
        ("Glider serial", a.get("glider_serial", "UNK")),
        ("Start date", a.get("start_date", "UNK")),
        ("Contributor", a.get("contributor_name", "UNK")),
        ("Profiles", n_prof),
        ("Variables", ", ".join(sorted(ds.data_vars))),
    ]
    body = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows
    )
    return f"<table class='meta'>{body}</table>"


PANELS: dict[str, Panel] = {
    "mission_meta": Panel(id="mission_meta", kind="html", render=lambda c: _mission_table(c.ds)),
    "track": Panel(id="track", render=lambda c: _plots.track(c.ds), caption="Glider track"),
    "basic_vars": Panel(
        id="basic_vars",
        render=lambda c: _plots.basic_vars(c.ds),
        caption="Core variables versus depth",
    ),
    "ts": Panel(id="ts", render=lambda c: _plots.ts(c.ds), caption="Temperature–salinity diagram"),
    "max_depth": Panel(
        id="max_depth",
        render=lambda c: _plots.max_depth(c.ds),
        caption="Maximum depth per profile",
    ),
}

PROFILE = Profile(
    entries=(
        Section(id="mission", title="Mission", panels=("mission_meta",)),
        Section(id="track", title="Track", panels=("track",)),
        Section(id="hydrography", title="Hydrography", panels=("basic_vars", "ts")),
        Section(id="sampling", title="Sampling", panels=("max_depth",)),
    ),
)


def build(ds: xr.Dataset) -> ResolvedReport:
    """Resolve the mission profile against *ds* into a numbered report."""
    return resolve(PROFILE, Ctx(ds=ds), PANELS)
