"""The mission report page: render context, panel registry, and section profile.

Defines what a glidertest mission report contains — a masthead meta-grid plus Metadata, Track,
Hydrography, Sampling and QC sections — by binding the plot adapters to panels and ordering them in
a profile. The page is resolved against a dataset by :func:`build`, then rendered by
:func:`glidertest.reports.report`.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .. import tools
from . import _plots, inventory, metadata, qc_section
from ._env import get_template
from ._manifest import Panel, Profile, ResolvedReport, Section, resolve

if TYPE_CHECKING:
    from collections.abc import Callable

    import xarray as xr


@dataclass(frozen=True)
class Ctx:
    """Render context for the mission page: the dataset every panel reads."""

    ds: xr.Dataset


def _duration(t0: np.datetime64, t1: np.datetime64) -> str:
    """Return a ``Nd Nh`` duration string between two datetimes, or ``UNK`` if either is NaT."""
    hours = float((t1 - t0) / np.timedelta64(1, "h"))
    if not np.isfinite(hours):
        return "UNK"
    days, rem = divmod(round(hours), 24)
    return f"{days}d {rem}h"


def header_card(ds: xr.Dataset) -> list[tuple[str, str]]:
    """Return (label, value) pairs for the masthead meta-grid: platform serial plus a mission summary.

    The report title is the OG1 ``id`` (set separately); this grid carries the serial and the
    overview statistics, derived from the data where possible. A field whose source is absent in the
    file is skipped; a field that cannot be computed (all-NaN, all-NaT) shows ``UNK``.
    """
    fields: list[tuple[str, str]] = []
    if "PLATFORM_SERIAL_NUMBER" in ds:
        sv = np.atleast_1d(ds["PLATFORM_SERIAL_NUMBER"].values)
        first = sv.ravel()[0] if sv.size else None
        serial = "UNK" if first is None else str(first)
        if isinstance(first, (float, np.floating)) and not np.isfinite(first):
            serial = "UNK"
        fields.append(("Platform serial", serial))
    if "PROFILE_NUMBER" in ds:
        pn = np.asarray(ds["PROFILE_NUMBER"].values)
        finite = np.isfinite(pn)
        n_prof = int(np.unique(pn[finite]).size)
        label = str(n_prof)
        if "PROFILE_DIRECTION" in ds:
            # Each profile is one cast: a dive (downcast, direction -1) or a climb (upcast, +1).
            pdir = np.asarray(ds["PROFILE_DIRECTION"].values)
            m = finite & np.isfinite(pdir)
            _uniq, idx = np.unique(pn[m], return_index=True)
            prof_dir = pdir[m][idx]
            n_dive = int((prof_dir == -1).sum())
            n_climb = int((prof_dir == 1).sum())
            label = f"{n_prof} ({n_dive} dive, {n_climb} climb)"
        fields.append(("Profiles", label))
    if "TIME" in ds:
        t0 = ds["TIME"].min().values
        t1 = ds["TIME"].max().values
        if np.isnat(t0) or np.isnat(t1):
            fields += [("Start", "UNK"), ("End", "UNK"), ("Duration", "UNK")]
        else:
            fields.append(("Start", str(t0.astype("datetime64[m]")).replace("T", " ")))
            fields.append(("End", str(t1.astype("datetime64[m]")).replace("T", " ")))
            fields.append(("Duration", _duration(t0, t1)))
        dt = np.diff(np.asarray(ds["TIME"].values))  # timedelta64; NaT where either end is NaT
        valid = dt[~np.isnat(dt)]
        med = np.median(valid.astype("timedelta64[s]").astype(float)) if valid.size else np.nan
        fields.append(("Sampling", f"{med:.0f} s" if np.isfinite(med) else "UNK"))
    if "DEPTH" in ds and "PROFILE_NUMBER" in ds:
        md = tools.max_depth_per_profile(ds)
        lo, hi = float(md.min()), float(md.max())
        depth = f"{int(lo)}–{int(hi)} m" if np.isfinite(lo) and np.isfinite(hi) else "UNK"
        fields.append(("Dive depth", depth))
    if "N_MEASUREMENTS" in ds.sizes:
        fields.append(("Records", f"{ds.sizes['N_MEASUREMENTS']:,}"))
    return fields


_QC_VARS = ("TEMP", "PSAL", "DOXY", "CHLA")


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
    "metadata": Panel(
        id="metadata",
        kind="html",
        render=_guarded(lambda c: get_template("_metadata.html").render(**metadata.metadata_data(c.ds))),
    ),
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
    "qc": Panel(
        id="qc",
        kind="html",
        render=_guarded(lambda c: get_template("_qc.html").render(**qc_section.qc_section_data(c.ds))),
    ),
    "file_contents": Panel(
        id="file_contents",
        kind="html",
        render=_guarded(lambda c: get_template("_inventory.html").render(**inventory.inventory_data(c.ds))),
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
