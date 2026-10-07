"""The mission report page: render context, panel registry, and section profile.

Defines what a glidertest mission report contains — a masthead meta-grid plus Metadata, Track,
Hydrography, Sampling and QC sections — by binding the plot adapters to panels and ordering them in
a profile. The page is resolved against a dataset by :func:`build`, then rendered by
:func:`glidertest.reports.report`.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .. import tools
from . import _plots, inventory, metadata, qc_section, sensors
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
    if "LATITUDE" in ds and "LONGITUDE" in ds:
        la = np.asarray(ds["LATITUDE"].values)
        lo_ = np.asarray(ds["LONGITUDE"].values)
        la, lo_ = la[np.isfinite(la)], lo_[np.isfinite(lo_)]
        if la.size and lo_.size:
            # Signed degrees, no hemisphere letter: a negative longitude labelled "°E" would be wrong.
            fields.append(("Extent", f"{la.min():.2f}–{la.max():.2f} lat, {lo_.min():.2f}–{lo_.max():.2f} lon"))
    if "N_MEASUREMENTS" in ds.sizes:
        fields.append(("Records", f"{ds.sizes['N_MEASUREMENTS']:,}"))
    # Source file + size, from the path xarray recorded when the dataset was opened; UNK for an
    # in-memory dataset (report(ds) takes a Dataset, so a file is not guaranteed).
    source = ds.encoding.get("source")
    if source:
        path = Path(str(source))
        fields.append(("Source", path.name))
        try:
            fields.append(("Size", f"{path.stat().st_size / 1e6:.1f} MB"))
        except OSError:
            fields.append(("Size", "UNK"))
    else:
        fields.append(("Source", "UNK"))
    return fields


#: Placeholder mission id when the dataset carries neither an ``id`` nor a source file. Named to flag
#: the problem in the directory name and masthead rather than hide it. Two such datasets written into
#: one root collide (the second overwrites); set an ``id`` to keep them apart.
MISSING_ID = "mission (id missing)"


def mission_id(ds: xr.Dataset) -> str:
    """Return the mission identifier used as the report's subdirectory name.

    The OG1 ``id`` global attribute when present and non-empty, else the source file stem (from
    ``ds.encoding["source"]``), else :data:`MISSING_ID`. Deterministic from the data, so the report
    writes into a predictably named ``<root>/<mission_id>/`` rather than a directory the caller picks.
    """
    mid = ds.attrs.get("id")
    if mid is not None and str(mid).strip():
        return str(mid)
    source = ds.encoding.get("source")
    if source:
        return Path(str(source)).stem
    return MISSING_ID


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
        caption="Mission-mean profiles (1 m bins)",
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
    "og1_conformance": Panel(
        id="og1_conformance",
        kind="html",
        render=_guarded(
            lambda c: get_template("_og1_conformance.html").render(**metadata.conformance_data(c.ds))
        ),
    ),
    "file_contents": Panel(
        id="file_contents",
        kind="html",
        render=_guarded(lambda c: get_template("_inventory.html").render(**inventory.inventory_data(c.ds))),
    ),
}


def _sensor_row_panel(pid: str, variables: tuple[str, ...]) -> Panel:
    """Return an html panel showing the SENSOR_* catalog entries behind *variables* (page header)."""
    return Panel(
        id=pid,
        kind="html",
        render=_guarded(
            lambda c: get_template("_sensor_row.html").render(**sensors.sensor_row_data(c.ds, variables))
        ),
    )


#: Canonical variable precedence on the sensor pages — plots (and the sensor-row columns) follow it.
VARIABLE_ORDER: tuple[str, ...] = ("TEMP", "PSAL", "CNDC", "DOXY", "CHLA", "BBP700")

#: Canonical plot-type order within a sensor-page section (by the adapter's name). Panels in a
#: section are listed in this order, then by :data:`VARIABLE_ORDER`; the sections themselves run in
#: the fixed order Sensor, Sections, Drift, Dive–climb bias, Day/night offset, QC checks. Enforced by
#: ``test_sensor_page_panels_in_canonical_order``.
DIAGNOSTIC_ORDER: tuple[str, ...] = (
    "process_optics", "temporal_drift", "updown_bias", "hysteresis",
    "daynight", "quench", "global_range", "sampling_period_var",
)

#: Variable-parameterised figure panels for the sensor pages: (id, adapter, var, caption, slot).
#: Generated into PANELS below so the sensor pages do not each hand-write near-identical Panel
#: definitions. The slot is the one source of truth for the panel's width: it sets Panel.slot and is
#: passed into the adapter call, so the display width and the render width can never diverge.
_VAR_FIGURE_PANELS: tuple[tuple[str, Callable[[xr.Dataset, str], str | None], str, str, str], ...] = (
    ("oxy_hysteresis", _plots.hysteresis, "DOXY", "Dissolved oxygen hysteresis (dive vs climb)", "full"),
    ("oxy_updown", _plots.updown_bias, "DOXY", "Dissolved oxygen up/down-cast bias", "half"),
    ("oxy_drift", _plots.temporal_drift, "DOXY", "Dissolved oxygen temporal drift", "full"),
    ("oxy_global_range", _plots.global_range, "DOXY", "Dissolved oxygen global-range check", "half"),
    ("oxy_sampling", _plots.sampling_period_var, "DOXY", "Dissolved oxygen sampling period", "half"),
    ("ctd_hyst_temp", _plots.hysteresis, "TEMP", "Temperature hysteresis (dive vs climb)", "full"),
    ("ctd_hyst_psal", _plots.hysteresis, "PSAL", "Salinity hysteresis (dive vs climb)", "full"),
    ("ctd_updown_temp", _plots.updown_bias, "TEMP", "Temperature up/down-cast bias", "half"),
    ("ctd_updown_psal", _plots.updown_bias, "PSAL", "Salinity up/down-cast bias", "half"),
    ("ctd_drift_temp", _plots.temporal_drift, "TEMP", "Temperature temporal drift", "full"),
    ("ctd_drift_psal", _plots.temporal_drift, "PSAL", "Salinity temporal drift", "full"),
    ("ctd_global_temp", _plots.global_range, "TEMP", "Temperature global-range check", "half"),
    ("ctd_global_psal", _plots.global_range, "PSAL", "Salinity global-range check", "half"),
    ("ctd_daynight_psal", _plots.daynight, "PSAL", "Salinity day/night average", "half"),
    ("ctd_sampling_temp", _plots.sampling_period_var, "TEMP", "Temperature sampling period", "half"),
    ("ctd_sampling_psal", _plots.sampling_period_var, "PSAL", "Salinity sampling period", "half"),
    ("opt_process_chla", _plots.process_optics, "CHLA", "Optics assessment (deep drift and negatives)", "half"),
    ("opt_quench_chla", _plots.quench, "CHLA", "Chlorophyll quenching assessment", "full"),
    ("opt_daynight_chla", _plots.daynight, "CHLA", "Chlorophyll day/night average", "half"),
    ("opt_hyst_chla", _plots.hysteresis, "CHLA", "Chlorophyll hysteresis (dive vs climb)", "full"),
    ("opt_hyst_bbp", _plots.hysteresis, "BBP700", "Backscatter hysteresis (dive vs climb)", "full"),
    ("opt_updown_chla", _plots.updown_bias, "CHLA", "Chlorophyll up/down-cast bias", "half"),
    ("opt_updown_bbp", _plots.updown_bias, "BBP700", "Backscatter up/down-cast bias", "half"),
    ("opt_drift_chla", _plots.temporal_drift, "CHLA", "Chlorophyll temporal drift", "full"),
    ("opt_drift_bbp", _plots.temporal_drift, "BBP700", "Backscatter temporal drift", "full"),
)

PANELS["oxy_sensor"] = _sensor_row_panel("oxy_sensor", ("DOXY",))
PANELS["ctd_sensor"] = _sensor_row_panel("ctd_sensor", ("TEMP", "PSAL"))
PANELS["opt_sensor"] = _sensor_row_panel("opt_sensor", ("CHLA", "BBP700"))
PANELS["section_cndc"] = Panel(
    id="section_cndc",
    render=lambda c: _plots.section(c.ds, "CNDC"),
    caption="Conductivity section",
    applies_to=_has("CNDC"),
)
PANELS["section_bbp700"] = Panel(
    id="section_bbp700",
    render=lambda c: _plots.section(c.ds, "BBP700"),
    caption="Backscatter section",
    applies_to=_has("BBP700"),
)
PANELS["flight_vspeed"] = Panel(
    id="flight_vspeed",
    render=lambda c: _plots.vertical_speeds(c.ds),
    caption="Vertical speeds and histograms (glider dz/dt versus the flight-model velocity)",
    applies_to=_has("GLIDER_VERT_VELO_MODEL"),
)
# Convective resistance is a mixed-layer diagnostic (needs TEMP+PSAL for SIGMA_1), so it lives on
# the CTD page, not the flight page.
PANELS["ctd_cr"] = Panel(
    id="ctd_cr",
    render=lambda c: _plots.convective_resistance(c.ds),
    caption="Convective resistance for the deepest profile (mixed-layer diagnostic); the plot title names the profile number",
    applies_to=lambda c: "TEMP" in c.ds and "PSAL" in c.ds,
)
for _pid, _adapter, _var, _cap, _slot in _VAR_FIGURE_PANELS:
    # The slot sets both the Panel's display width and the adapter's render width (passed through),
    # so a narrow plot (updown_bias → half) renders small and tiles at that same width. The panel
    # applies only when its variable is present, so a page never lists a panel for an absent sibling
    # variable (a PSAL-only CTD page drops the TEMP panels rather than relying on the plotter to raise).
    PANELS[_pid] = Panel(
        id=_pid,
        render=(lambda c, a=_adapter, v=_var, s=_slot: a(c.ds, v, slot=s)),
        caption=_cap,
        slot=_slot,
        applies_to=_has(_var),
    )

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

# The inventory page: everything about the *file* rather than the mission — the OG1 global-attribute
# conformance (merged with the attribute values) and the full variable/sensor inventory. Split off
# the landing page so the landing stays about the mission (ctdcast's index/inventory division).
INVENTORY = Profile(
    entries=(
        Section(id="og1", title="Global attributes", panels=("og1_conformance",)),
        Section(id="file_contents", title="File contents", panels=("file_contents",)),
    ),
)


@dataclass(frozen=True)
class Page:
    """One output page: filename, nav title + role, the Profile it renders, and when it applies.

    ``applies_to`` decides whether the page is written for a given dataset (e.g. an oxygen page only
    when DOXY is present); ``role`` selects the nav button's CSS class (landing / component / map).
    """

    filename: str
    title: str
    role: str
    profile: Profile
    applies_to: Callable[[Ctx], bool]


# Sensor pages share a typed-subsection shape (each becomes an in-page jump-nav entry): Sensor,
# Sections (the depth–time fields), Drift, Dive–climb bias, Day/night offset, QC checks. A section
# with no panels for a given page is simply omitted.
CTD = Profile(
    entries=(
        Section(id="sensor", title="Sensor", panels=("ctd_sensor",)),
        Section(id="sections", title="Sections", panels=("ts", "section_temp", "section_psal", "section_cndc")),
        Section(id="drift", title="Drift", panels=("ctd_drift_temp", "ctd_drift_psal")),
        Section(id="bias", title="Dive–climb bias",
                panels=("ctd_updown_temp", "ctd_updown_psal", "ctd_hyst_temp", "ctd_hyst_psal")),
        Section(id="offset", title="Day/night offset", panels=("ctd_daynight_psal",)),
        Section(id="mld", title="Mixed layer", panels=("ctd_cr",)),
        Section(id="qc", title="QC checks",
                panels=("ctd_global_temp", "ctd_global_psal", "ctd_sampling_temp", "ctd_sampling_psal")),
    ),
)

OXYGEN = Profile(
    entries=(
        Section(id="sensor", title="Sensor", panels=("oxy_sensor",)),
        Section(id="sections", title="Sections", panels=("section_doxy",)),
        Section(id="drift", title="Drift", panels=("oxy_drift",)),
        Section(id="bias", title="Dive–climb bias", panels=("oxy_updown", "oxy_hysteresis")),
        Section(id="qc", title="QC checks", panels=("oxy_global_range", "oxy_sampling")),
    ),
)

OPTICS = Profile(
    entries=(
        Section(id="sensor", title="Sensor", panels=("opt_sensor",)),
        Section(id="sections", title="Sections", panels=("section_chla", "section_bbp700")),
        Section(id="drift", title="Drift", panels=("opt_process_chla", "opt_drift_chla", "opt_drift_bbp")),
        Section(id="bias", title="Dive–climb bias",
                panels=("opt_updown_chla", "opt_updown_bbp", "opt_hyst_chla", "opt_hyst_bbp")),
        Section(id="offset", title="Day/night offset", panels=("opt_daynight_chla", "opt_quench_chla")),
    ),
)

# Flight needs the glider flight-model velocity (GLIDER_VERT_VELO_MODEL), which SeaExplorer sample
# data lacks — the page applies only to datasets that carry it (e.g. Seaglider).
FLIGHT = Profile(
    entries=(Section(id="velocity", title="Vertical velocity", panels=("flight_vspeed",)),),
)


#: The report's pages. The landing page (``index.html``) and the sensor pages are the nav pills;
#: the inventory (``role="inventory"``) is linked from a strip below the masthead, not a pill, and is
#: listed last so the landing page stays first (the returned path and the nav's leading pill).
PAGES: tuple[Page, ...] = (
    Page("index.html", "Mission", "landing", PROFILE, lambda _c: True),
    Page("ctd.html", "CTD", "component", CTD, lambda c: "TEMP" in c.ds or "PSAL" in c.ds),
    Page("oxygen.html", "Oxygen", "component", OXYGEN, _has("DOXY")),
    Page("optics.html", "Optics", "component", OPTICS,
         lambda c: any(v in c.ds for v in ("CHLA", "BBP700"))),
    Page("flight.html", "Flight", "component", FLIGHT, _has("GLIDER_VERT_VELO_MODEL")),
    Page("inventory.html", "File contents", "inventory", INVENTORY, lambda _c: True),
)


def build(ds: xr.Dataset, profile: Profile) -> ResolvedReport:
    """Resolve *profile* against *ds* into a numbered report."""
    return resolve(profile, Ctx(ds=ds), PANELS)
