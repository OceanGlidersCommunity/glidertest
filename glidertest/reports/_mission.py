"""The mission report page: render context, panel registry, and section profile.

Defines what a glidertest mission report contains — a masthead meta-grid plus Metadata, Track,
Hydrography, Sampling and QC sections — by binding the plot adapters to panels and ordering them in
a profile. The page is resolved against a dataset by :func:`build`, then rendered by
:func:`glidertest.reports.report`.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import og1_attrs, qc
from . import _plots, inventory, metadata, qc_section, sensors
from ._env import get_template
from ._vendored.page_model import Panel, Profile, ResolvedReport, Section, resolve

if TYPE_CHECKING:
    from collections.abc import Callable

    import xarray as xr


@dataclass
class Ctx:
    """Render context for a page: the dataset, plus the mission facts and file inventory.

    ``facts`` and ``inv`` are computed once, on first access, and reused by every panel and
    predicate — the inventory page would otherwise rebuild the inventory for each subsection and
    again in each ``applies_to``, and ``mission_facts`` would run for the masthead, the manifest and
    the spatiotemporal table separately. One :class:`Ctx` is shared across a report's pages, so each
    is computed once per report.

    ``qc_config`` holds the resolved QC thresholds for this report (a report config's ``qc:`` block
    deep-merged over the built-in defaults, from :func:`glidertest.config.loader.resolve_qc`), or
    ``None`` for the built-in defaults. It is set once in :func:`glidertest.reports.report` and
    threaded into every QC panel, so overridden thresholds reach the checks, plots and tables and no
    panel reads the module defaults by reference. ``bbox`` is the configured region box
    (``[west, south, east, north]``) for the mission-level region test, or ``None`` when no region is
    configured (then no region row is shown).
    """

    ds: xr.Dataset
    qc_config: dict | None = None
    bbox: list | None = None

    @cached_property
    def facts(self) -> dict[str, Any]:
        """Mission summary (:func:`glidertest.reports.metadata.mission_facts`), computed once."""
        return metadata.mission_facts(self.ds)

    @cached_property
    def inv(self) -> dict[str, Any]:
        """File inventory (:func:`glidertest.reports.inventory.inventory_data`), computed once."""
        return inventory.inventory_data(self.ds)

    @cached_property
    def _sample_tests(self) -> dict[str, tuple]:
        """Per-variable QARTOD flag cache (populated by :meth:`sample_tests`)."""
        return {}

    def sample_tests(self, var: str) -> tuple:
        """Return ``qc.sample_tests(ds, var)``, computed once per variable and reused across panels.

        A QC section has ~5 figure panels per variable; without this each would recompute all three
        QARTOD tests on the full record. Threaded into the panels via ``flags=`` (see
        :mod:`glidertest.reports._plots`). Uses this report's resolved :attr:`qc_config` thresholds.
        """
        if var not in self._sample_tests:
            self._sample_tests[var] = qc.sample_tests(self.ds, var, config=self.qc_config)
        return self._sample_tests[var]

    @cached_property
    def region(self) -> dict[str, Any] | None:
        """The mission-level region-test result for this report's :attr:`bbox`, computed once.

        The qc_glidertest panel renders on both the landing and QC pages off one shared :class:`Ctx`,
        so this caches :func:`glidertest.reports.qc_section.region_result` rather than scanning the full
        position arrays twice per report. ``None`` when no region box is configured.
        """
        return qc_section.region_result(self.ds, self.bbox) if self.bbox is not None else None


def _deg_range(lo: float, hi: float, pos: str, neg: str) -> str:
    """Format a degree range with per-bound hemisphere letters, e.g. ``58.12°N–59.03°N``.

    *pos*/*neg* are the hemisphere letters for non-negative/negative values (``"N"``/``"S"`` for
    latitude, ``"E"``/``"W"`` for longitude), so a negative value is never mislabelled.
    """

    def one(v: float) -> str:
        return f"{abs(v):.2f}°{pos if v >= 0 else neg}"

    return f"{one(lo)}–{one(hi)}"


def header_card(ds: xr.Dataset, facts: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    """Return (label, value) pairs for the masthead meta-grid: platform serial plus a mission summary.

    The overview statistics come from :func:`glidertest.reports.metadata.mission_facts` (the same
    source the manifest reads, so the two cannot disagree). A field whose source is absent is skipped;
    one that cannot be computed shows ``UNK`` (or ``—`` for profile counts when PROFILE_NUMBER is
    all-NaN). The file size is read here; the file name itself is the masthead subtitle.

    *facts* is a prebuilt :func:`mission_facts` result; when None it is computed from *ds*.
    """
    f = metadata.mission_facts(ds) if facts is None else facts
    fields: list[tuple[str, str]] = []
    if "PLATFORM_SERIAL_NUMBER" in ds:
        fields.append(("Platform serial", f["platform_serial"] or "UNK"))
    if "PROFILE_NUMBER" in ds:
        if f["n_profiles"] is None:
            fields.append(("Profiles", "—"))  # all-NaN PROFILE_NUMBER
        elif f["n_dive"] is not None:
            fields.append(("Profiles", f"{f['n_profiles']} ({f['n_dive']} dive, {f['n_climb']} climb)"))
        else:
            fields.append(("Profiles", str(f["n_profiles"])))
    if "TIME" in ds:
        if f["t0"] is None:
            fields += [("Start", "UNK"), ("End", "UNK"), ("Duration", "UNK")]
        else:
            fields.append(("Start", metadata.iso_minute(f["t0"]).replace("T", " ")))
            fields.append(("End", metadata.iso_minute(f["t1"]).replace("T", " ")))
            fields.append(("Duration", metadata.fmt_duration(f["duration_s"])))
        dt = np.diff(np.asarray(ds["TIME"].values))  # timedelta64; NaT where either end is NaT
        valid = dt[~np.isnat(dt)]
        med = np.median(valid.astype("timedelta64[s]").astype(float)) if valid.size else np.nan
        sampling = f"{med:.0f} s" if np.isfinite(med) else "UNK"
    if f["depth_min"] is not None:
        fields.append(("Dive depth", f"{int(f['depth_min'])}–{int(f['depth_max'])} m"))
    if f["lat_min"] is not None:
        fields.append(("Lat", _deg_range(f["lat_min"], f["lat_max"], "N", "S")))
        fields.append(("Lon", _deg_range(f["lon_min"], f["lon_max"], "E", "W")))
    if sampling is not None:  # sampling period sits after lat/lon in the masthead grid
        fields.append(("Sampling", sampling))
    if "N_MEASUREMENTS" in ds.sizes:
        fields.append(("Records", f"{f['n_records']:,}"))
    source = ds.encoding.get("source")
    if source:
        try:
            fields.append(("Size", f"{Path(str(source)).stat().st_size / 1e6:.1f} MB"))
        except OSError:
            fields.append(("Size", "UNK"))
    return fields


#: Placeholder mission id when the dataset carries neither an ``id`` nor a source file. Named to flag
#: the problem in the directory name and masthead rather than hide it. Two such datasets written into
#: one root collide (the second overwrites); set an ``id`` to keep them apart.
MISSING_ID = "mission (id missing)"


def _safe_dirname(name: str) -> str:
    """Neutralise path separators and traversal in *name* so it is safe as a directory component.

    The mission id comes from dataset metadata (the OG1 ``id`` attribute), which is untrusted: an
    ``id`` like ``../../etc`` or ``/abs`` would otherwise let the report escape its root. Path
    separators and NULs are replaced with ``_``; a name that is empty or only dots falls back to
    :data:`MISSING_ID`. Ordinary ids (``sea045_20230604T1253_delayed``) are returned unchanged.
    """
    slug = name.replace("/", "_").replace("\\", "_").replace("\x00", "").strip()
    if slug in ("", ".", "..") or slug.startswith("~"):
        return MISSING_ID
    return slug


def mission_id(ds: xr.Dataset) -> str:
    """Return the mission identifier used as the report's subdirectory name.

    The OG1 ``id`` global attribute when present and non-empty, else the source file stem (from
    ``ds.encoding["source"]``), else :data:`MISSING_ID`; always passed through :func:`_safe_dirname`
    so an ``id`` from untrusted metadata cannot escape the report root. Deterministic from the data,
    so the report writes into a predictably named ``<root>/<mission_id>/``, not a caller-chosen dir.
    """
    mid = ds.attrs.get("id")
    if mid is not None and str(mid).strip():
        return _safe_dirname(str(mid))
    source = ds.encoding.get("source")
    if source:
        return _safe_dirname(Path(str(source)).stem)
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


def _qc_or_region_applies(c: Ctx) -> bool:
    """The glidertest-diagnostics section/page applies when QC applies or a region box is configured.

    The mission-level region test is position-only, so it must reach the page even for a file with a
    configured ``region.bbox`` but no standard QC variable (then the only row shown is the region one).
    """
    return _qc_applies(c) or c.bbox is not None


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
    "ts": Panel(
        id="ts",
        render=lambda c: _plots.ts(c.ds),
        caption="Temperature–salinity diagram",
        applies_to=lambda c: "TEMP" in c.ds and "PSAL" in c.ds,
    ),
    "section_temp": Panel(
        id="section_temp",
        render=lambda c: _plots.section(c.ds, "TEMP"),
        caption="Temperature section",
        applies_to=_has("TEMP"),
    ),
    "section_psal": Panel(
        id="section_psal",
        render=lambda c: _plots.section(c.ds, "PSAL"),
        caption="Salinity section",
        applies_to=_has("PSAL"),
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
    "qc_delivered": Panel(
        id="qc_delivered",
        kind="html",
        render=_guarded(
            lambda c: get_template("_qc_delivered.html").render(**qc_section.delivered_data(c.ds))
        ),
    ),
    "qc_glidertest": Panel(
        id="qc_glidertest",
        kind="html",
        render=_guarded(
            lambda c: get_template("_qc_glidertest.html").render(
                **qc_section.diagnostics_data(c.ds, c.qc_config, c.bbox, region=c.region)
            )
        ),
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
    ("oxy_sampling", _plots.sampling_period_var, "DOXY", "Dissolved oxygen sampling period", "half"),
    ("ctd_hyst_temp", _plots.hysteresis, "TEMP", "Temperature hysteresis (dive vs climb)", "full"),
    ("ctd_hyst_psal", _plots.hysteresis, "PSAL", "Salinity hysteresis (dive vs climb)", "full"),
    ("ctd_updown_temp", _plots.updown_bias, "TEMP", "Temperature up/down-cast bias", "half"),
    ("ctd_updown_psal", _plots.updown_bias, "PSAL", "Salinity up/down-cast bias", "half"),
    ("ctd_drift_temp", _plots.temporal_drift, "TEMP", "Temperature temporal drift", "full"),
    ("ctd_drift_psal", _plots.temporal_drift, "PSAL", "Salinity temporal drift", "full"),
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
    ("opt_sampling_chla", _plots.sampling_period_var, "CHLA", "Chloropyll sampling period", "half"),
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
    render=lambda c: _plots.convective_resistance(c.ds, slot="half"),
    caption="Convective resistance for the deepest profile (mixed-layer diagnostic); the plot title names the profile number",
    slot="half",
    applies_to=lambda c: "TEMP" in c.ds and "PSAL" in c.ds,
)
# QC page (#295): per-variable figure panels visualising where each glidertest QC test fires and
# whether a test fits the sampling (the flat-line / spike diagnostics). Same (id, adapter, var,
# caption, slot) shape as the sensor panels, so they go through the same loop below.
_QC_FIGURE_PANELS: tuple[tuple[str, Callable[[xr.Dataset, str], str | None], str, str, str], ...] = tuple(
    panel
    for _v in _QC_VARS
    for panel in (
        (f"qc_{_v}_hits", _plots.test_hits, _v,
         f"{_v}: where glidertest's QC tests fire (gross range, spike, flat line) over the full record", "full"),
        (f"qc_{_v}_profile", _plots.flag_profile, _v,
         f"{_v} against depth: where the QC tests fire in the water column", "half"),
        (f"qc_{_v}_flat", _plots.flat_runs, _v,
         f"{_v}: flat-run durations against the test windows and the median sampling interval", "half"),
        (f"qc_{_v}_change", _plots.flat_change, _v,
         f"{_v}: start-to-end change of each flat run against the tolerance (stuck sensor vs slow signal)", "half"),
        (f"qc_{_v}_excerpt", _plots.flat_excerpt, _v,
         f"{_v}: zoom on the longest flagged flat run, with the ±tolerance band", "half"),
        (f"qc_{_v}_spike", _plots.spike_stat, _v,
         f"{_v}: spike statistic against the suspect and fail thresholds", "half"),
    )
)

# Gross range on the T–S plane (once per page; needs TEMP and PSAL): the suspect/fail spans as
# nested rectangles over the cloud, so a reader sees whether the limits are regional or the hits are
# outliers.
PANELS["qc_ts_gross"] = Panel(
    id="qc_ts_gross",
    render=lambda c: _plots.ts_gross_range(c.ds, slot="half", config=c.qc_config),
    caption="Gross-range suspect and fail spans on the T–S diagram (TEMP × PSAL)",
    slot="half",
    applies_to=lambda c: "TEMP" in c.ds and "PSAL" in c.ds,
)
# The suggested gross-range limits for TEMP and PSAL, as a half-slot note beside the T–S plot.
PANELS["qc_gross_ts_note"] = Panel(
    id="qc_gross_ts_note",
    kind="html",
    render=_guarded(lambda c: qc_section.gross_range_ts_note(c.ds, c.qc_config)),
    slot="half",
    applies_to=lambda c: "TEMP" in c.ds and "PSAL" in c.ds,
)

# Per-variable "suggested limits" note (gross range + flat line) derived from this mission's data —
# an html panel heading each variable's QC section; forward to a config-yaml snippet (#267).
for _v in _QC_VARS:
    PANELS[f"qc_{_v}_suggest"] = Panel(
        id=f"qc_{_v}_suggest",
        kind="html",
        render=_guarded(lambda c, var=_v: qc_section.suggested_limits_note(c.ds, var, c.qc_config)),
        slot="half",  # sits beside the half-width profile, as the gross-range note sits beside the T–S plot
        applies_to=_has(_v),
    )


# Unnumbered h3 sub-headings inside a variable's QC section — grouping, not navigation (the vendored
# _manifest numbers sections flatly and gives one jump-nav entry per Section, so a true 6.1/6.2 would
# need an upstream engine change). An html panel at full width forces its own line in the fig-row.
def _subheading(text: str) -> Callable[[Ctx], str]:
    """Return an html-panel render emitting *text* as a full-width ``<h3>`` QC sub-heading."""
    return lambda _c: f'<h3 class="qc-subsection">{text}</h3>'


for _v in _QC_VARS:
    PANELS[f"qc_{_v}_sub_flat"] = Panel(id=f"qc_{_v}_sub_flat", kind="html",
                                        render=_subheading("Flat-line runs"), applies_to=_has(_v))
    PANELS[f"qc_{_v}_sub_spike"] = Panel(id=f"qc_{_v}_sub_spike", kind="html",
                                         render=_subheading("Spike"), applies_to=_has(_v))

# Link from the landing page's QC summary to the full QC page (only on the landing — the QC page's
# own glidertest section does not carry it).
PANELS["full_qc_link"] = Panel(
    id="full_qc_link",
    kind="html",
    render=lambda c: (
        '<p class="caption"><a href="qc.html">Full QC →</a> per-variable diagnostics: where each '
        "glidertest test fires, and whether its thresholds fit the sampling.</p>"
    ),
    applies_to=_qc_or_region_applies,
)

# global_range is a sensor-page adapter that consults the gross-range span (for the histogram extent),
# so it declares a `config` parameter; it goes through the same loop as the other figure panels.
_GLOBAL_RANGE_PANELS: tuple[tuple[str, Callable[..., str | None], str, str, str], ...] = (
    ("oxy_global_range", _plots.global_range, "DOXY", "Dissolved oxygen global-range check", "half"),
    ("ctd_global_temp", _plots.global_range, "TEMP", "Temperature global-range check", "half"),
    ("ctd_global_psal", _plots.global_range, "PSAL", "Salinity global-range check", "half"),
    ("opt_global_chla", _plots.global_range, "CHLA", "Chlorophyll-a global-range check", "half"),
)


def _fig_render(
    adapter: Callable[..., str | None], var: str, slot: str, needs: tuple[str, ...]
) -> Callable[[Ctx], str | None]:
    """Return a Panel render callable that threads *var*/*slot* and the Ctx values named in *needs*.

    *needs* declares explicitly which of ``config`` (this report's resolved thresholds) and ``flags``
    (the variable's cached QARTOD flags) the adapter takes — a declaration on the panel group, not
    runtime signature introspection, so an adapter cannot silently stop receiving config when it is
    renamed or wrapped (the silent-default class this registration exists to remove).
    """
    wants_config, wants_flags = "config" in needs, "flags" in needs

    def render(c: Ctx) -> str | None:
        kwargs: dict[str, Any] = {}
        if wants_config:
            kwargs["config"] = c.qc_config
        if wants_flags:
            kwargs["flags"] = c.sample_tests(var)
        return adapter(c.ds, var, slot=slot, **kwargs)

    return render


# One registration loop for every variable-parameterised figure panel, each group declaring the Ctx
# values its adapters take (sensor figures need nothing; global_range needs config; the QC panels need
# config + the variable's cached flags) — explicit, not introspected. The slot sets both the Panel's
# display width and the adapter's render width, and each panel applies only when its variable is present.
for _group, _needs in ((_VAR_FIGURE_PANELS, ()), (_GLOBAL_RANGE_PANELS, ("config",)), (_QC_FIGURE_PANELS, ("config", "flags"))):
    for _pid, _adapter, _var, _cap, _slot in _group:
        PANELS[_pid] = Panel(
            id=_pid,
            render=_fig_render(_adapter, _var, _slot, _needs),
            caption=_cap,
            slot=_slot,
            applies_to=_has(_var),
        )

PROFILE = Profile(
    entries=(
        Section(id="track", title="Track", panels=("track",)),
        Section(id="payload", title="Payload", panels=("metadata",)),
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
            id="qc_delivered",
            title="QC — as delivered",
            panels=("qc_delivered",),
            applies_to=_qc_applies,
        ),
        Section(
            id="qc_glidertest",
            title="QC — glidertest diagnostics",
            panels=("qc_glidertest", "full_qc_link"),
            applies_to=_qc_or_region_applies,
        ),
    ),
)

# The inventory page: everything about the *file* rather than the mission. One Section per
# subsection, so each gets its own jump-nav entry (the bar is one-entry-per-Section). Each renders a
# thin slice of the existing builders (metadata.attr_category_data / inventory.inventory_slice). The
# four OG1 attribute categories always render — an empty category with its mandatory rows in amber is
# the strongest finding on the page — so only Other / other-dimension variables / scalars carry
# applies_to; the sensor catalog stays with a stub reason when a file has no SENSOR_* variables.
_ATTR_CATEGORY_IDS: dict[str, str] = {
    "Identity & discovery": "og1_identity",
    "Spatiotemporal coverage": "og1_coverage",
    "People & institutions": "og1_people",
    "Provenance & processing": "og1_provenance",
    og1_attrs.OTHER_GROUP: "og1_other",
}

_ATTR_SECTIONS: list[Section] = []
for _title, _ in og1_attrs.ATTR_GROUPS:
    _sid = _ATTR_CATEGORY_IDS[_title]
    PANELS[_sid] = Panel(
        id=_sid,
        kind="html",
        render=_guarded(
            lambda c, t=_title: get_template("_og1_conformance.html").render(
                **metadata.attr_category_data(c.ds, t, c.facts)
            )
        ),
    )
    _ATTR_SECTIONS.append(Section(id=_sid, title=_title, panels=(_sid,)))

PANELS["og1_other"] = Panel(
    id="og1_other",
    kind="html",
    render=_guarded(
        lambda c: get_template("_og1_conformance.html").render(
            **metadata.attr_category_data(c.ds, og1_attrs.OTHER_GROUP)
        )
    ),
    applies_to=lambda c: metadata.has_other_attrs(c.ds),
)


def _inv_render(which: str) -> Callable[[Ctx], str | None]:
    """Return an html-panel render for one inventory slice (coords / measurements / …)."""
    return _guarded(
        lambda c: get_template("_inventory.html").render(**inventory.inventory_slice(c.ds, which, data=c.inv))
    )


PANELS["coords"] = Panel(id="coords", kind="html", render=_inv_render("coords"))
PANELS["measurements"] = Panel(id="measurements", kind="html", render=_inv_render("measurements"))
PANELS["other_dims"] = Panel(
    id="other_dims",
    kind="html",
    render=_inv_render("other_dims"),
    applies_to=lambda c: bool(inventory.inventory_slice(c.ds, "other_dims", data=c.inv)["groups"]),
)
PANELS["scalars"] = Panel(
    id="scalars",
    kind="html",
    render=_inv_render("scalars"),
    applies_to=lambda c: bool(inventory.inventory_slice(c.ds, "scalars", data=c.inv)["groups"]),
)
PANELS["sensors"] = Panel(
    id="sensors",
    kind="html",
    render=_inv_render("sensors"),
    unavailable_if=lambda c: (
        None
        if inventory.inventory_slice(c.ds, "sensors", data=c.inv)["sensors"]
        else "no SENSOR_* variables in the file"
    ),
)

INVENTORY = Profile(
    entries=(
        *_ATTR_SECTIONS,
        Section(id="og1_other", title=og1_attrs.OTHER_GROUP, panels=("og1_other",)),
        Section(id="coords", title="Coordinates", panels=("coords",)),
        Section(id="measurements", title="Variables on N_MEASUREMENTS", panels=("measurements",)),
        Section(id="other_dims", title="Variables on other dimensions", panels=("other_dims",)),
        Section(id="scalars", title="Scalar variables", panels=("scalars",)),
        Section(id="sensors", title="Sensor catalog", panels=("sensors",)),
    ),
)


@dataclass(frozen=True)
class Page:
    """One output page: filename, nav title + role, the Profile it renders, and when it applies.

    ``applies_to`` decides whether the page is written for a given dataset (e.g. an oxygen page only
    when DOXY is present); ``role`` is the pill's ``ROLE_ACCENT`` colour role; ``nav_group`` is the
    masthead nav row it sits in (``summary`` / ``reports`` / ``derived`` / ``inventory``);
    ``type_label`` is the masthead's top-right page label ("Mission report", "CTD", "netCDF Inventory").
    """

    filename: str
    title: str
    type_label: str
    role: str
    nav_group: str
    profile: Profile
    applies_to: Callable[[Ctx], bool]
    #: Optional page intro rendered above the jump-nav (page-level prose that belongs to no one
    #: section, e.g. the inventory page's OG1 verdict and variable counts). ``Ctx -> html``.
    lead: Callable[[Ctx], str] | None = None


# Sensor pages share a typed-subsection shape (each becomes an in-page jump-nav entry): Sensor,
# Sections (the depth–time fields), Drift, Dive–climb bias, Day/night offset, QC checks. A section
# with no panels for a given page is simply omitted.
CTD = Profile(
    entries=(
        Section(id="sensor", title="Sensor", panels=("ctd_sensor",)),
        Section(id="sections", title="Sections", panels=("section_temp", "section_psal", "section_cndc")),
        Section(id="ts", title="T–S", panels=("ts",)),
        Section(id="drift", title="Drift",
                intro="Temporal drift — each variable's evolution in time over the mission.",
                panels=("ctd_drift_temp", "ctd_drift_psal")),
        Section(id="bias", title="Dive–climb bias",
                panels=("ctd_updown_temp", "ctd_updown_psal", "ctd_hyst_temp", "ctd_hyst_psal")),
        Section(id="offset", title="Day/night offset", panels=("ctd_daynight_psal",)),
        Section(id="mld", title="Mixed layer", panels=("ctd_cr",)),
        Section(id="qc", title="QC checks", panels=("ctd_global_temp", "ctd_global_psal")),
        Section(id="sample_rate", title="Sample rate",
                panels=("ctd_sampling_temp", "ctd_sampling_psal")),
    ),
)

OXYGEN = Profile(
    entries=(
        Section(id="sensor", title="Sensor", panels=("oxy_sensor",)),
        Section(id="sections", title="Sections", panels=("section_doxy",)),
        Section(id="drift", title="Drift", panels=("oxy_drift",)),
        Section(id="bias", title="Dive–climb bias", panels=("oxy_updown", "oxy_hysteresis")),
        Section(id="qc", title="QC checks", panels=("oxy_global_range",)),
        Section(id="sample_rate", title="Sample rate", panels=("oxy_sampling",)),
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
        Section(id="qc", title="QC checks", panels=("opt_global_chla",)),
        Section(id="sample_rate", title="Sample rate", panels=("opt_sampling_chla",)),
    ),
)

# Flight needs the glider flight-model velocity (GLIDER_VERT_VELO_MODEL), which SeaExplorer sample
# data lacks — the page applies only to datasets that carry it (e.g. Seaglider).
FLIGHT = Profile(
    entries=(Section(id="velocity", title="Vertical velocity", panels=("flight_vspeed",)),),
)

# The QC page (#295): the QC glidertest *applies* and why it flagged — the glidertest diagnostics
# matrix (same panel the landing page shows, one data build, so the two cannot drift), with
# per-variable "where it fires / why" plot sections added below (see feat-qc-page-plan.md). The
# as-delivered census stays on the landing page only (it is the provider's QC, not glidertest's).
QC = Profile(
    entries=(
        Section(
            id="qc_glidertest",
            title="QC — glidertest diagnostics",
            panels=("qc_glidertest",),
            applies_to=_qc_or_region_applies,
        ),
        Section(
            id="qc_gross_ts",
            title="Gross range (T–S)",
            panels=("qc_ts_gross", "qc_gross_ts_note"),
            applies_to=lambda c: "TEMP" in c.ds and "PSAL" in c.ds,
        ),
        *(
            Section(
                id=f"qc_{_v}",
                title=f"QC — {_v}",
                panels=(
                    f"qc_{_v}_hits", f"qc_{_v}_profile", f"qc_{_v}_suggest",
                    f"qc_{_v}_sub_flat", f"qc_{_v}_flat", f"qc_{_v}_change", f"qc_{_v}_excerpt",
                    f"qc_{_v}_sub_spike", f"qc_{_v}_spike",
                ),
                applies_to=_has(_v),
            )
            for _v in _QC_VARS
        ),
    ),
)


def _inventory_lead(ctx: Ctx) -> str:
    """Inventory page intro (above the jump-nav): the OG1 verdict and the variable/QC counts."""
    return get_template("_inventory_intro.html").render(
        verdict=metadata.verdict_line(ctx.ds), **ctx.inv
    )


#: The report's pages. The landing page (``index.html``) and the sensor pages are the nav pills;
#: the inventory (``role="inventory"``) is linked from a strip below the masthead, not a pill, and is
#: listed last so the landing page stays first (the returned path and the nav's leading pill).
# Pill colours (the role) per nav group — bright and varied in the oceanarray style: the mission
# summary blue, the sensor reports green, the derived flight purple. (Provisional: the role=colour
# mapping is reconciled when the masthead nav is vendored, per the shared-nav plan §3.2.)
PAGES: tuple[Page, ...] = (
    Page("index.html", "Mission", "Mission report", "landing", "summary", PROFILE, lambda _c: True),
    Page("qc.html", "QC", "QC", "aggregate-a", "summary", QC, _qc_or_region_applies),
    Page("ctd.html", "CTD", "CTD", "aggregate-b", "reports", CTD,
         lambda c: "TEMP" in c.ds or "PSAL" in c.ds),
    Page("oxygen.html", "Oxygen", "Oxygen", "aggregate-b", "reports", OXYGEN, _has("DOXY")),
    Page("optics.html", "Optics", "Optics", "aggregate-b", "reports", OPTICS,
         lambda c: any(v in c.ds for v in ("CHLA", "BBP700"))),
    Page("flight.html", "Flight", "Flight", "aggregate-a", "derived", FLIGHT,
         _has("GLIDER_VERT_VELO_MODEL")),
    Page("inventory.html", "Inventory", "netCDF Inventory", "inventory", "inventory", INVENTORY,
         lambda _c: True, lead=_inventory_lead),
)


def build(ds: xr.Dataset, profile: Profile, *, ctx: Ctx | None = None) -> ResolvedReport:
    """Resolve *profile* against *ds* into a numbered report.

    *ctx* is a shared :class:`Ctx` to resolve against; when None a fresh one is built from *ds*. The
    report passes one ``Ctx`` across all pages so ``mission_facts`` and the inventory are built once.
    """
    return resolve(profile, ctx or Ctx(ds=ds), PANELS)
