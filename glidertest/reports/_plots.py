"""Plot adapters: glidertest plotters → base64 report panels at the slot width.

Each adapter unwraps the plotter's ``(fig, ax)`` return to the Figure the slot layer needs and
routes it through :func:`glidertest.reports._slots.render`. Adapters never pass a width to the
plotter — glidertest's plot functions do not accept one; the slot layer forces the width after the
draw. Each returns a base64 PNG, or ``None`` when the plot could not be produced (the panel then
drops out of the page).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .. import plots, qc, tools
from . import _slots

if TYPE_CHECKING:
    import xarray as xr


def track(ds: xr.Dataset) -> str | None:
    """Render the glider track map panel."""
    return _slots.render(lambda: plots.plot_glider_track(ds)[0], source="plot_glider_track", optional=True)


def basic_vars(ds: xr.Dataset) -> str | None:
    """Render the depth-profile panel of the core variables (mission-mean profiles)."""
    return _slots.render(lambda: plots.plot_basic_vars(ds)[0], source="plot_basic_vars", optional=True)


def ts(ds: xr.Dataset) -> str | None:
    """Render the temperature–salinity diagram panel."""
    return _slots.render(lambda: plots.plot_ts(ds)[0], source="plot_ts", optional=True)


def max_depth(ds: xr.Dataset) -> str | None:
    """Render the maximum-depth-per-profile panel."""
    return _slots.render(lambda: plots.plot_max_depth_per_profile(ds)[0], source="plot_max_depth_per_profile", optional=True)


def section(ds: xr.Dataset, var: str) -> str | None:
    """Render a depth–time section panel for *var* over the whole mission (pcolormesh).

    ``plot_section`` returns ``(ax, cbar, time_ax)`` rather than ``(fig, ax)``, so the figure is
    taken from the axes.
    """
    return _slots.render(
        lambda: plots.plot_section(ds, vars=[var], method="pcolormesh")[0].get_figure(), source="plot_section", optional=True
    )


def grid_spacing(ds: xr.Dataset) -> str | None:
    """Render the horizontal/vertical grid-spacing panel."""
    return _slots.render(lambda: plots.plot_grid_spacing(ds)[0], source="plot_grid_spacing", optional=True)


def sampling_period(ds: xr.Dataset) -> str | None:
    """Render the sampling-period panel."""
    return _slots.render(lambda: plots.plot_sampling_period_all(ds)[0], source="plot_sampling_period_all", optional=True)


def prof_monotony(ds: xr.Dataset) -> str | None:
    """Render the profile-number monotonicity panel."""
    return _slots.render(lambda: plots.plot_prof_monotony(ds)[0], source="plot_prof_monotony", optional=True)


# --- variable-parameterised adapters (sensor pages) ---------------------------------------------


def hysteresis(ds: xr.Dataset, var: str, slot: str = "full") -> str | None:
    """Render the dive–climb hysteresis panel for *var*."""
    return _slots.render(lambda: plots.plot_hysteresis(ds, var=var)[0], slot=slot, source="plot_hysteresis", optional=True)


def updown_bias(ds: xr.Dataset, var: str, slot: str = "half") -> str | None:
    """Render the up/down-cast bias panel for *var* (a narrow profile plot, half width by default)."""
    return _slots.render(lambda: plots.plot_updown_bias(ds, var=var)[0], slot=slot, source="plot_updown_bias", optional=True)


def temporal_drift(ds: xr.Dataset, var: str, slot: str = "full") -> str | None:
    """Render the temporal-drift panel for *var*."""
    return _slots.render(lambda: plots.check_temporal_drift(ds, var=var)[0], slot=slot, source="check_temporal_drift", optional=True)


def global_range(ds: xr.Dataset, var: str, slot: str = "half") -> str | None:
    """Render the global-range histogram for *var*, using its qc.configs suspect span as the range.

    The plotter's default range is oxygen-shaped (−5..600); using the per-variable suspect span puts
    the limit lines where they belong. Falls back to the plotter default if *var* has no config.
    """
    span = qc.configs.get(var, {}).get("gross_range_test", {}).get("suspect_span")
    kw = {"min_val": span[0], "max_val": span[1]} if span else {}
    return _slots.render(
        lambda: plots.plot_global_range(ds, var=var, **kw)[0],
        slot=slot,
        source="plot_global_range",
        optional=True,
    )


def sampling_period_var(ds: xr.Dataset, var: str, slot: str = "half") -> str | None:
    """Render the per-variable sampling-period panel for *var* (plotter returns the axes)."""
    return _slots.render(
        lambda: plots.plot_sampling_period(ds, variable=var).get_figure(), slot=slot, source="plot_sampling_period", optional=True
    )


def daynight(ds: xr.Dataset, var: str, slot: str = "half") -> str | None:
    """Render the day/night average panel for *var*."""
    return _slots.render(lambda: plots.plot_daynight_avg(ds, var=var)[0], slot=slot, source="plot_daynight_avg", optional=True)


def quench(ds: xr.Dataset, var: str, slot: str = "full") -> str | None:
    """Render the chlorophyll quenching-assessment panel for *var*."""
    return _slots.render(lambda: plots.plot_quench_assess(ds, var)[0], slot=slot, source="plot_quench_assess", optional=True)


def process_optics(ds: xr.Dataset, var: str, slot: str = "half") -> str | None:
    """Render the optics-assessment panel (deep drift and negatives) for *var*."""
    return _slots.render(lambda: plots.process_optics_assess(ds, var=var)[0], slot=slot, source="process_optics_assess", optional=True)


# --- flight (vertical velocity) ----------------------------------------------------------------


def vertical_speeds(ds: xr.Dataset, slot: str = "full") -> str | None:
    """Render the vertical-speeds + histograms panel.

    The plotter needs the measured dz/dt velocity (``GLIDER_VERT_VELO_DZDT``), the glider
    flight-model velocity (``GLIDER_VERT_VELO_MODEL``), and their difference the vertical seawater
    velocity (``VERT_CURR_MODEL``). ``calc_w_meas`` adds the first from ``DEPTH``/``TIME``;
    ``calc_w_sw`` derives ``VERT_CURR_MODEL`` from the model and dz/dt velocities. The panel is
    gated (in ``PAGES``/``_VAR_FIGURE_PANELS``) on ``GLIDER_VERT_VELO_MODEL``, which both calcs need,
    so it drops out for datasets without the flight-model velocity.
    """
    return _slots.render(
        lambda: plots.plot_vertical_speeds_with_histograms(tools.calc_w_sw(tools.calc_w_meas(ds)))[0],
        slot=slot,
        source="plot_vertical_speeds_with_histograms",
        optional=True,
    )


# --- mixed layer (convective resistance) -------------------------------------------------------


def convective_resistance(ds: xr.Dataset, slot: str = "full") -> str | None:
    """Render the convective-resistance (mixed-layer) panel for the deepest profile.

    Picks the profile reaching the greatest ``DEPTH`` — deterministic, and CR is most meaningful on
    a full-depth profile. Computes ``SIGMA_1`` first (via :func:`glidertest.tools.add_sigma_1`),
    which ``plot_CR`` needs. Returns ``None`` when there is no finite profile to pick, or when CR
    cannot be computed for the chosen profile (so the panel drops rather than rendering blank).
    """

    def draw() -> object | None:
        pnum = np.asarray(ds["PROFILE_NUMBER"].values)
        depth = np.asarray(ds["DEPTH"].values)
        finite = np.isfinite(pnum) & np.isfinite(depth)
        if not finite.any():
            return None
        pnum, depth = pnum[finite], depth[finite]
        profiles = np.unique(pnum)
        rep = int(profiles[np.argmax([np.nanmax(depth[pnum == p]) for p in profiles])])
        # add_sigma_1 assigns SIGMA_1 in place; copy first so the caller's dataset is never modified
        # (diagnose-only contract). Deeper fix — make tools.add_sigma_1 non-mutating — is for tools.py.
        ds2 = tools.add_sigma_1(ds.copy())
        profile = ds2.where(ds2["PROFILE_NUMBER"] == rep, drop=True)
        if tools.calculate_CR_for_all_depth(profile).empty:
            return None  # no computable CR on the deepest profile -> drop the panel
        return plots.plot_CR(ds2, profile_num=rep)[0]

    return _slots.render(draw, slot=slot, source="plot_CR", optional=True)
