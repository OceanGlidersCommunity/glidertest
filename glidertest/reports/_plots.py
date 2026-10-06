"""Plot adapters: glidertest plotters → base64 report panels at the slot width.

Each adapter unwraps the plotter's ``(fig, ax)`` return to the Figure the slot layer needs and
routes it through :func:`glidertest.reports._slots.render`. Adapters never pass a width to the
plotter — glidertest's plot functions do not accept one; the slot layer forces the width after the
draw. Each returns a base64 PNG, or ``None`` when the plot could not be produced (the panel then
drops out of the page).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import plots
from . import _slots

if TYPE_CHECKING:
    import xarray as xr


def track(ds: xr.Dataset) -> str | None:
    """Render the glider track map panel."""
    return _slots.render(lambda: plots.plot_glider_track(ds)[0], optional=True)


def basic_vars(ds: xr.Dataset) -> str | None:
    """Render the depth–time panel of the core variables."""
    return _slots.render(lambda: plots.plot_basic_vars(ds)[0], optional=True)


def ts(ds: xr.Dataset) -> str | None:
    """Render the temperature–salinity diagram panel."""
    return _slots.render(lambda: plots.plot_ts(ds)[0], optional=True)


def max_depth(ds: xr.Dataset) -> str | None:
    """Render the maximum-depth-per-profile panel."""
    return _slots.render(lambda: plots.plot_max_depth_per_profile(ds)[0], optional=True)


def section(ds: xr.Dataset, var: str) -> str | None:
    """Render a depth–time section panel for *var* over the whole mission (pcolormesh).

    ``plot_section`` returns ``(ax, cbar, time_ax)`` rather than ``(fig, ax)``, so the figure is
    taken from the axes.
    """
    return _slots.render(
        lambda: plots.plot_section(ds, var, method="pcolormesh")[0].get_figure(), optional=True
    )


def grid_spacing(ds: xr.Dataset) -> str | None:
    """Render the horizontal/vertical grid-spacing panel."""
    return _slots.render(lambda: plots.plot_grid_spacing(ds)[0], optional=True)


def sampling_period(ds: xr.Dataset) -> str | None:
    """Render the sampling-period panel."""
    return _slots.render(lambda: plots.plot_sampling_period_all(ds)[0], optional=True)


def prof_monotony(ds: xr.Dataset) -> str | None:
    """Render the profile-number monotonicity panel."""
    return _slots.render(lambda: plots.plot_prof_monotony(ds)[0], optional=True)
