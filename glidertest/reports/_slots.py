"""Slot layer: render a plot through the report style at a fixed slot width.

The report builds every figure under its own style spec. Each plotting function wraps its body in
``plt.style.context(plots._style())`` — an inner context that overrides any outer one — so the only
way the report's style reaches a figure is to set ``plots._ACTIVE_STYLE`` for the duration of the
draw. :func:`render` does that, forces the figure to the slot width, and encodes it through the
vendored encoder. Writing the PNG to disk is the page builder's job, keyed on the panel id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from .. import plots
from ..config.report_tokens import MPLSTYLE_PATH, SLOTS
from . import _figdebug

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure

#: Default figure aspect (height / width) for bare ``plt.subplots()`` figures. The plotters lay
#: themselves out against the package ``glidertest.mplstyle`` (``figure.figsize: 14, 10``), so bare
#: figures are created at that 10/14 proportion to keep their authored spacing; the report overrides
#: only the width. Figures that set their own size keep their own aspect through :func:`_force_width`.
_DEFAULT_ASPECT: float = 10 / 14


def _report_spec(width_in: float) -> list:
    """Return the style spec for a *width_in*-inch figure: the report mplstyle plus a width override."""
    return [str(MPLSTYLE_PATH), {"figure.figsize": (width_in, width_in * _DEFAULT_ASPECT)}]


def _force_width(fig: Figure | None, width_in: float) -> Figure | None:
    """Resize *fig* to *width_in* inches keeping its aspect, and opt it out of ``tight_layout``.

    glidertest's plotters hand-lay their figures (``subplots_adjust``, colorbar ``set_position``),
    which the vendored encoder's ``tight_layout`` would undo — resetting hand-placed axes and pushing
    colorbar labels off the canvas. Setting ``fig._manual_layout`` makes the encoder skip
    ``tight_layout`` (it honours that flag); glidertest plotters never relied on it. Also exists so a
    plotter that sets its own ``figsize`` renders at the slot width. No-op if already at width or
    ``None``.
    """
    if fig is None:
        return None
    fig._manual_layout = True
    w, h = fig.get_size_inches()
    if w > 0 and abs(w - width_in) > 1e-6:
        fig.set_size_inches(width_in, width_in * h / w)
    return fig


def render(
    draw: Callable[..., Figure | None],
    /,
    *args: Any,  # noqa: ANN401  # forwarded verbatim to *draw*
    slot: str = "full",
    source: str = "",
    optional: bool = False,
    **kwargs: Any,  # noqa: ANN401  # forwarded verbatim to *draw*
) -> str | None:
    """Render *draw* at the *slot* width under the report style, returning a base64 PNG.

    Sets ``plots._ACTIVE_STYLE`` to the report spec for the duration of the draw (restored in a
    ``finally``) so the plotter's inner style context picks it up, forces the figure to the slot
    width, and delegates encoding to :func:`glidertest.reports._figdebug.render_b64` (the drop-in
    for the vendored encoder that records figure geometry when ``GLIDERTEST_REPORT_DEBUG`` is set).

    Parameters
    ----------
    draw : Callable
        A callable returning a Matplotlib ``Figure`` (or ``None`` when the required data is absent).
        Must return a ``Figure``, not a ``(fig, ax)`` tuple — the plot adapters unwrap glidertest's
        plotters before passing them here.
    slot : str
        Slot name from :data:`SLOTS` (``"full"``, ``"half"``, …); sets the render width.
    source : str
        The glidertest plotter function name that produced the figure, recorded against the PNG and
        shown as a ``source:`` line under it. Empty to omit.
    optional : bool
        Forwarded to the encoder; when ``True`` a ``None`` figure is dropped silently.

    Returns
    -------
    str or None
        The base64 PNG, or ``None`` when the figure was absent.
    """
    width_in = SLOTS[slot][1]
    original = plots._ACTIVE_STYLE
    plots._ACTIVE_STYLE = _report_spec(width_in)
    b64 = None
    try:
        with plt.ioff():
            b64 = _figdebug.render_b64(
                lambda *a, **k: _force_width(draw(*a, **k), width_in),
                *args,
                optional=optional,
                **kwargs,
            )
    finally:
        plots._ACTIVE_STYLE = original
    _figdebug.record_source(b64, source)  # the plotter name for the figure's "source:" line
    return b64
