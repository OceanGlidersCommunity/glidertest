"""The encoder — the single place a matplotlib Figure becomes base64 PNG bytes.

This module is written to be **vendored** across the packages that share the
report design system.  It is package-neutral — it references shared values only
through the sibling ``design_tokens`` module (textually identical in each package) and
names no package — so the copies can be frozen and checked later; for now this
document and convention are the shared reference (no cross-repo hash test yet).

One choke point (:func:`render_b64`) is where the report mplstyle is applied,
the layout guard runs, dpi and palette quantization are chosen, memory is
bounded, and error policy is decided.  Layer 3 in the spec's layer model.
"""

from __future__ import annotations

import base64
import io
import warnings
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image

from . import design_tokens as _dt
from .design_tokens import FIG_DPI, MPLSTYLE_PATH, PNG_PALETTE_COLORS


def _manages_own_layout(fig: Any) -> bool:
    """Return True when ``tight_layout()`` must be skipped for *fig*.

    Two cases, both mutually exclusive with ``tight_layout``:

    - a ``constrained_layout`` figure (matplotlib warns and mis-renders),
    - a figure containing a polar axis (bounding box is mis-computed).

    When a figure uses constrained layout — as the shared design intends — this
    returns True and ``tight_layout`` is skipped.  Figures that still rely on
    ``tight_layout`` return False here (True only for a polar axis).  The former outside-legend branch is withdrawn: growing the
    image to fit a legend is exactly what breaks the displayed-type invariant, so
    the fix is a fixed canvas with ``constrained_layout`` shrinking the axes
    instead — or, for an aspect-locked figure that is exempt from the exact-width
    invariant, an explicit ``bbox_inches="tight"`` in :func:`_fig_to_base64`.
    """
    if getattr(fig, "_manual_layout", False):
        return True
    engine = getattr(fig.get_layout_engine(), "__class__", None)
    if engine is not None and "Constrained" in engine.__name__:
        return True
    return any(ax.name == "polar" for ax in fig.axes)


def _fig_to_base64(fig: Any, *, bbox_inches: str | None = None) -> str:
    """Render *fig* to a quantized PNG and return its base64 string.

    By default (``bbox_inches=None``) saves the **full canvas** at
    :data:`FIG_DPI`, so ``png_px == round(fig_in × dpi)`` exactly — the
    displayed-type invariant that keeps figure text the same on-screen size
    across slot figures.  Pass ``bbox_inches="tight"`` only for an aspect-locked
    figure that is exempt from that invariant (a map, a section) and that draws a
    decoration outside its fixed canvas — e.g. the all-sections overview map's
    legend anchored east of the axes, which the full canvas would otherwise clip.

    Composites onto white to drop the alpha channel (report backgrounds are
    white), quantizes to :data:`PNG_PALETTE_COLORS` (these figures are few-colour
    by construction — line art plus discrete colorbars — so an 8-bit palette is
    visually lossless and ~3.6× smaller), and encodes the result.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, bbox_inches=bbox_inches)
    buf.seek(0)
    im = Image.open(buf).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(bg, im).convert(
        "RGB"
    )  # alpha must go before quantizing
    pal = rgb.quantize(colors=PNG_PALETTE_COLORS, method=Image.FASTOCTREE)
    out = io.BytesIO()
    pal.save(out, "PNG", optimize=True)
    return base64.b64encode(out.getvalue()).decode("ascii")


#: Base64 PNG -> source caption (``"<module-path>.py:<name>"`` of the draw function).
#: Populated by :func:`render_b64`, read by a report template via :func:`source_of`.
_SOURCE: dict[str, str] = {}


def source_for(draw: Callable[..., Any]) -> str:
    """Return ``"<module-path>.py:<name>"`` for a draw callable, closures collapsed.

    ``__module__`` becomes a slash path (``pkg.plotters.plots`` ->
    ``pkg/plotters/plots.py``); the name is ``__qualname__`` with any ``.<locals>.``
    closure prefix dropped, so an inner wrapper reports its enclosing
    ``_make_*`` / ``draw_*`` name rather than the closure's.
    """
    module = getattr(draw, "__module__", "") or ""
    name = getattr(draw, "__qualname__", None) or getattr(draw, "__name__", "draw")
    if ".<locals>." in name:
        name = name.split(".<locals>.")[0]
    name = name.rsplit(".", 1)[-1]
    path = f"{module.replace('.', '/')}.py" if module else "?"
    return f"{path}:{name}"


def source_of(b64: str | None) -> str | None:
    """Return the recorded source caption for a rendered PNG, or ``None``."""
    return _SOURCE.get(b64) if b64 else None


def record_source(b64: str | None, source: str) -> None:
    """Record *source* as the caption for a PNG built outside :func:`render_b64`.

    For a maker with a custom encode path (one that calls the figure-to-base64
    step itself, bypassing :func:`render_b64`) so its figure still carries a
    ``source:`` line.  Derive *source* with :func:`source_for`.
    """
    if b64:
        _SOURCE[b64] = source


def clear_sources() -> None:
    """Reset the base64 -> source registry — call at the start of each report build."""
    _SOURCE.clear()


def render_b64(
    draw: Callable[..., Any],
    /,
    *args: Any,
    optional: bool = False,
    on_error: Callable[[BaseException], None] | None = None,
    source: str | None = None,
    **kwargs: Any,
) -> str | None:
    """Run *draw* under the report mplstyle and return a base64 PNG, or None.

    Parameters
    ----------
    draw : callable
        A ``draw_*`` function returning a :class:`matplotlib.figure.Figure`, or
        ``None`` when the dataset lacks the required variables.
    *args, **kwargs
        Forwarded to *draw*.
    optional : bool
        ``True`` when this panel is legitimately absent for some inputs (no
        secondary sensor fitted, a biogeochemical variable not measured).  When
        ``False`` (default) and :data:`design_tokens.RAISE_ON_PLOT_ERROR` is set,
        a ``None`` return raises so tests catch silently dropped required panels.
    on_error : callable or None
        Called with the caught exception when *draw* raises and
        :data:`design_tokens.RAISE_ON_PLOT_ERROR` is unset — before the warning —
        so a caller can capture the cause (e.g. to turn a bare stub into one that
        says why).  Not called when ``RAISE_ON_PLOT_ERROR`` re-raises.  ``None``
        (default) keeps every existing call site valid.
    source : str or None
        The ``source:`` caption to record for the rendered PNG (``"<module>.py:<name>"``
        of the draw function).  ``None`` (default) derives it from *draw* via
        :func:`source_for`; pass an explicit string when the drawer is wrapped.

    Returns
    -------
    str or None
        Base64-encoded PNG bytes, or ``None`` when *draw* returned ``None`` or
        raised (unless ``RAISE_ON_PLOT_ERROR`` is set).
    """
    fig = None
    try:
        with plt.style.context(str(MPLSTYLE_PATH)):
            fig = draw(*args, **kwargs)
            if fig is None:
                if _dt.RAISE_ON_PLOT_ERROR and not optional:
                    raise RuntimeError(
                        f"{draw.__name__} returned None for a required panel; "
                        "check that all required variables are present in the dataset."
                    )
                return None
            if not _manages_own_layout(fig):
                fig.tight_layout()
            b64 = _fig_to_base64(fig)
            _SOURCE[b64] = source if source is not None else source_for(draw)
            return b64
    except Exception as exc:  # noqa: BLE001
        if _dt.RAISE_ON_PLOT_ERROR:
            raise
        if on_error is not None:
            on_error(exc)
        warnings.warn(
            f"{draw.__name__} failed; panel omitted: {type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    finally:
        if fig is not None:
            plt.close(fig)
