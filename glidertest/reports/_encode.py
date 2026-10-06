"""The encoder — the single place a matplotlib Figure becomes base64 PNG bytes.

This module is written to be **vendored** across the packages that share the
report design system.  It is package-neutral — it references shared values only
through ``..config.report_tokens`` (textually identical in each package) and
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

from ..config import report_tokens as _tok
from ..config.report_tokens import FIG_DPI, MPLSTYLE_PATH, PNG_PALETTE_COLORS


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


def render_b64(
    draw: Callable[..., Any],
    /,
    *args: Any,
    optional: bool = False,
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
        ``False`` (default) and :data:`report_tokens.RAISE_ON_PLOT_ERROR` is set,
        a ``None`` return raises so tests catch silently dropped required panels.

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
                if _tok.RAISE_ON_PLOT_ERROR and not optional:
                    raise RuntimeError(
                        f"{draw.__name__} returned None for a required panel; "
                        "check that all required variables are present in the dataset."
                    )
                return None
            if not _manages_own_layout(fig):
                fig.tight_layout()
            return _fig_to_base64(fig)
    except Exception:  # noqa: BLE001
        if _tok.RAISE_ON_PLOT_ERROR:
            raise
        warnings.warn(
            f"{draw.__name__} failed; panel omitted",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    finally:
        if fig is not None:
            plt.close(fig)
