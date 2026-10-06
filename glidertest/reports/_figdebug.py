"""Opt-in per-figure debug overlay for reports (``GLIDERTEST_REPORT_DEBUG=1``).

When enabled, every embedded report figure gets a small ``.debug`` line under it showing
**slot | draw-func · figsize · png_px**, so a figsize-vs-display-slot mismatch (the class
of bug that shrinks fonts when a wide PNG is dropped in a narrow slot) is visible.

Entirely package-local — it wraps the vendored encoder (:mod:`glidertest.reports._encode`)
without editing it, and the ``.debug`` CSS lives in ``base.html``'s local style block, not
in the vendored ``emit_css``. Zero cost when off: :func:`record` no-ops and the template
macro emits nothing (it guards on :func:`figdbg` being non-empty).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ..config.report_tokens import FIG_DPI
from . import _encode

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure

#: b64 PNG string -> {"func": str, "figsize_in": (w, h), "png_px": (w, h)}.
_COLLECTED: dict[str, dict[str, Any]] = {}


def enabled() -> bool:
    """Return True when the ``GLIDERTEST_REPORT_DEBUG`` env var is set to a truthy value."""
    return os.environ.get("GLIDERTEST_REPORT_DEBUG", "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )


def clear() -> None:
    """Reset the collected registry — call once at the start of each report build.

    Without this the ``b64 -> debug`` map accumulates across builds and could hand back
    a stale entry for a byte-identical PNG produced by a later run.
    """
    _COLLECTED.clear()


def _draw_name(draw: Callable[..., object]) -> str:
    """Return a readable name for *draw*, collapsing closures to their enclosing scope.

    Uses ``__qualname__`` and, if it contains ``.<locals>.`` (a closure such as the
    encoder wrapper), keeps the part before it — so a figure drawn by an inner ``_wrapped``
    reports its enclosing ``_make_*`` / ``draw_*_fig`` name instead of the closure's.
    """
    name = getattr(draw, "__qualname__", None) or getattr(draw, "__name__", "draw")
    if ".<locals>." in name:
        name = name.split(".<locals>.")[0]
    return name.rsplit(".", 1)[-1]


def record(b64: str | None, func: str, fig: Figure | None) -> None:
    """Store the figsize/png geometry for *b64* under label *func* (no-op when disabled)."""
    if not enabled() or not b64 or fig is None:
        return
    w_in, h_in = (float(v) for v in fig.get_size_inches())
    _COLLECTED[b64] = {
        "func": func,
        "figsize_in": (round(w_in, 2), round(h_in, 2)),
        "png_px": (round(w_in * FIG_DPI), round(h_in * FIG_DPI)),
    }


def figdbg(b64: str | None) -> str:
    """Return the one-line debug string for a recorded *b64*, or ``""`` if none.

    Registered as a Jinja global; the template macro emits nothing when this is empty,
    so the whole feature is a no-op when debug is off (no debug flag needs threading).
    """
    d = _COLLECTED.get(b64 or "")
    if not d:
        return ""
    fw, fh = d["figsize_in"]
    pw, ph = d["png_px"]
    return f"{d['func']} · figsize {fw}×{fh} in · png {pw}×{ph} px"


def render_b64(
    draw: Callable[..., object],
    /,
    *args: Any,  # noqa: ANN401  # forwarded verbatim to *draw*
    optional: bool = False,
    **kwargs: Any,  # noqa: ANN401  # forwarded verbatim to *draw*
) -> str | None:
    """Encode *draw* via the vendored encoder; when debug is on, record its geometry.

    Drop-in for :func:`glidertest.reports._encode.render_b64`. When disabled, delegates
    straight through (zero overhead). When enabled, wraps *draw* to capture the Figure it
    returns, then records the figsize/png_px under the returned base64 PNG.
    """
    if not enabled():
        return _encode.render_b64(draw, *args, optional=optional, **kwargs)

    captured: dict[str, Any] = {}

    def _wrapped(
        *a: Any,  # noqa: ANN401  # forwarded verbatim to *draw*
        **k: Any,  # noqa: ANN401  # forwarded verbatim to *draw*
    ) -> Any:  # noqa: ANN401  # returns whatever *draw* returns (Figure or None)
        fig = draw(*a, **k)
        captured["fig"] = fig
        return fig

    b64 = _encode.render_b64(_wrapped, *args, optional=optional, **kwargs)
    record(b64, _draw_name(draw), captured.get("fig"))
    return b64
