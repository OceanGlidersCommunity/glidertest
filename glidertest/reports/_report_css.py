"""glidertest's concrete report stylesheet, from the package-neutral ``emit_css``.

This is the package-specific wiring the vendored :mod:`glidertest.reports._css`
deliberately omits: the single place glidertest's accent is applied to the shared
generator. The vendored ``_css.py`` carries no package-specific edit.
"""

from __future__ import annotations

from ._css import _JS_TOP_LINKS, emit_css

#: glidertest's package accent — VOTO blue, sampled from the VOTO wordmark.
#: Applied to the wordmark, table header and footer rule; change this one line to rebrand.
PACKAGE_ACCENT: str = "#021489"

#: The generated stylesheet, ready to concatenate into a page ``<style>`` block.
SHARED_CSS: str = emit_css(PACKAGE_ACCENT)

__all__ = ["SHARED_CSS", "_JS_TOP_LINKS"]
