"""Shared report design tokens — the single source of every presentation value.

This module is written to be **vendored** across the packages that share the
report design system.  It is
package-neutral — it names no package and holds only data (plus the mplstyle
path, resolved relative to this file so no package name appears in the text) — so
the copies can be frozen and checked later; for now this document and convention
are the shared reference (no cross-repo hash test yet).  A value that belongs to
one package (a scientific variable registry) does **not** live here.

Layering: this is a leaf, and it lives in ``config/`` for that reason — its only
import is :mod:`pathlib`.  Both the plotters (which size figures from
:data:`SLOTS`) and the report/CSS layer read it; it reads nothing of theirs.
Placing it under ``reports/`` would form an import cycle
(``plotters.plots -> reports -> reports._index -> plotters.plots``).

Part II of ``2026-08-12-report-spec.md`` is the prose behind these numbers.
Templates and plotters read them; they never restate them.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Style file and error policy
# ---------------------------------------------------------------------------
# The report mplstyle sits next to this module under config/, with a
# package-neutral name so this line is byte-identical across packages.
MPLSTYLE_PATH: Path = Path(__file__).with_name("report.mplstyle")

# When True, a plotting failure (or a None from a required panel) re-raises
# instead of being swallowed.  Test infrastructure toggles this at runtime; the
# default False is identical across packages.  Read at call time by both the
# plotters and the encoder so a single flag governs both.
RAISE_ON_PLOT_ERROR: bool = False

# ---------------------------------------------------------------------------
# Geometry (spec §11)
# ---------------------------------------------------------------------------
CONTENT_MAX_PX: int = 1150  # body max-width
CONTENT_PAD_PX: int = 32  # body padding, each side
USABLE_PX: int = CONTENT_MAX_PX - 2 * CONTENT_PAD_PX  # 1086
W_FULL: float = 9.0  # full-slot figure width in inches (the basis of SLOTS)
FIG_DPI: int = 150  # savefig dpi; with W_FULL this fixes every PNG width
# Oversample (png_px / display_px) is DERIVED, not a free knob: W_FULL, FIG_DPI
# and USABLE_PX over-determine it, so declaring all three independently would
# guarantee a contradiction.  The true value is ≈1.243, not a round 1.25 — the
# difference is invisible, and keeping W_FULL and FIG_DPI clean (they are the two
# that appear in code, as figsize and savefig dpi) reproduces the package's existing
# PNG widths exactly.  Nothing reads this; it is documentation.
OVERSAMPLE: float = W_FULL * FIG_DPI / USABLE_PX  # ≈ 1.2431
PNG_PALETTE_COLORS: int = 256  # 8-bit palette quantization in the encoder

# ---------------------------------------------------------------------------
# Slot table (spec §14)
# ---------------------------------------------------------------------------
# name -> (fraction of USABLE_PX, figure width in inches).
# Invariant asserted by the slot-contract test: inches == W_FULL * fraction, so
# display_px / fig_in is identical for every figure and one font size renders at
# one on-screen size everywhere.  Test 2 asserts each saved PNG is exactly
# round(inches * FIG_DPI) px wide (1350 / 900 / 810 / 675 / 540 / 450 / 338).
SLOTS: dict[str, tuple[float, float]] = {
    "full": (1.0, 9.0),
    "twothirds": (2 / 3, 6.0),
    "three-fifths": (0.6, 5.4),
    "half": (0.5, 4.5),
    "two-fifths": (0.4, 3.6),
    "third": (1 / 3, 3.0),
    "quarter": (0.25, 2.25),
}

# Ergonomic width aliases (inches) for plotter call sites; derived from SLOTS.
W_TWOTHIRDS: float = SLOTS["twothirds"][1]
W_THREE_FIFTHS: float = SLOTS["three-fifths"][1]
W_HALF: float = SLOTS["half"][1]
W_TWO_FIFTHS: float = SLOTS["two-fifths"][1]
W_THIRD: float = SLOTS["third"][1]
W_QUARTER: float = SLOTS["quarter"][1]

# Aspect-locked figure constants (spec §14).
SECTION_STRETCH: float = (
    16.0  # calibrated: 416 dbar × 94 km → 2.5 in tall at full width
)
MAX_SECTION_H: float = 5.2  # height cap; tall/narrow sections get a narrower fig_w
MIN_SECTION_H: float = 3.0  # height floor

# Colorbar geometry in inches — a *fixed* width (not a fraction of the axes) so the
# bar and its right margin are identical across every field figure regardless of
# slot width, matching the fixed-width ethos of SLOTS.
CBAR_WIDTH_IN: float = 0.12  # colorbar bar thickness
CBAR_PAD_IN: float = 0.10  # gap between the axes and the colorbar

# ---------------------------------------------------------------------------
# Figure annotation font sizes (points, spec §13.3)
# ---------------------------------------------------------------------------
# The only per-call font sizes a plotter may set: matplotlib does not route
# annotation text through a style key, so these cannot come from the mplstyle.
# Everything else (axes/tick/legend/title sizes) is the mplstyle's job and must
# not be set per call.  The "no stray typography" test allow-lists exactly these
# three names.
CLABEL_FS: int = 8  # ax.clabel() contour labels
ANNOT_FS: int = 9  # colorbar unit titles + in-axes annotation / panel-label text
CAST_LABEL_FS: int = 6  # dense in-axes cast-number labels on maps and sections

# ---------------------------------------------------------------------------
# Figure line widths — GMT named pen-width ladder (points)
# ---------------------------------------------------------------------------
# matplotlib linewidths are in points, the same unit as GMT's pen widths, so
# plotters name line weights from GMT's ladder (``pen("thin")``) instead of
# hardcoding numbers.  Values verified against the GMT line tutorial:
#   https://www.generic-mapping-tools.org/gmt-examples/tutorials/basics/line.html
# The "no stray typography" test allow-lists the pen() helper.
GMT_PEN: dict[str, float] = {
    "faint": 0.0,
    "thinnest": 0.25,
    "default": 0.25,
    "thinner": 0.5,
    "thin": 0.75,
    "thick": 1.0,
    "thicker": 1.5,
    "thickest": 2.0,
    "fat": 3.0,
    "fatter": 6.0,
    "fattest": 10.0,
    "wide": 18.0,
}


def pen(name: str) -> float:
    """Return a matplotlib linewidth in points for a GMT pen-width *name*.

    Borrows GMT's named pen ladder (``faint`` … ``wide``) so plotters name line
    weights (``lw=pen("thin")``) rather than hardcoding numbers.
    """
    return GMT_PEN[name]


# ---------------------------------------------------------------------------
# Spacing and radii (spec §15)  [data; applied by emit_css() in rep/vis-system]
# ---------------------------------------------------------------------------
SPACE: dict[str, str] = {
    "1": "4px",
    "2": "8px",
    "3": "12px",
    "4": "16px",
    "5": "24px",
    "6": "32px",
    "7": "48px",
}
RADII: dict[str, str] = {"card": "8px", "btn": "4px", "pill": "999px"}

# ---------------------------------------------------------------------------
# Typography (spec §13.2)  [data; applied by emit_css() in rep/vis-system]
# ---------------------------------------------------------------------------
# THE single source of page font sizes.  Nothing else in the repository sets a
# page font size.  Values in rem (the browser's 16px root, exactly as the current
# CSS uses them) plus one absolute px for the body base.  This is the established,
# reviewed page scale (spec §13.2) — the values to match, not a redesign.  Keyed
# by role, matching the CSS custom
# properties emit_css() generates.  Figure font sizes are a *separate* knob (the
# mplstyle, in points).
TYPE: dict[str, dict[str, str]] = {
    "root": {"size": "14px", "line": "1.5"},  # body base
    "h1": {"size": "1.75rem", "weight": "700"},  # masthead title
    "type": {"size": "1.35rem", "weight": "700"},  # .masthead-type page label
    "h2": {"size": "1rem"},  # section headings — colour/weight/underline, not size
    "meta": {"size": "0.84rem"},  # meta-grid <dd>
    "note": {"size": "0.82rem"},  # .note, .caption, .explainer
    "nav": {"size": "0.8rem"},  # jump-nav, .btn-nav
    "cap": {"size": "0.76rem"},  # figcaption
    "xs": {"size": "0.75rem"},  # breadcrumb, footer
    "top": {"size": "0.72rem"},  # ↑ top link
    "dt": {"size": "0.7rem"},  # meta-grid <dt>, jump-nav ▸
}

# Page font stacks (spec §13.1).  The CSS uses the native `system-ui` stack —
# zero-install, matches the host OS.  The *figure* font is separate (the mplstyle
# names a Helvetica stack); figure text is baked into a raster, so the two need
# not match.
FONT_SANS: str = 'system-ui, -apple-system, "Segoe UI", sans-serif'
FONT_MONO: str = (
    'ui-monospace, "SF Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace'
)

# ---------------------------------------------------------------------------
# Colour — base tokens (spec §12.1)  [data; applied in rep/vis-system]
# ---------------------------------------------------------------------------
COLORS: dict[str, str] = {
    "ocean": "#1a3a5c",  # headings, structural dark
    "seafoam": "#e8f4f8",  # h2 underline, jump-nav background
    "muted": "#95a5a6",  # footer, breadcrumb separators, ↑ top
    "text": "#2c3e50",  # body text
    "rule": "#dfe6e9",  # table borders, hairlines
    "bg": "#ffffff",  # page background
    "bg-sunken": "#f7f9fa",  # table zebra, card interiors
    "warn": "#e67e22",  # .warn border and icon
    "warn-bg": "#fdf3e7",  # .warn background
    "error": "#c0392b",  # failed QC, sentinel values
}

# Role accent (spec §12.2): masthead background + nav pill.  `landing` is the
# summary/index accent; entity/collection/component share the structural dark
# --ocean; the aggregate and map roles have their own colours.  The `landing`
# value is a per-package choice (see the spec's role table).
ROLE_ACCENT: dict[str, str] = {
    "landing": "#2980b9",  # summary / index accent
    "entity": "#1a3a5c",
    "collection": "#1a3a5c",
    "component": "#1a3a5c",
    "aggregate-a": "#8e44ad",  # sections
    "aggregate-b": "#27ae60",  # timeseries
    "map": "#ee3377",
}

# The package accent (spec §12.2) — a small masthead wordmark, the table header
# and the footer's border-top — is NOT a value in this vendored file.  Each
# package chooses it locally and passes it to ``emit_css(package_accent)`` (see
# each repo's ``reports/_report_css.py``), so no per-package colour is encoded here.

# Neutral gray scale — consolidates the ad-hoc grays that the per-page template
# <style> blocks used (text shades, hairlines, sunken fills).  Emitted as
# --gray-1 (lightest) … --gray-7 (near-black).  A few template literals shift to
# the nearest step here; the changes are small and were signed off on OdB.
GRAYS: dict[str, str] = {
    "gray-1": "#f5f7fa",  # lightest sunken fill (cast-note / card backgrounds)
    "gray-2": "#e0e0e0",  # hairlines, borders
    "gray-3": "#aaaaaa",  # faint text ("not generated" placeholders)
    "gray-4": "#888888",  # soft secondary text, <details> summaries
    "gray-5": "#555555",  # secondary body text, descriptions
    "gray-6": "#333333",  # strong text (cast/trim notes, filenames)
    "gray-7": "#111111",  # darkest text (interactive-map labels)
}

# Semantic status colours used by badges and banners.  --error/--warn already
# exist in COLORS; --ok is new (the LADCP-present badge green).
SEMANTIC: dict[str, str] = {
    "ok": "#2c6e49",  # LADCP-present badge; "good/available" green
}
# Note: leaflet.html is a standalone page that does not load this shared CSS, so
# its dark-UI palette (#1a1a2e / #aed6f1 / #4a6fa5) stays as literals there —
# tokenizing it would need the :root injected into that page.
