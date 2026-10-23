"""Shared report design tokens — the single source of every presentation value.

This module is written to be **vendored** across the packages that share the
report design system.  It is
package-neutral — it names no package and holds only data (plus the mplstyle
path, resolved relative to this file so no package name appears in the text) — so
the copies can be frozen and checked later; for now this document and convention
are the shared reference (no cross-repo hash test yet).  A value that belongs to
one package (a scientific variable registry) does **not** live here.

Layering: this is a leaf — it imports only :mod:`pathlib` at module load.
``cmocean`` (which provides the ``cmo.*`` entries in :data:`CMAPS`) is imported
lazily by :func:`register_cmocean`, only when such a name is actually requested,
so no CSS- or geometry-only importer pays its cost.  Both the plotters (which
size figures from :data:`SLOTS` and read :data:`CMAPS`) and the report/CSS layer
read this module; it reads nothing of theirs.

Templates and plotters read these values; they never restate them.
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

#: Vendored default for the per-figure ``source:`` caption (which draw function
#: produced each panel).  OFF here — a package turns it on in its owned design
#: module (ctdcast: ``reports/design.py`` ``FIGURE_SOURCE``), never in the shared
#: table.  Read into the report env as the ``show_source`` template global.
SHOW_FIGURE_SOURCE: bool = False

# ---------------------------------------------------------------------------
# Geometry
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
# Slot table
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

# Aspect-locked figure constants.
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
# Figure annotation font sizes (points)
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
# Spacing and radii  [data; applied by emit_css()]
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
# Typography  [data; applied by emit_css()]
# ---------------------------------------------------------------------------
# THE single source of page font sizes.  Nothing else in the repository sets a
# page font size.  Values in rem (the browser's 16px root, exactly as the current
# CSS uses them) plus one absolute px for the body base.  This is the established,
# reviewed page scale — the values to match, not a redesign.  Keyed
# by role, matching the CSS custom
# properties emit_css() generates.  Figure font sizes are a *separate* knob (the
# mplstyle, in points).
TYPE: dict[str, dict[str, str]] = {
    "root": {"size": "14px", "line": "1.5"},  # body base
    "h1": {"size": "1.75rem", "weight": "700"},  # masthead title
    "type": {"size": "1.35rem", "weight": "700"},  # .masthead-type page label
    "h2": {"size": "1rem"},  # section headings — colour/weight/underline, not size
    "h3": {"size": "0.88rem"},  # subsection headings inside a panel (un-numbered)
    "meta": {"size": "0.84rem"},  # meta-grid <dd>
    "note": {"size": "0.92rem"},  # .note, .caption, .explainer
    "nav": {"size": "0.8rem"},  # jump-nav, .btn-nav
    "cap": {"size": "0.92rem"},  # figcaption
    "xs": {"size": "0.75rem"},  # breadcrumb, footer
    "top": {"size": "0.72rem"},  # ↑ top link
    "dt": {"size": "0.7rem"},  # meta-grid <dt>, jump-nav ▸
}

# Page font stacks.  The CSS uses the native `system-ui` stack —
# zero-install, matches the host OS.  The *figure* font is separate (the mplstyle
# names a Helvetica stack); figure text is baked into a raster, so the two need
# not match.
FONT_SANS: str = 'system-ui, -apple-system, "Segoe UI", sans-serif'
FONT_MONO: str = (
    'ui-monospace, "SF Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace'
)

# ---------------------------------------------------------------------------
# Colour — base tokens  [data; applied]
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

# Role accent: masthead background + nav pill.  `landing` is the
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
    "up": "#5d6d7e",  # back / up-link pill (masthead nav)
}

# Okabe–Ito qualitative palette for track / section colours — colour-blind-safe,
# in a fixed order so the Nth track is the same colour across packages.  Data
# only; a package assigns them (e.g. per section / per track) at its own call
# sites.  Yellow is last: faint as a thin line on a light map.
TRACK_COLORS: list[str] = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#F0E442",  # yellow
]

# Per-quantity colormaps — the family default for each physical quantity, by a
# short quantity word.  A package's variable registry points its "cmap" at a
# quantity (resolved through here) or carries a literal colormap name; a package
# may override a default in its own reports/design.py.  QC_COLORS and
# TRACK_COLORS are deliberately NOT here — they encode meaning, not a scale.
# cmo.* names come from cmocean (imported above).  Values: family-colormaps note.
CMAPS: dict[str, str] = {
    "temperature": "RdYlBu_r",
    "salinity": "YlGnBu_r",
    "conductivity": "GnBu",
    "density": "BuPu",
    "depth": "cmo.deep",
    "velocity": "RdBu_r",  # diverging about zero
    "speed": "plasma",
    "oxygen": "cmo.oxy",
    "oxygen_saturation": "BrBG",  # plain scale over range, not centred on 100 %
    "chlorophyll": "YlGn",
    "turbidity": "YlOrBr",
    "backscatter": "OrRd",
    "par": "cmo.solar",
    "stratification": "plasma",
    "direction": "twilight",  # cyclic
}

# Bridge from a variable's CF standard_name to a quantity, so a package whose
# registry carries standard_names resolves to a colormap without restating the
# quantity per variable.  A variable with no standard name gives an explicit
# "cmap" (a quantity word or a literal) in its own registry instead.
QUANTITY_BY_STANDARD_NAME: dict[str, str] = {
    "sea_water_temperature": "temperature",
    "sea_water_conservative_temperature": "temperature",
    "sea_water_potential_temperature": "temperature",
    "sea_water_practical_salinity": "salinity",
    "sea_water_absolute_salinity": "salinity",
    "sea_water_electrical_conductivity": "conductivity",
    "sea_water_sigma_theta": "density",
    "sea_water_density": "density",
    "sea_water_pressure": "depth",
    "depth": "depth",
    "eastward_sea_water_velocity": "velocity",
    "northward_sea_water_velocity": "velocity",
    "upward_sea_water_velocity": "velocity",
    "sea_water_speed": "speed",
    "sea_water_velocity_to_direction": "direction",
    "sea_water_velocity_from_direction": "direction",
    "direction_of_sea_water_velocity": "direction",  # deprecated CF name, mapped for old files
    "moles_of_oxygen_per_unit_mass_in_sea_water": "oxygen",
    "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water": "oxygen",
    "mass_concentration_of_chlorophyll_in_sea_water": "chlorophyll",
    "concentration_of_chlorophyll_in_sea_water": "chlorophyll",
    "sea_water_turbidity": "turbidity",
    "square_of_brunt_vaisala_frequency_in_sea_water": "stratification",
}

# QARTOD / OceanSITES QC-flag colours — colour-blind-safe (Okabe-Ito good/suspect/
# fail: teal green / orange / vermillion, which the old red/green pair was not).
# Keyed by flag value.  Like TRACK_COLORS these encode meaning, so they are NOT
# overridable — a flag colour must read the same in every report.  The colour is
# carried by the bar/marker/chip, never on its own (tables also carry the word);
# flag 2 "probably good" is sky blue where a solid fill is needed.  emit_css()
# mirrors this into the ``--qc-<flag>`` CSS variables from the same dict.
QC_COLORS: dict[int, str] = {
    0: "#999999",  # no QC performed — neutral grey
    1: "#009E73",  # good — teal green
    2: "#56B4E9",  # probably good — sky blue
    3: "#E69F00",  # suspect — orange
    4: "#D55E00",  # fail — vermillion
    7: "#CC79A7",  # nominal — reddish purple
    8: "#0072B2",  # interpolated — blue
    9: "#bdbdbd",  # missing — light grey
}

# Darkened text variants, for a QC colour used as small text ink rather than a
# fill (orange is only 2.25:1 on white as text; this is 4.7:1).  Only suspect
# needs one; vermillion and teal are acceptable for bold/large text.
QC_TEXT_COLORS: dict[int, str] = {
    3: "#9A6B00",  # suspect orange, darkened for text
}

#: Grey fallback for a flag token no convention below names.
QC_FLAG_DEFAULT_COLOR: str = "#999999"

#: Flag-meaning token → its flag value, for each convention a family writer emits, so the
#: token-keyed colour table is built from the value-keyed :data:`QC_COLORS` (one colour source,
#: no drift).  QARTOD (ctdcast ``processors/qc.py``) and OceanSITES (oceanarray
#: ``config/parameters.py``); a token in both maps to the same value, so to the same colour.
_QC_FLAG_VALUE: dict[str, int] = {
    # QARTOD
    "pass": 1,
    "not_evaluated": 2,
    "suspect_or_of_high_interest": 3,
    "fail": 4,
    "missing_data": 9,
    # OceanSITES
    "unknown": 0,
    "good_data": 1,
    "probably_good_data": 2,
    "potentially_correctable_bad_data": 3,
    "bad_data": 4,
    "nominal_value": 7,
    "interpolated_value": 8,
    "missing_value": 9,
}

#: QC flag colour keyed by meaning token (so a report colours a flag by what it *means*, not by
#: a value that differs between conventions), derived from :data:`QC_COLORS`.  A table looks a
#: token up here; an unlisted token gets :data:`QC_FLAG_DEFAULT_COLOR`.
QC_FLAG_COLORS: dict[str, str] = {
    token: QC_COLORS[value] for token, value in _QC_FLAG_VALUE.items()
}

#: Short display label keyed by meaning token — a shortening of the CF word, nothing invented;
#: an unlisted token falls back to the token with underscores as spaces.
QC_FLAG_LABELS: dict[str, str] = {
    # QARTOD
    "pass": "pass",
    "not_evaluated": "not eval",
    "suspect_or_of_high_interest": "suspect",
    "fail": "fail",
    "missing_data": "missing",
    # OceanSITES
    "unknown": "unknown",
    "good_data": "good",
    "probably_good_data": "probably good",
    "potentially_correctable_bad_data": "correctable",
    "bad_data": "bad",
    "nominal_value": "nominal",
    "interpolated_value": "interpolated",
    "missing_value": "missing",
}

_CMOCEAN_REGISTERED = False


def register_cmocean() -> None:
    """Register cmocean's ``cmo.*`` colormaps with matplotlib, once.

    Called only when a ``cmo.*`` name is actually requested (from a colormap
    resolver, or a config validator's colormap check), so neither the import
    cost nor cmocean's own import-time deprecation warnings are paid on an
    import of this module that never touches a cmocean map.  cmocean 3.x builds
    its colormaps with the deprecated ``ListedColormap(N=...)`` (its code, not
    ours), emitting one ``MatplotlibDeprecationWarning`` per map; those are
    silenced here at their source — the import — rather than by a global filter.
    """
    global _CMOCEAN_REGISTERED
    if _CMOCEAN_REGISTERED:
        return
    import warnings

    import matplotlib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", matplotlib.MatplotlibDeprecationWarning)
        import cmocean  # noqa: F401  # the import registers the cmo.* names
    _CMOCEAN_REGISTERED = True


# The package accent — a small masthead wordmark, the table header
# and the footer's border-top — is NOT a value in this vendored file.  Each
# package chooses it locally and passes it to ``emit_css(package_accent)`` (see
# each repo's ``reports/design.py``), so no per-package colour is encoded here.

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
