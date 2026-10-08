"""Shared CSS for the report HTML pages, generated from the design tokens.

``emit_css(package_accent)`` builds the whole stylesheet string from
``..config.report_tokens`` — colours, typography, spacing, radii and the slot
table — so no value is restated here.  Its output is concatenated into each page
template's ``<style>`` block (``{{ css | safe }}``).

This module is written to be **vendored** across the packages that share the
report design system: it names no package and holds no per-package colour table,
reading shared values only through ``..config.report_tokens``.  The accent
colour to render is an argument to ``emit_css``, supplied by the
(package-specific) caller — so the file itself carries no package-specific edit
(Phase A; no cross-repo hash test yet, spec §9).
"""

from __future__ import annotations

from ..config.report_tokens import (
    COLORS,
    CONTENT_MAX_PX,
    FONT_MONO,
    FONT_SANS,
    GRAYS,
    RADII,
    ROLE_ACCENT,
    SEMANTIC,
    SLOTS,
    SPACE,
    TYPE,
)


def _emit_root(package_accent: str) -> str:
    """Return the ``:root { … }`` custom-property block built from the tokens."""
    lines: list[str] = [":root {"]
    for table in (COLORS, GRAYS, SEMANTIC):
        for key, value in table.items():
            lines.append(f"  --{key}: {value};")
    for key, value in ROLE_ACCENT.items():
        lines.append(f"  --role-{key}: {value};")
    lines.append(f"  --package-accent: {package_accent};")
    lines.append(f"  --font-sans: {FONT_SANS};")
    lines.append(f"  --font-mono: {FONT_MONO};")
    for name, spec in TYPE.items():
        lines.append(f"  --fs-{name}: {spec['size']};")
    lines.append(f"  --lh-root: {TYPE['root']['line']};")
    for key, value in SPACE.items():
        lines.append(f"  --sp-{key}: {value};")
    for key, value in RADII.items():
        lines.append(f"  --radius-{key}: {value};")
    lines.append("}")
    return "\n".join(lines)


def _emit_slots() -> str:
    """Return the ``.slot-*`` width classes generated from the slot table.

    Each width is ``fraction`` of the row minus a ``(1 - fraction) rem`` share of
    the 1 rem flex gap, so the columns of a ``.fig-row`` tile without overflow.
    """
    out: list[str] = []
    for name, (frac, _inch) in SLOTS.items():
        if frac >= 1.0:
            out.append(f".slot-{name}         {{ width: 100%; }}")
        else:
            out.append(
                f".slot-{name:<14s}{{ width: calc({frac * 100:.4g}% - {1 - frac:.2f}rem); }}"
            )
    return "\n".join(out)


def emit_css(package_accent: str) -> str:
    """Return the full shared stylesheet as a string, generated from the tokens.

    Reproduces the established page look from the design tokens (spec §12–15 v2.0
    are the current values, not a redesign).  The only additions over
    the previous hand-written stylesheet are the package accent's two homes — a
    masthead ``.wordmark`` and the footer's ``border-top`` — plus the ``78ch``
    reading measure and the ``@media print`` rules for browser printing.  (The
    PDF renderer additionally injects :func:`print_css`, which layers ``@page``
    geometry on top for the WeasyPrint path.)

    Parameters
    ----------
    package_accent : str
        The accent colour to render as ``--package-accent`` — any CSS colour
        value, e.g. a hex literal or ``var(--ocean)``.  Chosen by the
        package-specific caller; the function itself names no package and holds
        no per-package colour table.
    """
    root = _emit_root(package_accent)
    slots = _emit_slots()
    return f"""\
{root}
* {{ box-sizing: border-box; }}
body {{
  font-family: var(--font-sans);
  font-size: var(--fs-root); color: var(--text);
  max-width: {CONTENT_MAX_PX}px; margin: 0 auto;
  padding: 1.5rem 2rem 4rem; line-height: var(--lh-root);
}}
p, li {{ max-width: 78ch; }}
.masthead {{
  background: var(--ocean); color: #fff; position: relative;
  padding: 1.6rem 2rem; border-radius: var(--radius-card); margin-bottom: 0.9rem;
}}
.masthead h1 {{ margin: 0 0 0.3rem; font-size: var(--fs-h1); font-weight: 700; }}
.masthead .sub {{ font-size: var(--fs-meta); opacity: 0.85; margin: 0 0 0.15rem; max-width: none; }}
.wordmark {{
  position: absolute; right: 2rem; bottom: 1.2rem;
  font-size: var(--fs-dt); font-weight: 700; letter-spacing: 0.04em;
  color: var(--package-accent); background: #fff; opacity: 0.85;
  padding: 0.1rem 0.5rem; border-radius: var(--radius-pill); text-decoration: none;
}}
.wordmark:hover {{ opacity: 1; }}
.meta-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  gap: 0.5rem 2rem; font-size: var(--fs-meta); margin-top: 0.9rem;
}}
.meta-grid dt {{
  opacity: 0.7; text-transform: uppercase; font-size: var(--fs-dt);
  letter-spacing: 0.06em; margin-bottom: 0.1rem;
}}
.meta-grid dd {{ margin: 0; font-weight: 600; }}
h2 {{
  color: var(--ocean); font-size: var(--fs-h2);
  border-bottom: 2px solid var(--seafoam);
  padding-bottom: 0.3rem; margin: 2.5rem 0 1rem;
  display: flex; justify-content: space-between; align-items: baseline;
}}
.top-link {{
  font-size: var(--fs-top); font-weight: 400; color: var(--muted);
  text-decoration: none; margin-left: auto;
}}
.top-link:hover {{ color: var(--ocean); text-decoration: underline; }}
.caption {{
  color: var(--gray-5); font-size: var(--fs-note);
  margin-top: 0.75rem; margin-bottom: 0.75rem;
}}
/* A section intro (the caption directly under a heading) is a lede for the whole
   figure block, so it spans the content width rather than the 78ch prose measure a
   paragraph would otherwise inherit — which left it stranded under full-width figures. */
h2 + .caption {{ margin-top: -0.5rem; max-width: none; }}
.explainer {{
  font-size: var(--fs-note); color: var(--text);
  background: var(--bg-sunken); border-left: 3px solid var(--muted);
  padding: var(--sp-3); margin: -0.25rem 0 0.75rem; border-radius: var(--radius-btn);
}}
.warn {{
  font-size: var(--fs-note); color: var(--text);
  background: var(--warn-bg); border-left: 3px solid var(--warn);
  padding: var(--sp-3); margin-bottom: 0.5rem; border-radius: var(--radius-btn);
}}
.warn::before {{ content: "⚠ "; }}
.warn.error {{ border-left-color: var(--error); }}
.jump-nav {{
  background: var(--seafoam); padding: 0.55rem 1rem;
  border-radius: 6px; margin-bottom: 1.5rem;
  font-size: var(--fs-nav); line-height: 2.2;
}}
.jump-nav::before {{ content: "Jump to: "; opacity: 0.55; font-size: var(--fs-xs); margin-right: 0.25rem; }}
.jump-nav a {{
  color: var(--ocean); text-decoration: none;
  font-weight: 600; margin: 0 0.5rem 0 0;
}}
.jump-nav a::before {{ content: "▸ "; font-size: var(--fs-dt); }}
.fig-row {{
  display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem;
  align-items: flex-start;
}}
.fig-col {{ display: flex; flex-direction: column; gap: 0.75rem; }}
.fig-row.center {{ justify-content: center; }}
/* A multi-table panel (processing provenance) is one full-width flex item so its
   headings and tables stack instead of tiling across the row. */
.prov-panel {{ flex: 1 1 100%; }}
/* Conformance status pips — a green ✓ (matches), a red ✗ (differs) and a neutral
   ringed dash (no reference), matching the LADCP-present/absent badges. */
.conf {{
  display: inline-flex; align-items: center; justify-content: center;
  width: 1.4em; height: 1.4em; border-radius: 50%;
  font-size: var(--fs-xs); font-weight: 700; line-height: 1;
}}
.conf-match {{ background: var(--ok); color: #fff; }}
.conf-differ {{ background: var(--error); color: #fff; }}
.conf-none {{ background: #fff; color: #000; border: 1px solid #000; }}
figure {{ margin: 0; }}
figure img {{
  border: 1px solid var(--rule); border-radius: var(--radius-btn);
  display: block; width: 100%; height: auto;
}}
figcaption {{ font-size: var(--fs-cap); color: var(--gray-5); margin-top: 0.25rem; }}
/* A stub panel's warning fills the figure slot it stands in for, rather than the
   78ch prose reading measure it would otherwise inherit as a paragraph. */
figure .warn {{ max-width: none; }}
table {{
  width: 100%; border-collapse: collapse;
  font-size: var(--fs-note); margin: 0.6rem 0 1.2rem;
}}
th {{
  background: var(--package-accent); color: #fff; text-align: left;
  font-weight: 600; padding: 0.4rem 0.65rem;
}}
th.num {{ text-align: right; }}
td {{
  padding: 0.35rem 0.65rem; border-bottom: 1px solid var(--rule);
  vertical-align: top;
}}
tr:nth-child(even) td {{ background: var(--bg-sunken); }}
td.num, .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
td.mono {{ font-family: var(--font-mono); font-size: var(--fs-xs); }}
{slots}
.masthead-header {{
  display: flex; justify-content: space-between; align-items: flex-start;
  margin-bottom: 0.3rem;
}}
.masthead-header h1 {{ margin: 0 0 0.1rem; font-size: var(--fs-h1); font-weight: 700; }}
.masthead-type {{
  font-size: var(--fs-type); font-weight: 700; opacity: 0.88; line-height: 1.35;
  padding-top: 0.25rem;
}}
.nav-btns {{ display: flex; gap: 0.5rem; }}
.btn-nav {{
  background: var(--ocean); color: #fff; padding: 0.25rem 0.75rem;
  border-radius: var(--radius-pill); text-decoration: none; font-size: var(--fs-nav);
}}
.btn-nav:hover {{ opacity: 0.85; }}
footer {{
  text-align: center; padding: 1rem;
  border-top: 1px solid var(--package-accent);
  font-size: var(--fs-xs); color: var(--muted);
}}
footer a {{ color: var(--muted); }}
@media print {{
  body {{ max-width: 100%; }}
  .masthead {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; padding: 0.9rem 1.25rem; }}
  .masthead-header h1, .masthead h1 {{ font-size: 1.35rem; }}
  .meta-grid {{ grid-template-columns: repeat(4, 1fr); gap: 0.3rem 1rem; font-size: var(--fs-xs); }}
  h2 {{ page-break-after: avoid; }}
  .fig-row, figure, table {{ break-inside: avoid; }}
  .jump-nav {{ display: none; }}
}}
"""


def print_css(*, terse: bool = False) -> str:
    """Return the print stylesheet, injected at PDF render time (spec §8.1).

    These WeasyPrint-specific rules are kept out of the screen stylesheet so
    screen output is byte-identical; a PDF renderer appends them at render time.
    The shared stylesheet keeps only the structural ``@media print`` rules (page
    breaks).  With *terse* ``True`` the ``.explainer`` prose is hidden (the
    ``--pdf-terse`` variant); by default nothing is hidden — captions, explainers
    and warnings all print.
    """
    css = """\
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
body { max-width: 100%; padding: 0; }
p, li, .explainer { max-width: 78ch; }
figure .warn { max-width: none; }
img { max-width: 100%; height: auto; }
.masthead { -webkit-print-color-adjust: exact; print-color-adjust: exact; padding: 0.9rem 1.25rem; }
.masthead-header h1, .masthead h1 { font-size: var(--fs-h1); }
.meta-grid { grid-template-columns: repeat(4, 1fr); gap: 0.3rem 1rem; font-size: var(--fs-xs); }
.jump-nav, .nav-btns, .btn-nav, .top-link { display: none; }
"""
    if terse:
        css += ".explainer { display: none; }\n"
    return css


_JS_TOP_LINKS: str = """\
<script>
document.querySelectorAll('h2').forEach(h => {
  const a = document.createElement('a');
  a.href = '#top'; a.className = 'top-link'; a.textContent = '↑ top';
  h.appendChild(a);
});
</script>
"""
