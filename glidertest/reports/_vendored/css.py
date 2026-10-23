"""Shared CSS for the report HTML pages, generated from the design tokens.

``emit_css(package_accent)`` builds the whole stylesheet string from
the sibling ``design_tokens`` module — colours, typography, spacing, radii and the slot
table — so no value is restated here.  Its output is concatenated into each page
template's ``<style>`` block (``{{ css | safe }}``).

This module is written to be **vendored** across the packages that share the
report design system: it names no package and holds no per-package colour table,
reading shared values only through the sibling ``design_tokens`` module.  The accent
colour to render is an argument to ``emit_css``, supplied by the
(package-specific) caller — so the file itself carries no package-specific edit.
"""

from __future__ import annotations

from .design_tokens import (
    COLORS,
    CONTENT_MAX_PX,
    FONT_MONO,
    FONT_SANS,
    GRAYS,
    QC_COLORS,
    QC_TEXT_COLORS,
    RADII,
    ROLE_ACCENT,
    SEMANTIC,
    SLOTS,
    SPACE,
    TYPE,
)


#: UI tokens a package may recolour through ``overrides`` (one allow-list, validated in
#: ``emit_css``): one ``role-<key>`` per :data:`ROLE_ACCENT` role, the ``(_QC)`` marker colour
#: ``qc-mark``, and ``package-accent`` for symmetry.  Data palettes (QC flag colours, track
#: colours, colormaps) are deliberately *not* here — a flag colour must read the same everywhere.
def _override_tokens() -> set[str]:
    """Return the set of CSS-var suffixes a package may override via ``overrides``."""
    return {f"role-{key}" for key in ROLE_ACCENT} | {"qc-mark", "package-accent"}


def _emit_root(package_accent: str, overrides: dict[str, str] | None = None) -> str:
    """Return the ``:root { … }`` custom-property block built from the tokens.

    ``overrides`` rewrites individual overridable tokens (``role-<key>``, ``qc-mark``,
    ``package-accent``), so a package recolours a role (pills + masthead together) or the
    ``(_QC)`` marker without editing the shared table.
    """
    ov = overrides or {}
    lines: list[str] = [":root {"]
    for table in (COLORS, GRAYS, SEMANTIC):
        for key, value in table.items():
            lines.append(f"  --{key}: {value};")
    for key, value in ROLE_ACCENT.items():
        lines.append(f"  --role-{key}: {ov.get(f'role-{key}', value)};")
    # The (_QC) companion marker colour (css `.var-qc`); the "present/available" green by default.
    lines.append(f"  --qc-mark: {ov.get('qc-mark', 'var(--ok)')};")
    lines.append(f"  --package-accent: {ov.get('package-accent', package_accent)};")
    lines.append(f"  --font-sans: {FONT_SANS};")
    lines.append(f"  --font-mono: {FONT_MONO};")
    for name, spec in TYPE.items():
        lines.append(f"  --fs-{name}: {spec['size']};")
    lines.append(f"  --lh-root: {TYPE['root']['line']};")
    for key, value in SPACE.items():
        lines.append(f"  --sp-{key}: {value};")
    for key, value in RADII.items():
        lines.append(f"  --radius-{key}: {value};")
    # QC-flag colours, mirrored from the same dict so HTML (``--qc-<flag>``) and
    # matplotlib (QC_COLORS) cannot drift.  ``--qc-<flag>-text`` are the darkened
    # text variants.
    for flag, value in QC_COLORS.items():
        lines.append(f"  --qc-{flag}: {value};")
    for flag, value in QC_TEXT_COLORS.items():
        lines.append(f"  --qc-{flag}-text: {value};")
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


def _emit_nav_pills() -> str:
    """Return one ``.nav-<role>`` background rule per :data:`ROLE_ACCENT` key.

    The masthead page-nav pills take their colour from their *role*, never their
    page, so the class set is generated from the same table as ``--role-*`` — a
    new role in the tokens yields its pill rule with no edit here.
    """
    return "\n".join(
        f".nav-{key} {{ background: var(--role-{key}); }}" for key in ROLE_ACCENT
    )


def _emit_masthead_roles() -> str:
    """Return one ``.masthead.role-<role>`` background rule per :data:`ROLE_ACCENT` key.

    A page's masthead takes the colour of the nav pill that points at it, so the class set is
    generated from the same table as ``--role-*`` and ``.nav-<role>`` — a new role needs no edit
    here, and a ``role_overrides`` recolour moves the pill and the masthead together.
    """
    return "\n".join(
        f".masthead.role-{key} {{ background: var(--role-{key}); }}"
        for key in ROLE_ACCENT
    )


def emit_css(package_accent: str, *, overrides: dict[str, str] | None = None) -> str:
    """Return the full shared stylesheet as a string, generated from the tokens.

    Reproduces the established page look from the design tokens (the current
    values, not a redesign).  The only additions over
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
    overrides : dict of str to str, optional
        ``{token: colour}`` rewriting an overridable UI token — ``role-<key>`` for each
        :data:`~design_tokens.ROLE_ACCENT` role, ``qc-mark`` (the ``(_QC)`` marker), or
        ``package-accent``.  An unknown key raises (never silently ignored), so pills/mastheads
        of a role recolour together.  Data palettes (QC flag colours, track colours) stay fixed.
    """
    if overrides:
        unknown = set(overrides) - _override_tokens()
        if unknown:
            raise ValueError(
                f"overrides keys are not overridable UI tokens: {sorted(unknown)}"
            )
    root = _emit_root(package_accent, overrides)
    slots = _emit_slots()
    nav_pills = _emit_nav_pills()
    masthead_roles = _emit_masthead_roles()
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
/* Subsection headings inside a panel's own HTML — un-numbered, outside the jump
   nav (the page model numbers Sections, not these).  Same navy as h2, smaller. */
h3 {{
  color: var(--ocean); font-size: var(--fs-h3); margin: 1.2rem 0 0.4rem;
}}
h3 + .caption {{ margin-top: -0.2rem; }}
figure h3, .fig-row > h3 {{ margin-top: 0.6rem; }}  /* first heading inside a panel sits closer */
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
figcaption {{ font-size: var(--fs-cap); color: var(--gray-5); margin-bottom: 0.25rem; }}
/* Optional provenance line under a figure: which draw function produced it.
   Off by default (vendored SHOW_FIGURE_SOURCE); a package turns it on in its
   owned design module. */
.fig-source {{
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: var(--fs-xs); color: var(--muted); margin-top: 0.15rem;
}}
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
/* netCDF inventory tables (vendored nc_tables): variable / coord / scalar / sensor /
   global tables.  Package-neutral — the markup is emitted by templates/_nc_tables.html. */
.var-table code {{ font-family: var(--font-mono); font-size: var(--fs-xs); }}
.muted {{ color: var(--muted); }}
.var-qc {{ color: var(--qc-mark); font-size: var(--fs-xs); }}
/* The original source name on its own muted line above the canonical name (name_cell), so the
   Variable column does not widen to fit both on one line. */
.var-orig {{ color: var(--muted); font-size: var(--fs-xs); }}
/* A long unbroken token (a tracking_id, a path, a dotted standard_name) in a scalar value or a
   standard/long-name cell would widen the whole table; cap the width and break inside the word. */
.cell-wrap {{ overflow-wrap: anywhere; max-width: 28rem; }}
.cell-bad {{ color: var(--warn); font-variant-numeric: tabular-nums; }}
tr.units th {{
  background: transparent; color: var(--muted); font-weight: 400;
  font-size: var(--fs-xs); padding-top: 0;
}}
.attr-disclosure summary {{ cursor: pointer; color: var(--muted); font-size: var(--fs-xs); }}
.attr-disclosure table {{ margin: 0.3rem 0; width: auto; }}
.attr-disclosure th {{ background: transparent; color: var(--text); font-weight: 600; }}
/* QC flag-count distribution bar and legend (qc_counts_table). Flag colours are inline on each
   segment (from the vendored QC_FLAG_COLORS), so these rules carry only the layout. */
.qc-bar {{ display: flex; width: 200px; height: 13px; border-radius: 3px; overflow: hidden; gap: 1px; background: #ecf0f1; }}
.qc-bar div {{ height: 100%; }}
.qc-legend {{ display: flex; flex-wrap: wrap; gap: 0.3rem 0.9rem; font-size: 0.75rem; margin: 0.2rem 0 0.6rem; }}
.qc-legend span {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; vertical-align: middle; margin-right: 3px; }}
/* The source: caption hugs the table it belongs to, not the next one: a table has 1.2rem
   bottom margin, so pull the caption up under it and keep the gap below before the next table. */
table + .fig-source {{ margin-top: -0.9rem; margin-bottom: 1.2rem; }}
/* Click-to-sort affordance: _JS_SORT_TABLES gives every sortable heading a tabindex, so
   keying on th[tabindex] styles exactly the headings it wired (a data-nosort table or th
   gets none).  The heading is a pointer; the active one shows the direction. */
th[tabindex] {{ cursor: pointer; user-select: none; }}
th.sorted-asc::after {{ content: " \\25B2"; font-size: 0.8em; }}
th.sorted-desc::after {{ content: " \\25BC"; font-size: 0.8em; }}
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
/* Masthead page-nav (vendored): the row of role pills under the masthead, and the
   "Data inventory:" file strip.  A pill's colour is its role, never its page; the
   current page's pill and an already-viewed file pill are muted the same way. */
.page-nav {{ margin: 0.65rem 0 0.5rem; }}
/* Consecutive nav rows (a multi-row masthead) stack tighter than the first row's
   top margin, which is sized for a single row. */
.page-nav + .page-nav {{ margin-top: 0.15rem; }}
.page-nav-label {{ font-size: var(--fs-top); opacity: 0.75; margin-right: 0.3rem; }}
.nav-pill {{
  display: inline-block; padding: 0.2em 0.65em; border-radius: var(--radius-btn);
  font-size: var(--fs-nav); font-weight: 700; text-decoration: none; color: #fff;
  margin: 0 0.2rem 0.25rem 0;
  /* A contrasting inner edge keeps a pill visible when its role colour matches the
     masthead background, so colour-is-role holds on any masthead. */
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.55);
}}
{nav_pills}
{masthead_roles}
.nav-current {{ opacity: 0.5; pointer-events: none; }}
.nav-missing {{ background: var(--gray-3); opacity: 0.5; pointer-events: none; }}
.inventory-strip {{ margin: 0 0 1rem; }}
.inventory-strip .label {{ font-size: var(--fs-top); opacity: 0.75; margin-right: 0.3rem; }}
.file-pill {{
  display: inline-block; background: var(--role-component); color: #fff;
  padding: 0.2rem 0.7rem; border-radius: var(--radius-pill);
  font-family: var(--font-mono); font-size: 0.78rem;
  text-decoration: none; margin: 0.1rem 0.05rem;
}}
.file-pill:hover {{ opacity: 0.85; }}
.file-pill-active {{ opacity: 0.5; pointer-events: none; }}
/* Opt-in plain masthead (vendored) for landing / navigator pages: white panel,
   ink text, rule border, wordmark hidden.  No ctdcast page uses it yet. */
.masthead-plain {{ background: #fff; color: var(--text); border: 1px solid var(--rule); }}
.masthead-plain .wordmark {{ display: none; }}
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
  h3 {{ page-break-after: avoid; }}
  .fig-row, figure, table {{ break-inside: avoid; }}
  .jump-nav {{ display: none; }}
}}
"""


def print_css(*, terse: bool = False) -> str:
    """Return the print stylesheet, injected at PDF render time.

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

#: Click-to-sort for every ``<table>`` on a page — every table sorts unless it opts out.
#: Vanilla, dependency-free and inline, so it works offline — the family offline-at-view-time
#: rule.  A ``<th>`` with ``data-nosort`` (the Attributes column), a ``<table data-nosort>``
#: (the attribute disclosure sub-tables), and a table with no header row (a key/value list)
#: are left alone.  A column sorts numerically when every non-empty
#: cell parses as a number or carries ``data-sort`` (which also lets ``1,234`` and ``a / b``
#: cells sort on a raw value); otherwise it sorts case-insensitively by visible text.  First
#: click ascending, second reverses; the active heading shows ▲/▼ via ``.sorted-asc`` /
#: ``.sorted-desc``.  Headings are keyboard-operable (Enter / Space).
_JS_SORT_TABLES: str = """\
<script>
document.querySelectorAll('table').forEach(function (table) {
  if (table.hasAttribute('data-nosort')) { return; }
  var head = table.tHead && table.tHead.rows[0];
  var body = table.tBodies[0];
  if (!head || !body) { return; }
  function cellKey(row, idx) {
    var cell = row.cells[idx];
    if (!cell) { return ''; }
    var ds = cell.getAttribute('data-sort');
    return ds !== null ? ds : cell.textContent.trim();
  }
  function numeric(idx) {
    var rows = body.rows, seen = false;
    for (var i = 0; i < rows.length; i++) {
      var k = cellKey(rows[i], idx);
      if (k === '') { continue; }
      if (!/^[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?$/.test(k)) { return false; }
      seen = true;
    }
    return seen;
  }
  Array.prototype.forEach.call(head.cells, function (th, idx) {
    if (th.hasAttribute('data-nosort')) { return; }
    th.tabIndex = 0;
    function sort() {
      var asc = !(th.classList.contains('sorted-asc'));
      var isNum = numeric(idx);
      var rows = Array.prototype.slice.call(body.rows);
      rows.sort(function (a, b) {
        var ka = cellKey(a, idx), kb = cellKey(b, idx);
        if (isNum) { ka = parseFloat(ka) || 0; kb = parseFloat(kb) || 0; }
        else { ka = ka.toLowerCase(); kb = kb.toLowerCase(); }
        if (ka < kb) { return asc ? -1 : 1; }
        if (ka > kb) { return asc ? 1 : -1; }
        return 0;
      });
      rows.forEach(function (r) { body.appendChild(r); });
      Array.prototype.forEach.call(head.cells, function (o) {
        o.classList.remove('sorted-asc', 'sorted-desc');
      });
      th.classList.add(asc ? 'sorted-asc' : 'sorted-desc');
    }
    th.addEventListener('click', sort);
    th.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); sort(); }
    });
  });
});
</script>
"""
