"""HTML report generation: slot layer, plot adapters, and the public report entry point."""

from __future__ import annotations

import base64
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._manifest import validate_nav

if TYPE_CHECKING:
    import xarray as xr

    from ._mission import Page

__all__ = ["navigator", "report", "resolve_mission_id"]

#: Masthead nav rows, in order: (``Page.nav_group`` key, row label). The inventory group is a
#: separate strip, not a row. Built in the planned shared-design-system shape
#: (``notes/2026-10-07-shared-masthead-nav-plan.md`` §2) so the eventual vendored macro is a drop-in.
_NAV_ROWS: tuple[tuple[str, str], ...] = (("summary", "Summary"), ("reports", "Reports"), ("derived", "Derived"))


def _build_nav(pages: list[Page], current: Page, source_name: str, *, back: bool) -> dict[str, Any]:
    """Build the masthead nav for the *current* page: grouped pill rows, inventory strip, back pill.

    Shape matches the planned vendored contract (notes/2026-10-07-shared-masthead-nav-plan §2):
    ``{"rows": [{"label", "pills": [{"label", "href", "role", "state"}]}], "back": pill|None,
    "inventory": [pill]}``; ``state`` is ``"current"`` for *current*, else ``"link"``.
    """

    def pill(p: Page) -> dict[str, str]:
        return {
            "label": p.title,
            "href": p.filename,
            "role": p.role,
            "state": "current" if p is current else "link",
        }

    rows = [
        {"label": label, "pills": [pill(p) for p in group]}
        for key, label in _NAV_ROWS
        if (group := [p for p in pages if p.nav_group == key])
    ]
    # Inventory pills render as .file-pill, not .nav-<role>, so their role is cosmetic — but
    # validate_nav checks every pill's role against ROLE_ACCENT, so use a valid key ("component",
    # as ctdcast's _navs.py does) rather than the inventory Page's own "inventory" role.
    inventory = [
        {**pill(p), "label": source_name, "role": "component"}
        for p in pages
        if p.nav_group == "inventory"
    ]
    up_pill = (
        {"label": "← All missions", "href": "../index.html", "role": "up", "state": "link"}
        if back
        else None
    )
    # The up-link renders as a breadcrumb above the masthead (base.html topbar), not as an inline
    # nav pill, so it goes under "up", not "back" — the vendored page_nav draws "back" in the first
    # row, which we don't want. "back" stays None so page_nav omits it.
    nav = {"rows": rows, "back": None, "inventory": inventory, "up": up_pill}
    validate_nav(nav)  # fail loudly at build time on a malformed nav (bad role/state, >1 current)
    return nav


def resolve_mission_id(ds: xr.Dataset | None, mission_id: str | None = None) -> str:
    """Return the sanitised mission directory name: *mission_id* if given, else derived from *ds*.

    The single source for turning a dataset (and an optional caller override) into the subdirectory
    name used under a report root, shared by :func:`report` and the command-line interface so the
    two cannot disagree about where a mission is written. With *mission_id* the override is sanitised
    (path separators neutralised); *ds* is not read. Otherwise the name is the OG1 ``id`` attribute,
    else the source file stem (see :func:`glidertest.reports._mission.mission_id`).

    Parameters
    ----------
    ds : xarray.Dataset or None
        The dataset to derive the name from; may be None only when *mission_id* is given.
    mission_id : str, optional
        An explicit name to use instead of the derived one.

    Returns
    -------
    str
        The sanitised mission directory name.
    """
    from ._mission import _safe_dirname
    from ._mission import mission_id as _derive

    return _safe_dirname(mission_id) if mission_id else _derive(ds)


def report(
    ds: xr.Dataset,
    outdir: Path | str,
    *,
    navigator: bool = True,
    mission_id: str | None = None,
    layout: str = "root",
) -> Path:
    """Write a self-contained HTML report for *ds* under *outdir* and return the landing page path.

    With *layout* ``"root"`` (the default) *outdir* is treated as a **root**: the report is written
    into ``<outdir>/<mission_id>/`` (created if absent), so a root can hold many missions side by
    side. With *layout* ``"flat"`` the pages are written directly into *outdir* and no navigator is
    built (*navigator* is ignored) — one mission, one self-contained folder.

    The ``mission_id`` subdirectory name is the OG1 ``id`` attribute when present, else the source
    file stem (see :func:`glidertest.reports._mission.mission_id`); pass *mission_id* to override it
    (run through the same directory-name sanitising). Each applicable page in
    :data:`glidertest.reports._mission.PAGES` is written to ``<page>.html``; the pages share one
    ``figures/`` with page-prefixed slugs (``<page>_<panel>.png``) and a cross-page nav. A
    machine-readable ``report.json`` manifest is written beside the landing page.

    With *navigator* ``True`` (the default, root layout only), ``<outdir>/index.html`` — a fleet
    navigator over every ``<outdir>/*/report.json`` — is rebuilt after the mission is written (see
    :func:`navigator`). The rebuild redraws the fleet track map, so for a **batch** pass
    ``navigator=False`` in the loop and call :func:`navigator` once at the end rather than rebuilding
    on every mission.

    Figures are rendered under the non-interactive ``Agg`` backend for the duration of the build,
    then the caller's backend is restored. glidertest's plotters call ``plt.show()`` when they draw
    their own figure, which would pop a window per panel on an interactive backend (e.g. ``macosx``);
    ``Agg`` makes that a no-op. Switching the backend closes any figures the caller had open.

    Parameters
    ----------
    ds : xarray.Dataset
        An OG1 glider dataset.
    outdir : pathlib.Path or str
        The root directory (``layout="root"``) or the mission's own directory (``layout="flat"``).
    navigator : bool, default True
        Rebuild ``<outdir>/index.html`` (the fleet navigator) after writing the mission. Ignored
        when ``layout="flat"``.
    mission_id : str, optional
        Subdirectory name for this mission, overriding the name derived from *ds*; run through the
        same sanitising as the derived name. Must not be combined with ``layout="flat"`` (there is
        no subdirectory to name) — doing so raises :class:`ValueError`.
    layout : {"root", "flat"}, default "root"
        ``"root"`` writes the mission into ``<outdir>/<mission_id>/``; ``"flat"`` writes the pages
        directly into *outdir* and builds no navigator.

    Returns
    -------
    pathlib.Path
        The path to the written landing page (``<outdir>/<mission_id>/index.html`` for root layout,
        ``<outdir>/index.html`` for flat).
    """
    import matplotlib
    import matplotlib.pyplot as plt

    from .._version import __version__
    from . import _figdebug, paths
    from ._env import get_template
    from ._mission import PAGES, Ctx, build, header_card
    from ._report_css import _JS_TOP_LINKS, PACKAGE_ACCENT, SHARED_CSS
    from .manifest import mission_manifest

    if layout not in ("root", "flat"):
        msg = f"layout must be 'root' or 'flat', got {layout!r}"
        raise ValueError(msg)
    if mission_id is not None and layout == "flat":
        msg = "mission_id cannot be combined with layout='flat' (there is no subdirectory to name)"
        raise ValueError(msg)
    root = Path(outdir)
    if layout == "flat" and paths.is_report_root(root):
        msg = (
            f"{root} is a report root (it already holds mission subdirectories); writing a flat "
            f"report here would overwrite its fleet index.html. Use layout='root', or a fresh "
            f"directory."
        )
        raise ValueError(msg)
    mid = resolve_mission_id(ds, mission_id)
    missiondir = root if layout == "flat" else paths.mission_dir(root, mid)
    do_navigator = navigator and layout == "root"
    (missiondir / "figures").mkdir(parents=True, exist_ok=True)

    ctx = Ctx(ds=ds)
    pages = [p for p in PAGES if p.applies_to(ctx)]
    source = ds.encoding.get("source")
    source_name = Path(source).name if source else f"{ds.attrs.get('id') or mid}.nc"
    source_size = None
    if source:
        try:
            source_size = Path(source).stat().st_size
        except OSError:
            source_size = None

    # Overwriting on a re-run (same source file) is intended and silent; overwriting with a
    # *different* source file is worth a warning, whose cause differs by layout: in root layout two
    # files share one OG1 id (a metadata problem); in flat layout the output directory was reused.
    existing = paths.manifest_in(missiondir)
    if existing.exists():
        try:
            prev = json.loads(existing.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prev = {}
        prev_source = prev.get("source_file")
        if prev_source and prev_source != source_name:
            if layout == "flat":
                msg = (
                    f"{root} already holds a report for {prev_source!r}; "
                    f"overwriting with {source_name!r}."
                )
            else:
                msg = (
                    f"mission id {mid!r} already reported from {prev_source!r}; overwriting with "
                    f"{source_name!r}. Two files sharing one OG1 id is a metadata problem — check "
                    f"the 'id' attribute of both files."
                )
            warnings.warn(msg, stacklevel=2)

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    common = {
        "css": SHARED_CSS,
        "header": header_card(ds, facts=ctx.facts),
        "mission_id": mid,
        "source_name": source_name,
        "version": __version__,
        "generated_at": generated_at,
        "masthead_bg": PACKAGE_ACCENT,
        "js_top_links": _JS_TOP_LINKS,
    }
    template = get_template("mission.html")

    # Render figures headless so the plotters' plt.show() calls never pop a window; restore after.
    orig_backend = matplotlib.get_backend()
    switch = orig_backend.lower() != "agg"
    if switch:
        plt.switch_backend("Agg")
    try:
        _figdebug.clear()
        for page in pages:
            slug = page.filename.rsplit(".", 1)[0]
            resolved = build(ds, page.profile, ctx=ctx)
            for section in resolved.sections:
                for panel in section.panels:
                    if panel.kind == "figure" and panel.payload is not None:
                        (missiondir / "figures" / f"{slug}_{panel.id}.png").write_bytes(
                            base64.b64decode(panel.payload)
                        )
            # Root layout always links back to the fleet page at ../index.html (the root convention),
            # whether this call rebuilds it or a later navigator() does; flat -o has no fleet page.
            nav = _build_nav(pages, page, source_name, back=(layout == "root"))
            rendered = template.render(
                report=resolved,
                nav=nav,
                page_title=page.title,
                page_type=page.type_label,
                page_landing=page.role == "landing",
                page_intro=(page.lead(ctx) if page.lead else ""),
                **common,
            )
            # page has ✓/✗/⚠/– glyphs; Windows default is cp1252
            (missiondir / page.filename).write_text(rendered, encoding="utf-8")

        manifest = mission_manifest(
            ds,
            mission_id=mid,
            source_name=source_name,
            source_size_bytes=source_size,
            pages=[(p.filename, p.title) for p in pages],
            version=__version__,
            generated_at=generated_at,
            facts=ctx.facts,
        )
        existing.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if do_navigator:
            _build_navigator(root)  # draws the tracks map under the same Agg backend
    finally:
        if switch:
            plt.switch_backend(orig_backend)

    # The landing page is the first entry in PAGES (always applicable); return its path even if the
    # filtered `pages` were somehow empty, so the contract never depends on an IndexError-free slice.
    return missiondir / PAGES[0].filename


def navigator(root: Path | str, title: str | None = None) -> Path:
    """Rebuild ``<root>/index.html`` — a fleet navigator over every ``<root>/*/report.json``.

    Reads only the per-mission manifests (never a NetCDF file, never a mission's HTML), so it is a
    fast, idempotent re-index of whatever missions a root currently holds.

    Parameters
    ----------
    root : pathlib.Path or str
        A root directory holding ``<mission_id>/report.json`` subdirectories.
    title : str, optional
        The navigator's masthead title; defaults to the root directory name.

    Returns
    -------
    pathlib.Path
        The path to the written ``<root>/index.html``.

    Raises
    ------
    FileNotFoundError
        If *root* is not an existing directory.
    ValueError
        If *root* is itself a mission report directory (it holds a top-level ``report.json``, as a
        ``layout="flat"`` report does); indexing it would find no missions and overwrite its landing
        page. Pass the parent directory instead.
    """
    from . import paths

    root = Path(root)
    if not root.is_dir():
        msg = f"report root not found: {root}"
        raise FileNotFoundError(msg)
    if paths.manifest_in(root).exists():
        msg = (
            f"{root} is a mission report directory (it holds a top-level report.json), not a fleet "
            f"root; its index.html is the mission landing page — pass the parent directory."
        )
        raise ValueError(msg)

    import matplotlib
    import matplotlib.pyplot as plt

    orig_backend = matplotlib.get_backend()
    switch = orig_backend.lower() != "agg"
    if switch:
        plt.switch_backend("Agg")
    try:
        return _build_navigator(root, title)
    finally:
        if switch:
            plt.switch_backend(orig_backend)


def _build_navigator(root: Path, title: str | None = None) -> Path:
    """Render ``<root>/index.html`` from the manifests; caller owns the Matplotlib backend."""
    from ._navigator import build_navigator

    return build_navigator(root, title)
