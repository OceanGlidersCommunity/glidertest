"""HTML report generation: slot layer, plot adapters, and the public report entry point."""

from __future__ import annotations

import base64
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import xarray as xr

    from ._mission import Page

__all__ = ["navigator", "report"]

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
    inventory = [
        {**pill(p), "label": source_name} for p in pages if p.nav_group == "inventory"
    ]
    back_pill = (
        {"label": "← All missions", "href": "../index.html", "role": "up", "state": "link"}
        if back
        else None
    )
    return {"rows": rows, "back": back_pill, "inventory": inventory}


def report(ds: xr.Dataset, outdir: Path | str, *, navigator: bool = True) -> Path:
    """Write a self-contained HTML report for *ds* under *outdir* and return the landing page path.

    *outdir* is treated as a **root**: the report is written into ``<outdir>/<mission_id>/`` (created
    if absent), never directly into *outdir*, so a root can hold many missions side by side. The
    ``mission_id`` is the OG1 ``id`` attribute when present, else the source file stem (see
    :func:`glidertest.reports._mission.mission_id`). Each applicable page in
    :data:`glidertest.reports._mission.PAGES` is written to ``<mission_id>/<page>.html``; the pages
    share one ``figures/`` with page-prefixed slugs (``<page>_<panel>.png``) and a cross-page nav. A
    machine-readable ``report.json`` manifest is written beside the landing page.

    With *navigator* ``True`` (the default), ``<outdir>/index.html`` — a fleet navigator over every
    ``<outdir>/*/report.json`` — is rebuilt after the mission is written (see :func:`navigator`).

    Figures are rendered under the non-interactive ``Agg`` backend for the duration of the build,
    then the caller's backend is restored. glidertest's plotters call ``plt.show()`` when they draw
    their own figure, which would pop a window per panel on an interactive backend (e.g. ``macosx``);
    ``Agg`` makes that a no-op. Switching the backend closes any figures the caller had open.

    Parameters
    ----------
    ds : xarray.Dataset
        An OG1 glider dataset.
    outdir : pathlib.Path or str
        The root directory; the mission subdirectory is created under it.
    navigator : bool, default True
        Rebuild ``<outdir>/index.html`` (the fleet navigator) after writing the mission.

    Returns
    -------
    pathlib.Path
        The path to the written landing page (``<outdir>/<mission_id>/index.html``).
    """
    import matplotlib
    import matplotlib.pyplot as plt

    from .._version import __version__
    from . import _figdebug
    from ._env import get_template
    from ._mission import PAGES, Ctx, build, header_card, mission_id
    from ._report_css import _JS_TOP_LINKS, PACKAGE_ACCENT, SHARED_CSS
    from .manifest import mission_manifest

    root = Path(outdir)
    mid = mission_id(ds)
    missiondir = root / mid
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

    # Overwriting a mission on a re-run is intended; overwriting with a *different* source file under
    # the same OG1 id is a metadata problem (two files claiming one id) — surface it, do not hide it.
    existing = missiondir / "report.json"
    if existing.exists():
        try:
            prev = json.loads(existing.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prev = {}
        if prev.get("source_file") and prev["source_file"] != source_name:
            warnings.warn(
                f"mission id {mid!r} already reported from {prev['source_file']!r}; "
                f"overwriting with {source_name!r}. Two files sharing one OG1 id is a metadata "
                f"problem — check the 'id' attribute of both files.",
                stacklevel=2,
            )

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    common = {
        "css": SHARED_CSS,
        "header": header_card(ds),
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
            resolved = build(ds, page.profile)
            for section in resolved.sections:
                for panel in section.panels:
                    if panel.kind == "figure" and panel.payload is not None:
                        (missiondir / "figures" / f"{slug}_{panel.id}.png").write_bytes(
                            base64.b64decode(panel.payload)
                        )
            nav = _build_nav(pages, page, source_name, back=navigator)
            rendered = template.render(
                report=resolved,
                nav=nav,
                page_title=page.title,
                page_type=page.type_label,
                page_landing=page.role == "landing",
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
        )
        (missiondir / "report.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if navigator:
            _build_navigator(root)  # draws the tracks map under the same Agg backend
    finally:
        if switch:
            plt.switch_backend(orig_backend)

    # The landing page is the first entry in PAGES (always applicable); return its path even if the
    # filtered `pages` were somehow empty, so the contract never depends on an IndexError-free slice.
    return missiondir / PAGES[0].filename


def navigator(root: Path | str) -> Path:
    """Rebuild ``<root>/index.html`` — a fleet navigator over every ``<root>/*/report.json``.

    Reads only the per-mission manifests (never a NetCDF file, never a mission's HTML), so it is a
    fast, idempotent re-index of whatever missions a root currently holds.

    Parameters
    ----------
    root : pathlib.Path or str
        A root directory holding ``<mission_id>/report.json`` subdirectories.

    Returns
    -------
    pathlib.Path
        The path to the written ``<root>/index.html``.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    orig_backend = matplotlib.get_backend()
    switch = orig_backend.lower() != "agg"
    if switch:
        plt.switch_backend("Agg")
    try:
        return _build_navigator(Path(root))
    finally:
        if switch:
            plt.switch_backend(orig_backend)


def _build_navigator(root: Path) -> Path:
    """Render ``<root>/index.html`` from the manifests; caller owns the Matplotlib backend."""
    from .navigator import build_navigator

    return build_navigator(root)
