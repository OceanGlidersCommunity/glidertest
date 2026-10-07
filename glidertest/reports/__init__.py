"""HTML report generation: slot layer, plot adapters, and the public report entry point."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr

__all__ = ["report"]


def report(ds: xr.Dataset, outdir: Path | str) -> Path:
    """Write a self-contained HTML report for *ds* into *outdir* and return the landing page path.

    Each page in :data:`glidertest.reports._mission.PAGES` that applies to *ds* is written to
    ``<outdir>/<page>.html``; the pages share one ``<outdir>/figures/`` with page-prefixed slugs
    (``<page>_<panel>.png``) and a cross-page nav. Today that is the single Mission page.

    Figures are rendered under the non-interactive ``Agg`` backend for the duration of the build,
    then the caller's backend is restored. glidertest's plotters call ``plt.show()`` when they draw
    their own figure, which would pop a window per panel on an interactive backend (e.g. ``macosx``);
    ``Agg`` makes that a no-op. Switching the backend closes any figures the caller had open.

    Parameters
    ----------
    ds : xarray.Dataset
        An OG1 glider dataset.
    outdir : pathlib.Path or str
        Output directory; created if absent.

    Returns
    -------
    pathlib.Path
        The path to the written landing page (the first applicable page).
    """
    import matplotlib
    import matplotlib.pyplot as plt

    from .._version import __version__
    from . import _figdebug
    from ._env import get_template
    from ._mission import PAGES, Ctx, build, header_card
    from ._report_css import PACKAGE_ACCENT, SHARED_CSS

    outdir = Path(outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    ctx = Ctx(ds=ds)
    pages = [p for p in PAGES if p.applies_to(ctx)]
    template = get_template("mission.html")
    common = {
        "css": SHARED_CSS,
        "header": header_card(ds),
        "mission_id": str(ds.attrs.get("id", "mission")),
        "version": __version__,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "masthead_bg": PACKAGE_ACCENT,
    }

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
                        (outdir / "figures" / f"{slug}_{panel.id}.png").write_bytes(
                            base64.b64decode(panel.payload)
                        )
            nav = [
                {"label": p.title, "href": p.filename, "role": p.role, "current": p is page}
                for p in pages
            ]
            rendered = template.render(report=resolved, nav=nav, **common)
            # page has ✓/✗/⚠/– glyphs; Windows default is cp1252
            (outdir / page.filename).write_text(rendered, encoding="utf-8")
    finally:
        if switch:
            plt.switch_backend(orig_backend)

    # The landing page is the first entry in PAGES (always applicable); return its path even if the
    # filtered `pages` were somehow empty, so the contract never depends on an IndexError-free slice.
    return outdir / PAGES[0].filename
