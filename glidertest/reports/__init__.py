"""HTML report generation: slot layer, plot adapters, and the public report entry point."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr

__all__ = ["report"]


def report(ds: xr.Dataset, outdir: Path | str) -> Path:
    """Write a self-contained mission HTML report for *ds* into *outdir*.

    Builds the mission page (Mission, Track, Hydrography, Sampling sections), writes each figure
    panel to ``<outdir>/figures/<id>.png`` alongside the copy embedded in the page, and writes the
    page to ``<outdir>/mission.html``.

    This function does not set a Matplotlib backend; call it from a context that has one (the CLI
    forces ``Agg``). Figure rendering is suppressed from interactive display during the build.

    Parameters
    ----------
    ds : xarray.Dataset
        An OG1 glider dataset.
    outdir : pathlib.Path or str
        Output directory; created if absent.

    Returns
    -------
    pathlib.Path
        The path to the written ``mission.html``.
    """
    from .._version import __version__
    from . import _figdebug
    from ._env import get_template
    from ._mission import build
    from ._report_css import PACKAGE_ACCENT, SHARED_CSS

    outdir = Path(outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    _figdebug.clear()
    resolved = build(ds)
    for section in resolved.sections:
        for panel in section.panels:
            if panel.kind == "figure" and panel.payload is not None:
                (outdir / "figures" / f"{panel.id}.png").write_bytes(base64.b64decode(panel.payload))

    rendered = get_template("mission.html").render(
        report=resolved,
        css=SHARED_CSS,
        mission_id=str(ds.attrs.get("id", "mission")),
        version=__version__,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        masthead_bg="#021489",
    )
    out = outdir / "mission.html"
    out.write_text(rendered)
    return out
