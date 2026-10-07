"""The fleet navigator: an index page over many mission reports in one root.

:func:`build_navigator` reads every ``<root>/*/report.json`` manifest (never a NetCDF file, never a
mission's HTML), draws a map of all tracks, and renders ``<root>/index.html`` — a table with one row
per mission and a sensor-completeness matrix. Idempotent: it re-indexes whatever missions the root
currently holds.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _slots, metadata

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

#: Sensor rows of the completeness matrix: (display label, manifest ``sensors`` key).
_SENSORS = (("CTD", "ctd"), ("Oxygen", "oxygen"), ("Optics", "optics"), ("Flight", "flight"))


def _load_manifests(root: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Return ``(manifests, orphan_dirs, unreadable_dirs)`` for *root*.

    A directory with no ``report.json`` is an *orphan* (never a mission); one whose ``report.json``
    exists but cannot be read or parsed is *unreadable* — a warning is logged and it is surfaced
    separately, so a corrupt manifest is not silently dropped from the fleet counts as if it were a
    stray folder.
    """
    manifests: list[dict[str, Any]] = []
    orphans: list[str] = []
    unreadable: list[str] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        mf = sub / "report.json"
        if not mf.exists():
            orphans.append(sub.name)
            continue
        try:
            manifests.append(json.loads(mf.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            logger.warning("unreadable report.json in %s: %s", sub.name, exc)
            unreadable.append(sub.name)
    return manifests, orphans, unreadable


def _track_map(missions: list[dict[str, Any]]) -> str | None:
    """Render the all-tracks map as a base64 PNG, one colour per mission, or None if no track exists.

    Uses the same cartopy projection and coastline as :func:`glidertest.plots.plot_glider_track`
    (PlateCarree with LAND/OCEAN/COASTLINE), drawing only the decimated manifest tracks.
    """
    tracks = [(m["id"], np.asarray(m["track"], dtype=float)) for m in missions if m.get("track")]
    if not tracks:
        return None
    all_lon = np.concatenate([t[:, 0] for _, t in tracks])
    all_lat = np.concatenate([t[:, 1] for _, t in tracks])

    def draw() -> Figure:
        """Plot every mission track on one PlateCarree map, coloured and labelled by id."""
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt

        from ..config.report_tokens import MPLSTYLE_PATH

        pc = ccrs.PlateCarree()
        with plt.style.context(str(MPLSTYLE_PATH)):
            fig, ax = plt.subplots(subplot_kw={"projection": pc})
            cmap = plt.get_cmap("tab20" if len(tracks) > 10 else "tab10")
            for i, (mid, pts) in enumerate(tracks):
                lon, lat = pts[:, 0], pts[:, 1]
                color = cmap(i % cmap.N)
                ax.plot(lon, lat, color=color, lw=1.3, transform=pc)
                mp = len(lon) // 2
                ax.text(lon[mp], lat[mp], mid, fontsize=6, color=color, transform=pc)
            # Pad 1°, clamped to valid ranges so a high-latitude mission does not push the bound
            # past ±90° (which PlateCarree rejects).
            ax.set_extent(
                [
                    max(all_lon.min() - 1, -180.0),
                    min(all_lon.max() + 1, 180.0),
                    max(all_lat.min() - 1, -90.0),
                    min(all_lat.max() + 1, 90.0),
                ],
                crs=pc,
            )
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.OCEAN)
            ax.add_feature(cfeature.COASTLINE)
            gl = ax.gridlines(draw_labels=True, color="black", alpha=0.5, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False
            return fig

    return _slots.render(draw, slot="full")


def _qc_cell(qc: dict[str, Any]) -> str:
    """Return the worst-QC summary for a mission row.

    ``—`` when no ``*_QC`` was delivered or the worst evaluated flag is clean; ``not evaluated`` when
    flags are present but none were evaluated; else ``<var> N% bad``.
    """
    if not qc.get("delivered"):
        return "—"
    pct = qc.get("worst_bad_pct")
    if pct is None:
        return "not evaluated"
    if pct == 0:
        return "—"
    return f"{qc.get('worst_var')} {pct:.0f}% bad"


def _mission_row(m: dict[str, Any]) -> dict[str, Any]:
    """Return one missions-table row from a manifest, formatted for display.

    The report link is a single pill to the mission's landing page (``<id>/index.html``); the sensor
    pages are reached from that page's own nav, keeping the fleet table's report column narrow.
    """
    og1 = m.get("og1", {})
    present, total = og1.get("mandatory_present", 0), og1.get("mandatory_total", 0)
    depth = m.get("max_depth_m")
    return {
        "id": m["id"],
        "start": (m.get("start") or "")[:10] or "UNK",  # date only; drop the HH:MM
        "duration": metadata.fmt_duration(m.get("duration_s")),
        "profiles": str(m.get("n_profiles", 0)),
        "max_depth": f"{depth:.0f} m" if depth is not None else "—",
        "og1_text": f"{present}/{total}",
        "og1_ok": present >= total > 0,
        "qc": _qc_cell(m.get("qc", {})),
        "report_href": f"{m['id']}/index.html",
        "generated": (m.get("generated_at") or "")[:10],  # date only; drop version + HH:MM
    }


def navigator_data(root: Path) -> dict[str, Any]:
    """Return the navigator page as data: masthead counts, mission rows, completeness matrix, map."""
    from ._mission import _deg_range

    missions, orphans, unreadable = _load_manifests(root)
    missions.sort(key=lambda m: m.get("start") or "")

    # Unique vehicles by PLATFORM_SERIAL_NUMBER (not the free-text platform attribute).
    vehicles = {m.get("platform_serial") for m in missions}
    vehicles.discard(None)
    starts = sorted(m["start"] for m in missions if m.get("start"))
    ends = sorted(m["end"] for m in missions if m.get("end"))
    total_profiles = sum(m.get("n_profiles", 0) for m in missions)
    lats = [m[k] for m in missions for k in ("lat_min", "lat_max") if m.get(k) is not None]
    lons = [m[k] for m in missions for k in ("lon_min", "lon_max") if m.get(k) is not None]
    counts = [
        ("Missions", str(len(missions))),
        ("Vehicles", str(len(vehicles))),
        ("Total profiles", f"{total_profiles:,}"),
        ("Start", starts[0][:10] if starts else "UNK"),
        ("End", ends[-1][:10] if ends else "UNK"),
        ("Lat", _deg_range(min(lats), max(lats), "N", "S") if lats else "UNK"),
        ("Lon", _deg_range(min(lons), max(lons), "E", "W") if lons else "UNK"),
    ]

    matrix = {
        "columns": [m.get("platform_serial") or m["id"][:12] for m in missions],
        "rows": [
            {"label": label, "cells": [bool(m.get("sensors", {}).get(key)) for m in missions]}
            for label, key in _SENSORS
        ],
    }
    return {
        "counts": counts,
        "rows": [_mission_row(m) for m in missions],
        "matrix": matrix,
        "map_png": _track_map(missions),
        "orphans": orphans,
        "unreadable": unreadable,
    }


def build_navigator(root: Path | str, title: str | None = None) -> Path:
    """Render ``<root>/index.html`` from the mission manifests and return its path.

    The caller is responsible for the Matplotlib backend (the map is drawn here); use the public
    :func:`glidertest.reports.navigator` wrapper, which switches to ``Agg`` first.

    Parameters
    ----------
    root : pathlib.Path or str
        A root directory holding ``<mission_id>/report.json`` subdirectories.
    title : str, optional
        The navigator's masthead title; defaults to the root directory name (e.g. pass
        ``"VOTO 2023"`` rather than ``_smoke``).

    Returns
    -------
    pathlib.Path
        The path to the written ``<root>/index.html``.
    """
    from .._version import __version__
    from ._env import get_template
    from ._report_css import _JS_TOP_LINKS, PACKAGE_ACCENT, SHARED_CSS

    root = Path(root)
    data = navigator_data(root)
    html = get_template("navigator.html").render(
        css=SHARED_CSS,
        masthead_bg=PACKAGE_ACCENT,
        js_top_links=_JS_TOP_LINKS,
        version=__version__,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        mission_id=title or root.name or "missions",
        source_name="",
        nav={"rows": [], "back": None, "inventory": []},
        header=data["counts"],
        rows=data["rows"],
        matrix=data["matrix"],
        map_png=data["map_png"],
        orphans=data["orphans"],
        unreadable=data["unreadable"],
    )
    out = root / "index.html"
    out.write_text(html, encoding="utf-8")
    return out
