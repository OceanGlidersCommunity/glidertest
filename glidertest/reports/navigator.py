"""The fleet navigator: an index page over many mission reports in one root.

:func:`build_navigator` reads every ``<root>/*/report.json`` manifest (never a NetCDF file, never a
mission's HTML), draws a map of all tracks, and renders ``<root>/index.html`` — a table with one row
per mission and a sensor-completeness matrix. Idempotent: it re-indexes whatever missions the root
currently holds.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _slots

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: Sensor rows of the completeness matrix: (display label, manifest ``sensors`` key).
_SENSORS = (("CTD", "ctd"), ("Oxygen", "oxygen"), ("Optics", "optics"), ("Flight", "flight"))


def _load_manifests(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Return ``(manifests, orphan_dirs)`` for *root*: parsed ``report.json``s and dirs lacking one."""
    manifests: list[dict[str, Any]] = []
    orphans: list[str] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        mf = sub / "report.json"
        if not mf.exists():
            orphans.append(sub.name)
            continue
        try:
            manifests.append(json.loads(mf.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            orphans.append(sub.name)
    return manifests, orphans


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
            ax.set_extent(
                [all_lon.min() - 1, all_lon.max() + 1, all_lat.min() - 1, all_lat.max() + 1], crs=pc
            )
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.OCEAN)
            ax.add_feature(cfeature.COASTLINE)
            gl = ax.gridlines(draw_labels=True, color="black", alpha=0.5, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False
            return fig

    return _slots.render(draw, slot="full")


def _fmt_duration(seconds: int | None) -> str:
    """Return a ``Nd Nh`` duration, or ``UNK`` when the span is unknown."""
    if seconds is None:
        return "UNK"
    days, rem = divmod(int(seconds), 86_400)
    return f"{days}d {rem // 3600}h"


def _qc_cell(qc: dict[str, Any]) -> str:
    """Return the worst-QC summary for a mission row (e.g. ``DOXY 100% bad``)."""
    if not qc.get("delivered"):
        return "no flags"
    var, pct = qc.get("worst_var"), qc.get("worst_bad_pct")
    if var is None or pct is None:
        return "—"
    if pct == 0:
        return "clean"
    return f"{var} {pct:.0f}% bad"


def _mission_row(m: dict[str, Any], roles: dict[str, str]) -> dict[str, Any]:
    """Return one missions-table row from a manifest, formatted for display."""
    og1 = m.get("og1", {})
    present, total = og1.get("mandatory_present", 0), og1.get("mandatory_total", 0)
    depth = m.get("max_depth_m")
    return {
        "id": m["id"],
        "platform": m.get("platform_serial") or m.get("platform") or "—",
        "start": (m.get("start") or "")[:10] or "UNK",  # date only; drop the HH:MM
        "end": (m.get("end") or "")[:10] or "UNK",
        "duration": _fmt_duration(m.get("duration_s")),
        "profiles": str(m.get("n_profiles", 0)),
        "max_depth": f"{depth:.0f} m" if depth is not None else "—",
        "sensors": m.get("sensors", {}),
        "og1_text": f"{present}/{total}",
        "og1_ok": present >= total > 0,
        "qc": _qc_cell(m.get("qc", {})),
        "pages": [
            {"href": f"{m['id']}/{p['file']}", "label": p["label"], "role": roles.get(p["file"], "component")}
            for p in m.get("pages", [])
        ],
        "generated": f"{m.get('glidertest_version', '?')} · {(m.get('generated_at') or '')[:10]}",
    }


def navigator_data(root: Path) -> dict[str, Any]:
    """Return the navigator page as data: masthead counts, mission rows, completeness matrix, map."""
    from ._mission import PAGES, _deg_range

    roles = {p.filename: p.role for p in PAGES}
    missions, orphans = _load_manifests(root)
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
        "rows": [_mission_row(m, roles) for m in missions],
        "matrix": matrix,
        "map_png": _track_map(missions),
        "orphans": orphans,
    }


def build_navigator(root: Path | str) -> Path:
    """Render ``<root>/index.html`` from the mission manifests and return its path.

    The caller is responsible for the Matplotlib backend (the map is drawn here); use the public
    :func:`glidertest.reports.navigator` wrapper, which switches to ``Agg`` first.

    Parameters
    ----------
    root : pathlib.Path or str
        A root directory holding ``<mission_id>/report.json`` subdirectories.

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
        mission_id=root.name or "missions",
        source_name="",
        nav={"rows": [], "back": None, "inventory": []},
        header=data["counts"],
        rows=data["rows"],
        matrix=data["matrix"],
        map_png=data["map_png"],
        orphans=data["orphans"],
    )
    out = root / "index.html"
    out.write_text(html, encoding="utf-8")
    return out
