"""The fleet navigator: an index page over many mission reports in one root.

:func:`build_navigator` reads every ``<root>/*/report.json`` manifest (never a NetCDF file, never a
mission's HTML), draws a map of all tracks, and renders ``<root>/index.html`` — a table with one row
per mission. Idempotent: it re-indexes whatever missions the root
currently holds.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _slots, paths
from ._vendored.design_tokens import USABLE_PX

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

#: Map container height (px) — one source of truth: the template sets its ``--map-height`` CSS
#: variable from this and the coastline aspect below is derived from it, so the two cannot desync.
_MAP_HEIGHT_PX: int = 460

#: Map container aspect (width / height), used to grow the clipped coastline so land fills the map.
#: Width is the full content column (:data:`~glidertest.reports._vendored.design_tokens.USABLE_PX`); using the
#: widest layout over-covers on narrower screens, which is harmless (extra coastline off-view).
_MAP_ASPECT: float = USABLE_PX / _MAP_HEIGHT_PX


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
        mf = paths.manifest_in(sub)
        if not mf.exists():
            orphans.append(sub.name)
            continue
        try:
            manifests.append(json.loads(mf.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            logger.warning("unreadable report.json in %s: %s", sub.name, exc)
            unreadable.append(sub.name)
    return manifests, orphans, unreadable


def _track_map(missions: list[dict[str, Any]], colors: list[str]) -> str | None:
    """Render the all-tracks map as a base64 PNG, or None if no track exists.

    The print / no-JavaScript fallback for the interactive map: same cartopy projection and coastline
    as :func:`glidertest.plots.plot_glider_track` (PlateCarree with LAND/OCEAN/COASTLINE), drawing
    only the decimated manifest tracks. Each mission keeps the colour it was assigned in display
    order (*colors* is parallel to *missions*), so the PNG, the live map and the row swatch match.
    """
    tracks = [
        (m["id"], np.asarray(m["track"], dtype=float), colors[i])
        for i, m in enumerate(missions)
        if m.get("track")
    ]
    if not tracks:
        return None
    all_lon = np.concatenate([t[:, 0] for _, t, _ in tracks])
    all_lat = np.concatenate([t[:, 1] for _, t, _ in tracks])

    def draw() -> Figure:
        """Plot every mission track on one PlateCarree map, coloured and labelled by id."""
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt

        from ._vendored.design_tokens import MPLSTYLE_PATH

        pc = ccrs.PlateCarree()
        with plt.style.context(str(MPLSTYLE_PATH)):
            fig, ax = plt.subplots(subplot_kw={"projection": pc})
            for mid, pts, color in tracks:
                lon, lat = pts[:, 0], pts[:, 1]
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


def _mission_row(m: dict[str, Any], color: str) -> dict[str, Any]:
    """Return one missions-table row from a manifest, formatted for display.

    The mission id is the link to its landing page (``<id>/index.html``), and the whole row is a
    click target for the same destination; the sensor pages are reached from that page's own nav.
    *color* is the mission's assigned track colour, drawn as a swatch in the id cell so the table
    ties to the map.
    """
    og1 = m.get("og1", {})
    present, total = og1.get("mandatory_present", 0), og1.get("mandatory_total", 0)
    depth = m.get("max_depth_m")
    dur_s = m.get("duration_s")
    qcd = m.get("qc", {})
    # Worst-QC sort key: not-delivered (-2) < delivered-but-not-evaluated (-1) < 0% … worst % last, so
    # ascending runs clean→bad. The *_sort fields feed the table's client-side column sort (data-sort).
    if not qcd.get("delivered"):
        qc_sort = -2.0
    elif qcd.get("worst_bad_pct") is None:
        qc_sort = -1.0
    else:
        qc_sort = float(qcd["worst_bad_pct"])
    return {
        "id": m["id"],
        "color": color,
        "has_track": bool(m.get("track")),
        "start": (m.get("start") or "")[:10] or "UNK",  # date only; drop the HH:MM
        # Duration and max depth are bare numbers (units in the table's second header row), so the
        # columns stay narrow and sort numerically. Duration is fractional days; the masthead keeps
        # the "216d 11h" form via metadata.fmt_duration.
        "duration": f"{dur_s / 86400:.1f}" if dur_s is not None else "—",
        "duration_sort": dur_s if dur_s is not None else -1,
        "profiles": str(m.get("n_profiles", 0)),
        "profiles_sort": m.get("n_profiles", 0),
        "max_depth": f"{depth:.0f}" if depth is not None else "—",
        "depth_sort": depth if depth is not None else -1,
        "og1_text": f"{present}/{total}",
        "og1_ok": present >= total > 0,
        "og1_sort": (present / total) if total else -1,
        "qc": _qc_cell(qcd),
        "qc_sort": qc_sort,
        "report_href": f"{m['id']}/index.html",
        "generated": (m.get("generated_at") or "")[:10],  # date only; drop version + HH:MM
    }


def _map_context(
    missions: list[dict[str, Any]], colors: list[str]
) -> dict[str, Any] | None:
    """Return the interactive-map context (payload, bounds, coastline, fallback PNG), or None.

    None when no mission has a track, so the caller omits the map entirely. *colors* is parallel to
    *missions* so each mission's track keeps its assigned colour. Longitudes are used as stored: a
    fleet that straddles the antimeridian (tracks near +180 and −180) fits a wide view rather than a
    tight one — handling that properly without breaking a genuinely globe-spanning fleet is deferred
    (tracked as a follow-up issue).
    """
    from ._map_payloads import fleet_payload
    from ._vendored import leaflet as _leaflet

    payload = fleet_payload(missions, colors)
    if not payload:
        return None
    all_lat = [pt[0] for t in payload for pt in t["track"]]
    all_lon = [pt[1] for t in payload for pt in t["track"]]
    bounds = [[min(all_lat), min(all_lon)], [max(all_lat), max(all_lon)]]
    coastline = _leaflet.coastline_geojson(
        (min(all_lon), min(all_lat), max(all_lon), max(all_lat)), aspect=_MAP_ASPECT
    )
    return {
        "payload": payload,
        "bounds": bounds,
        "coastline": coastline,
        "png": _track_map(missions, colors),
    }


def navigator_data(root: Path) -> dict[str, Any]:
    """Return the navigator page as data: masthead counts, mission rows and the tracks map."""
    from ._map_payloads import track_color
    from ._mission import _deg_range

    missions, orphans, unreadable = _load_manifests(root)
    missions.sort(key=lambda m: m.get("start") or "")
    colors = [track_color(i) for i in range(len(missions))]

    # Unique vehicles by PLATFORM_SERIAL_NUMBER (not the free-text platform attribute).
    vehicles = {m.get("platform_serial") for m in missions}
    vehicles.discard(None)
    starts = sorted(m["start"] for m in missions if m.get("start"))
    ends = sorted(m["end"] for m in missions if m.get("end"))
    total_profiles = sum(m.get("n_profiles", 0) for m in missions)
    total_records = sum(m.get("n_records", 0) for m in missions)
    lats = [m[k] for m in missions for k in ("lat_min", "lat_max") if m.get(k) is not None]
    lons = [m[k] for m in missions for k in ("lon_min", "lon_max") if m.get(k) is not None]
    # Order for a 4-column meta grid: the four bulk numbers fill row 1 (most important leftmost), and
    # Start, End, Lat, Lon fill row 2 so each pair shares a row and cannot split.
    counts = [
        ("Missions", str(len(missions))),
        ("Vehicles", str(len(vehicles))),
        ("Total profiles", f"{total_profiles:,}"),
        ("Total records", f"{total_records:,}"),
        ("Start", starts[0][:10] if starts else "UNK"),
        ("End", ends[-1][:10] if ends else "UNK"),
        ("Lat", _deg_range(min(lats), max(lats), "N", "S") if lats else "UNK"),
        ("Lon", _deg_range(min(lons), max(lons), "E", "W") if lons else "UNK"),
    ]

    return {
        "counts": counts,
        "rows": [_mission_row(m, colors[i]) for i, m in enumerate(missions)],
        "map": _map_context(missions, colors),
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
    from ._vendored import leaflet as _leaflet
    from .design import _JS_TOP_LINKS, PACKAGE_ACCENT, SHARED_CSS

    root = Path(root)
    data = navigator_data(root)
    mapc = data["map"]
    # Only load the bundled Leaflet assets when there is a map to embed, so a track-less fleet (or an
    # install missing the assets) still renders its table.
    leaflet_js, leaflet_css = _leaflet.load_leaflet() if mapc else ("", "")
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
        map=mapc,
        map_height=_MAP_HEIGHT_PX,
        payload_json=_leaflet.safe_json(mapc["payload"]) if mapc else "[]",
        bounds_json=_leaflet.safe_json(mapc["bounds"]) if mapc else "null",
        coastline_json=mapc["coastline"] if mapc else "null",
        leaflet_js=leaflet_js,
        leaflet_css=leaflet_css,
        orphans=data["orphans"],
        unreadable=data["unreadable"],
    )
    out = root / "index.html"
    out.write_text(html, encoding="utf-8")
    return out
