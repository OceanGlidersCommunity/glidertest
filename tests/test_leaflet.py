"""Tests for the interactive Leaflet fleet map (stage 1: tracks + coastline, offline).

All offline: the coastline tests monkeypatch cartopy's shapefile reader so no Natural Earth download
is triggered, and the asset/payload tests touch no network. The navigator-rendering tests reuse the
locally cached cartopy data (as the existing navigator PNG test does).
"""

import json

from glidertest.reports import _map_payloads
from glidertest.reports._vendored import leaflet as _leaflet


def _manifest(mid, *, track):
    """Return a minimal report.json dict for a mission with the given track ([[lon, lat], ...])."""
    return {
        "manifest_version": 1,
        "id": mid,
        "platform_serial": "sea001",
        "start": "2023-06-01T00:00",
        "end": "2023-06-10T00:00",
        "duration_s": 777600,
        "n_profiles": 120,
        "n_dive": 60,
        "n_climb": 60,
        "lat_min": 58.0,
        "lat_max": 58.5,
        "lon_min": 10.0,
        "lon_max": 10.5,
        "max_depth_m": 500.0,
        "n_records": 1000,
        "source_file": f"{mid}.nc",
        "source_size_bytes": 1,
        "sensors": {"ctd": True, "oxygen": True, "optics": False, "flight": False},
        "og1": {"mandatory_present": 16, "mandatory_total": 16, "missing": []},
        "qc": {"delivered": True, "worst_var": "TEMP", "worst_bad_pct": 0.0},
        "pages": [{"file": "index.html", "label": "Mission"}],
        "track": track,
        "glidertest_version": "0.0",
        "generated_at": "2026-10-07 00:00 UTC",
    }


def _write_fleet(root, missions):
    """Write each (mid, track) as <root>/<mid>/report.json."""
    for mid, track in missions:
        d = root / mid
        d.mkdir()
        (d / "report.json").write_text(json.dumps(_manifest(mid, track=track)))


# --- bundled assets ----------------------------------------------------------


def test_leaflet_assets_bundled():
    """Leaflet JS/CSS ship as package data next to the module."""
    assert (_leaflet.LEAFLET_DIR / "leaflet.min.js").exists()
    assert (_leaflet.LEAFLET_DIR / "leaflet.min.css").exists()
    assert (_leaflet.LEAFLET_DIR / "LICENSE").exists()


def test_load_leaflet_returns_both():
    """load_leaflet returns non-empty JS and CSS; no CDN fallback is reachable here."""
    js, css = _leaflet.load_leaflet()
    assert "Leaflet" in js and _leaflet.LEAFLET_VERSION in js
    assert len(css) > 100


# --- safe_json ---------------------------------------------------------------


def test_safe_json_escapes_close_tag():
    """safe_json escapes <, > and & as \\uXXXX so a payload cannot break out of a <script> block."""
    payload = {"html": "a</script>b & <!--c-->"}
    out = _leaflet.safe_json(payload)
    # No raw <, > or & survives: none of </script>, </SCRIPT>, <!-- or a bare & can close the
    # block early or open a comment/CDATA.
    assert "<" not in out and ">" not in out and "&" not in out
    assert "\\u003c" in out  # '<' escaped
    # A JSON parser decodes the escapes back, so the data round-trips unchanged.
    assert json.loads(out) == payload


# --- track colours & payload -------------------------------------------------


def test_track_color_cycles_at_eight():
    """Eight Okabe–Ito colours, repeating from the ninth mission."""
    assert len(_map_payloads.TRACK_COLORS) == 8
    assert _map_payloads.track_color(0) == _map_payloads.track_color(8)
    assert _map_payloads.track_color(1) != _map_payloads.track_color(0)


def test_fleet_payload_swaps_lonlat_and_keeps_colour_and_href():
    """Payload swaps [lon, lat] → [lat, lon], carries the assigned colour, and omits track-less missions."""
    missions = [
        _manifest("m_a", track=[[10.0, 58.0], [10.5, 58.5]]),
        _manifest(
            "m_b", track=[]
        ),  # no track → omitted from the map, colour still consumed
        _manifest("m_c", track=[[11.0, 59.0]]),
    ]
    colors = [_map_payloads.track_color(i) for i in range(3)]
    payload = _map_payloads.fleet_payload(missions, colors)

    assert [p["id"] for p in payload] == ["m_a", "m_c"]  # m_b dropped
    assert payload[0]["track"] == [[58.0, 10.0], [58.5, 10.5]]  # lat, lon
    assert payload[0]["href"] == "m_a/index.html"
    assert payload[0]["color"] == colors[0]
    assert payload[1]["color"] == colors[2]  # m_c keeps its own colour, not m_b's


# --- coastline ---------------------------------------------------------------


class _FakeReader:
    """Stand-in for cartopy's shapefile Reader yielding one fixture land polygon."""

    def __init__(self, geoms):
        self._geoms = geoms

    def geometries(self):
        return iter(self._geoms)


def test_coastline_clips_and_rounds(monkeypatch):
    """A fixture land polygon is clipped to the padded extent and rounded to 3 dp."""
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import box

    land = box(0.123456, 0.123456, 5.0, 5.0)  # overlaps the extent; extends past it
    monkeypatch.setattr(shpreader, "natural_earth", lambda **k: "fake-path")
    monkeypatch.setattr(shpreader, "Reader", lambda _p: _FakeReader([land]))

    gj = _leaflet.coastline_geojson((0.0, 0.0, 1.0, 1.0), pad_deg=1.0)
    fc = json.loads(gj)
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    coords = fc["features"][0]["geometry"]["coordinates"][0]
    # Clipped to the padded extent [-1, 2]; every coordinate rounded to 3 decimals.
    assert all(-1.0001 <= x <= 2.0001 and -1.0001 <= y <= 2.0001 for x, y in coords)
    assert all(round(x, 3) == x and round(y, 3) == y for x, y in coords)


def test_coastline_whole_globe_for_large_span(monkeypatch):
    """A large span draws the whole globe (no clip): land far from the data is still included."""
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import box

    far_land = box(150.0, 70.0, 160.0, 80.0)  # nowhere near the data extent below
    monkeypatch.setattr(shpreader, "natural_earth", lambda **k: "fake-path")
    monkeypatch.setattr(shpreader, "Reader", lambda _p: _FakeReader([far_land]))

    gj = _leaflet.coastline_geojson(
        (-100.0, -60.0, 100.0, 60.0)
    )  # 200° span → 110m → whole globe
    fc = json.loads(gj)
    assert len(fc["features"]) == 1  # kept despite being outside the data box


def test_coastline_aspect_widens_clip(monkeypatch):
    """A wide aspect grows a tall, narrow clip in longitude so side land is included."""
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import box

    side_land = box(5.0, 4.0, 6.0, 6.0)  # east of a tall, narrow data extent
    monkeypatch.setattr(shpreader, "natural_earth", lambda **k: "fake-path")
    monkeypatch.setattr(shpreader, "Reader", lambda _p: _FakeReader([side_land]))

    tall_narrow = (0.0, 0.0, 0.5, 10.0)  # 10° span → 50m → clipped
    assert (
        _leaflet.coastline_geojson(tall_narrow) == "null"
    )  # side land outside the plain clip
    gj = _leaflet.coastline_geojson(tall_narrow, aspect=2.3)  # widened to a wide map
    assert len(json.loads(gj)["features"]) == 1  # now inside the clip


def test_coastline_returns_null_when_unavailable(monkeypatch):
    """A missing/unreadable shapefile yields the string "null", never an exception."""
    import cartopy.io.shapereader as shpreader

    def _boom(**_k):
        raise OSError("no shapefile")

    monkeypatch.setattr(shpreader, "natural_earth", _boom)
    assert _leaflet.coastline_geojson((0.0, 0.0, 1.0, 1.0)) == "null"


# --- rendered navigator ------------------------------------------------------


def test_navigator_renders_map_and_clickable_rows(tmp_path, monkeypatch):
    """The navigator embeds the live map, the fallback PNG, and rows wired to their mission."""
    monkeypatch.setattr(
        _leaflet, "coastline_geojson", lambda *a, **k: "null"
    )  # offline, deterministic
    from glidertest.reports import navigator

    _write_fleet(
        tmp_path,
        [
            ("m_a", [[10.0, 58.0], [10.2, 58.2]]),
            ("m_b", [[10.3, 58.3], [10.5, 58.5]]),
        ],
    )
    html = navigator(tmp_path).read_text(encoding="utf-8")

    assert 'id="track-map"' in html  # live map div
    assert 'id="track-map-fallback"' in html  # static PNG fallback inside it
    assert "data:image/png;base64" in html  # the fallback PNG actually rendered
    assert "L.map('track-map'" in html  # Leaflet inlined and wired
    # Print CSS: hide the live map, show the PNG.
    assert "@media print" in html and "#track-map { display: none" in html
    # Every body row carries its mission id and links to the mission via the id cell (the whole-row
    # click is wired in JS from that link's href — no mission id is interpolated into script).
    for mid in ("m_a", "m_b"):
        assert f'data-track-id="{mid}"' in html
        assert f'href="{mid}/index.html"' in html
    # Payload colour matches the row swatch for the first mission.
    first = _map_payloads.track_color(0)
    assert f"background:{first}" in html  # swatch
    assert f'"color": "{first}"' in html  # payload


def test_map_context_uses_raw_longitudes(monkeypatch):
    """A globe-spanning fleet keeps raw longitudes — no +360 unwrap (which blanked the map by
    pushing fit-bounds past the world) and every track longitude stays in [-180, 180]."""
    monkeypatch.setattr(_leaflet, "coastline_geojson", lambda *a, **k: "null")
    from glidertest.reports._navigator import _map_context

    missions = [
        _manifest("west", track=[[-60.0, -73.0], [-50.0, -70.0]]),
        _manifest("east", track=[[170.0, 60.0], [173.0, 67.0]]),
    ]
    colors = [_map_payloads.track_color(i) for i in range(2)]
    ctx = _map_context(missions, colors)

    (_lat_min, lon_min), (_lat_max, lon_max) = ctx["bounds"]
    assert lon_min == -60.0 and lon_max == 173.0  # raw span, not shifted past 180
    assert all(
        -180.0 <= lon <= 180.0 for t in ctx["payload"] for _lat, lon in t["track"]
    )


def test_navigator_omits_map_when_no_tracks(tmp_path, monkeypatch):
    """A fleet whose manifests carry no track gets no map, but still lists the missions."""
    monkeypatch.setattr(_leaflet, "coastline_geojson", lambda *a, **k: "null")
    from glidertest.reports import navigator

    _write_fleet(tmp_path, [("m_a", []), ("m_b", [])])
    html = navigator(tmp_path).read_text(encoding="utf-8")

    assert 'id="track-map"' not in html
    assert "m_a" in html and "m_b" in html  # rows still present


def test_navigator_single_point_track(tmp_path, monkeypatch):
    """A one-position track gives a degenerate but finite extent; the navigator still renders a map.

    The JS caps fitBounds with maxZoom so the zero-size extent cannot blank the map.
    """
    monkeypatch.setattr(_leaflet, "coastline_geojson", lambda *a, **k: "null")
    from glidertest.reports import navigator

    _write_fleet(tmp_path, [("m_pt", [[10.0, 58.0]])])  # single [lon, lat]
    html = navigator(tmp_path).read_text(encoding="utf-8")

    assert 'id="track-map"' in html
    assert "var BOUNDS = [[58.0, 10.0], [58.0, 10.0]]" in html  # finite, degenerate
    assert "maxZoom: 12" in html  # the cap that keeps a zero-size extent from blanking the map


def test_navigator_table_units_numeric_and_sortable(tmp_path, monkeypatch):
    """Units move to a second header row; duration is fractional days, depth bare; columns sort."""
    monkeypatch.setattr(_leaflet, "coastline_geojson", lambda *a, **k: "null")
    from glidertest.reports import navigator

    # duration_s 777600 = 9.0 days; max_depth_m 500.0 → "500" with no unit in the cell.
    _write_fleet(tmp_path, [("m_a", [[10.0, 58.0], [10.2, 58.2]])])
    html = navigator(tmp_path).read_text(encoding="utf-8")

    assert '<tr class="units">' in html  # second header row
    assert ">days<" in html and ">m<" in html  # units live there, not in the cells
    assert ">9.0<" in html  # duration as fractional days
    assert 'data-sort="777600"' in html  # duration sort key is seconds
    assert (
        ">500<" in html and ">500 m<" not in html
    )  # depth numeric, no unit in the cell
    assert "sort the missions table" in html  # the client-side column sort is wired


def test_mission_landing_has_no_leaflet(subset_report):
    """Mission pages keep their static track panel and carry no interactive map (stage 1 scope)."""
    html = (subset_report / "index.html").read_text(encoding="utf-8")
    assert 'id="track-map"' not in html
    assert "L.map(" not in html
