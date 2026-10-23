"""Package-neutral Leaflet building blocks for the report maps.

Part of the vendored report layer (see ``MANIFEST.txt``), authored in glidertest
first (Eleanor, 2026-10-10) and kept byte-identical across ctdcast, oceanarray and
glidertest, with ctdcast ``main`` as the source of truth.  It names no package
and knows nothing about missions, dives or casts: it ships the bundled Leaflet
assets, JSON-encodes a payload for embedding in a ``<script>`` block, and turns a
geographic extent into coastline GeoJSON.  The glidertest-specific payload builder
lives in the owned :mod:`glidertest.reports._map_payloads`.

Original Author: Eleanor Frajka-Williams (glidertest), porting the Leaflet asset
loader from ctdcast ``reports/_leaflet.py``.
"""

from __future__ import annotations

import functools
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from shapely.geometry import Polygon
    from shapely.geometry.base import BaseGeometry

#: Bundled Leaflet assets live next to this module (shipped as package data).
LEAFLET_DIR: Path = Path(__file__).parent / "leaflet"
LEAFLET_VERSION: str = "1.9.4"


def safe_json(obj: object) -> str:
    """Return ``obj`` as JSON safe to embed in a ``<script>`` block.

    Escapes every ``<``, ``>`` and ``&`` as its ``\\uXXXX`` form, so no string value can close
    the block early (``</script>`` in any case, ``</SCRIPT>``) or open a comment / CDATA
    (``<!--``).  A JSON parser decodes the escapes back, so the data is unchanged.

    Parameters
    ----------
    obj : object
        Any JSON-serialisable value (the map payload: lists of tracks, colours, HTML strings).

    Returns
    -------
    str
        A JSON string safe to drop between ``<script>`` tags.
    """
    return (
        json.dumps(obj, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


@functools.cache
def load_leaflet() -> tuple[str, str]:
    """Return ``(js_text, css_text)`` for the bundled Leaflet assets.

    Cached: the ~150 KB of JS/CSS is read once per process, not re-read for every page in a build.
    Unlike ctdcast's loader there is **no CDN fallback**: a missing asset is a packaging bug, not a
    reason to need the internet at build time, so it raises rather than silently fetching.

    Returns
    -------
    tuple of (str, str)
        The JavaScript and CSS source read from the bundled files, inlined into the page so the map
        works offline at view time.

    Raises
    ------
    FileNotFoundError
        If either bundled asset is absent (a packaging error a test should catch).
    """
    js = (LEAFLET_DIR / "leaflet.min.js").read_text(encoding="utf-8")
    css = (LEAFLET_DIR / "leaflet.min.css").read_text(encoding="utf-8")
    return js, css


def _resolution_for_span(span_deg: float) -> str:
    """Return the Natural Earth land resolution (``"10m"``/``"50m"``/``"110m"``) for a degree span.

    Finer detail for a small extent (a single coastal mission), coarser for a fleet spanning
    degrees — a fleet root is usually the ``50m``/``110m`` case.
    """
    if span_deg < 3.0:
        return "10m"
    if span_deg <= 30.0:
        return "50m"
    return "110m"


#: Simplify tolerance (degrees) per resolution — coarse enough to shrink the embedded JSON, fine
#: enough that the coastline still reads at that extent.
_SIMPLIFY_TOL: dict[str, float] = {"10m": 0.003, "50m": 0.01, "110m": 0.02}


def _polygons(geom: BaseGeometry) -> Iterator[Polygon]:
    """Yield each shapely ``Polygon`` in *geom*, descending into multi/collection geometries.

    Dropping non-polygonal parts (the stray lines/points a box intersection can leave) so the
    coastline layer carries only filled land.
    """
    gt = geom.geom_type
    if gt == "Polygon":
        yield geom
    elif gt in ("MultiPolygon", "GeometryCollection"):
        for g in geom.geoms:
            yield from _polygons(g)


def _round_nested(seq: Any, nd: int) -> Any:  # nested coordinate sequence
    """Round a nested GeoJSON coordinate sequence to *nd* decimals, bottoming out at ``[x, y]``."""
    if seq and isinstance(seq[0], (int, float)):
        return [round(float(v), nd) for v in seq]
    return [_round_nested(s, nd) for s in seq]


def _expand_to_aspect(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float, aspect: float
) -> tuple[float, float, float, float]:
    """Grow the box about its centre so its Web-Mercator aspect matches *aspect* (width / height).

    The map container is wider than it is tall, so fitting a tall, narrow data extent into it leaves
    blank sides; widening the clip box to the container's aspect means land fills those sides instead
    of empty ocean. Mercator stretches latitude by ``1 / cos(lat)``, so the comparison is made in
    projected units. Only ever enlarges, symmetric about the centre.
    """
    lon_span, lat_span = lon_max - lon_min, lat_max - lat_min
    coslat = max(0.1, math.cos(math.radians((lat_min + lat_max) / 2.0)))
    proj_w, proj_h = lon_span, lat_span / coslat
    if proj_w <= 0 or proj_h <= 0:
        return lon_min, lat_min, lon_max, lat_max
    if proj_w / proj_h < aspect:  # too narrow for the container → widen longitude
        pad = (aspect * proj_h - lon_span) / 2.0
        lon_min, lon_max = lon_min - pad, lon_max + pad
    else:  # too wide → heighten latitude
        pad = (proj_w / aspect * coslat - lat_span) / 2.0
        lat_min, lat_max = lat_min - pad, lat_max + pad
    return lon_min, lat_min, lon_max, lat_max


def coastline_geojson(
    bounds: tuple[float, float, float, float],
    *,
    pad_deg: float = 1.0,
    round_ndigits: int = 3,
    aspect: float | None = None,
) -> str:
    """Return land-polygon GeoJSON for *bounds*, or the string ``"null"`` if unavailable.

    Requires cartopy (imported lazily); returns ``"null"`` when it is unavailable — the owning
    caller must log that.  Natural Earth land polygons (via cartopy's shapereader) are clipped to
    the padded extent, simplified, and rounded to *round_ndigits* decimals (3 ≈ 110 m, finer than
    Natural Earth ever resolves).  The resolution is chosen from the extent span by
    :func:`_resolution_for_span`; a span large enough to resolve to ``110m`` draws the whole globe
    (no clip) so land stays visible when the viewer pans or zooms out.

    Parameters
    ----------
    bounds : tuple of float
        ``(lon_min, lat_min, lon_max, lat_max)`` in degrees — the data extent to draw land around.
    pad_deg : float, default 1.0
        Degrees of context added on every side before clipping, so land does not stop at the data.
    round_ndigits : int, default 3
        Decimal places kept on each coordinate; 3 (~110 m) keeps the embedded JSON small.
    aspect : float, optional
        Width / height of the map container. When given (and the extent is clipped, not whole-globe),
        the clip box is grown to this aspect so land fills a wide map instead of leaving blank sides
        where the view letterboxes a tall, narrow extent.

    Returns
    -------
    str
        A GeoJSON ``FeatureCollection`` string, or ``"null"`` when the shapefile is unavailable
        (offline, not yet downloaded, or a read error) — the map then draws tracks on a plain
        background rather than raising.
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import box, mapping

        resolution = _resolution_for_span(max(lon_max - lon_min, lat_max - lat_min))
        tol = _SIMPLIFY_TOL[resolution]
        # A large span resolves to 110m, where the whole-world land file is small: draw the entire
        # globe (no clip) so land stays visible wherever the viewer pans or zooms out, while the view
        # still opens on the data (the caller fits to the data min/max). A small/medium extent is
        # clipped to the padded data extent (grown to the map's aspect, if given) so a coastal
        # deployment keeps fine detail without embedding the fine-resolution whole world.
        if resolution == "110m":
            clip = None
        else:
            lo0, la0, lo1, la1 = bounds
            if aspect:
                lo0, la0, lo1, la1 = _expand_to_aspect(lo0, la0, lo1, la1, aspect)
            clip = box(
                lo0 - pad_deg,
                max(la0 - pad_deg, -90.0),
                lo1 + pad_deg,
                min(la1 + pad_deg, 90.0),
            )
        path = shpreader.natural_earth(
            resolution=resolution, category="physical", name="land"
        )
        reader = shpreader.Reader(path)

        features: list[dict[str, Any]] = []
        for geom in reader.geometries():
            if clip is None:
                shape = geom
            elif geom.intersects(clip):
                shape = geom.intersection(clip)
            else:
                continue
            shape = shape.simplify(tol, preserve_topology=True)
            for poly in _polygons(shape):
                if poly.is_empty:
                    continue
                gj = mapping(poly)
                gj["coordinates"] = _round_nested(gj["coordinates"], round_ndigits)
                features.append({"type": "Feature", "geometry": gj, "properties": {}})

        if not features:
            return "null"
        return safe_json({"type": "FeatureCollection", "features": features})
    except ImportError:
        return (
            "null"  # cartopy / shapely not installed — plain background, as documented
        )
    except OSError:
        return "null"  # the Natural Earth shapefile is absent or unreadable offline
    except Exception as exc:  # noqa: BLE001  # a geometry error must not blank the map, but do not swallow it silently
        logger.debug("coastline geometry failed, using plain background: %s", exc)
        return "null"
