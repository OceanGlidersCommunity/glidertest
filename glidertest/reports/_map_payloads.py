"""glidertest-specific map payloads for the vendored Leaflet layer.

Owned (not vendored): this turns mission manifests into the generic payload the package-neutral
:mod:`glidertest.reports._vendored.leaflet` and ``_vendored/templates/_map.html`` consume — a list of tracks, each with
a colour, a link and an info-panel HTML string. The vendored side knows nothing about missions.

Original Author: Eleanor Frajka-Williams (glidertest).
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

from . import metadata
from ._vendored.design_tokens import TRACK_COLORS as _TRACK_COLORS

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Shared track colours
# ---------------------------------------------------------------------------
# The Okabe–Ito colour-blind-safe set (strongest contrasts first, yellow last), sourced from the
# shared vendored palette so the family's packages cannot drift. One colour per mission in mission
# order; it repeats from the ninth mission. The same list colours the fallback PNG, the live-map
# polylines and the row swatch, so print and screen match.
TRACK_COLORS: tuple[str, ...] = tuple(_TRACK_COLORS)


def track_color(index: int) -> str:
    """Return the track colour for the *index*-th mission, cycling :data:`TRACK_COLORS`."""
    return TRACK_COLORS[index % len(TRACK_COLORS)]


def _info_html(m: dict[str, Any]) -> str:
    """Return the info-panel HTML for a mission: id then start, duration, profiles, max depth."""
    depth = m.get("max_depth_m")
    lines = [
        f"<strong>{html.escape(str(m['id']))}</strong>",
        (m.get("start") or "")[:10] or "UNK",
        metadata.fmt_duration(m.get("duration_s")),
        f"{m.get('n_profiles', 0)} profiles",
        f"{depth:.0f} m max depth" if depth is not None else "— max depth",
    ]
    return "<br>".join(lines)


def fleet_payload(
    missions: Sequence[dict[str, Any]],
    colors: Sequence[str],
) -> list[dict[str, Any]]:
    """Return the per-mission map payload for the fleet navigator.

    Parameters
    ----------
    missions : sequence of dict
        The mission manifests, in display order (the order the colours were assigned in).
    colors : sequence of str
        One colour per mission, parallel to *missions* (so a mission keeps its colour whether or not
        it has a track).

    Returns
    -------
    list of dict
        One entry per mission that has a track, each ``{id, href, color, track, info}`` where
        ``track`` is ``[[lat, lon], ...]`` (the manifest stores ``[lon, lat]`` — swapped here) and
        ``href`` matches the row's report link. Missions with no track are omitted from the map.
    """
    payload: list[dict[str, Any]] = []
    for m, color in zip(missions, colors):
        track = m.get("track") or []
        latlon = [[lat, lon] for lon, lat in track]
        if not latlon:
            continue
        payload.append(
            {
                "id": m["id"],
                "href": f"{m['id']}/index.html",
                "color": color,
                "track": latlon,
                "info": _info_html(m),
            }
        )
    return payload
