"""OG1 global-attribute registry, transcribed from the OceanGliders format user manual.

Source: ``OG_Format.adoc``, "Global attributes" table, OceanGlidersCommunity/OG-format-user-manual.
The registry (:data:`ATTR_GROUPS`) is the single source of truth: it lists all 41 OG1 global
attributes, grouped into four categories, each attribute carrying its requirement tier (mandatory /
highly desirable / suggested) from the manual. :data:`GLOBAL_ATTR_ORDER` and :data:`MANDATORY_GLOBALS`
are derived from it.

Presence only — attribute *values* are not format-checked. A present value is reported as present
whatever its content, so the report's amber marking means one thing: a mandatory attribute is
missing. This does not reproduce the full OG1 compliance checker.

The manual states several contributor/institution rows conditionally ("PI name is mandatory",
"Operator role is mandatory"); those are recorded here at the mandatory tier for the PI- and
operator-level fields, matching the manual's minimum-conformance set of 16 mandatory attributes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

    import xarray as xr

#: Requirement tier of an OG1 global attribute (manual "Requirement status" column).
Tier = Literal["mandatory", "highly_desirable", "suggested"]

#: The OG1 global-attribute registry: the single source of truth for which attributes exist, their
#: category, their canonical order within the category, and their requirement tier. Categories and
#: the order within each are glidertest's grouping of the manual's flat table; the tier of each
#: attribute is transcribed from the manual's "Requirement status" column. :data:`GLOBAL_ATTR_ORDER`
#: (flat order) and :data:`MANDATORY_GLOBALS` are derived from this, so they cannot drift from it.
ATTR_GROUPS: tuple[tuple[str, tuple[tuple[str, Tier], ...]], ...] = (
    (
        "Identity & discovery",
        (
            ("title", "mandatory"),
            ("platform", "mandatory"),
            ("platform_vocabulary", "mandatory"),
            ("id", "mandatory"),
            ("naming_authority", "highly_desirable"),
            ("institution", "highly_desirable"),
            ("internal_mission_identifier", "highly_desirable"),
            ("site", "highly_desirable"),
            ("site_vocabulary", "highly_desirable"),
            ("program", "highly_desirable"),
            ("program_vocabulary", "highly_desirable"),
            ("project", "suggested"),
            ("network", "suggested"),
        ),
    ),
    (
        "Spatiotemporal coverage",
        (
            ("geospatial_lat_min", "suggested"),
            ("geospatial_lat_max", "suggested"),
            ("geospatial_lon_min", "suggested"),
            ("geospatial_lon_max", "suggested"),
            ("geospatial_vertical_min", "suggested"),
            ("geospatial_vertical_max", "suggested"),
            ("time_coverage_start", "highly_desirable"),
            ("time_coverage_end", "highly_desirable"),
        ),
    ),
    (
        "People & institutions",
        (
            ("contributor_name", "mandatory"),
            ("contributor_email", "mandatory"),
            ("contributor_id", "highly_desirable"),
            ("contributor_role", "mandatory"),
            ("contributor_role_vocabulary", "mandatory"),
            ("contributing_institutions", "mandatory"),
            ("contributing_institutions_vocabulary", "highly_desirable"),
            ("contributing_institutions_role", "mandatory"),
            ("contributing_institutions_role_vocabulary", "mandatory"),
        ),
    ),
    (
        "Provenance & processing",
        (
            ("uri", "suggested"),
            ("data_url", "highly_desirable"),
            ("doi", "highly_desirable"),
            ("rtqc_method", "mandatory"),
            ("rtqc_method_doi", "highly_desirable"),
            ("web_link", "suggested"),
            ("comment", "suggested"),
            ("start_date", "mandatory"),
            ("date_created", "mandatory"),
            ("featureType", "mandatory"),
            ("Conventions", "mandatory"),
        ),
    ),
)

#: Title for file attributes not named in :data:`ATTR_GROUPS` (never dropped).
OTHER_GROUP = "Other"

#: Tier of each registered attribute, keyed by name. Derived from :data:`ATTR_GROUPS`.
_TIER_OF: dict[str, Tier] = {
    name: tier for _title, names in ATTR_GROUPS for name, tier in names
}

#: Canonical order of OG1 global attributes: category order, then the canonical order within each
#: category. Derived from :data:`ATTR_GROUPS` so it is one list, not a second hand-kept copy.
GLOBAL_ATTR_ORDER: tuple[str, ...] = tuple(_TIER_OF)

#: The mandatory OG1 global attributes (tier ``"mandatory"`` in :data:`ATTR_GROUPS`), in canonical
#: order. The manual's minimum-conformance set.
MANDATORY_GLOBALS: tuple[str, ...] = tuple(
    name for name, tier in _TIER_OF.items() if tier == "mandatory"
)

Status = Literal["match", "none"]


def _present(attrs: Mapping[str, object], name: str) -> bool:
    """Return whether *name* is present and non-empty in *attrs*."""
    raw = attrs.get(name)
    return raw is not None and str(raw).strip() != ""


def check_globals(ds: xr.Dataset) -> list[tuple[str, Status, str]]:
    """Check whether *ds* carries each mandatory OG1 global attribute.

    Presence only — attribute values are not format-checked (see the module docstring).

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset whose ``attrs`` are checked.

    Returns
    -------
    list of (str, {"match", "none"}, str)
        One ``(attribute, status, value)`` tuple per mandatory attribute, in canonical order.
        ``"match"`` — present and non-empty; ``"none"`` — absent or present-but-empty.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams.
    """
    results: list[tuple[str, Status, str]] = []
    for attr in MANDATORY_GLOBALS:
        raw = ds.attrs.get(attr)
        value = "" if raw is None else str(raw).strip()
        status: Status = "none" if value == "" else "match"
        results.append((attr, status, "" if raw is None else str(raw)))
    return results


def order_globals(attrs: Mapping[str, object]) -> list[str]:
    """Return *attrs*' keys in OG1 canonical order, with non-OG1 keys after, in their given order.

    Keys present in :data:`GLOBAL_ATTR_ORDER` come first, in that order; any remaining key (not an
    OG1 global attribute) follows in *attrs*' own iteration order. No key is added or dropped — only
    reordered.

    Parameters
    ----------
    attrs : collections.abc.Mapping
        A dataset's global attributes (``ds.attrs``).

    Returns
    -------
    list of str
        The keys of *attrs*, canonical OG1 attributes first, then the rest in file order.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams.
    """
    canonical = set(GLOBAL_ATTR_ORDER)
    ordered = [k for k in GLOBAL_ATTR_ORDER if k in attrs]
    ordered += [k for k in attrs if k not in canonical]
    return ordered


def group_globals(attrs: Mapping[str, object]) -> list[dict[str, object]]:
    """Group *attrs* by OG1 category for the inventory's merged conformance-and-value tables.

    Walks :data:`ATTR_GROUPS` (not *attrs*), so the result includes rows for attributes the file
    *lacks* — this is what lets one table carry both the value (when present) and the conformance
    status (present / missing). The display rule follows the registry tier of each row:

    - mandatory — always shown; ``present`` is ``False`` when absent (the caller marks it amber);
    - highly desirable — always shown; absent rows carry ``present=False`` and no value;
    - suggested — shown only when present; an absent suggested attribute is omitted entirely.

    File attributes not in the registry are collected into a trailing :data:`OTHER_GROUP`, in file
    order, never dropped. Empty groups are omitted.

    Parameters
    ----------
    attrs : collections.abc.Mapping
        A dataset's global attributes (``ds.attrs``).

    Returns
    -------
    list of dict
        ``[{"title": str, "rows": [{"name", "value", "tier", "present"}, ...]}, ...]``. ``value`` is
        the stringified attribute value when present, else ``None``; ``tier`` is ``None`` for rows in
        :data:`OTHER_GROUP`.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams.
    """
    groups: list[dict[str, object]] = []
    for title, names in ATTR_GROUPS:
        rows: list[dict[str, object]] = []
        for name, tier in names:
            present = _present(attrs, name)
            if not present and tier == "suggested":
                continue
            rows.append(
                {
                    "name": name,
                    "value": str(attrs[name]) if present else None,
                    "tier": tier,
                    "present": present,
                }
            )
        if rows:
            groups.append({"title": title, "rows": rows})

    other = [
        {"name": k, "value": str(v), "tier": None, "present": True}
        for k, v in attrs.items()
        if k not in _TIER_OF
    ]
    if other:
        groups.append({"title": OTHER_GROUP, "rows": other})
    return groups


def conformance_summary(attrs: Mapping[str, object]) -> dict[str, int]:
    """Return counts for the index's one-line OG1 conformance verdict.

    Parameters
    ----------
    attrs : collections.abc.Mapping
        A dataset's global attributes (``ds.attrs``).

    Returns
    -------
    dict of str to int
        ``mandatory_present`` / ``mandatory_total`` and ``highly_desirable_missing`` — enough to
        render "OG1: N/M mandatory present · K highly-desirable missing".

    Notes
    -----
    Original Author: Eleanor Frajka-Williams.
    """
    mandatory_present = sum(1 for n in MANDATORY_GLOBALS if _present(attrs, n))
    highly_desirable = [n for n, t in _TIER_OF.items() if t == "highly_desirable"]
    highly_desirable_missing = sum(1 for n in highly_desirable if not _present(attrs, n))
    return {
        "mandatory_present": mandatory_present,
        "mandatory_total": len(MANDATORY_GLOBALS),
        "highly_desirable_missing": highly_desirable_missing,
    }
