"""OG1 global-attribute requirements, transcribed from the OceanGliders format user manual.

Source: ``OG_Format.adoc``, "Global attributes" table, OceanGlidersCommunity/OG-format-user-manual.
Lists the 16 mandatory global attributes and the value checks the manual states: ``start_date`` and
``date_created`` are datetime strings ``YYYYmmddTHHMMss``; ``featureType`` has the fixed value
``"trajectory"``. :func:`check_globals` reports, per attribute, whether it is present and valid.

A present-but-empty-string value counts as missing. This checks presence and the two documented
value formats only — it does not reproduce the full OG1 compliance checker.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

    import xarray as xr

#: The 16 mandatory OG1 global attributes (manual: "Global attributes" table). "PI" and "Operator"
#: rows in the manual are mandatory for the contributor and institution fields listed here.
MANDATORY_GLOBALS: tuple[str, ...] = (
    "title",
    "platform",
    "platform_vocabulary",
    "id",
    "contributor_name",
    "contributor_email",
    "contributor_role",
    "contributor_role_vocabulary",
    "contributing_institutions",
    "contributing_institutions_role",
    "contributing_institutions_role_vocabulary",
    "rtqc_method",
    "start_date",
    "date_created",
    "featureType",
    "Conventions",
)

#: Mandatory attributes whose value must be a ``YYYYmmddTHHMMss`` datetime string.
DATETIME_FORMAT_GLOBALS: tuple[str, ...] = ("start_date", "date_created")

#: Mandatory attributes with a fixed required value.
FIXED_VALUE_GLOBALS: dict[str, str] = {"featureType": "trajectory"}

#: Canonical order of OG1 global attributes, transcribed from ``OG_Format.adoc``'s "Global
#: attributes" table (document order, mandatory through suggested). Used to present a dataset's
#: global attributes predictably: :func:`order_globals` lists those present in this order, then any
#: attribute not in this list in the file's own order.
GLOBAL_ATTR_ORDER: tuple[str, ...] = (
    "title",
    "platform",
    "platform_vocabulary",
    "id",
    "naming_authority",
    "institution",
    "internal_mission_identifier",
    "geospatial_lat_min",
    "geospatial_lat_max",
    "geospatial_lon_min",
    "geospatial_lon_max",
    "geospatial_vertical_min",
    "geospatial_vertical_max",
    "time_coverage_start",
    "time_coverage_end",
    "site",
    "site_vocabulary",
    "program",
    "program_vocabulary",
    "project",
    "network",
    "contributor_name",
    "contributor_email",
    "contributor_id",
    "contributor_role",
    "contributor_role_vocabulary",
    "contributing_institutions",
    "contributing_institutions_vocabulary",
    "contributing_institutions_role",
    "contributing_institutions_role_vocabulary",
    "uri",
    "data_url",
    "doi",
    "rtqc_method",
    "rtqc_method_doi",
    "web_link",
    "comment",
    "start_date",
    "date_created",
    "featureType",
    "Conventions",
)

_DATETIME_RE = re.compile(r"^\d{8}T\d{6}$")  # YYYYmmddTHHMMss

Status = Literal["match", "differ", "none"]


def check_globals(ds: xr.Dataset) -> list[tuple[str, Status, str]]:
    """Check *ds*'s global attributes against the 16 mandatory OG1 attributes.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset whose ``attrs`` are checked.

    Returns
    -------
    list of (str, {"match", "differ", "none"}, str)
        One ``(attribute, status, value)`` tuple per mandatory attribute, in manual order.
        ``"match"`` — present and any format/value check passes; ``"differ"`` — present but the
        format or fixed value is wrong; ``"none"`` — absent or present-but-empty.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams.
    """
    results: list[tuple[str, Status, str]] = []
    for attr in MANDATORY_GLOBALS:
        raw = ds.attrs.get(attr)
        value = "" if raw is None else str(raw).strip()
        status: Status
        if value == "":
            status = "none"
        elif attr in DATETIME_FORMAT_GLOBALS and not _DATETIME_RE.match(value):
            status = "differ"
        elif attr in FIXED_VALUE_GLOBALS and value != FIXED_VALUE_GLOBALS[attr]:
            status = "differ"
        else:
            status = "match"
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
