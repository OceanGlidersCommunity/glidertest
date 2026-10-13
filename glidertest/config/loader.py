"""Load, validate, and resolve an optional glidertest report config file.

A report config is one optional YAML file (or a plain :class:`dict`) passed to
``glidertest report --config report.yaml`` or ``reports.report(ds, outdir, config=...)``. It
overrides the built-in QC thresholds (:data:`glidertest.qc.configs`) per variable and per test,
and sets the region for the QARTOD location test. It never modifies the data — it only changes
which thresholds the diagnostics use and what the report records.

Resolution has three layers, merged in order (spec: the family config-overrides note): the built-in
defaults in :data:`glidertest.qc.configs`, then the user's ``qc:`` block deep-merged over them, with
the user winning field by field. A variable or test the file does not mention keeps its default.

The file is validated at load, before any figure is drawn: an unknown top-level key, an unknown
variable, an unknown test or field name, or a span that is not a two-element increasing numeric list
raises :class:`ConfigError` naming the allowed set. A threshold for a variable the dataset lacks is a
warning, not an error. The ``display:``, ``mission:`` and ``pages:`` blocks are recognised but not
yet applied (they land with ``feat/display-config``); their presence is reported as a warning rather
than silently ignored.

Original Author: Eleanor Frajka-Williams.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import yaml

from glidertest import qc

if TYPE_CHECKING:
    import xarray as xr

#: Severity labels for a :class:`ValidationIssue`.
ERROR = "ERROR"
WARNING = "WARNING"

#: The QARTOD tests glidertest can override, each mapped to the field names it accepts — ioos_qc's
#: own test and argument vocabulary (not its ``qartod:``-nested file format), so the same keys read
#: the same across the package family.
_QC_TEST_FIELDS: Mapping[str, frozenset[str]] = {
    "gross_range_test": frozenset({"suspect_span", "fail_span"}),
    "spike_test": frozenset({"suspect_threshold", "fail_threshold", "method"}),
    "flat_line_test": frozenset({"suspect_threshold", "fail_threshold", "tolerance"}),
}

#: Package-specific tests that sit beside the variables under ``qc:`` (not keyed by variable).
_PACKAGE_TEST_FIELDS: Mapping[str, frozenset[str]] = {
    "hysteresis": frozenset({"threshold"}),
}

#: Fields whose value must be a two-element increasing numeric list ``[min, max]``.
_SPAN_FIELDS = frozenset({"suspect_span", "fail_span"})

#: Top-level keys the file may carry. Only the active set is applied in this release; the rest are
#: parsed and carried (so the file round-trips) but reported as not-yet-applied.
_ACTIVE_KEYS = frozenset({"config_version", "source", "qc", "region"})
_DEFERRED_KEYS = frozenset({"display", "mission", "pages"})
_TOP_LEVEL_KEYS = _ACTIVE_KEYS | _DEFERRED_KEYS

#: Config file schema version this loader understands.
_SUPPORTED_VERSION = 1


class ConfigError(ValueError):
    """Raised when a report config file has one or more validation errors."""


@dataclasses.dataclass(frozen=True)
class ValidationIssue:
    """One problem found in a report config, with its severity and location.

    Attributes
    ----------
    severity : str
        :data:`ERROR` (the file is rejected) or :data:`WARNING` (applied, but flagged).
    message : str
        Human-readable description, naming the allowed set where a value was out of it.
    path : str
        Dotted path to the offending key (e.g. ``"qc.TEMP.gross_range_test.suspect_span"``),
        empty for a whole-file issue.
    """

    severity: str
    message: str
    path: str = ""

    def __str__(self) -> str:
        """Return ``SEVERITY at path: message`` (the ``at path`` omitted for a whole-file issue)."""
        where = f" at {self.path}" if self.path else ""
        return f"{self.severity}{where}: {self.message}"


@dataclasses.dataclass(frozen=True)
class Config:
    """A loaded, immutable report config.

    Holds the user's overrides as given; use :func:`resolve_qc` to merge the ``qc`` block over the
    built-in defaults, and read :attr:`region` for the location test. The ``display``, ``mission``
    and ``pages`` blocks are carried verbatim but not applied in this release.

    Attributes
    ----------
    config_version : int
        Schema version declared by the file (defaults to the supported version when absent).
    source : str or None
        Free provenance string printed on the report page and recorded in ``report.json``.
    qc : Mapping
        The user's ``qc:`` block, variable-first (plus an optional ``hysteresis`` test block), as
        given — not yet merged over the defaults.
    region : Mapping or None
        The ``region:`` block (``{"bbox": [...]}``) for the location test, or ``None`` when absent
        (then the location test is not run).
    display, mission, pages : Mapping or None
        Recognised but not applied in this release.
    path : pathlib.Path or None
        The file the config was loaded from, or ``None`` when loaded from a dict.
    extra_keys : tuple of str
        Top-level keys the file carried that glidertest does not recognise; :func:`validate` reports
        each as an error (they are not dropped silently).
    """

    config_version: int = _SUPPORTED_VERSION
    source: str | None = None
    qc: Mapping[str, Any] = dataclasses.field(default_factory=lambda: MappingProxyType({}))
    region: Mapping[str, Any] | None = None
    display: Mapping[str, Any] | None = None
    mission: Mapping[str, Any] | None = None
    pages: Mapping[str, Any] | None = None
    path: Path | None = None
    extra_keys: tuple[str, ...] = ()

    @property
    def bbox(self) -> list[float] | None:
        """Return the location-test bounding box ``[lon_min, lat_min, lon_max, lat_max]`` or None."""
        if not self.region:
            return None
        bbox = self.region.get("bbox")
        return list(bbox) if bbox is not None else None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Return *value* as a read-only mapping, or an empty one when it is not a mapping."""
    return MappingProxyType(dict(value)) if isinstance(value, Mapping) else MappingProxyType({})


def load(source: str | Path | Mapping[str, Any]) -> Config:
    """Load a report config from a YAML file path, or a plain dict, into a :class:`Config`.

    When *source* is a path, the file is read as YAML. When the loaded document (or the dict) has a
    top-level ``glidertest:`` key, only that block is used and the rest of the document is ignored —
    so a glidertest config may ride inside another tool's combined YAML. This does not validate; call
    :func:`validate` (or :func:`check`) on the result before use.

    Parameters
    ----------
    source : str or pathlib.Path or Mapping
        A path to a YAML file, or an already-parsed mapping.

    Returns
    -------
    Config
        The parsed config. Unknown keys are carried through to :func:`validate`, not dropped here.

    Raises
    ------
    ConfigError
        If a file's top-level YAML is not a mapping.
    """
    path: Path | None = None
    if isinstance(source, Mapping):
        raw: Mapping[str, Any] = source
    else:
        path = Path(source)
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, Mapping):
            msg = f"{path}: top-level YAML must be a mapping of config keys, got {type(loaded).__name__}"
            raise ConfigError(msg)
        raw = loaded

    # A glidertest config may be nested under a 'glidertest:' key inside another tool's file; when it
    # is, read nothing outside that block.
    if isinstance(raw.get("glidertest"), Mapping):
        raw = raw["glidertest"]

    return Config(
        config_version=raw.get("config_version", _SUPPORTED_VERSION),
        source=raw.get("source"),
        qc=_as_mapping(raw.get("qc")),
        region=_as_mapping(raw["region"]) if isinstance(raw.get("region"), Mapping) else None,
        display=_as_mapping(raw["display"]) if isinstance(raw.get("display"), Mapping) else None,
        mission=_as_mapping(raw["mission"]) if isinstance(raw.get("mission"), Mapping) else None,
        pages=_as_mapping(raw["pages"]) if isinstance(raw.get("pages"), Mapping) else None,
        path=path,
        extra_keys=tuple(k for k in raw if k not in _TOP_LEVEL_KEYS),
    )


def _is_number(value: Any) -> bool:
    """Return True when *value* is a real number (int or float, not bool)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _span_issue(span: Any) -> str | None:
    """Return why *span* is not a valid ``[min, max]`` span, or None when it is valid."""
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return "must be a two-element list [min, max]"
    lo, hi = span
    if not (_is_number(lo) and _is_number(hi)):
        return "both bounds must be numbers"
    if lo >= hi:
        return f"min must be less than max, got [{lo}, {hi}]"
    return None


def _allowed_vars(ds: xr.Dataset | None) -> set[str]:
    """Return the variable names a ``qc:`` block may key on: the built-in set plus any in *ds*."""
    allowed = set(qc.configs)
    if ds is not None:
        allowed |= set(ds.data_vars)
    return allowed


def _validate_test(
    test: str, fields: Mapping[str, Any], path: str, allowed: Mapping[str, frozenset[str]]
) -> list[ValidationIssue]:
    """Validate one test block's field names and span shapes against *allowed*."""
    issues: list[ValidationIssue] = []
    if test not in allowed:
        issues.append(
            ValidationIssue(ERROR, f"unknown test {test!r}; allowed: {_fmt(allowed)}", path)
        )
        return issues
    if not isinstance(fields, Mapping):
        issues.append(ValidationIssue(ERROR, f"{test} must be a mapping of fields", path))
        return issues
    for field, value in fields.items():
        fpath = f"{path}.{field}"
        if field not in allowed[test]:
            issues.append(
                ValidationIssue(
                    ERROR, f"unknown field {field!r}; allowed: {_fmt(allowed[test])}", fpath
                )
            )
            continue
        if field in _SPAN_FIELDS and (why := _span_issue(value)) is not None:
            issues.append(ValidationIssue(ERROR, f"{field} {why}", fpath))
    # A suspect span must sit inside the fail span: suspect flags a narrower band than fail.
    susp, fail = fields.get("suspect_span"), fields.get("fail_span")
    if _span_issue(susp) is None and _span_issue(fail) is None and (susp[0] < fail[0] or susp[1] > fail[1]):
        issues.append(
            ValidationIssue(
                ERROR,
                f"suspect_span {list(susp)} must sit inside fail_span {list(fail)}",
                f"{path}.suspect_span",
            )
        )
    return issues


def _validate_bbox(region: Mapping[str, Any]) -> list[ValidationIssue]:
    """Validate a ``region`` block: only ``bbox``, a 4-element increasing lon/lat box."""
    issues: list[ValidationIssue] = []
    for key in region:
        if key != "bbox":
            issues.append(ValidationIssue(ERROR, f"unknown region key {key!r}; allowed: bbox", f"region.{key}"))
    bbox = region.get("bbox")
    if bbox is None:
        return issues
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or not all(_is_number(v) for v in bbox):
        issues.append(
            ValidationIssue(ERROR, "bbox must be four numbers [lon_min, lat_min, lon_max, lat_max]", "region.bbox")
        )
        return issues
    lon_min, lat_min, lon_max, lat_max = bbox
    if lon_min >= lon_max or lat_min >= lat_max:
        issues.append(
            ValidationIssue(
                ERROR,
                f"bbox must be increasing: lon_min<lon_max and lat_min<lat_max, got {list(bbox)}",
                "region.bbox",
            )
        )
    return issues


def validate(cfg: Config, ds: xr.Dataset | None = None) -> list[ValidationIssue]:
    """Return the list of problems in *cfg*, each an :class:`ValidationIssue` (ERROR or WARNING).

    Checks the schema version, the top-level keys, every ``qc:`` variable/test/field, the span
    shapes, and the ``region.bbox``. When *ds* is given, a threshold set for a variable the dataset
    does not contain is a WARNING; without *ds*, variable names are checked only against the built-in
    set. An empty list means the config is clean.

    Parameters
    ----------
    cfg : Config
        The config to check (from :func:`load`).
    ds : xarray.Dataset, optional
        The dataset the config will run against; lets the check accept OG1 variables beyond the
        built-in set and warn about thresholds for absent variables.

    Returns
    -------
    list of ValidationIssue
        Every problem found, in file order; ERRORs reject the file, WARNINGs are advisory.
    """
    issues: list[ValidationIssue] = []

    for key in cfg.extra_keys:
        issues.append(
            ValidationIssue(ERROR, f"unknown top-level key {key!r}; allowed: {_fmt(_TOP_LEVEL_KEYS)}", key)
        )

    if cfg.config_version != _SUPPORTED_VERSION:
        issues.append(
            ValidationIssue(
                ERROR,
                f"config_version {cfg.config_version!r} not supported; this release reads version {_SUPPORTED_VERSION}",
                "config_version",
            )
        )

    for key in _DEFERRED_KEYS:
        if getattr(cfg, key) is not None:
            issues.append(
                ValidationIssue(WARNING, f"{key!r} is recognised but not applied in this release", key)
            )

    allowed_vars = _allowed_vars(ds)
    for name, block in cfg.qc.items():
        path = f"qc.{name}"
        if name in _PACKAGE_TEST_FIELDS:
            issues += _validate_test(name, block, path, _PACKAGE_TEST_FIELDS)
            continue
        if name not in allowed_vars:
            issues.append(
                ValidationIssue(ERROR, f"unknown variable {name!r}; allowed: {_fmt(allowed_vars)}", path)
            )
            continue
        if ds is not None and name not in ds.data_vars:
            issues.append(
                ValidationIssue(WARNING, f"{name!r} is not in this dataset; its thresholds will not be used", path)
            )
        if not isinstance(block, Mapping):
            issues.append(ValidationIssue(ERROR, f"{name} must be a mapping of tests", path))
            continue
        for test, fields in block.items():
            issues += _validate_test(test, fields, f"{path}.{test}", _QC_TEST_FIELDS)

    if cfg.region is not None:
        issues += _validate_bbox(cfg.region)

    return issues


def _fmt(names: Any) -> str:
    """Return a comma-separated, sorted list of *names* for an error message."""
    return ", ".join(sorted(names))


def check(cfg: Config, ds: xr.Dataset | None = None) -> Config:
    """Validate *cfg*, warn on WARNINGs, raise :class:`ConfigError` on any ERROR, and return it.

    The convenience the report entry point and the CLI use: run :func:`validate`, emit each WARNING
    through :mod:`warnings`, and if any ERROR is present raise a single :class:`ConfigError` listing
    them all (so the user sees every problem at once, not one per run).

    Parameters
    ----------
    cfg : Config
        The config to check.
    ds : xarray.Dataset, optional
        The dataset the config will run against (see :func:`validate`).

    Returns
    -------
    Config
        *cfg* unchanged, once it has passed (no ERRORs).

    Raises
    ------
    ConfigError
        If :func:`validate` returns any ERROR-severity issue.
    """
    import warnings

    issues = validate(cfg, ds)
    for issue in issues:
        if issue.severity == WARNING:
            warnings.warn(f"report config: {issue}", stacklevel=2)
    errors = [i for i in issues if i.severity == ERROR]
    if errors:
        where = f" {cfg.path}" if cfg.path else ""
        listed = "\n  ".join(str(e) for e in errors)
        msg = f"report config{where} has {len(errors)} error(s):\n  {listed}"
        raise ConfigError(msg)
    return cfg


def _deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Return *base* with *over* merged in recursively; *over* wins at every leaf."""
    merged = {k: dict(v) if isinstance(v, Mapping) else v for k, v in base.items()}
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = dict(value) if isinstance(value, Mapping) else value
    return merged


def resolve_qc(cfg: Config | None) -> dict[str, Any]:
    """Return the effective QC thresholds: *cfg*'s ``qc:`` block deep-merged over the built-in defaults.

    This is the one dict the QC checks and plots read, so the report shows exactly what ran. A
    ``None`` config (or one with no ``qc:`` block) returns a copy of :data:`glidertest.qc.configs`
    unchanged. The ``hysteresis`` test block, if present, is carried through under the ``hysteresis``
    key (it is not a variable); the location test is driven by :attr:`Config.bbox`, not by this dict.

    Parameters
    ----------
    cfg : Config or None
        The loaded config, or ``None`` for the built-in defaults only.

    Returns
    -------
    dict
        A fresh, mutable copy of the merged thresholds — safe for callers to read without touching
        the module-level defaults.
    """
    base = copy.deepcopy(qc.configs)  # fully independent of the module defaults at every level
    if cfg is None or not cfg.qc:
        return base
    return _deep_merge(base, cfg.qc)
