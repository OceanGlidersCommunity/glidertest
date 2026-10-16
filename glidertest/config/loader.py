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
import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import yaml

from glidertest import qc

if TYPE_CHECKING:
    import xarray as xr
    from ruamel.yaml.comments import CommentedMap

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

#: The spike-detection methods ioos_qc's ``spike_test`` accepts; a ``spike_test.method`` outside this
#: set is rejected at validation rather than raising deep inside qartod at render time.
_SPIKE_METHODS = frozenset({"average", "differential"})

#: Fields whose value must be a two-element increasing numeric list ``[min, max]``.
_SPAN_FIELDS = frozenset({"suspect_span", "fail_span"})

#: Top-level keys the file may carry. Only the active set is applied in this release; the rest are
#: parsed and carried (so the file round-trips) but reported as not-yet-applied.
_ACTIVE_KEYS = frozenset({"config_version", "source", "qc", "region"})
_DEFERRED_KEYS = frozenset({"display", "mission", "pages"})
_TOP_LEVEL_KEYS = _ACTIVE_KEYS | _DEFERRED_KEYS

#: Each block key mapped to the shape a wrong-typed value is reported against (``validate`` ERROR).
_BLOCK_SHAPES = {
    "qc": "a mapping of variables (or 'hysteresis') to tests",
    "region": "a mapping with a 'bbox' key",
    "display": "a mapping",
    "mission": "a mapping",
    "pages": "a mapping",
}

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
        if not isinstance(self.region, Mapping):  # absent, or wrong-typed (reported by validate)
            return None
        bbox = self.region.get("bbox")
        return list(bbox) if bbox is not None else None

    def user_blocks(self) -> dict[str, Any]:
        """Return the user's overrides as plain JSON-serialisable dicts, for ``report.json`` provenance.

        ``{"qc": {...}, "region": {...} or None, "source": str or None}`` — the ``qc:`` and
        ``region:`` blocks exactly as given (one level of nested mappings materialised to plain
        dicts), so a reader of the manifest sees what was overridden without diffing against the
        defaults.
        """
        qc = self.qc.items() if isinstance(self.qc, Mapping) else ()
        return {
            "qc": {k: dict(v) if isinstance(v, Mapping) else v for k, v in qc},
            "region": dict(self.region) if isinstance(self.region, Mapping) else None,
            "source": self.source,
        }


def _block(raw: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Return ``raw[key]`` as a read-only mapping, *default* when absent, or the raw value unchanged.

    A known block present with the wrong type (``region: [10, 50, 25, 70]``, ``qc: TEMP``) is passed
    through as-is — never silently coerced to an empty mapping — so :func:`validate` can report it as an
    ERROR alongside the config's other problems (one multi-issue report), rather than load raising on
    the first wrong type before the rest is seen.
    """
    if key not in raw or raw[key] is None:
        return default
    value = raw[key]
    return MappingProxyType(dict(value)) if isinstance(value, Mapping) else value


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
        If the file cannot be read, its YAML does not parse, its top-level document is not a mapping,
        or a known block (``qc``, ``region``, ``display``, ``mission``, ``pages``) has the wrong type.
    """
    path: Path | None = None
    if isinstance(source, Mapping):
        raw: Mapping[str, Any] = source
    else:
        path = Path(source)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            msg = f"{path}: cannot read config file ({exc.strerror or exc})"
            raise ConfigError(msg) from exc
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            msg = f"{path}: invalid YAML ({exc})"
            raise ConfigError(msg) from exc
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
        qc=_block(raw, "qc", MappingProxyType({})),
        region=_block(raw, "region"),
        display=_block(raw, "display"),
        mission=_block(raw, "mission"),
        pages=_block(raw, "pages"),
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
        if field == "method" and value not in _SPIKE_METHODS:
            issues.append(
                ValidationIssue(ERROR, f"method {value!r} not supported; allowed: {_fmt(_SPIKE_METHODS)}", fpath)
            )
    # The suspect-inside-fail cross-check is done on the *merged* spans in validate(), not here: a
    # partial override that sets only one span must be checked against the default it merges with.
    return issues


def _gross_span_issue(var: str, block: Mapping[str, Any]) -> ValidationIssue | None:
    """Return an issue when *var*'s effective gross-range suspect span is not inside its fail span.

    Checks the *merged* spans — the user's ``gross_range_test`` fields over the built-in defaults — so
    overriding only ``suspect_span`` (or only ``fail_span``) cannot leave a resolved config whose
    suspect band spills past its fail band. Returns ``None`` when either span is absent or malformed
    (the per-field shape check in :func:`_validate_test` reports those).
    """
    default = qc.configs.get(var, {}).get("gross_range_test", {})
    user = block.get("gross_range_test", {}) if isinstance(block, Mapping) else {}
    if not isinstance(user, Mapping):
        return None
    susp = user.get("suspect_span", default.get("suspect_span"))
    fail = user.get("fail_span", default.get("fail_span"))
    if _span_issue(susp) is not None or _span_issue(fail) is not None:
        return None
    if susp[0] < fail[0] or susp[1] > fail[1]:
        return ValidationIssue(
            ERROR,
            f"effective suspect_span {list(susp)} must sit inside fail_span {list(fail)} "
            "(after merging the override with the defaults)",
            f"qc.{var}.gross_range_test.suspect_span",
        )
    return None


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
        The dataset the config will run against. Without it the check is shape-only (any variable
        name is accepted); with it, a threshold set for a variable the file lacks is a WARNING.

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

    for key, shape in _BLOCK_SHAPES.items():
        value = getattr(cfg, key)
        if value is not None and not isinstance(value, Mapping):
            issues.append(ValidationIssue(ERROR, f"{key}: expected {shape}, got {type(value).__name__}", key))

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

    # A non-mapping qc block is already reported above; iterate nothing rather than crash on .items().
    for name, block in (cfg.qc.items() if isinstance(cfg.qc, Mapping) else ()):
        path = f"qc.{name}"
        if name in _PACKAGE_TEST_FIELDS:
            issues += _validate_test(name, block, path, _PACKAGE_TEST_FIELDS)
            # hysteresis validates for shape but nothing applies it yet, so warn rather than let the
            # report silently claim a threshold it never used (see resolve_qc, which drops it).
            issues.append(
                ValidationIssue(WARNING, f"qc.{name} is recognised but not applied in this release", path)
            )
            continue
        # A variable name is not checked against a vocabulary: the built-in four are defaults, not the
        # allowed set, so a user may configure CNDC/BBP700/… thresholds. Shape is still checked here
        # (no dataset needed); whether the file carries the variable is a WARNING, never an error that
        # blocks the run. With the file open that is a concrete "not in this dataset"; without it (the
        # CLI pre-check), a name outside the built-ins is flagged as a likely typo — it only applies if
        # the file turns out to have the variable.
        if ds is not None:
            if name not in ds.data_vars:
                issues.append(
                    ValidationIssue(WARNING, f"{name!r} is not in this dataset; its thresholds will not be used", path)
                )
        elif name not in qc.configs:
            issues.append(
                ValidationIssue(
                    WARNING,
                    f"{name!r} is not a built-in QC variable; its thresholds apply only if the file has it "
                    "(check for a typo)",
                    path,
                )
            )
        if not isinstance(block, Mapping):
            issues.append(ValidationIssue(ERROR, f"{name} must be a mapping of tests", path))
            continue
        for test, fields in block.items():
            issues += _validate_test(test, fields, f"{path}.{test}", _QC_TEST_FIELDS)
        if (issue := _gross_span_issue(name, block)) is not None:
            issues.append(issue)

    if isinstance(cfg.region, Mapping):  # a non-mapping region is already reported above
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
    merged = _deep_merge(base, cfg.qc)
    # hysteresis is a recognised-but-not-applied test block (validate() warns); it is not an applied
    # per-variable threshold, so it must not appear in the effective thresholds the report records.
    merged.pop("hysteresis", None)
    return merged


#: Built-in QC variables an init template seeds a ``qc:`` block for (those present in the file).
_INIT_VARS = ("TEMP", "PSAL", "DOXY", "CHLA")

def _finite_or(value: float, fallback: float) -> float:
    """Return *value* unless it is NaN, else *fallback* (for suggesters that return NaN on short data)."""
    return fallback if math.isnan(value) else value


def _init_cmap_name(var: str) -> str:
    """Return the colormap name glidertest draws *var* with (from :data:`glidertest.utilities.label_dict`).

    Derived from the actual default rather than a frozen literal, so the seeded commented ``display:``
    stub cannot advertise a colormap that no longer matches what the plots use. Falls back to ``viridis``.
    """
    from glidertest import utilities

    cmap = utilities.label_dict.get(var, {}).get("colormap")
    return getattr(cmap, "name", "viridis")


def _seeded_qc(ds: xr.Dataset, present: list[str]) -> CommentedMap:
    """Build the ``qc:`` block as a ruamel ``CommentedMap`` — suggested value, default in an eol comment.

    ruamel is imported here (only :func:`init_template` calls this), not at module load.
    """
    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    def flow_seq(values: list) -> CommentedSeq:
        seq = CommentedSeq(values)
        seq.fa.set_flow_style()
        return seq

    def num(x: float) -> int | float:
        f = float(x)
        return int(f) if f.is_integer() else f

    qc_map = CommentedMap()
    for var in present:
        d = qc.configs[var]
        gd, sd, fd = d["gross_range_test"], d["spike_test"], d["flat_line_test"]
        gr = qc.suggest_gross_range(ds, var)
        fl = qc.suggest_flat_line(ds, var)
        susp, fail = (gr["suspect"], gr["fail"]) if gr else (gd["suspect_span"], gd["fail_span"])
        # A near-constant variable gives a degenerate [v, v] span; fall back to the built-in default
        # spans so the generated template validates (min<max, suspect inside fail) as promised.
        if _span_issue(susp) is not None or _span_issue(fail) is not None or susp[0] < fail[0] or susp[1] > fail[1]:
            susp, fail = gd["suspect_span"], gd["fail_span"]
        gross = CommentedMap()
        gross["suspect_span"] = flow_seq([num(susp[0]), num(susp[1])])
        gross.yaml_add_eol_comment(f"default: [{gd['suspect_span'][0]}, {gd['suspect_span'][1]}]", "suspect_span")
        gross["fail_span"] = flow_seq([num(fail[0]), num(fail[1])])
        gross.yaml_add_eol_comment(f"default: [{gd['fail_span'][0]}, {gd['fail_span'][1]}]", "fail_span")
        spike = CommentedMap()
        spike["suspect_threshold"] = num(sd["suspect_threshold"])
        spike.yaml_add_eol_comment("default (no mission-derived suggestion)", "suspect_threshold")
        spike["fail_threshold"] = num(sd["fail_threshold"])
        spike.yaml_add_eol_comment("default (no mission-derived suggestion)", "fail_threshold")
        flat = CommentedMap()
        flat["suspect_threshold"] = num(_finite_or(fl["suspect"], fd["suspect_threshold"]))
        flat.yaml_add_eol_comment(f"seconds; default: {num(fd['suspect_threshold'])}", "suspect_threshold")
        flat["fail_threshold"] = num(_finite_or(fl["fail"], fd["fail_threshold"]))
        flat.yaml_add_eol_comment(f"seconds; default: {num(fd['fail_threshold'])}", "fail_threshold")
        flat["tolerance"] = fd["tolerance"]
        flat.yaml_add_eol_comment(f"default: {fd['tolerance']:g}", "tolerance")
        block = CommentedMap()
        block["gross_range_test"] = gross
        block["spike_test"] = spike
        block["flat_line_test"] = flat
        qc_map[var] = block
    return qc_map


def init_template(ds: xr.Dataset) -> str:
    """Return a commented report-config YAML for *ds*, seeded from its mission-derived suggestions.

    For each built-in QC variable present in *ds*, the ``qc:`` block carries glidertest's suggested
    gross-range and flat-line thresholds for *this* mission (:func:`glidertest.qc.suggest_gross_range`,
    :func:`~glidertest.qc.suggest_flat_line`) as the value, with the built-in default shown in a
    comment beside each so the user sees both. Spike thresholds have no mission-derived suggestion, so
    they take the built-in default. ``region.bbox`` is left commented out — a box drawn from the data
    cannot fail its own region test, so the user must set it deliberately. The override categories the
    loader recognises but does not yet apply (``qc.hysteresis`` and the top-level ``display``,
    ``mission`` and ``pages`` blocks) are emitted commented-out, so the template shows the whole menu
    without tripping their not-applied warnings. ``glidertest init`` writes this for the user to edit
    and pass back with ``glidertest report -c``.

    The ``config_version``/``source``/``qc`` data block is emitted with ``ruamel.yaml`` (not string
    concatenation), so the per-value default comments are attached as real YAML comments and a value
    carrying a YAML-special character (e.g. a ``source`` with a colon) round-trips through a later edit.

    Parameters
    ----------
    ds : xarray.Dataset
        An OG1 glider dataset.

    Returns
    -------
    str
        The YAML template text (with comments); loads and validates clean as written.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams.
    """
    import io

    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap

    present = [v for v in _INIT_VARS if v in ds.data_vars]
    doc = CommentedMap()
    doc["config_version"] = _SUPPORTED_VERSION
    doc["source"] = ""
    doc.yaml_add_eol_comment("free text recorded in report.json (qc_config_source)", "source")
    doc["qc"] = _seeded_qc(ds, present)

    buf = io.StringIO()
    yaml_rt = YAML()
    yaml_rt.indent(mapping=2, sequence=2, offset=0)
    yaml_rt.dump(doc, buf)
    body = buf.getvalue()

    header = (
        "# glidertest report config — generated by `glidertest init`.\n"
        "# Edit the values to override, then run: glidertest report -c <this file> <mission.nc>\n"
        "# Thresholds are in each variable's OG1 units. Values below are glidertest's suggestions for\n"
        "# THIS mission (from its data distribution); the built-in default is in a comment beside each.\n"
        "#\n"
        "# region:   # uncomment and set to run the region test (samples outside the box are flagged fail)\n"
        "#   bbox: [west, south, east, north]   # e.g. [10, 50, 25, 70] = lon_min, lat_min, lon_max, lat_max\n"
    )
    # Override categories the loader recognises but does not apply in this release (qc.hysteresis and the
    # top-level display/mission/pages blocks): appended commented-out so the template shows the whole menu
    # without tripping the not-applied warnings — and so the generated file still round-trips to zero issues.
    menu = [
        "",
        "  # hysteresis — sits beside the variables, not keyed by a variable. Recognised and recorded",
        "  # for provenance; NOT applied yet (uncommenting it makes `report -c` print a warning).",
        "  # hysteresis:",
        "  #   threshold: 0.5",
        "",
        "# display — per-variable plot overrides. Recognised and recorded; NOT applied yet. Shape follows",
        "# the package family: display.variables.<var> with cmap (a cmocean/matplotlib name) and",
        "# vmin/vmax (soft axis limits). The cmap shown is glidertest's current default for the variable.",
        "# display:",
        "#   variables:",
    ]
    for v in present:
        menu += [f"#     {v}:", f"#       cmap: {_init_cmap_name(v)}", "#       vmin:", "#       vmax:"]
    menu += [
        "",
        "# mission — display-only metadata (title, project, fleet label), shown with provenance, never",
        "# merged into the QC verdict. Recognised and recorded; NOT applied yet.",
        "# pages — which pages to build, e.g. {ctd, oxygen, optics, flight, inventory}: true/false.",
        "# Recognised and recorded; NOT applied yet.",
    ]
    return header + body + "\n".join(menu) + "\n"
