"""Shared units machinery — resolve a reader spelling, merge a package's view, label a unit.

Package-neutral and vendored byte-identical across the packages that share the report design
layer (see ``MANIFEST.txt``); the data it reads live beside it in
:mod:`ctdcast.reports._vendored.units_tables`.  Stdlib-only (no numpy, no xarray — nothing
here touches a dataset), so it imports with no third-party dependency and the import-purity
test guards it.

Three functions, over a frozen :class:`Tables`:

:func:`resolve_units`
    ``(units, standard_name) -> (canonical, factor)`` — what a reader spelling *is* (its
    stored spelling) and *by how much* it must be scaled to become that spelling.  The
    storage pass (each writing package's own ``to_canonical_units``) is a loop over this; it
    is kept out of here so a display-only consumer need not carry it.
:func:`merge_tables`
    a package's (or a user's) view layered over a base :class:`Tables` — the same fold at
    every layer.  A redundant override warns; a factor conflict raises.
:func:`format_units`
    the display label for a units string: it resolves first and labels the canonical for a
    ``×1`` spelling (so ``degC`` on a foreign file reads ``°C``), labels the string as given
    for a factor ``≠ 1`` spelling (``S/m`` stays ``S/m`` — the values it labels are still
    S/m), and never raises (a label must not crash a report).

:data:`DEFAULT_TABLES` is the family view, before any package override.  A package reads it
beside its own ``TABLES`` (``merge_tables(DEFAULT_TABLES, ...)``); the reserved user layer is
one more ``merge_tables`` over that.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

from . import units_tables as _t

#: Digits 0-9 mapped to their Unicode superscripts, for the exponent rule.
_SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


class UnitsError(ValueError):
    """A units string cannot be related to a canonical spelling.

    Raised by :func:`resolve_units` for a *present* spelling that is not a recognised member
    of its canonical's alias group, and by :func:`merge_tables` for an alias factor that
    disagrees with the base.  A dedicated :class:`ValueError` subclass so a caller can catch
    *this* specifically without swallowing unrelated ``ValueError`` exceptions.
    """


@dataclass(frozen=True)
class Tables:
    """A resolved view of the units tables: the family defaults, or a package's over them.

    Immutable so a view can be shared without a caller mutating a shared dict.  Built from
    :mod:`ctdcast.reports._vendored.units_tables` for :data:`DEFAULT_TABLES`, or returned by
    :func:`merge_tables` for a package or user view.

    Attributes
    ----------
    canonical : dict of str to str
        CF ``standard_name`` → stored spelling.
    aliases : dict of str to (dict of str to float)
        canonical spelling → ``{reader spelling: factor}``.
    scoped : dict of str to (dict of str to float)
        CF ``standard_name`` → ``{spelling: factor}`` for a quantity-specific spelling.
    labels : dict of str to str
        whole units string → display symbol.
    labels_by_standard_name : dict of str to str
        CF ``standard_name`` → fixed scale label.
    tokens : dict of str to str
        substring substitutions applied before the exponent rule.
    """

    canonical: dict[str, str]
    aliases: dict[str, dict[str, float]]
    scoped: dict[str, dict[str, float]]
    labels: dict[str, str]
    labels_by_standard_name: dict[str, str]
    tokens: dict[str, str]


#: The family view, before any package override.  A package builds its own ``TABLES`` with
#: :func:`merge_tables` over this; a display-only consumer uses it directly.
DEFAULT_TABLES = Tables(
    canonical=_t.CANONICAL,
    aliases=_t.ALIASES,
    scoped=_t.SCOPED_ALIASES,
    labels=_t.LABELS,
    labels_by_standard_name=_t.LABELS_BY_STANDARD_NAME,
    tokens=_t.TOKENS,
)


def merge_tables(
    base: Tables,
    *,
    canonical: dict[str, str] | None = None,
    aliases: dict[str, dict[str, float]] | None = None,
    scoped: dict[str, dict[str, float]] | None = None,
    labels: dict[str, str] | None = None,
) -> Tables:
    """Return *base* with a package's (or a user's) overrides folded in.

    The same fold at every layer, with four rules: an override entry identical to the base is
    a no-op and warns (a restatement that would silently shadow a future base change — remove
    it); a *different* canonical for a quantity the base has is a deliberate override and is
    accepted; an alias **factor** that disagrees with the base for the same spelling raises
    :class:`UnitsError` (a factor conflict is a bug, never a preference); a canonical, alias
    group, scoped group or label the base lacks is added.  Alias and scoped groups merge per
    key, so a package adds one spelling without restating the family's.

    Parameters
    ----------
    base : Tables
        The view to layer over (:data:`DEFAULT_TABLES`, or a package's ``TABLES``).
    canonical : dict of str to str, optional
        Stored-spelling overrides, keyed by CF ``standard_name``.
    aliases : dict of str to (dict of str to float), optional
        Extra reader spellings, keyed by canonical spelling.
    scoped : dict of str to (dict of str to float), optional
        Extra quantity-specific spellings, keyed by CF ``standard_name``.
    labels : dict of str to str, optional
        Display-label overrides, keyed by units string.

    Returns
    -------
    Tables
        A new frozen view; *base* is not mutated.

    Raises
    ------
    UnitsError
        If an override alias or scoped entry gives a factor that disagrees with the base for
        the same spelling.
    """
    new_canonical = dict(base.canonical)
    for sn, spelling in (canonical or {}).items():
        if base.canonical.get(sn) == spelling:
            _warn_redundant("canonical", sn)
        new_canonical[sn] = spelling

    new_labels = dict(base.labels)
    for units_str, label in (labels or {}).items():
        if base.labels.get(units_str) == label:
            _warn_redundant("label", units_str)
        new_labels[units_str] = label

    new_aliases = _merge_factor_groups(base.aliases, aliases, "alias")
    new_scoped = _merge_factor_groups(base.scoped, scoped, "scoped")

    return Tables(
        canonical=new_canonical,
        aliases=new_aliases,
        scoped=new_scoped,
        labels=new_labels,
        labels_by_standard_name=dict(base.labels_by_standard_name),
        tokens=dict(base.tokens),
    )


def _merge_factor_groups(
    base: dict[str, dict[str, float]],
    extra: dict[str, dict[str, float]] | None,
    kind: str,
) -> dict[str, dict[str, float]]:
    """Merge ``{key: {spelling: factor}}`` *extra* over *base*, per spelling.

    A spelling already in the base with the same factor warns (redundant); with a different
    factor raises :class:`UnitsError`; a new spelling is added.  *kind* names the group in the
    messages (``"alias"`` or ``"scoped"``).
    """
    merged = {key: dict(group) for key, group in base.items()}
    for key, group in (extra or {}).items():
        target = merged.setdefault(key, {})
        for spelling, factor in group.items():
            if spelling in target:
                if target[spelling] != factor:
                    raise UnitsError(
                        f"{kind} factor conflict for {spelling!r} under {key!r}: "
                        f"base {target[spelling]:g} vs override {factor:g}; a factor is a "
                        f"physical fact, not a package preference."
                    )
                _warn_redundant(kind, f"{key}:{spelling}")
            target[spelling] = factor
    return merged


def _warn_redundant(kind: str, key: str) -> None:
    """Warn that an override restates the base and will shadow a future base change."""
    warnings.warn(
        f"redundant {kind} override {key!r}: it restates the base value and will silently "
        f"shadow a future change to it — remove it from the package tables.",
        UserWarning,
        stacklevel=3,
    )


def resolve_units(
    units: str,
    standard_name: str | None = None,
    *,
    tables: Tables = DEFAULT_TABLES,
    canonical: str | None = None,
) -> tuple[str, float]:
    """Return ``(canonical spelling, factor)`` for a reader *units* string.

    The canonical is *canonical* when given (the storage pass supplies the registry spelling
    for a variable, which is known before a ``standard_name`` attribute is stamped), else
    ``tables.canonical[standard_name]`` when the quantity is known, else — for display of an
    arbitrary string — the string is matched back against the alias groups so a bare ``degC``
    resolves to ``degree_Celsius``, and failing that is returned unchanged as a
    package-decided unit.  The accepted group for a known canonical is the canonical (``×1``)
    plus :data:`Tables.aliases` for it plus :data:`Tables.scoped` for *standard_name*; a
    *units* string outside that group raises.

    Parameters
    ----------
    units : str
        The reader's units string.  An empty string raises :class:`UnitsError`.
    standard_name : str or None, optional
        CF standard name, used to find the canonical and the scoped group.
    tables : Tables, optional
        The view to resolve against; defaults to :data:`DEFAULT_TABLES`.
    canonical : str or None, optional
        The canonical spelling when the caller already knows it (by variable name); skips the
        ``standard_name`` lookup.

    Returns
    -------
    tuple of (str, float)
        The canonical spelling and the factor by which *units* values must be multiplied to
        reach it (``1.0`` for a pure spelling rewrite).

    Raises
    ------
    UnitsError
        If *units* is empty, or is a present spelling that is not a member of its canonical's
        accepted group (only when the canonical is known by *canonical* or *standard_name*).
    """
    if not units:
        raise UnitsError("empty units string cannot be resolved")

    canon = canonical
    if canon is None and standard_name is not None:
        canon = tables.canonical.get(standard_name)

    if canon is None:
        # No canonical known (display of an arbitrary string): recover it from the alias
        # groups so `degC` reads as `degree_Celsius`; otherwise the string is its own
        # canonical (a package-decided unit) and nothing is refused.
        for base_canon, group in tables.aliases.items():
            if units == base_canon:
                return base_canon, 1.0
            if units in group:
                return base_canon, group[units]
        return units, 1.0

    group = {canon: 1.0, **tables.aliases.get(canon, {})}
    if standard_name is not None:
        group.update(tables.scoped.get(standard_name, {}))
    if units not in group:
        raise UnitsError(
            f"units {units!r} is not a recognised spelling of {canon!r}"
            + (f" (standard_name {standard_name!r})" if standard_name else "")
            + "; add it to the alias tables with a verified factor or fix the reader "
            "— no silent pass-through."
        )
    return canon, group[units]


def _superscript_term(term: str) -> str:
    """Superscript a single UDUNITS term's exponent in place.

    ``m-2`` → ``m⁻²``, ``rad2`` → ``rad²``; a bare ``m`` or ``µmol`` is returned unchanged.
    Used for a multi-denominator unit, where the single ``X/Y`` slash form does not apply.
    """
    neg = re.fullmatch(r"([A-Za-z]+)-(\d+)", term)
    if neg:
        return neg.group(1) + "⁻" + neg.group(2).translate(_SUPERSCRIPT_DIGITS)
    pos = re.fullmatch(r"([A-Za-z]+)([2-9])", term)
    if pos:
        return pos.group(1) + pos.group(2).translate(_SUPERSCRIPT_DIGITS)
    return term


def format_units(
    units: str,
    standard_name: str | None = None,
    *,
    tables: Tables = DEFAULT_TABLES,
) -> str:
    """Return the display label for a CF/UDUNITS *units* string.

    The string is resolved first (:func:`resolve_units`): a ``×1`` spelling is labelled by its
    *canonical* (so a foreign file's ``degC`` reads ``°C``), a factor ``≠ 1`` spelling is
    labelled as given (``S/m`` stays ``S/m`` — the values it labels have not been scaled), and
    an unknown string is labelled as it stands (resolution never crashes a label).  The label
    itself is then: the ``standard_name`` scale label (:data:`Tables.labels_by_standard_name`),
    then the whole-string :data:`Tables.labels`, then the token substitutions and the exponent
    rule — ``X Y-1`` → ``X/Y``, ``X Y-n`` → ``X/Yⁿ``, a bare ``Y-n`` → ``Y⁻ⁿ``, a trailing
    power concatenated on a letter numerator token superscripted (``rad2 s-2`` → ``rad²/s²``),
    and every term superscripted in place for a two-or-more-denominator unit (``umol m-2 s-1``
    → ``µmol m⁻² s⁻¹``).  A time-coordinate encoding (``days since 1970-01-01``) and any
    string the rule does not match are returned unchanged.

    Parameters
    ----------
    units : str
        CF/UDUNITS units string (e.g. ``"mS cm-1"``).  An empty string returns ``""``.
    standard_name : str or None, optional
        CF standard name, used for the dimensionless turbidity / practical salinity labels and
        to resolve a scoped spelling.
    tables : Tables, optional
        The view to label against; defaults to :data:`DEFAULT_TABLES`.

    Returns
    -------
    str
        The display units string; ``""`` when *units* is empty.
    """
    if not units:
        return ""
    try:
        canonical, factor = resolve_units(units, standard_name, tables=tables)
    except UnitsError:
        canonical, factor = units, 1.0
    s = canonical if factor == 1.0 else units
    # The standard_name scale label names the canonical scale (turbidity NTU, dimensionless
    # salinity PSU); it does not apply to a not-yet-scaled factor != 1 spelling.
    if (
        factor == 1.0
        and standard_name in tables.labels_by_standard_name
        and (s == "1" or standard_name == "sea_water_turbidity")
    ):
        return tables.labels_by_standard_name[standard_name]
    if s in tables.labels:
        return tables.labels[s]
    for token, display in tables.tokens.items():
        s = s.replace(token, display)
    # A time-coordinate encoding ("<unit> since <epoch>") is not a scale unit; the exponent
    # rule would mangle the date, so leave it untouched.
    if " since " in s:
        return s
    # Two or more denominator terms (e.g. `umol m-2 s-1`) cannot be written as a single
    # `X/Y`; the single-denominator regex below would match greedily and half-convert it, so
    # superscript every term in place instead (`µmol m⁻² s⁻¹`).
    if len([t for t in s.split() if re.fullmatch(r"\S+?-\d+", t)]) >= 2:
        return " ".join(_superscript_term(t) for t in s.split())
    m = re.match(r"^(?:(.+?)\s+)?(\S+?)-(\d+)$", s)
    if not m:
        return s
    num, den, exp = m.group(1), m.group(2), m.group(3)
    # A positive power is written concatenated in UDUNITS (`rad2` = rad²), so superscript a
    # trailing single digit on a letter-only numerator token — the same rule the denominator
    # exponent gets.  Restricted to `[A-Za-z]+[2-9]` so a bare number or `m` is untouched.
    if num and re.fullmatch(r"[A-Za-z]+[2-9]", num):
        num = num[:-1] + num[-1].translate(_SUPERSCRIPT_DIGITS)
    if num:
        return (
            f"{num}/{den}"
            if exp == "1"
            else f"{num}/{den}{exp.translate(_SUPERSCRIPT_DIGITS)}"
        )
    return f"{den}⁻{exp.translate(_SUPERSCRIPT_DIGITS)}"
