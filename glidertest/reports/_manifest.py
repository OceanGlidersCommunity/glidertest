"""Section-manifest model and resolver for report pages.

A report page is described by a :class:`Profile`: an ordered sequence of
:class:`Section` (and :class:`Expand`) entries, each naming :class:`Panel` ids
that render figures or tables.  :func:`resolve` walks a profile once against a
render context and returns a :class:`ResolvedReport` whose sections are numbered
over the *rendered subset* — absent sections leave no gap, and identity is the
section id (a stable slug), not the integer.

This module holds only the model and the resolution algorithm — it is
package-neutral (names no variable, page, or science) and is vendored
byte-identical to the sister repos.  Each page's concrete registry (its
``Ctx`` builder, panels, sections, and profiles) lives in that page's own
module — grid's in ``reports/_grid.py``, and so on — never in a plural
``_manifests.py`` companion.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

#: A page's render context, threaded opaque through the resolver: the concrete keys are
#: decided by each page's own registry (grid's ``Ctx``, etc.), never by this package-neutral
#: module.  Named once so the resolver signatures read ``ctx: RenderContext`` with no
#: per-signature noqa, and any future narrowing lands in one place.
RenderContext = Any

#: Panel content kinds.  ``"figure"`` renders a base64 PNG (escaped as an image
#: ``src``); ``"html"`` and ``"table"`` render pre-built markup that the template
#: macro emits with ``|safe``.  The discriminator keeps the ``autoescape=True``
#: boundary to one auditable branch — a figure payload is never ``|safe``-d.
PanelKind = Literal["figure", "html", "table"]


def _always(_ctx: RenderContext) -> bool:
    """Return True for any context (the default ``applies_to`` predicate)."""
    return True


def _unavailable_never(_ctx: RenderContext) -> str | None:
    """Return None for any context (the default ``unavailable_if`` predicate)."""
    return None


# ---------------------------------------------------------------------------
# Authored model — what a profile is built from
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Panel:
    """One figure or table, addressable by id.

    Parameters
    ----------
    id : str
        Unique panel identifier, e.g. ``"temperature_field"``.  Also the anchor
        a caption or cross-reference can point at.
    render : Callable
        Adapter that returns the panel's payload — a base64 PNG string for a
        ``"figure"``, or ready-to-emit markup for ``"html"``/``"table"`` — or
        ``None`` when the panel applies but no output could be produced (a plot
        raised, or the variable is absent), which the resolver turns into a
        ``.warn`` stub.  Figure adapters wrap the existing ``_make_*_b64``
        functions unchanged.
    kind : {"figure", "html", "table"}, optional
        Content discriminator.  The template macro branches on it so that only
        ``"html"``/``"table"`` payloads are emitted ``|safe``; a ``"figure"``
        payload is always escaped into an image ``src``.
    slot : str or Callable or None, optional
        ``SLOTS`` key giving the panel's display width, or a callable
        ``ctx -> slot`` that computes it (e.g. from a section's aspect ratio).
        Belongs to the panel, not the section, so a panel is the same width
        wherever it is placed.  The resolver calls it when callable.  ``None``
        marks a figure that is not rendered through the slot system: it is
        emitted as a bare ``.fig`` that fills the content column (no
        ``slot-*`` class, so no rendered-width contract).
    caption : str, optional
        Caption text rendered beneath the panel.
    applies_to : Callable, optional
        Predicate deciding whether the panel is attempted at all.  ``False``
        omits the panel silently (it is excluded, not merely unavailable).
    unavailable_if : Callable, optional
        Precondition checked *before* ``render``: given the context, return a
        reason string when the panel applies but cannot be produced (e.g. a
        required metadata field is missing), or ``None`` to proceed.  A returned
        reason becomes a ``.warn`` stub *with that reason* and ``render`` is not
        called.  This is the channel for defects knowable from context; a
        ``render`` that still returns ``None`` gets the generic stub reason,
        because that is the "should have worked and didn't" case.

    """

    id: str
    render: Callable[[Any], str | None]
    kind: PanelKind = "figure"
    slot: str | Callable[[Any], str] | None = "full"
    caption: str | None = None
    applies_to: Callable[[Any], bool] = _always
    unavailable_if: Callable[[Any], str | None] = _unavailable_never


@dataclass(frozen=True)
class PanelGroup:
    """A :attr:`Section.panels` entry that expands to N panels at resolution time.

    Places a data-driven run of panels — one per item — under a single heading
    without inflating the section count.  Use this (not :class:`Expand`) when the
    run is over a *numeric or unbounded* set, e.g. one panel per isopycnal;
    reserve :class:`Expand`'s section-level expansion for a *closed editorial
    vocabulary*.  Numbering reflects editorial structure, not data cardinality.

    Parameters
    ----------
    over : Callable
        Returns the sequence of items to expand over, in display order.
    panel : Callable
        Builds one :class:`Panel` per item yielded by ``over``.

    """

    over: Callable[[Any], Sequence[Any]]
    panel: Callable[[Any], Panel]


@dataclass(frozen=True)
class Section:
    """A numbered heading and the panels beneath it.

    Parameters
    ----------
    id : str
        Unique section identifier, e.g. ``"hydrography"``.  Doubles as the anchor
        and the cross-reference key.
    title : str
        Human-readable heading text (without a number — the number is computed).
    panels : tuple of (str or PanelGroup)
        Panel ids, in display order.  An entry may instead be a
        :class:`PanelGroup`, which expands to a data-driven run of panels under
        this one heading.
    level : int, optional
        Heading level (``2`` for ``<h2>``).
    intro : str, optional
        Introductory prose rendered under the heading, before the panels.
    applies_to : Callable, optional
        Predicate deciding whether the section is included.  When ``None``
        (default), the section applies if *any* of its panels apply.  A section
        dropped here is named in the report's "not applicable" footer line.
    role : str, optional
        ``"content"`` (numbered ``1..N``) or ``"appendix"`` (numbered ``A..``),
        so appendix material such as a NetCDF-variable table does not pad the
        science numbering.
    layout : str, optional
        Within-section panel arrangement.  ``None`` (default) stacks panels
        vertically; ``"row"`` lays them out in a wrapping flex row, each panel in
        a cell sized by its ``slot`` (so two ``slot="half"`` figures sit
        side-by-side and a ``slot="full"`` one wraps to its own line).

    """

    id: str
    title: str
    panels: tuple[str | PanelGroup, ...]
    level: int = 2
    intro: str | None = None
    applies_to: Callable[[Any], bool] | None = None
    role: str = "content"
    layout: str | None = None


@dataclass(frozen=True)
class Expand:
    """A single manifest entry that becomes N sections at resolution time.

    Parameters
    ----------
    over : Callable
        Returns the sequence of items to expand over (e.g. the variables present
        on the page, in display order).
    section : Callable
        Builds one :class:`Section` per item yielded by ``over``.

    """

    over: Callable[[Any], Sequence[Any]]
    section: Callable[[Any], Section]


@dataclass(frozen=True)
class Profile:
    """An ordered page description plus a numbering policy.

    Parameters
    ----------
    numbering : str
        ``"flat"`` (content ``1..N``, appendices ``A..``), ``"none"`` (no
        numbers), or ``"grouped"`` (reserved — not yet implemented).
    entries : tuple of (Section or Expand)
        The page's sections, in order.

    """

    numbering: str = "flat"
    entries: tuple[Section | Expand, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Resolved model — what the template renders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedPanel:
    """A panel after rendering: a payload (figure b64 or markup) or a ``.warn`` stub."""

    id: str
    kind: PanelKind
    slot: str | None
    payload: str | None
    caption: str | None
    stub_reason: str | None = None

    @property
    def is_stub(self) -> bool:
        """True when the panel applied but produced no output."""
        return self.payload is None


@dataclass(frozen=True)
class ResolvedSection:
    """A section after resolution: numbered heading plus resolved panels."""

    id: str
    number: str
    title: str
    level: int
    intro: str | None
    panels: tuple[ResolvedPanel, ...]
    role: str
    layout: str | None = None


@dataclass(frozen=True)
class ResolvedReport:
    """The full resolved page: numbered sections plus the not-applicable list."""

    sections: tuple[ResolvedSection, ...]
    not_applicable: tuple[str, ...]

    def anchor(self, section_id: str) -> str | None:
        """Return the display number for *section_id*, or None if not rendered."""
        for sec in self.sections:
            if sec.id == section_id:
                return sec.number
        return None


# ---------------------------------------------------------------------------
# Resolution algorithm
# ---------------------------------------------------------------------------

_STUB_REASON = "applicable but unavailable"


def _letter(n: int) -> str:
    """Return the appendix letter for a zero-based index (0 -> 'A', 25 -> 'Z', 26 -> 'AA')."""
    letters = ""
    n += 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


def _expand(entries: Sequence[Section | Expand], ctx: RenderContext) -> list[Section]:
    """Splice every :class:`Expand` into the concrete sections it yields."""
    out: list[Section] = []
    for entry in entries:
        if isinstance(entry, Expand):
            out.extend(entry.section(item) for item in entry.over(ctx))
        else:
            out.append(entry)
    return out


def _section_panels(
    sec: Section, ctx: RenderContext, panels: dict[str, Panel]
) -> list[Panel]:
    """Flatten a section's entries to concrete panels: ids looked up, groups expanded.

    A :class:`PanelGroup` entry is expanded by calling ``over(ctx)`` and building
    one panel per item, spliced in place; a string entry is looked up in the
    registry.
    """
    out: list[Panel] = []
    for entry in sec.panels:
        if isinstance(entry, PanelGroup):
            out.extend(entry.panel(item) for item in entry.over(ctx))
        else:
            out.append(panels[entry])
    return out


def _section_applies(
    sec: Section, ctx: RenderContext, panels: dict[str, Panel]
) -> bool:
    """Decide whether *sec* is included: explicit predicate, else any panel applies."""
    if sec.applies_to is not None:
        return bool(sec.applies_to(ctx))
    return any(p.applies_to(ctx) for p in _section_panels(sec, ctx, panels))


def _resolve_slot(panel: Panel, ctx: RenderContext) -> str:
    """Return the panel's slot, calling it with *ctx* when it is a callable."""
    return panel.slot(ctx) if callable(panel.slot) else panel.slot


def _resolve_panels(
    sec: Section,
    ctx: RenderContext,
    panels: dict[str, Panel],
) -> tuple[ResolvedPanel, ...]:
    """Render each applicable panel of a kept section; None output becomes a stub.

    A kept section whose panels all drop out (none apply, or all render None)
    keeps one stub so the heading is never empty.
    """
    concrete = _section_panels(sec, ctx, panels)
    resolved: list[ResolvedPanel] = []
    for panel in concrete:
        if not panel.applies_to(ctx):
            continue
        reason = panel.unavailable_if(ctx)
        if reason is not None:
            # Precondition failed: stub with the specific reason, do not render.
            payload, stub_reason = None, reason
        else:
            payload = panel.render(ctx)
            stub_reason = None if payload is not None else _STUB_REASON
        resolved.append(
            ResolvedPanel(
                id=panel.id,
                kind=panel.kind,
                slot=_resolve_slot(panel, ctx),
                payload=payload,
                caption=panel.caption,
                stub_reason=stub_reason,
            )
        )
    if not resolved:
        # Kept but empty (case 2): keep the heading with a single stub.
        first = concrete[0] if concrete else None
        resolved.append(
            ResolvedPanel(
                id=first.id if first else f"{sec.id}_stub",
                kind=first.kind if first else "figure",
                slot=_resolve_slot(first, ctx) if first else "full",
                payload=None,
                caption=first.caption if first else None,
                stub_reason=_STUB_REASON,
            )
        )
    return tuple(resolved)


def resolve(
    profile: Profile,
    ctx: RenderContext,
    panels: dict[str, Panel],
    *,
    drop_stub: bool = False,
) -> ResolvedReport:
    """Resolve *profile* against *ctx* into a numbered :class:`ResolvedReport`.

    One pass: expand :class:`Expand` entries, drop sections whose ``applies_to``
    is false (collecting them for the not-applicable footer), resolve each kept
    section's panels (``None`` render → ``.warn`` stub), then number the
    survivors — ``content`` sections ``1..N`` and ``appendix`` sections ``A..``.

    Parameters
    ----------
    profile : Profile
        The page's ordered section description and numbering policy.
    ctx : Any
        The render context passed to every predicate and render callable
        (dataset, config, paths — whatever the page's panels need).
    panels : dict of str to Panel
        The panel registry; section ``panels`` entries are ids into this map.
    drop_stub : bool, optional
        When True, a section that *applies* but whose panels are *all* stubs
        (applicable-but-entirely-unavailable) is dropped and the survivors
        renumber over it, so the page closes cleanly.  Default False keeps the
        heading with its stub, which surfaces the failure rather than hiding it.
        A section dropped this way is **not** added to ``not_applicable`` — that
        list stays reserved for genuinely-not-applicable-to-this-deployment
        sections, so a plot failure never reads as "not applicable".  A section
        with any non-stub panel is never dropped.

    Returns
    -------
    ResolvedReport
        Numbered, rendered sections plus the titles of any omitted sections.

    Raises
    ------
    NotImplementedError
        If ``profile.numbering == "grouped"`` (reserved for a later branch).
    KeyError
        If a section references a panel id absent from *panels*.

    """
    if profile.numbering not in ("flat", "none"):
        raise NotImplementedError(
            f"numbering={profile.numbering!r} is not implemented; use 'flat' or 'none'"
        )

    sections = _expand(profile.entries, ctx)

    kept: list[tuple[Section, tuple[ResolvedPanel, ...]]] = []
    not_applicable: list[str] = []
    for sec in sections:
        if not _section_applies(sec, ctx, panels):
            not_applicable.append(sec.title)
            continue
        rpanels = _resolve_panels(sec, ctx, panels)
        if drop_stub and all(p.is_stub for p in rpanels):
            # Applicable but entirely unavailable: drop silently, renumber over
            # it.  Not added to not_applicable — that is case-1 only.
            continue
        kept.append((sec, rpanels))

    resolved: list[ResolvedSection] = []
    content_n = 0
    appendix_n = 0
    for sec, rpanels in kept:
        if profile.numbering == "none":
            number = ""
        elif sec.role == "appendix":
            number = _letter(appendix_n)
            appendix_n += 1
        else:
            content_n += 1
            number = str(content_n)
        resolved.append(
            ResolvedSection(
                id=sec.id,
                number=number,
                title=sec.title,
                level=sec.level,
                intro=sec.intro,
                panels=rpanels,
                role=sec.role,
                layout=sec.layout,
            )
        )

    return ResolvedReport(
        sections=tuple(resolved), not_applicable=tuple(not_applicable)
    )
