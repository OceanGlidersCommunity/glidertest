"""Small reusable HTML snippets for report pages (owned, not part of the vendored layer).

These are glidertest-authored building blocks shared across report sections, so a given bit of
presentation has one definition. Each returns a ``markupsafe.Markup`` HTML fragment and is also
registered as a Jinja global in :mod:`glidertest.reports._env`, so a template can call it with
``{{ name(...) }}`` and a Python-built note can call it directly — one implementation either way.

Original Author: Eleanor Frajka-Williams (glidertest).
"""

from __future__ import annotations

from markupsafe import Markup, escape

#: Default caption above a :func:`yaml_box`.
YAML_BOX_LABEL = "Suggested — copy into your report config"


def yaml_box(yaml_text: str, *, label: str = YAML_BOX_LABEL) -> Markup:
    """Return a labelled, one-click-selectable copy box for a YAML config snippet.

    A read-only ``<textarea>`` (it selects on click, wraps nothing, and scrolls/resizes for long
    snippets) under a small caption. Styled by the ``.yaml-box`` / ``.yaml-box-label`` rules in
    ``templates/base.html``. Use it anywhere the report offers config to paste — the suggested QC
    limits, a region box, a future threshold — not just one section.

    Parameters
    ----------
    yaml_text : str
        The YAML snippet shown verbatim (HTML-escaped). Its line count sets the box height.
    label : str, optional
        The caption above the box.

    Returns
    -------
    markupsafe.Markup
        An HTML fragment: a ``<p class="yaml-box-label">`` caption and a read-only ``<textarea>``.
    """
    text = yaml_text.strip("\n")
    rows = text.count("\n") + 1
    return Markup(
        f'<p class="yaml-box-label">{escape(label)}</p>'
        f'<textarea class="yaml-box" readonly rows="{rows}" '
        f'onclick="this.select()">{escape(text)}</textarea>'
    )
