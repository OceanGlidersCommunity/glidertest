"""Jinja2 environment factory for glidertest report templates.

Loads templates from the ``templates/`` subdirectory of this package.

Usage::

    from glidertest.reports._env import get_template
    html = get_template("mission.html").render(**ctx)
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template

from . import _figdebug

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_env: Environment = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=True,
)
# Per-figure debug overlay (opt-in via GLIDERTEST_REPORT_DEBUG); figdbg() returns "" when off,
# so the template macro guarding on it emits nothing in normal builds.
_env.globals["figdbg"] = _figdebug.figdbg


def get_template(name: str) -> Template:
    """Return an autoescaped Jinja2 template by filename."""
    return _env.get_template(name)
