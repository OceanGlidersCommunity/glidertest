"""Report-tree layout rules: where a mission's directory and manifest live under a root.

One definition of the layout, shared by :func:`glidertest.reports.report`,
:func:`glidertest.reports.navigator` and the command-line interface (its duplicate-id and
skip-existing checks), so the convention cannot drift between writer and reader.
"""

from __future__ import annotations

from pathlib import Path


def mission_dir(root: Path | str, mission_id: str) -> Path:
    """Return the directory a mission's pages are written to: ``<root>/<mission_id>/``.

    Parameters
    ----------
    root : pathlib.Path or str
        The report root holding one subdirectory per mission.
    mission_id : str
        The mission's subdirectory name (already sanitised; see
        :func:`glidertest.reports._mission._safe_dirname`).

    Returns
    -------
    pathlib.Path
        ``<root>/<mission_id>``.
    """
    return Path(root) / mission_id


def manifest_path(root: Path | str, mission_id: str) -> Path:
    """Return the path to a mission's ``report.json`` manifest: ``<root>/<mission_id>/report.json``.

    Parameters
    ----------
    root : pathlib.Path or str
        The report root.
    mission_id : str
        The mission's subdirectory name.

    Returns
    -------
    pathlib.Path
        ``<root>/<mission_id>/report.json``.
    """
    return mission_dir(root, mission_id) / "report.json"
