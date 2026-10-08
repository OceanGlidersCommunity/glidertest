"""Report-tree layout rules: where a mission's directory and manifest live under a root.

One definition of the layout, shared by :func:`glidertest.reports.report`,
:func:`glidertest.reports.navigator`, the navigator's manifest reader and the command-line
interface, so the convention cannot drift between writer and readers. The manifest filename lives in
:data:`MANIFEST_NAME` alone; every path helper derives from it.
"""

from __future__ import annotations

from pathlib import Path

#: Filename of the per-mission manifest, written inside each mission directory.
MANIFEST_NAME = "report.json"


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


def manifest_in(directory: Path | str) -> Path:
    """Return the manifest path inside a mission *directory*: ``<directory>/report.json``.

    Used by the writer and the navigator's reader, which hold the mission directory directly (in
    flat layout it is the root itself, so :func:`manifest_path` does not apply).

    Parameters
    ----------
    directory : pathlib.Path or str
        A mission directory.

    Returns
    -------
    pathlib.Path
        ``<directory>/report.json``.
    """
    return Path(directory) / MANIFEST_NAME


def manifest_path(root: Path | str, mission_id: str) -> Path:
    """Return a mission's manifest under *root*: ``<root>/<mission_id>/report.json``.

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
    return manifest_in(mission_dir(root, mission_id))


def is_report_root(root: Path | str) -> bool:
    """Return True if *root* already holds mission subdirectories (``<id>/report.json``).

    A report root carries a fleet ``index.html`` and one subdirectory per mission; writing a flat
    report directly into it would overwrite that fleet index. :func:`glidertest.reports.report`
    uses this to refuse ``layout="flat"`` into an existing root.

    Parameters
    ----------
    root : pathlib.Path or str
        A candidate directory.

    Returns
    -------
    bool
        True if any immediate subdirectory holds a ``report.json``.
    """
    root = Path(root)
    if not root.is_dir():
        return False
    return any((p / MANIFEST_NAME).exists() for p in root.iterdir() if p.is_dir())
