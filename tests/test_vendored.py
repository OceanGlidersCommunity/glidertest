"""Guard the vendored report-design files against a local edit that skipped re-vendor.

The vendored directory checks itself: ``check.main([])`` hashes every file under
``glidertest/reports/_vendored/`` against ``MANIFEST.txt`` and returns non-zero on any drift
— a file edited since ``--update``, a file on disk not recorded, or a recorded file missing.
Cross-repo equality — are the siblings in sync? — is ``check``'s ``../sibling`` mode, run in
review, not a test here.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from glidertest.reports._vendored import check

_REPO_ROOT = Path(__file__).resolve().parent.parent
_VENDORED = _REPO_ROOT / "glidertest" / "reports" / "_vendored"


def test_vendored_copy_matches_its_manifest(capsys):
    """Every vendored file matches MANIFEST.txt; names the drifted file on failure.

    This is the guard that makes a lint config unnecessary: a ``ruff --fix`` or ``ruff format``
    run that rewrites a vendored file fails the suite here, naming the file, rather than silently
    forking the shared layer.
    """
    rc = check.main([])
    out = capsys.readouterr().out
    assert rc == 0, f"vendored files drift from MANIFEST.txt:\n{out}"


def test_vendored_python_is_ruff_format_stable():
    """``ruff format`` leaves the vendored Python unchanged.

    The source-of-truth files are kept already formatted in ctdcast, so a consumer running a
    whole-repo ``ruff format`` must not rewrite a byte-identical copy (which would then fail its
    own ``check``). Skipped when ruff is not importable.
    """
    pytest.importorskip("ruff", reason="ruff not installed")
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", str(_VENDORED)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"vendored files are not ruff-format-stable:\n{result.stdout}{result.stderr}"
    )


def test_no_owned_template_shadows_a_vendored_one():
    """No template filename exists in both the owned and vendored template dirs.

    The report environment (:mod:`glidertest.reports._env`) loads owned templates first, so an
    owned partial with a vendored one's name would silently fork the shared copy.
    """
    owned = _REPO_ROOT / "glidertest" / "reports" / "templates"
    vendored = _VENDORED / "templates"
    owned_names = {p.name for p in owned.glob("*") if p.is_file()}
    vendored_names = {p.name for p in vendored.glob("*") if p.is_file()}
    assert not (owned_names & vendored_names), (
        f"owned templates shadow vendored ones: {sorted(owned_names & vendored_names)}"
    )
