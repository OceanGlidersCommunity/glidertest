"""Check the report-design files that are vendored byte-identical across packages.

ctdcast ``main`` is the source of truth for the shared report-design files
(the encoder, CSS generator, page-manifest model, design tokens, mplstyle, and
the masthead-nav template).  oceanarray and glidertest carry byte-identical
copies and re-vendor from here; they never edit their copies in place.

Usage
-----
Verify the local copies match the recorded hashes (what the test does)::

    venv/bin/python scripts/check_vendored.py

Compare the local copies against one or more sibling checkouts::

    venv/bin/python scripts/check_vendored.py ../ctdcast ../oceanarray

Regenerate ``VENDORED.txt`` after an intentional edit to a vendored file::

    venv/bin/python scripts/check_vendored.py --update

Exit code is 1 if any file drifts (local hash mismatch, or a sibling differs),
0 otherwise.  A sibling repo or file that is absent is reported and skipped, not
counted as a mismatch.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

#: Repo-relative paths of the vendored files.  The single source of this list.
VENDORED: tuple[str, ...] = (
    "glidertest/reports/_encode.py",
    "glidertest/reports/_css.py",
    "glidertest/reports/_manifest.py",
    "glidertest/config/report_tokens.py",
    "glidertest/config/report.mplstyle",
    "glidertest/reports/templates/_nav.html",
    # Leaflet 1.9.4 assets, shipped byte-identical with ctdcast. The map Python/template that use
    # them are authored in glidertest first and join this list when ctdcast adopts them.
    "glidertest/reports/leaflet/leaflet.min.js",
    "glidertest/reports/leaflet/leaflet.min.css",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "VENDORED.txt"


def sha256(path: Path) -> str:
    """Return the hex sha256 of a file's *content*, normalized to LF line endings.

    The vendored files are text, and "byte-identical across packages" means identical
    text — not identical line endings.  CRLF (a Windows checkout, or a sibling with a
    different ``core.autocrlf``) is normalized to LF before hashing so the hash is the same
    on every platform; otherwise the guard would fail on Windows CI for every file.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_expected() -> dict[str, str]:
    """Parse ``VENDORED.txt`` into a ``{relative-path: sha256}`` mapping."""
    expected: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(maxsplit=1)
        expected[rel] = digest
    return expected


def write_manifest() -> None:
    """Rewrite ``VENDORED.txt`` from the current local files."""
    lines = [
        "# sha256 of the report-design files vendored byte-identical across",
        "# ctdcast, oceanarray and glidertest.  Source of truth: ctdcast main.",
        "# Regenerate after an intentional edit: python scripts/check_vendored.py --update",
    ]
    lines += [f"{sha256(REPO_ROOT / rel)}  {rel}" for rel in VENDORED]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_local() -> int:
    """Check each local file against its recorded hash.  Return a failure count."""
    expected = load_expected()
    failures = 0
    for rel in VENDORED:
        want = expected.get(rel)
        have = sha256(REPO_ROOT / rel)
        if want is None:
            print(f"NOT RECORDED  {rel} (run --update)")
            failures += 1
        elif want != have:
            print(f"DRIFTED       {rel} (edited since --update)")
            failures += 1
    if not failures:
        print(f"OK            {len(VENDORED)} vendored files match VENDORED.txt")
    return failures


def compare_sibling(sibling: Path) -> int:
    """Byte-compare every vendored file against *sibling*.  Return a mismatch count.

    The vendored files live under each repo's own package directory, so the
    leading ``glidertest/`` of each path is mapped to the sibling's package name
    (taken from the checkout directory name — ``../ctdcast`` → ``ctdcast``).
    """
    print(f"\n== {sibling} ==")
    if not sibling.exists():
        print("  absent — skipped")
        return 0
    pkg = sibling.name
    mismatches = 0
    for rel in VENDORED:
        here = REPO_ROOT / rel
        there = sibling / rel.replace("glidertest/", f"{pkg}/", 1)
        if not there.exists():
            print(f"  MISSING   {there.relative_to(sibling)}")
            mismatches += 1
        elif sha256(here) != sha256(there):
            print(f"  DIFFERS   {there.relative_to(sibling)}")
            mismatches += 1
        else:
            print(f"  ok        {there.relative_to(sibling)}")
    return mismatches


def main(argv: list[str]) -> int:
    """Run the check; see the module docstring for the argument forms."""
    if "--update" in argv:
        write_manifest()
        print(f"wrote {MANIFEST.relative_to(REPO_ROOT)} ({len(VENDORED)} files)")
        return 0
    siblings = [a for a in argv if not a.startswith("-")]
    if not siblings:
        return 1 if verify_local() else 0
    total = sum(compare_sibling(Path(s)) for s in siblings)
    print(f"\n{total} mismatch(es) across {len(siblings)} sibling(s)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
