"""Self-check for the report-design files vendored byte-identical across packages.

This module lives *inside* the vendored directory it guards, so a consumer
(oceanarray, glidertest) takes the whole directory and nothing else — the hash
manifest and the gate travel with the files. ctdcast ``main`` is the source of
truth; siblings re-vendor from it and never edit their copies in place.

The vendored set is the *contents of this directory*: every file here except
``MANIFEST.txt`` (which records the others' hashes) and machine cruft
(``.DS_Store``, AppleDouble ``._*``, ``__pycache__``, ``*.pyc``). So ``check.py``
and ``README.md`` guard themselves too, and the whole directory is byte-identical
across the three repositories.

Paths in the manifest are relative to this directory, so the manifest is itself
identical across packages (no ``ctdcast/`` vs ``oceanarray/`` prefix).

Usage
-----
    python -m <pkg>.reports._vendored.check            # verify local copies
    python -m <pkg>.reports._vendored.check --update   # regenerate MANIFEST.txt
    python -m <pkg>.reports._vendored.check ../oceanarray ../glidertest
    python -m <pkg>.reports._vendored.check --compare ../oceanarray/.../MANIFEST.txt

Exit code is 1 on any drift (local hash mismatch, a recorded file missing, a
sibling file absent, extra, or differing, a sibling path that does not exist, a
sibling with no vendored copy, or two manifests that disagree); 0 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

#: The directory this file lives in — the vendored set is exactly its contents.
VENDORED_ROOT = Path(__file__).resolve().parent

#: Hash manifest, beside the files it records.  Excluded from the *local* hashed
#: set (a file cannot record its own hash); the cross-repo compare does check it,
#: because it too is byte-identical across packages.
MANIFEST = VENDORED_ROOT / "MANIFEST.txt"

#: Neutral invocation string — recorded in the manifest header and used as the
#: fallback hint.  Identical across packages, so the manifest stays byte-identical.
_NEUTRAL_INVOKE = "python -m <pkg>.reports._vendored.check"

#: Live invocation hint for printed messages: the running package resolves
#: ``__package__`` (the source text is identical across repos).
_INVOKE = f"python -m {__package__}.check" if __package__ else _NEUTRAL_INVOKE

#: Suffixes (and the bare name ``LICENSE``) whose content is text: hashed after
#: normalizing CRLF to LF so the hash is platform-independent.  Anything else —
#: a binary asset such as a Leaflet marker image — is hashed raw, byte-for-byte;
#: normalizing a binary would corrupt its hash.
_TEXT_SUFFIXES = frozenset({".py", ".md", ".txt", ".mplstyle", ".html", ".css", ".js"})
_TEXT_NAMES = frozenset({"LICENSE"})

#: Directories that never hold a vendored copy — skipped when locating a sibling's
#: ``check.py`` so an installed, built, or cached copy is not mistaken for it.
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "build",
        "dist",
        "site-packages",
        "node_modules",
        "__pycache__",
    }
)


def _is_cruft(rel: Path) -> bool:
    """Return True for machine cruft that is never part of the vendored set.

    Exactly ``.DS_Store``, AppleDouble ``._*``, ``__pycache__`` and ``*.pyc`` — a
    deliberately short list, so any *other* unexpected file (including a stray
    dotfile) is surfaced as ``NOT RECORDED`` rather than silently skipped.
    """
    return any(
        part in (".DS_Store", "__pycache__")
        or part.startswith("._")
        or part.endswith(".pyc")
        for part in rel.parts
    )


def _list_vendored(root: Path) -> list[str]:
    """Return every vendored file under *root*, path relative to *root*, sorted.

    Skips that root's ``MANIFEST.txt`` and machine cruft (:func:`_is_cruft`); used
    for the local set (``root == VENDORED_ROOT``) and for a sibling's set, so the
    cross-repo compare can take the *union* of both directories.
    """
    out: list[str] = []
    manifest = root / "MANIFEST.txt"
    for p in sorted(root.rglob("*")):
        if p.is_dir() or p == manifest:
            continue
        rel = p.relative_to(root)
        if _is_cruft(rel):
            continue
        out.append(rel.as_posix())
    return out


def vendored_files() -> list[str]:
    """Return every vendored file's path relative to this directory, sorted.

    Skips ``MANIFEST.txt`` and machine cruft (:func:`_is_cruft`); the result is
    exactly what must be byte-identical across packages.
    """
    return _list_vendored(VENDORED_ROOT)


def sha256(path: Path) -> str:
    """Return the hex sha256 of a file's content.

    Text files (by suffix, plus a bare ``LICENSE``) are normalized CRLF→LF before
    hashing, so "byte-identical across packages" means identical *text* regardless
    of a Windows checkout or a sibling's ``core.autocrlf``.  Any other file — a
    binary asset — is hashed raw, because normalizing a binary would corrupt it.
    """
    data = path.read_bytes()
    if path.suffix in _TEXT_SUFFIXES or path.name in _TEXT_NAMES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def parse_manifest(path: Path, *, key_basename: bool = False) -> dict[str, str]:
    """Parse a manifest file into a ``{key: sha256}`` mapping.

    The key is the recorded relative path, or its basename when *key_basename* is set
    (used by ``--compare`` so two manifests diff even if a sibling nests the files
    differently).  Exits 1 with a clear message on a missing manifest, a malformed
    line, or a duplicate key — a silent last-wins would hide a contradictory entry.
    """
    if not path.exists():
        raise SystemExit(f"{path} not found — run: {_INVOKE} --update")
    what = "basename" if key_basename else "path"
    expected: dict[str, str] = {}
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise SystemExit(
                f"{path.name}:{n}: malformed line "
                f"(expected '<sha256>  <path>'): {raw!r}"
            )
        digest, rel = parts
        key = Path(rel).name if key_basename else rel
        if key in expected:
            raise SystemExit(f"{path.name}:{n}: duplicate {what} {key}")
        expected[key] = digest
    return expected


def load_expected() -> dict[str, str]:
    """Parse the local ``MANIFEST.txt`` into a ``{relative-path: sha256}`` mapping."""
    return parse_manifest(MANIFEST)


def compare_manifests(other: Path) -> int:
    """Diff the local ``MANIFEST.txt`` against *other* by basename.  Return a mismatch count.

    Verifies sync from the two manifests alone — no sibling checkout needed.  A basename
    recorded in one manifest but not the other, or with a different hash, counts as drift.
    """
    print(f"\n== {MANIFEST} vs {other} ==")
    here = parse_manifest(MANIFEST, key_basename=True)
    there = parse_manifest(other, key_basename=True)
    mismatches = 0
    for key in sorted(set(here) | set(there)):
        if key not in there:
            print(f"  ONLY HERE  {key}")
            mismatches += 1
        elif key not in here:
            print(f"  ONLY THERE {key}")
            mismatches += 1
        elif here[key] != there[key]:
            print(f"  DIFFERS    {key}")
            mismatches += 1
        else:
            print(f"  ok         {key}")
    return mismatches


def sibling_vendored(repo: Path) -> Path | None:
    """Return *repo*'s copy of this directory, or ``None`` if it has none.

    Located by the sibling's own ``check.py`` at any depth
    (``rglob("_vendored/check.py")``, skipping build/venv/site-packages so an
    installed copy is not matched), so neither the import-package name — which
    need not equal the distribution name (``AMOCatlas`` installs as ``amocatlas``)
    — nor the ``reports`` vs ``report`` spelling is assumed.  More than one match
    is an error, not a silent pick.
    """
    matches = [
        p.parent
        for p in repo.rglob("_vendored/check.py")
        if not _SKIP_DIRS & set(p.parts)
    ]
    if len(matches) > 1:
        raise SystemExit(
            f"{repo}: multiple _vendored/check.py found: "
            f"{sorted(str(m) for m in matches)}"
        )
    return matches[0] if matches else None


def write_manifest() -> None:
    """Rewrite ``MANIFEST.txt`` from the current local files."""
    lines = [
        "# sha256 of the report-design files vendored byte-identical across",
        "# ctdcast, oceanarray and glidertest.  Source of truth: ctdcast main.",
        "# The set is the contents of this directory, minus this file.",
        f"# Regenerate after an intentional edit: {_NEUTRAL_INVOKE} --update",
    ]
    lines += [f"{sha256(VENDORED_ROOT / rel)}  {rel}" for rel in vendored_files()]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_local() -> int:
    """Check each local file against its recorded hash.  Return a failure count.

    Flags both directions: a file on disk not in ``MANIFEST.txt`` (ran without
    ``--update``) and a recorded file missing from disk.
    """
    expected = load_expected()
    present = vendored_files()
    failures = 0
    for rel in present:
        want = expected.get(rel)
        if want is None:
            print(f"NOT RECORDED  {rel} (run --update)")
            failures += 1
        elif want != sha256(VENDORED_ROOT / rel):
            print(f"DRIFTED       {rel} (edited since --update)")
            failures += 1
    for rel in expected:
        if rel not in present:
            print(f"MISSING       {rel} (recorded but not on disk)")
            failures += 1
    if not failures:
        print(f"OK            {len(present)} vendored files match {MANIFEST.name}")
    return failures


def compare_sibling(repo: Path) -> int:
    """Byte-compare every vendored file against *repo*'s copy.  Return a mismatch count.

    Iterates the *union* of both directories (plus ``MANIFEST.txt``), so a file that
    exists only in the sibling is caught as drift, not silently ignored.  ``MANIFEST.txt``
    is compared too — it is byte-identical across packages, so a sibling with a stale
    manifest is caught.  A sibling file that is absent, extra, or differs counts as drift;
    a path that does not exist, or a repo with no ``_vendored`` copy, is an error (not
    silently skipped — else a typo'd path would read as "in sync").
    """
    print(f"\n== {repo} ==")
    if not repo.exists():
        print("  ERROR path not found — not compared")
        return 1
    there_root = sibling_vendored(repo)
    if there_root is None:
        print("  no _vendored/ copy found — not synced")
        return 1
    union = set(vendored_files()) | set(_list_vendored(there_root)) | {MANIFEST.name}
    mismatches = 0
    for rel in sorted(union):
        here = VENDORED_ROOT / rel
        there = there_root / rel
        if not here.exists():
            print(f"  EXTRA     {there} (in sibling, not vendored here)")
            mismatches += 1
        elif not there.exists():
            print(f"  MISSING   {there}")
            mismatches += 1
        elif sha256(here) != sha256(there):
            print(f"  DIFFERS   {there}")
            mismatches += 1
        else:
            print(f"  ok        {rel}")
    return mismatches


def main(argv: list[str]) -> int:
    """Run the check: verify locally, regenerate the manifest, or compare.

    Mutually exclusive modes — no arguments verifies the local copy against
    ``MANIFEST.txt``; ``--update`` regenerates it; one or more repo paths compares
    this copy against each sibling (union of both directories); ``--compare
    OTHER/MANIFEST.txt`` diffs the two manifests by basename without a checkout.
    Returns the failure/mismatch count (0 = ok).
    """
    parser = argparse.ArgumentParser(
        prog=_INVOKE,
        description="Check the vendored report layer is byte-identical across packages.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="regenerate MANIFEST.txt from the current files",
    )
    parser.add_argument(
        "--compare",
        metavar="OTHER/MANIFEST.txt",
        help="diff this MANIFEST.txt against another by basename (no checkout needed)",
    )
    parser.add_argument(
        "siblings",
        nargs="*",
        metavar="REPO",
        help="sibling repo checkout(s) to compare this copy against",
    )
    args = parser.parse_args(argv)
    modes = sum((args.update, args.compare is not None, bool(args.siblings)))
    if modes > 1:
        parser.error("--update, --compare and sibling paths are mutually exclusive")
    if args.update:
        write_manifest()
        print(f"wrote {MANIFEST.name} ({len(vendored_files())} files)")
        return 0
    if args.compare is not None:
        return 1 if compare_manifests(Path(args.compare)) else 0
    if not args.siblings:
        return 1 if verify_local() else 0
    total = sum(compare_sibling(Path(s)) for s in args.siblings)
    print(f"\n{total} mismatch(es) across {len(args.siblings)} sibling(s)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
