"""Command-line entry point for glidertest: ``glidertest <command>``.

Two subcommands, each a thin wrapper over the library:

``report``
    Write an HTML report for one or more OG1 netCDF files and rebuild the fleet page.
``navigator``
    Rebuild the fleet page over a report root from the per-mission manifests, reading no data.

No report logic lives here; the heavy imports (matplotlib, xarray, the report subsystem) are
deferred into the command functions so ``glidertest --help`` stays fast and backend-free.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from glidertest import __version__

_EPILOG = """\
Typical workflow:
  glidertest report mission.nc --report-dir reports/   one mission into reports/<id>/, fleet page rebuilt
  glidertest report data/ --report-dir reports/        every *.nc in data/, fleet page rebuilt once
  glidertest report mission.nc -o out/                 one mission's pages directly in out/, no fleet page
  glidertest navigator reports/                        re-index after adding or removing missions

Run 'glidertest <command> --help' for command-specific options.
"""

_REPORT_EPILOG = """\
Examples:
  # One mission into a report tree; the fleet page is rebuilt afterwards:
  glidertest report sea045_20230604T1253_delayed.nc --report-dir reports/

  # A directory of missions; the fleet page is rebuilt once at the end:
  glidertest report data/ --report-dir reports/

  # Name the subdirectory yourself when the file has no usable OG1 id:
  glidertest report odd_file.nc --report-dir reports/ --mission-id sea045_2023_leg2

  # Re-run over a folder, reporting only missions not yet in the tree:
  glidertest report data/ --report-dir reports/ --skip-existing

  # One mission's pages straight into a folder, no tree and no fleet page:
  glidertest report sea045_20230604T1253_delayed.nc -o sea045_report/

Exit status: 0 all files reported; 1 one or more files failed (the rest were still written,
each failure one line on stderr); 2 usage error.
"""


def _expand_inputs(inputs: list[str], pattern: str) -> list[Path]:
    """Expand *inputs* (files, directories, wildcards) into a deduplicated file list.

    A directory contributes ``sorted(dir.glob(pattern))``; an argument that does not exist but
    contains a wildcard (``*``, ``?`` or ``[``) is globbed in Python (Windows shells do not expand
    wildcards); anything else is taken verbatim as a path. Duplicates (by resolved path) are
    dropped, keeping the first occurrence.

    Parameters
    ----------
    inputs : list of str
        The positional FILE arguments.
    pattern : str
        Filename glob applied inside directory arguments (not recursive).

    Returns
    -------
    list of pathlib.Path
        The expanded files, in argument order, deduplicated.
    """
    out: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            matches = sorted(p.glob(pattern))
        elif not p.exists() and any(c in item for c in "*?["):
            matches = [Path(m) for m in sorted(glob.glob(item))]
        else:
            matches = [p]
        for m in matches:
            resolved = m.resolve()
            if resolved not in seen:
                seen.add(resolved)
                out.append(m)
    return out


def _add_report_parser(sub: argparse._SubParsersAction) -> None:
    """Add the ``report`` subcommand and its arguments to *sub*."""
    p = sub.add_parser(
        "report",
        help="Write HTML mission report(s) for OG1 netCDF file(s).",
        description=(
            "Write an HTML mission report for each OG1 netCDF file: a landing page, one page per "
            "sensor present, a flight page when the data supports it, an inventory page, and a "
            "report.json manifest. The dataset is read, never modified."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_REPORT_EPILOG,
    )
    p.add_argument(
        "inputs",
        nargs="+",
        metavar="FILE",
        help="OG1 netCDF file(s), or directories of them (see --pattern).",
    )
    p.add_argument(
        "--mission-id",
        dest="mission_id",
        default=None,
        metavar="NAME",
        help="Subdirectory name for the mission, instead of the file's OG1 id (or its file stem "
        "when the id is missing). Single FILE, --report-dir only (not with -o).",
    )
    p.add_argument(
        "--title",
        default=None,
        metavar="TEXT",
        help="Title for the fleet page's masthead (default: the ROOT directory name). "
        "Ignored with --no-navigator or -o.",
    )
    p.add_argument(
        "--no-navigator",
        dest="no_navigator",
        action="store_true",
        help="Do not rebuild ROOT/index.html after writing the mission(s).",
    )
    p.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        help="Skip a FILE whose mission already has a ROOT/<mission_id>/report.json.",
    )
    p.add_argument(
        "--pattern",
        default="*.nc",
        metavar="GLOB",
        help="Filename glob applied inside each directory given as FILE (default: '*.nc'; "
        "not recursive).",
    )
    p.add_argument(
        "-n",
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Print what would be written without writing anything.",
    )
    out = p.add_argument_group("output location (one is required)")
    grp = out.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--report-dir",
        dest="report_dir",
        type=Path,
        default=None,
        metavar="ROOT",
        help="Central directory for all mission reports. Each mission's pages are written to "
        "ROOT/<mission_id>/ and ROOT/index.html is the fleet page, making the whole report "
        "tree portable.",
    )
    grp.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory for one mission's pages, written directly into DIR (no <mission_id>/ "
        "subdirectory, no fleet page). Single FILE only.",
    )
    p.set_defaults(func=cmd_report)


def _add_navigator_parser(sub: argparse._SubParsersAction) -> None:
    """Add the ``navigator`` subcommand and its arguments to *sub*."""
    p = sub.add_parser(
        "navigator",
        help="Rebuild the fleet page (index.html) over a report root; reads no data.",
        description=(
            "Rebuild ROOT/index.html, the fleet page, from the ROOT/<mission_id>/report.json "
            "manifests. Reads no netCDF and touches no mission pages. Run it after adding, "
            "removing or renaming mission directories. (The counterpart of 'oceanarray report "
            "--array'.)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "root",
        type=Path,
        metavar="ROOT",
        help="Report root holding <mission_id>/report.json subdirectories.",
    )
    p.add_argument(
        "--title",
        default=None,
        metavar="TEXT",
        help="Title for the fleet page's masthead (default: the ROOT directory name).",
    )
    p.set_defaults(func=cmd_navigator)


def cmd_report(args: argparse.Namespace) -> int:
    """Run the ``report`` subcommand; return a process exit code (0 ok, 1 failures, 2 usage)."""
    import matplotlib

    matplotlib.use("Agg")  # a CLI must not depend on the caller's backend; set before xarray/reports
    import xarray as xr

    from glidertest import reports
    from glidertest.reports import paths, resolve_mission_id

    flat = args.output_dir is not None
    root = args.output_dir if flat else args.report_dir

    if flat and args.mission_id:
        print(
            "glidertest report: error: --mission-id has no effect with -o/--output-dir "
            "(which writes no <mission_id>/ subdirectory); use --report-dir",
            file=sys.stderr,
        )
        return 2

    files = _expand_inputs(args.inputs, args.pattern)
    if not files:
        print("no files matched", file=sys.stderr)
        return 1
    if flat and len(files) > 1:
        print(
            "glidertest report: error: -o/--output-dir takes a single FILE; "
            "use --report-dir for several",
            file=sys.stderr,
        )
        return 2
    if args.mission_id and len(files) > 1:
        print(
            "glidertest report: error: --mission-id applies to a single FILE",
            file=sys.stderr,
        )
        return 2

    def resolve_name(path: Path) -> str | None:
        """The mission directory name for *path*, or None if the file cannot be opened."""
        if args.mission_id:
            return resolve_mission_id(None, args.mission_id)  # the override needs no dataset
        try:
            with xr.open_dataset(path) as ds:
                return resolve_mission_id(ds)
        except Exception:  # any unreadable file is reported in the main loop, so None here
            return None

    # Resolve each mission's directory name once, up front (root layout only; flat is single-file
    # and resolved inside report()). The name is reused by the collision, skip-existing and dry-run
    # checks and passed to report(), so a file's id is derived once per run, not three times.
    names: dict[Path, str | None] = {}
    if not flat:
        for path in files:
            names[path] = resolve_name(path)
        # Refuse a run where two files would overwrite each other under one mission id. A delayed
        # and a real-time file sharing one OG1 id, or a file and its _subset, would otherwise clobber.
        by_name: dict[str, list[Path]] = {}
        for path, name in names.items():
            if name is not None:
                by_name.setdefault(name, []).append(path)
        collisions = {n: ps for n, ps in by_name.items() if len(ps) > 1}
        if collisions:
            for name, ps in collisions.items():
                listed = ", ".join(str(p) for p in ps)
                print(
                    f"glidertest report: error: {len(ps)} files map to mission {name!r}: {listed}",
                    file=sys.stderr,
                )
            return 1

    failed = 0
    wrote = 0  # missions written (or, under --dry-run, that would be written)
    for path in files:
        if not path.is_file():
            print(f"{path}: no such file", file=sys.stderr)
            failed += 1
            continue
        name = names.get(path)  # None in flat layout (resolved inside report()) or if unreadable
        try:
            if flat:
                manifest = paths.manifest_in(root)  # flat writes the manifest at the top of DIR
            elif name is not None:
                manifest = paths.manifest_path(root, name)
            else:
                manifest = None
            if args.skip_existing and manifest is not None and manifest.exists():
                print(f"{path}: skipped ({name or root} already reported)")
                continue
            if args.dry_run:
                if flat:
                    print(f"{path} -> {root}/")
                elif name is not None:
                    print(f"{path} -> {paths.mission_dir(root, name)}/")
                else:
                    print(f"{path}: unreadable (would fail)", file=sys.stderr)
                    failed += 1
                    continue
                wrote += 1
                continue
            with xr.open_dataset(path) as ds:
                landing = reports.report(
                    ds,
                    root,
                    navigator=False,
                    mission_id=(None if flat else name),
                    layout="flat" if flat else "root",
                )
            print(landing)
            wrote += 1
        except Exception as exc:  # catch-all at the file boundary: one line per failure, run on
            print(f"{path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1
            continue

    # Rebuild the fleet page only when a mission was written (or would be, under --dry-run): an
    # all-skipped re-run leaves the existing fleet page untouched, keeping re-runs cheap, and an
    # all-failed run never calls navigator() on a root that may not exist.
    if not flat and not args.no_navigator:
        if args.dry_run:
            if wrote:
                print(f"would rebuild {root / 'index.html'}")
            else:
                print(f"nothing to write; {root / 'index.html'} would not be rebuilt")
        elif wrote:
            print(reports.navigator(root, title=args.title))

    return 1 if failed else 0


def cmd_navigator(args: argparse.Namespace) -> int:
    """Run the ``navigator`` subcommand; return a process exit code (0 ok, 1 on a bad root)."""
    import matplotlib

    matplotlib.use("Agg")  # the fleet map is matplotlib; do not touch the caller's backend

    from glidertest import reports

    try:
        print(reports.navigator(args.root, title=args.title))
    except (FileNotFoundError, ValueError) as exc:
        # Missing root, or a mission directory mistaken for a fleet root — report() owns the rule.
        print(exc, file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``glidertest`` argument parser (the ``report`` and ``navigator`` subcommands).

    Separated from :func:`main` so the documentation can render the parser with
    ``sphinx-argparse`` and the two cannot drift apart.

    Returns
    -------
    argparse.ArgumentParser
        The configured top-level parser.
    """
    parser = argparse.ArgumentParser(
        prog="glidertest",
        description="Diagnose and report on OG1 glider datasets: QC checks and self-contained HTML reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", title="commands", metavar="<command>")
    _add_report_parser(sub)
    _add_navigator_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse *argv* (default ``sys.argv``), dispatch to a subcommand, and exit with its code.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector excluding the program name; defaults to ``sys.argv[1:]``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        # No subcommand: show the full help (which lists report and navigator) rather than a terse
        # "the following arguments are required: <command>".
        parser.print_help(sys.stderr)
        sys.exit(2)
    sys.exit(args.func(args))
