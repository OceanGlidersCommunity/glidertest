"""Build the live demo report for the documentation site.

Writes a two-mission fleet — the SeaExplorer sample (sea045, Baltic) and the Seaglider sample
subset (sg014, Labrador Sea, which carries a flight-model velocity so the flight page appears)
— into ``docs/source/_static/demo/`` using the library, not the CLI, so the documentation
build does not depend on the console script being installed. The output is not committed;
the documentation workflows run this script before ``make html``.

Usage
-----
    python scripts/make_demo.py [--clean] [--out DIR]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MISSIONS: tuple[str, ...] = (
    "sea045_20230604T1253_delayed.nc",
    "sg014_20040924T182454_delayed_subset.nc",
)
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "docs" / "source" / "_static" / "demo"


def main(argv: list[str] | None = None) -> int:
    """Build the demo fleet; return a process exit code.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        ``0`` when every mission and the fleet page were written, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output root (default: {DEFAULT_OUT})")
    parser.add_argument("--clean", action="store_true", help="remove the output root first")
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")

    from glidertest import fetchers, reports

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    failed = 0
    for name in MISSIONS:
        try:
            ds = fetchers.load_sample_dataset(dataset_name=name)
            landing = reports.report(ds, args.out, navigator=False)
            print(landing)
        except Exception as exc:  # per-mission boundary: report the failure and carry on
            print(f"{name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1
    print(reports.navigator(args.out, title="glidertest demo"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
