"""Shared pytest setup and fixtures.

pytest imports this before any test module, so setting the non-interactive Matplotlib backend here
means test files need no ``matplotlib.use("agg")`` statement above their imports (and therefore no
E402 suppression on those imports). ``reports.report`` and the CLI also force Agg at runtime.

The fixtures serve the test-suite speed-up: render the committed subset once per session, open it
once, and (for the CLI flag tests) stub ``reports.report`` so the figure render is skipped while the
real manifest is still written. See ``tests/data/README.md`` for the subsets.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import pytest  # noqa: E402
import xarray as xr  # noqa: E402

DATA = Path(__file__).parent / "data"
SEA045_SUBSET = DATA / "sea045_subset.nc"  # CTD + oxygen + optics, 12 profiles
SG014_SUBSET = DATA / "sg014_subset.nc"  # flight (GLIDER_VERT_VELO_MODEL), 6 profiles


@pytest.fixture(scope="session")
def subset_path():
    """Path to the committed sea045 subset (CTD + oxygen + optics)."""
    return SEA045_SUBSET


@pytest.fixture(scope="session")
def sg014_subset_path():
    """Path to the committed sg014 subset (flight page)."""
    return SG014_SUBSET


@pytest.fixture(scope="session")
def subset_ds():
    """The sea045 subset opened once per session (read-only).

    For tests that add derived variables to the dataset (TEOS-10, sigma, DEPTH_Z) use the
    function-scoped ``fresh_subset`` / ``fresh_sg014`` fixtures instead, so one test's mutation
    cannot leak into another.
    """
    with xr.open_dataset(SEA045_SUBSET) as ds:
        yield ds


@pytest.fixture
def fresh_subset():
    """A fresh, writable sea045 subset, one per test (for code that mutates the dataset)."""
    with xr.open_dataset(SEA045_SUBSET) as ds:
        yield ds


@pytest.fixture
def fresh_sg014():
    """A fresh, writable sg014 subset, one per test (flight / vertical-velocity tools)."""
    with xr.open_dataset(SG014_SUBSET) as ds:
        yield ds


@pytest.fixture(scope="session")
def subset_report(tmp_path_factory):
    """Render the sea045 subset once; return its mission directory.

    Session-scoped, so **read-only consumers only** — a test that writes into or deletes from the
    returned tree must render its own report into its own ``tmp_path`` instead.
    """
    from glidertest.reports import report

    root = tmp_path_factory.mktemp("subset_report")
    with xr.open_dataset(SEA045_SUBSET) as ds:
        landing = report(ds, root, navigator=False)
    # report()'s return-path contract (root layout): <root>/<mission_id>/index.html.
    assert landing == root / "sea045_20230604T1253_delayed" / "index.html"
    return landing.parent


@pytest.fixture
def fake_report(monkeypatch):
    """Replace ``reports.report`` with a stub for the CLI flag/layout tests.

    The stub honours *layout* and *mission_id*, writes a placeholder landing page and a **real**
    ``mission_manifest`` (which renders nothing — mission facts, QC counts, a decimated track), and
    records each call in the returned list. ``navigator()`` then runs for real on real-shaped
    manifests. It skips only the figure/HTML render, which is what these tests do not assert.

    Returns
    -------
    list of dict
        One entry per call: ``{"source", "mdir", "navigator", "mission_id", "layout"}``.
    """
    calls = []

    def _stub(ds, outdir, *, navigator=True, mission_id=None, layout="root"):
        from glidertest.reports import manifest as _manifest
        from glidertest.reports import paths, resolve_mission_id

        name = resolve_mission_id(ds, mission_id)
        mdir = Path(outdir) if layout == "flat" else paths.mission_dir(Path(outdir), name)
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "index.html").write_text("<html>← All missions ../index.html</html>", encoding="utf-8")
        source = ds.encoding.get("source")
        source_name = Path(source).name if source else f"{ds.attrs.get('id') or name}.nc"
        man = _manifest.mission_manifest(
            ds,
            mission_id=name,
            source_name=source_name,
            source_size_bytes=None,
            pages=[("index.html", "Mission")],
            version="test",
            generated_at="2026-01-01 00:00 UTC",
        )
        paths.manifest_in(mdir).write_text(json.dumps(man), encoding="utf-8")
        calls.append(
            {"source": source, "mdir": mdir, "navigator": navigator, "mission_id": mission_id, "layout": layout}
        )
        return mdir / "index.html"

    monkeypatch.setattr("glidertest.reports.report", _stub)
    return calls
