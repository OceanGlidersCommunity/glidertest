from pathlib import Path

import matplotlib

matplotlib.use("agg")  # no display; the CLI forces this too

from PIL import Image  # noqa: E402

from glidertest import fetchers, plots  # noqa: E402
from glidertest.config.report_tokens import FIG_DPI, W_FULL  # noqa: E402
from glidertest.reports import _slots, report  # noqa: E402
from glidertest.reports._mission import build  # noqa: E402


def test_report_writes_files(tmp_path):
    ds = fetchers.load_sample_dataset()
    out = report(ds, tmp_path)
    assert out == Path(tmp_path) / "mission.html"
    assert out.exists()
    figures = list((Path(tmp_path) / "figures").glob("*.png"))
    assert len(figures) >= 4
    html = out.read_text(encoding="utf-8")
    assert "#07264f" in html  # package accent
    for section_id in ("metadata", "track", "hydrography", "sampling", "qc"):
        assert f'id="{section_id}"' in html


def test_sections_resolve_in_order():
    ds = fetchers.load_sample_dataset()
    resolved = build(ds)
    titles = [s.title for s in resolved.sections]
    assert titles[:5] == ["Metadata", "Track", "Hydrography", "Sampling", "QC"]
    # the metadata panel is html and always renders (never a stub)
    assert not resolved.sections[0].panels[0].is_stub


def test_report_style_reaches_figure():
    # Review 2.4 / PR #269: setting _ACTIVE_STYLE must drive the figure width through the
    # plotter's own inner style context, independent of _force_width.
    ds = fetchers.load_sample_dataset()
    original = plots._ACTIVE_STYLE
    plots._ACTIVE_STYLE = _slots._report_spec(W_FULL)
    try:
        fig, _ = plots.plot_glider_track(ds)
        assert abs(fig.get_size_inches()[0] - W_FULL) < 1e-6
    finally:
        plots._ACTIVE_STYLE = original


def test_png_width(tmp_path):
    ds = fetchers.load_sample_dataset()
    report(ds, tmp_path)
    expected = round(W_FULL * FIG_DPI)
    pngs = list((Path(tmp_path) / "figures").glob("*.png"))
    assert pngs
    for png in pngs:
        assert Image.open(png).size[0] == expected
