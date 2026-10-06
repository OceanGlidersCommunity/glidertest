from pathlib import Path

import matplotlib

matplotlib.use("agg")  # no display; the CLI forces this too

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from PIL import Image  # noqa: E402

from glidertest import fetchers, plots  # noqa: E402
from glidertest.config.report_tokens import FIG_DPI, W_FULL  # noqa: E402
from glidertest.reports import _slots, report  # noqa: E402
from glidertest.reports._mission import (  # noqa: E402
    _attrs_details,
    _file_contents,
    _fmt_scalar,
    _var_table,
    build,
)


def test_report_writes_files(tmp_path):
    ds = fetchers.load_sample_dataset()
    out = report(ds, tmp_path)
    assert out == Path(tmp_path) / "mission.html"
    assert out.exists()
    figures = list((Path(tmp_path) / "figures").glob("*.png"))
    assert len(figures) >= 4
    html = out.read_text(encoding="utf-8")
    assert "#07264f" in html  # package accent
    for section_id in ("metadata", "track", "hydrography", "sampling", "qc", "file_contents"):
        assert f'id="{section_id}"' in html


def test_sections_resolve_in_order():
    ds = fetchers.load_sample_dataset()
    resolved = build(ds)
    titles = [s.title for s in resolved.sections]
    assert titles[:5] == ["Metadata", "Track", "Hydrography", "Sampling", "QC"]
    assert titles[-1] == "File contents"
    # the metadata panel is html and always renders (never a stub)
    assert not resolved.sections[0].panels[0].is_stub


def test_payload_and_file_contents(tmp_path):
    ds = fetchers.load_sample_dataset()
    html = report(ds, tmp_path).read_text(encoding="utf-8")
    # Payload table: sensor labels, and the source SENSOR_* column links to the catalog.
    assert "Payload" in html
    for label in ("Temperature", "Salinity", "Chlorophyll", "Altimeter", "ADCP"):
        assert label in html
    assert "SENSOR_CTD_205048" in html  # TEMP's source sensor, shown in the payload Source column
    # File-contents inventory: dimension-grouped tables, a sensor catalog, and the attr dump.
    for heading in (
        "Coordinates on N_MEASUREMENTS",
        "Variables on N_MEASUREMENTS",
        "Sensor catalog",
        "Global attributes",
    ):
        assert f"<h3>{heading}</h3>" in html
    assert "Standard name" in html
    assert "TEMP" in html
    # The _QC companions are not listed as inventory rows (shown in the QC section instead).
    assert "QC-flag variables" in html


def test_qc_section_has_basic_checks_sentences(tmp_path):
    ds = fetchers.load_sample_dataset()
    html = report(ds, tmp_path).read_text(encoding="utf-8")
    assert "Profile number:" in html
    assert "Profile duration:" in html


def test_qc_delivered_covers_all_qc_variables(tmp_path):
    from glidertest.reports._mission import _qc_delivered

    ds = fetchers.load_sample_dataset()
    delivered = _qc_delivered(ds)
    # Every *_QC variable in the file must appear, not just the four with diagnostics thresholds.
    for parent in ("TEMP", "PSAL", "DOXY", "CHLA", "CNDC", "DENSITY", "POTDENS0", "THETA"):
        assert f"<td>{parent}</td>" in delivered
    # Column labels come from the file's flag_meanings: flag 2 is "Unknown" here, not QARTOD wording.
    assert "Unknown %" in delivered
    assert "Not eval %" not in delivered


def test_flag_labels_read_from_file():
    from glidertest import qc

    ds = fetchers.load_sample_dataset()
    labels = qc.flag_labels(ds["TEMP_QC"])
    # flag_values [1,2,3,4,9] / flag_meanings "GOOD UNKNOWN SUSPECT FAIL MISSING".
    assert labels[1] == "Good"
    assert labels[2] == "Unknown"
    assert labels[4] == "Fail"


def test_order_globals_canonical_then_file_order():
    from glidertest import og1_attrs

    attrs = {"glider_serial": "x", "Conventions": "CF", "title": "t", "zzz_custom": "q"}
    # title before Conventions (canonical order), then the two non-OG1 keys in file order.
    assert og1_attrs.order_globals(attrs) == ["title", "Conventions", "glider_serial", "zzz_custom"]


def test_flag_counts_sums_to_size_with_other_bucket():
    from glidertest import qc

    arr = np.array([1, 1, 0, 3, 4, 7, 9, 2])  # 0 and 7 are non-standard -> "other"
    c = qc.flag_counts(arr)
    assert sum(c.values()) == arr.size  # invariant: nothing dropped
    assert c["other"] == 2


def test_flag_scale_mismatch_and_label_fallback():
    from glidertest import qc

    good = xr.DataArray(
        np.array([1], dtype="int8"), attrs={"flag_values": [1, 2, 3], "flag_meanings": "a b c"}
    )
    bad = xr.DataArray(
        np.array([1], dtype="int8"), attrs={"flag_values": [1, 2, 3], "flag_meanings": "a b"}
    )
    assert qc.flag_scale_mismatch(good) is None
    assert qc.flag_scale_mismatch(bad) == (3, 2)
    # on a mismatch, flag_labels keeps the defaults rather than apply a partial scale
    assert qc.flag_labels(bad)[2] == "Not evaluated"


def test_flag_labels_fall_back_without_attrs():
    from glidertest import qc

    bare = xr.DataArray(np.array([1, 3, 4], dtype="int8"))  # no flag_values/flag_meanings
    labels = qc.flag_labels(bare)
    assert labels[2] == "Not evaluated"  # default from QC_FLAG_CATEGORIES


def test_fmt_scalar():
    assert _fmt_scalar(None) == "—"
    assert _fmt_scalar(float("nan")) == "—"
    assert _fmt_scalar(1.5) == "1.5"
    assert _fmt_scalar(1234.5678) == "1235"  # 4 significant figures
    assert len(_fmt_scalar("x" * 50)) == 40  # non-float falls back to a 40-char string


def test_attrs_details():
    assert _attrs_details({}) == "—"
    assert _attrs_details({"a": "1"}, exclude=("a",)) == "—"  # all attrs excluded
    two = _attrs_details({"a": "1", "b": "2"})
    assert "<details" in two and "2 attrs" in two
    one = _attrs_details({"a": "1", "b": "2"}, exclude=("a",))
    assert "1 attrs" in one


def test_var_table_empty():
    assert _var_table(xr.Dataset(), []) == ""


def test_file_contents_groups_each_dimension_signature():
    # A variable on a second dimension gets its own "Variables on ..." table.
    ds = xr.Dataset(
        {
            "TEMP": ("N_MEASUREMENTS", np.arange(5.0)),
            "ADCP_VEL": (("N_MEASUREMENTS", "N_CELLS"), np.zeros((5, 3))),
        },
        coords={"TIME": ("N_MEASUREMENTS", np.arange(5))},
    )
    html = _file_contents(ds)
    assert "<h3>Variables on N_MEASUREMENTS</h3>" in html
    assert "<h3>Variables on N_MEASUREMENTS, N_CELLS</h3>" in html


def test_header_card_degrades_on_nat_and_nan():
    from glidertest.reports._mission import header_card

    nat = np.array(["NaT", "NaT", "NaT"], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {
            "PLATFORM_SERIAL_NUMBER": ((), np.float64("nan")),
            "PROFILE_NUMBER": ("N_MEASUREMENTS", np.array([1.0, 1.0, 2.0])),
        },
        coords={"TIME": ("N_MEASUREMENTS", nat)},
    )
    fields = dict(header_card(ds))  # must not raise
    assert fields["Platform serial"] == "UNK"
    assert fields["Start"] == "UNK"
    assert fields["Duration"] == "UNK"
    assert fields["Sampling"] == "UNK"


def test_report_style_reaches_figure():
    # Setting _ACTIVE_STYLE must drive the figure width through the plotter's own inner
    # style context, independent of _force_width.
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
