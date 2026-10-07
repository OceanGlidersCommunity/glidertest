from pathlib import Path

import matplotlib

matplotlib.use("agg")  # no display; the CLI forces this too

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from PIL import Image  # noqa: E402

from glidertest import fetchers, plots  # noqa: E402
from glidertest.config.report_tokens import FIG_DPI, W_FULL  # noqa: E402
from glidertest.reports import _slots, report  # noqa: E402
from glidertest.reports._mission import PROFILE, build  # noqa: E402
from glidertest.reports.inventory import _fmt_scalar, inventory_data  # noqa: E402


def test_report_writes_files(tmp_path):
    ds = fetchers.load_sample_dataset()
    out = report(ds, tmp_path)
    assert out == Path(tmp_path) / "index.html"  # landing page is index.html
    assert out.exists()
    figures = list((Path(tmp_path) / "figures").glob("*.png"))
    assert len(figures) >= 4
    html = out.read_text(encoding="utf-8")
    assert "#07264f" in html  # package accent
    # The landing page is about the mission; File contents moved to inventory.html.
    for section_id in ("metadata", "track", "hydrography", "sampling", "qc"):
        assert f'id="{section_id}"' in html
    assert 'id="file_contents"' not in html
    inventory = (Path(tmp_path) / "inventory.html").read_text(encoding="utf-8")
    for section_id in ("og1", "file_contents"):
        assert f'id="{section_id}"' in inventory


def test_sensor_pages_rendered(tmp_path):
    ds = fetchers.load_sample_dataset()
    landing = report(ds, tmp_path)
    assert landing.name == "index.html"  # landing is the first applicable page
    for page in ("ctd.html", "oxygen.html", "optics.html"):
        p = tmp_path / page
        assert p.exists()  # the sensor's variables are present -> its page is written
        html = p.read_text(encoding="utf-8")
        assert "page-nav" in html  # cross-page nav appears with >1 page
        slug = page[:-5]
        figs = list((tmp_path / "figures").glob(f"{slug}_*.png"))
        assert len(figs) >= 3  # page-prefixed figures, sharing figures/


def test_sensor_page_panels_in_canonical_order():
    from glidertest.reports import _mission as m

    # Each variable-parameterised panel's rank: (plot-type order, variable order).
    rank = {
        pid: (m.DIAGNOSTIC_ORDER.index(adapter.__name__), m.VARIABLE_ORDER.index(var))
        for pid, adapter, var, _cap, _slot in m._VAR_FIGURE_PANELS
    }
    for profile in (m.CTD, m.OXYGEN, m.OPTICS):
        for section in profile.entries:
            ranked = [rank[p] for p in section.panels if p in rank]
            assert ranked == sorted(ranked), f"{section.id} panels are out of canonical order"


def test_flight_absent_and_cr_on_ctd(tmp_path):
    ds = fetchers.load_sample_dataset()
    report(ds, tmp_path)
    # The flight page needs the glider flight-model velocity, which the sample lacks -> no flight page.
    assert not (tmp_path / "flight.html").exists()
    # Convective resistance is a mixed-layer diagnostic and lives on the CTD page (needs TEMP+PSAL).
    ctd = (tmp_path / "ctd.html").read_text(encoding="utf-8")
    assert "Convective resistance" in ctd and "Mixed layer" in ctd
    assert list((tmp_path / "figures").glob("ctd_ctd_cr.png"))


def test_sections_resolve_in_order():
    ds = fetchers.load_sample_dataset()
    resolved = build(ds, PROFILE)
    titles = [s.title for s in resolved.sections]
    # The landing profile ends at QC; File contents moved to the inventory page.
    assert titles == ["Metadata", "Track", "Hydrography", "Sampling", "QC"]
    # the metadata panel is html and always renders (never a stub)
    assert not resolved.sections[0].panels[0].is_stub


def test_payload_and_file_contents(tmp_path):
    ds = fetchers.load_sample_dataset()
    report(ds, tmp_path)
    index = (Path(tmp_path) / "index.html").read_text(encoding="utf-8")
    inventory = (Path(tmp_path) / "inventory.html").read_text(encoding="utf-8")
    # Payload table stays on the landing page: sensor labels and the source SENSOR_* column.
    assert "Payload" in index
    for label in ("Temperature", "Salinity", "Chlorophyll", "Altimeter", "ADCP"):
        assert label in index
    assert "SENSOR_CTD_205048" in index  # TEMP's source sensor, shown in the payload Source column
    # File-contents inventory moved to inventory.html: dimension-grouped tables + sensor catalog.
    for heading in (
        "Coordinates on N_MEASUREMENTS",
        "Variables on N_MEASUREMENTS",
        "Sensor catalog",
    ):
        assert f"<h3>{heading}</h3>" in inventory
    assert "Standard / long name" in inventory  # combined standard+long name column
    assert "TEMP" in inventory
    # The _QC companions are not listed as inventory rows (shown in the QC section instead).
    assert "QC-flag variables" in inventory
    # The global-attribute conformance is categorised on the inventory page, not the landing page.
    assert "<h3>Identity &amp; discovery</h3>" in inventory


def test_inventory_strip_and_index_verdict(tmp_path):
    ds = fetchers.load_sample_dataset()
    report(ds, tmp_path)
    index = (Path(tmp_path) / "index.html").read_text(encoding="utf-8")
    inventory = (Path(tmp_path) / "inventory.html").read_text(encoding="utf-8")
    # The landing page carries the one-line OG1 verdict and links to the inventory, not the full table.
    assert "mandatory global attributes present" in index
    assert 'href="inventory.html' in index
    # The inventory link is a below-masthead strip (not a role pill), labelled with the file name,
    # active on the inventory page.
    assert "Data inventory:" in index
    assert "inventory-strip" in index
    assert "nav-inventory" not in index  # not rendered as a role pill
    assert "sea045_20230604T1253_delayed.nc" in index  # the source file name is the pill label
    assert "file-pill-active" in inventory
    # QC coverage line on the inventory.
    assert "data variables carry a" in inventory


def test_conformance_marks_missing_mandatory_amber():
    from glidertest.reports import metadata
    from glidertest.reports._env import get_template

    # Only `title` present: other mandatory attributes (id, Conventions, ...) are missing.
    ds = xr.Dataset(attrs={"title": "t"})
    html = get_template("_og1_conformance.html").render(**metadata.conformance_data(ds))
    assert "nonconform" in html  # a missing mandatory attribute is marked amber
    assert "✓ present" in html  # title is present
    # A missing suggested attribute (geospatial bounds) is not listed at all.
    assert "geospatial_lat_min</td>" not in html.split("Geospatial extent")[0]


def test_qc_section_has_basic_checks_sentences(tmp_path):
    ds = fetchers.load_sample_dataset()
    html = report(ds, tmp_path).read_text(encoding="utf-8")
    assert "Profile number:" in html
    assert "Profile duration:" in html


def test_qc_delivered_covers_all_qc_variables(tmp_path):
    ds = fetchers.load_sample_dataset()
    html = report(ds, tmp_path).read_text(encoding="utf-8")
    # Every *_QC variable in the file must appear, not just the four with diagnostics thresholds.
    for parent in ("TEMP", "PSAL", "DOXY", "CHLA", "CNDC", "DENSITY", "POTDENS0", "THETA"):
        assert f"<td>{parent}</td>" in html
    # Column labels come from the file's flag_meanings: flag 2 is "Unknown" here, not QARTOD wording.
    assert "Unknown %" in html
    assert "Not eval %" not in html


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


def test_inventory_data_groups_each_dimension_signature():
    # A variable on a second dimension gets its own "Variables on ..." group.
    ds = xr.Dataset(
        {
            "TEMP": ("N_MEASUREMENTS", np.arange(5.0)),
            "ADCP_VEL": (("N_MEASUREMENTS", "N_CELLS"), np.zeros((5, 3))),
        },
        coords={"TIME": ("N_MEASUREMENTS", np.arange(5))},
    )
    titles = [g["title"] for g in inventory_data(ds)["groups"]]
    assert "Variables on N_MEASUREMENTS" in titles
    assert "Variables on N_MEASUREMENTS, N_CELLS" in titles


def test_inventory_data_var_meta_fields():
    ds = fetchers.load_sample_dataset()
    data = inventory_data(ds)
    # TEMP is on N_MEASUREMENTS, has a TEMP_QC companion, and its attrs dropdown excludes the columns.
    temp = next(v for g in data["groups"] for v in g["variables"] if v["name"] == "TEMP")
    assert temp["has_qc"] is True
    assert "units" not in temp["attrs"] and "long_name" not in temp["attrs"]
    assert data["n_sensors"] == len(data["sensors"]) > 0


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
    from glidertest.config.report_tokens import SLOTS
    from glidertest.reports._mission import PANELS

    ds = fetchers.load_sample_dataset()
    report(ds, tmp_path)
    # Each figure PNG is rendered at exactly its panel's declared slot width (round(slot_in * dpi)),
    # not merely at some valid width: a half-slot panel mis-declared as full (or vice versa) must
    # fail here. The filename is "<page>_<panel.id>.png" and page slugs are single tokens, so the
    # panel id is everything after the first underscore.
    pngs = list((Path(tmp_path) / "figures").glob("*.png"))
    assert pngs
    for png in pngs:
        _page, pid = png.stem.split("_", 1)
        expected = round(SLOTS[PANELS[pid].slot][1] * FIG_DPI)
        assert Image.open(png).size[0] == expected, f"{pid} rendered at the wrong slot width"
