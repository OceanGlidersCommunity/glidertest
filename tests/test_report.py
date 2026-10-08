import json
import warnings

import numpy as np
import pytest
import xarray as xr
from PIL import Image

from glidertest import fetchers, plots
from glidertest.config.report_tokens import FIG_DPI, W_FULL
from glidertest.reports import _slots, report
from glidertest.reports._mission import PROFILE, build
from glidertest.reports.inventory import _fmt_scalar, inventory_data

# Most render-then-inspect tests read the one session render of the committed sea045 subset
# (`subset_report`, read-only); dataset-only tests open it once (`subset_ds`). The subset keeps its
# OG1 id, so the mission directory is still "sea045_20230604T1253_delayed", but its *source file* is
# "sea045_subset.nc" (what a few assertions check). See tests/data/README.md. The subset renders
# every diagnostic panel except the day/night offset pair (no night in 12 Baltic-June profiles),
# which keeps its own `slow` full-sample test below.


def test_report_writes_files(subset_report):
    mdir = subset_report  # report writes into <root>/<mission_id>/
    out = mdir / "index.html"
    assert mdir.name == "sea045_20230604T1253_delayed"
    assert out.exists()  # landing page is index.html
    figures = list((mdir / "figures").glob("*.png"))
    assert len(figures) >= 4
    html = out.read_text(encoding="utf-8")
    assert "#07264f" in html  # package accent
    # The landing page is about the mission; File contents moved to inventory.html. QC is split into
    # two jumpable sections (as delivered / glidertest diagnostics). The sensor list is "Payload".
    for section_id in ("track", "payload", "hydrography", "sampling", "qc_delivered", "qc_glidertest"):
        assert f'id="{section_id}"' in html
    assert 'id="file_contents"' not in html
    inventory = (mdir / "inventory.html").read_text(encoding="utf-8")
    for section_id in ("og1", "file_contents"):
        assert f'id="{section_id}"' in inventory


def test_sensor_pages_rendered(subset_report):
    mdir = subset_report
    for page in ("ctd.html", "oxygen.html", "optics.html"):
        p = mdir / page
        assert p.exists()  # the sensor's variables are present -> its page is written
        html = p.read_text(encoding="utf-8")
        assert "page-nav" in html  # cross-page nav appears with >1 page
        slug = page[:-5]
        figs = list((mdir / "figures").glob(f"{slug}_*.png"))
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


def test_flight_absent_and_cr_on_ctd(subset_report):
    mdir = subset_report
    # The flight page needs the glider flight-model velocity, which the sample lacks -> no flight page.
    assert not (mdir / "flight.html").exists()
    # Convective resistance is a mixed-layer diagnostic and lives on the CTD page (needs TEMP+PSAL).
    ctd = (mdir / "ctd.html").read_text(encoding="utf-8")
    assert "Convective resistance" in ctd and "Mixed layer" in ctd
    assert list((mdir / "figures").glob("ctd_ctd_cr.png"))


def test_sections_resolve_in_order(subset_ds):
    resolved = build(subset_ds, PROFILE)
    titles = [s.title for s in resolved.sections]
    # Track leads, then Payload (the sensor list); the landing ends with the two QC sections.
    assert titles == [
        "Track", "Payload", "Hydrography", "Sampling",
        "QC — as delivered", "QC — glidertest diagnostics",
    ]
    # the track panel is a figure and always resolves; the payload panel is html and never a stub
    assert not resolved.sections[1].panels[0].is_stub


def test_payload_and_file_contents(subset_report):
    mdir = subset_report
    index = (mdir / "index.html").read_text(encoding="utf-8")
    inventory = (mdir / "inventory.html").read_text(encoding="utf-8")
    # Payload table stays on the landing page: sensor labels and the source SENSOR_* column.
    assert "Payload" in index
    for label in ("Temperature", "Salinity", "Chlorophyll", "Altimeter", "ADCP"):
        assert label in index
    assert "SENSOR_CTD_205048" in index  # TEMP's source sensor, shown in the payload Source column
    # File-contents inventory moved to inventory.html: dimension-grouped tables + sensor catalog.
    for heading in ("On N_MEASUREMENTS", "Sensor catalog"):
        assert f"<h3>{heading}</h3>" in inventory
    # Coordinates and variables share the "On N_MEASUREMENTS" subheader, with sub-labels.
    assert "<strong>Coordinates</strong>" in inventory
    assert "<strong>Variables</strong>" in inventory
    assert "Standard / long name" in inventory  # combined standard+long name column
    assert "TEMP" in inventory
    # The _QC companions are not listed as inventory rows (shown in the QC section instead).
    assert "QC-flag variables" in inventory
    # The global-attribute conformance is categorised on the inventory page, not the landing page.
    assert "<h3>Identity &amp; discovery</h3>" in inventory


def test_inventory_strip_and_index_verdict(subset_report):
    mdir = subset_report
    index = (mdir / "index.html").read_text(encoding="utf-8")
    inventory = (mdir / "inventory.html").read_text(encoding="utf-8")
    # The landing page carries the one-line OG1 verdict and links to the inventory, not the full table.
    assert "mandatory global attributes present" in index
    assert 'href="inventory.html' in index
    # The inventory link is a below-masthead strip (not a role pill), labelled with the file name,
    # active on the inventory page.
    assert "Data inventory:" in index
    assert "inventory-strip" in index
    assert "nav-inventory" not in index  # not rendered as a role pill
    assert "sea045_subset.nc" in index  # the source file name is the pill label
    assert "file-pill-active" in inventory
    # QC coverage line on the inventory.
    assert "data variables carry a" in inventory


def test_conformance_marks_missing_mandatory_amber():
    from glidertest.reports import metadata
    from glidertest.reports._env import get_template

    # Only `title` present: other mandatory attributes (id, Conventions, ...) are missing.
    ds = xr.Dataset(attrs={"title": "t"})
    html = get_template("_og1_conformance.html").render(**metadata.conformance_data(ds))
    assert "<th>Status</th>" not in html  # no status column; presence is shown by the value/dash
    assert "nonconform" in html  # a missing mandatory attribute's dash cell is amber
    assert ">t</td>" in html  # title present -> its value is shown
    # A missing suggested attribute (geospatial bounds) is not listed in the attribute rows.
    assert "geospatial_lat_min</td>" not in html.split("Geospatial extent")[0]


def test_sg014_missing_id_marked_amber(sg014_subset_path):
    # The sg014 subset is a realistic imperfect OG1 file: 15/16 mandatory globals, missing `id`.
    # The missing mandatory attribute must render an amber (nonconform) dash cell.
    from glidertest import og1_attrs
    from glidertest.reports import metadata
    from glidertest.reports._env import get_template

    with xr.open_dataset(sg014_subset_path) as ds:
        assert og1_attrs.conformance_summary(ds.attrs)["mandatory_present"] == 15
        assert "id" not in ds.attrs
        html = get_template("_og1_conformance.html").render(**metadata.conformance_data(ds))
    assert "nonconform" in html


def test_qc_section_has_basic_checks_sentences(subset_report):
    html = (subset_report / "index.html").read_text(encoding="utf-8")
    assert "Profile number:" in html
    assert "Profile duration:" in html


def test_qc_delivered_covers_all_qc_variables(subset_report):
    html = (subset_report / "index.html").read_text(encoding="utf-8")
    # Every *_QC variable in the file must appear, not just those with diagnostic thresholds. The
    # subset keeps CNDC_QC (no diagnostic threshold) alongside the four thresholded vars to hold
    # that distinction; the subset drops the derived-variable QC (DENSITY/POTDENS0/THETA), which
    # test_qc_delivered_covers_derived_qc_on_full_sample checks on the full sample.
    for parent in ("TEMP", "PSAL", "DOXY", "CHLA", "CNDC"):
        assert f"<td>{parent}</td>" in html
    # Column labels come from the file's flag_meanings: flag 2 is "Unknown" here, not QARTOD wording.
    assert "Unknown %" in html
    assert "Not eval %" not in html


@pytest.mark.slow
def test_qc_delivered_covers_derived_qc_on_full_sample(tmp_path):
    # The subset drops the derived-variable QC companions; the full sample carries them, so the
    # delivered-QC table must list every _QC parent, derived (DENSITY/POTDENS0/THETA) included.
    ds = fetchers.load_sample_dataset()
    html = report(ds, tmp_path, navigator=False).read_text(encoding="utf-8")
    for parent in ("TEMP", "PSAL", "DOXY", "CHLA", "CNDC", "DENSITY", "POTDENS0", "THETA"):
        assert f"<td>{parent}</td>" in html


def test_flag_labels_read_from_file(subset_ds):
    from glidertest import qc

    labels = qc.flag_labels(subset_ds["TEMP_QC"])
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
    headers = [g["header"] for g in inventory_data(ds)["groups"]]
    assert "On N_MEASUREMENTS" in headers
    assert "On N_MEASUREMENTS, N_CELLS" in headers


def test_inventory_data_var_meta_fields(subset_ds):
    data = inventory_data(subset_ds)
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


def test_report_style_reaches_figure(subset_ds):
    # Setting _ACTIVE_STYLE must drive the figure width through the plotter's own inner
    # style context, independent of _force_width.
    original = plots._ACTIVE_STYLE
    plots._ACTIVE_STYLE = _slots._report_spec(W_FULL)
    try:
        fig, _ = plots.plot_glider_track(subset_ds)
        assert abs(fig.get_size_inches()[0] - W_FULL) < 1e-6
    finally:
        plots._ACTIVE_STYLE = original


def test_png_width(subset_report):
    from glidertest.config.report_tokens import SLOTS
    from glidertest.reports._mission import PANELS

    mdir = subset_report
    # Each figure PNG is rendered at exactly its panel's declared slot width (round(slot_in * dpi)),
    # not merely at some valid width: a half-slot panel mis-declared as full (or vice versa) must
    # fail here. The filename is "<page>_<panel.id>.png" and page slugs are single tokens, so the
    # panel id is everything after the first underscore.
    pngs = list((mdir / "figures").glob("*.png"))
    assert pngs
    for png in pngs:
        _page, pid = png.stem.split("_", 1)
        expected = round(SLOTS[PANELS[pid].slot][1] * FIG_DPI)
        assert Image.open(png).size[0] == expected, f"{pid} rendered at the wrong slot width"


def test_root_convention_and_manifest(subset_report):
    mdir = subset_report
    # The report writes into <root>/<mission_id>/, named from the data, not the caller's choice.
    assert mdir.name == "sea045_20230604T1253_delayed"
    assert (mdir / "index.html").exists()
    manifest = json.loads((mdir / "report.json").read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 1
    assert manifest["id"] == "sea045_20230604T1253_delayed"
    assert manifest["sensors"] == {"ctd": True, "oxygen": True, "optics": True, "flight": False}
    assert manifest["og1"]["mandatory_present"] == 16
    assert 0 < len(manifest["track"]) <= 200
    assert manifest["source_file"] == "sea045_subset.nc"


def test_mission_id_falls_back_to_labeled_placeholder():
    from glidertest.reports._mission import MISSING_ID, mission_id

    # No id and no source file -> a labelled placeholder (flags the problem, does not crash).
    assert mission_id(xr.Dataset(attrs={})) == MISSING_ID == "mission (id missing)"
    assert mission_id(xr.Dataset(attrs={"id": "foo_R"})) == "foo_R"
    ds = xr.Dataset()
    ds.encoding["source"] = "/data/bar_20230101_delayed.nc"
    assert mission_id(ds) == "bar_20230101_delayed"  # file stem when no id


def _fake_manifest(mid: str, serial: str, *, oxygen: bool) -> dict:
    """Return a minimal report.json dict for navigator tests (no NetCDF, no rendering)."""
    return {
        "manifest_version": 1, "id": mid, "platform_serial": serial,
        "start": "2023-01-01T00:00", "end": "2023-01-10T00:00", "duration_s": 777600,
        "n_profiles": 10, "n_dive": 5, "n_climb": 5,
        "lat_min": 58.0, "lat_max": 59.0, "lon_min": 10.0, "lon_max": 11.0,
        "max_depth_m": 500.0, "n_records": 1000,
        "source_file": f"{mid}.nc", "source_size_bytes": 1,
        "sensors": {"ctd": True, "oxygen": oxygen, "optics": False, "flight": False},
        "og1": {"mandatory_present": 16, "mandatory_total": 16, "missing": []},
        "qc": {"delivered": True, "worst_var": "TEMP", "worst_bad_pct": 0.0},
        "pages": [{"file": "index.html", "label": "Mission"}],
        "track": [[10.0, 58.0], [10.5, 58.5], [11.0, 59.0]],
        "glidertest_version": "0.0", "generated_at": "2026-10-07 00:00 UTC",
    }


def test_navigator_indexes_manifests(tmp_path):
    from glidertest.reports import navigator

    for mid, serial, oxygen in [("m_a", "sea001", True), ("m_b", "sg002", False)]:
        d = tmp_path / mid
        d.mkdir()
        (d / "report.json").write_text(json.dumps(_fake_manifest(mid, serial, oxygen=oxygen)))
    (tmp_path / "no_manifest").mkdir()  # a directory without a manifest

    out = navigator(tmp_path)
    assert out == tmp_path / "index.html"
    html = out.read_text(encoding="utf-8")
    assert "Mission navigator" in html
    assert "m_a" in html and "m_b" in html  # both missions listed
    assert "data:image/png;base64" in html  # the tracks map rendered
    assert 'href="m_a/index.html"' in html  # page buttons link into the mission subdir
    assert "no_manifest" in html  # a manifest-less directory is surfaced, not hidden


def test_titles_and_top_links(subset_report):
    mdir = subset_report
    index = (mdir / "index.html").read_text(encoding="utf-8")
    ctd = (mdir / "ctd.html").read_text(encoding="utf-8")
    # Title: page name first so tabs are distinguishable; the landing page is just the id.
    assert "<title>sea045_20230604T1253_delayed</title>" in index
    assert "<title>CTD — sea045_20230604T1253_delayed</title>" in ctd
    # The vendored "↑ top" per-section script is injected on every page.
    assert "↑ top" in index and "top-link" in index
    # Payload surfaces the sensor model (from the SENSOR_* catalog) as its own column.
    assert "<th>Model</th>" in index
    # Masthead type label (top-right) is per page.
    assert '<span class="masthead-type">Mission report</span>' in index
    assert '<span class="masthead-type">CTD</span>' in ctd
    # Tab titles: landing is just the id; the inventory tab reads "netCDF Inventory — <id>".
    inventory = (mdir / "inventory.html").read_text(encoding="utf-8")
    assert "<title>netCDF Inventory — sea045_20230604T1253_delayed</title>" in inventory
    # Masthead extent is split into Lat and Lon cells with hemisphere-formatted degrees.
    assert "<dt>Lat</dt>" in index and "<dt>Lon</dt>" in index
    assert "°N" in index or "°S" in index
    # Sampling-period moved out of QC into its own "Sample rate" section on the CTD page.
    assert 'id="sample_rate"' in ctd and "Sample rate" in ctd


def test_payload_model_from_sensor_catalog(subset_ds):
    from glidertest.reports.metadata import metadata_data

    temp = next(p for p in metadata_data(subset_ds)["payload"] if p["var"] == "TEMP")
    assert temp["model"]  # model read from the source SENSOR_* catalog entry
    assert isinstance(temp["attrs"], dict)


def test_scalar_table_shows_value_not_minmax():
    from glidertest.reports import inventory
    from glidertest.reports._env import get_template

    ds = xr.Dataset(
        {"TEMP": ("N_MEASUREMENTS", np.arange(5.0)), "WMO_IDENTIFIER": ((), "6801673")},
        coords={"TIME": ("N_MEASUREMENTS", np.arange(5))},
    )
    data = inventory.inventory_data(ds)
    scalar = next(g for g in data["groups"] if g.get("scalar"))
    assert scalar["variables"][0]["value"] == "6801673"
    html = get_template("_inventory.html").render(**data)
    # Scalars get their own subheader and a Value column instead of Min/Max + N (valid).
    assert "<h3>Scalar variables</h3>" in html
    assert "Value</th>" in html
    assert "6801673" in html  # the scalar value is shown
    assert "N (valid)" not in html.split("Scalar variables")[1]  # not in the scalar table


def test_mission_facts_degrades_on_all_nan_profile_number():
    from glidertest.reports.metadata import mission_facts

    # DEPTH present but PROFILE_NUMBER all-NaN: the max-depth groupby would raise — mission_facts
    # must degrade (None, not a substituted 0) rather than abort the report.
    ds = xr.Dataset(
        {
            "DEPTH": ("N_MEASUREMENTS", np.arange(10.0)),
            "PROFILE_NUMBER": ("N_MEASUREMENTS", np.full(10, np.nan)),
        },
        coords={"TIME": ("N_MEASUREMENTS", np.arange(10).astype("datetime64[s]"))},
    )
    f = mission_facts(ds)  # must not raise
    assert f["n_profiles"] is None and f["n_dive"] is None
    assert f["max_depth_m"] is None


def test_manifest_and_header_agree_on_counts(subset_ds):
    from glidertest.reports._mission import header_card
    from glidertest.reports.manifest import mission_manifest

    header = dict(header_card(subset_ds))
    manifest = mission_manifest(
        subset_ds, mission_id="x", source_name="x.nc", source_size_bytes=None,
        pages=[], version="0", generated_at="t",
    )
    # Both read metadata.mission_facts, so the masthead and the manifest cannot disagree.
    assert str(manifest["n_profiles"]) in header["Profiles"]
    assert manifest["start"].replace("T", " ") == header["Start"]


# --- subset representativeness and the panels it cannot cover ----------------


def test_subset_renders_every_panel_except_daynight(subset_ds):
    # Guard that the committed subset stays representative: every figure panel resolves to a real
    # figure except the day/night offset pair (needs night profiles the 12-profile subset lacks).
    # A regression here means make_subset's keep-list or profile count drifted.
    from glidertest.reports._mission import PAGES, Ctx

    ctx = Ctx(ds=subset_ds)
    stubbed, rendered = [], []
    for page in PAGES:
        if not page.applies_to(ctx):
            continue
        for section in build(subset_ds, page.profile).sections:
            for panel in section.panels:
                if panel.kind == "figure":
                    (stubbed if panel.is_stub else rendered).append(panel.id)
    assert rendered  # the subset exercises the diagnostic panels
    assert stubbed and all("daynight" in pid for pid in stubbed)  # only day/night is missing


@pytest.mark.slow
def test_daynight_panels_render_on_full_sample():
    # The day/night panels the subset cannot cover do render on the multi-day full sample.
    from glidertest.reports._mission import PAGES, Ctx

    ds = fetchers.load_sample_dataset()
    daynight = []
    ctx = Ctx(ds=ds)
    for page in PAGES:
        if not page.applies_to(ctx):
            continue
        for section in build(ds, page.profile).sections:
            daynight += [p for p in section.panels if p.kind == "figure" and "daynight" in p.id]
    assert daynight
    assert all(not p.is_stub for p in daynight)


@pytest.mark.slow
def test_flight_page_renders_on_sg014(tmp_path, sg014_subset_path):
    # The flight page is only reachable with glider flight-model velocity; the sg014 subset has it.
    with xr.open_dataset(sg014_subset_path) as ds:
        mdir = report(ds, tmp_path, navigator=False).parent
    assert (mdir / "flight.html").exists()
    assert list((mdir / "figures").glob("flight_*.png"))


@pytest.mark.slow
def test_report_flat_warns_only_on_different_source(tmp_path, subset_path):
    import shutil

    a, b = tmp_path / "a.nc", tmp_path / "b.nc"
    shutil.copy(subset_path, a)
    shutil.copy(subset_path, b)
    out = tmp_path / "out"
    with xr.open_dataset(a) as ds:
        report(ds, out, layout="flat")  # first write, no prior report: no warning
    with xr.open_dataset(a) as ds, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")  # re-run, same source file: no overwrite warning
        report(ds, out, layout="flat")
    assert not any("already holds a report" in str(w.message) for w in caught)
    with xr.open_dataset(b) as ds, pytest.warns(UserWarning, match="already holds a report"):
        report(ds, out, layout="flat")  # different source file into the same dir: warns


@pytest.mark.slow
def test_report_warns_when_id_reused_for_different_file(tmp_path, subset_ds):
    mdir = report(subset_ds, tmp_path, navigator=False).parent
    # Tamper the manifest to look as if a *different* file had claimed this id.
    manifest = json.loads((mdir / "report.json").read_text(encoding="utf-8"))
    manifest["source_file"] = "a_different_file.nc"
    (mdir / "report.json").write_text(json.dumps(manifest))
    with pytest.warns(UserWarning, match="metadata problem"):
        report(subset_ds, tmp_path, navigator=False)
