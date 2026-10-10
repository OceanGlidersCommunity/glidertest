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
    # The inventory page is one Section per subsection (each a jump-nav entry), no umbrella headings.
    for section_id in ("og1_identity", "og1_coverage", "coords", "measurements", "sensors"):
        assert f'id="{section_id}"' in inventory
    assert 'id="file_contents"' not in inventory and 'id="og1"' not in inventory


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
    # Each inventory subsection is its own Section (an <h2 id=…> jump-nav entry), not an <h3>: the
    # file-content groups and the attribute categories alike.
    for section_id in ("coords", "measurements", "scalars", "sensors", "og1_identity", "og1_coverage"):
        assert f'id="{section_id}"' in inventory
    assert "Variables on N_MEASUREMENTS" in inventory and "Sensor catalog" in inventory
    assert "Identity &amp; discovery" in inventory  # category heading (& autoescaped)
    assert "Standard / long name" in inventory  # combined standard+long name column
    assert "TEMP" in inventory
    # The _QC companions are not listed as inventory rows (shown in the QC section instead).
    assert "QC-flag variables" in inventory


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

    # Only `title` present: other mandatory attributes (id, ...) in Identity & discovery are missing.
    ds = xr.Dataset(attrs={"title": "t"})
    html = get_template("_og1_conformance.html").render(**metadata.attr_category_data(ds, "Identity & discovery"))
    assert "<th>Status</th>" not in html  # no status column; presence is shown by the value/dash
    assert "nonconform" in html  # a missing mandatory attribute's dash cell is amber
    assert ">t</td>" in html  # title present -> its value is shown


def test_sg014_missing_id_marked_amber(sg014_subset_path):
    # The sg014 subset is a realistic imperfect OG1 file: 15/16 mandatory globals, missing `id`.
    # The missing mandatory attribute must render an amber (nonconform) dash cell.
    from glidertest import og1_attrs
    from glidertest.reports import metadata
    from glidertest.reports._env import get_template

    with xr.open_dataset(sg014_subset_path) as ds:
        assert og1_attrs.conformance_summary(ds.attrs)["mandatory_present"] == 15
        assert "id" not in ds.attrs
        html = get_template("_og1_conformance.html").render(**metadata.attr_category_data(ds, "Identity & discovery"))
    assert "nonconform" in html


def test_inventory_sections_split_and_applies_to():
    from glidertest.reports._mission import INVENTORY

    # Minimal file: no SENSOR_*, no scalars, no extra-dimension vars, no non-OG1 attributes.
    ds = xr.Dataset(
        {"TEMP": ("N_MEASUREMENTS", np.arange(4.0)), "DEPTH": ("N_MEASUREMENTS", np.arange(4.0))},
        coords={"TIME": ("N_MEASUREMENTS", np.arange(4))},
        attrs={"title": "t"},
    )
    secs = build(ds, INVENTORY).sections
    ids = [s.id for s in secs]
    # The four OG1 categories always render (an empty category in amber is the strongest finding);
    # Other / other-dimension variables / scalars drop when absent.
    assert ids[:4] == ["og1_identity", "og1_coverage", "og1_people", "og1_provenance"]
    assert "og1_other" not in ids and "other_dims" not in ids and "scalars" not in ids
    assert "coords" in ids and "measurements" in ids
    # The sensor catalog stays as a stub (absence is a finding, not a non-event).
    sensors = next(s for s in secs if s.id == "sensors")
    assert sensors.panels[0].is_stub
    assert "SENSOR_*" in (sensors.panels[0].stub_reason or "")


def test_inventory_slice_selects_one_group():
    from glidertest.reports.inventory import inventory_slice

    ds = xr.Dataset(
        {
            "TEMP": ("N_MEASUREMENTS", np.arange(5.0)),
            "WMO_IDENTIFIER": ((), "6801673"),
            "SENSOR_CTD": ((), "x"),
        },
        coords={"TIME": ("N_MEASUREMENTS", np.arange(5))},
    )
    assert [g["label"] for g in inventory_slice(ds, "coords")["groups"]] == ["Coordinates"]
    assert inventory_slice(ds, "measurements")["groups"][0]["variables"][0]["name"] == "TEMP"
    assert inventory_slice(ds, "scalars")["groups"][0]["scalar"] and inventory_slice(ds, "scalars")["sensors"] == []
    assert inventory_slice(ds, "sensors")["sensors"] and inventory_slice(ds, "sensors")["groups"] == []
    assert inventory_slice(ds, "other_dims")["groups"] == []


def test_spatiotemporal_computed_time_and_vertical(subset_ds):
    from glidertest.reports.metadata import spatiotemporal_rows

    rows = {r["attr"]: r for r in spatiotemporal_rows(subset_ds)}
    assert rows["time_coverage_start"]["computed"].startswith("2023-06-04T")  # ISO to the second
    # Vertical extent is the shallowest/deepest sample (DEPTH min/max) — not the per-profile-max
    # range that mission_facts carries. min must be near the surface.
    vmin = float(rows["geospatial_vertical_min"]["computed"])
    vmax = float(rows["geospatial_vertical_max"]["computed"])
    assert vmin == float(np.nanmin(subset_ds["DEPTH"].values))
    assert vmax == float(np.nanmax(subset_ds["DEPTH"].values))
    assert vmin < 1.0 < vmax
    # sea045 declares no geospatial_vertical_positive, so the vertical rows note the ambiguity.
    assert "sign convention not declared" in rows["geospatial_vertical_min"]["note"]
    # No file attributes present -> every diff is empty ("—" in the table).
    assert all(r["diff"] == "" for r in rows.values())


def test_spatiotemporal_diff_amber_above_threshold(subset_path):
    from glidertest.reports.metadata import spatiotemporal_rows

    # The diff is always shown when both values exist; the cell turns amber only above the per-axis
    # threshold (5e-4° for latitude). Real subset, attribute nudged in memory.
    with xr.open_dataset(subset_path) as ds:
        computed = float(np.nanmax(ds["LATITUDE"].values))
        for delta, want_amber in ((0.01, True), (1e-6, False)):
            ds.attrs["geospatial_lat_max"] = repr(computed + delta)
            row = next(r for r in spatiotemporal_rows(ds) if r["attr"] == "geospatial_lat_max")
            assert row["diff"], f"diff always shown (delta={delta})"
            assert row["over"] is want_amber, f"delta={delta} -> over={row['over']}"


def test_spatiotemporal_caption_flags_subset(subset_path):
    from glidertest.reports.metadata import attr_category_data

    # The sea045 subset starts at profile 468, so the caption flags it as a subset and names the
    # first profile — the computed ranges need not match the full mission.
    with xr.open_dataset(subset_path) as ds:
        cap = attr_category_data(ds, "Spatiotemporal coverage")["spatiotemporal_caption"]
    assert "subset" in cap and "468" in cap


def test_spatiotemporal_caption_plain_when_first_profile_is_one():
    from glidertest.reports.metadata import _spatiotemporal_caption

    ds = xr.Dataset({"PROFILE_NUMBER": ("N_MEASUREMENTS", np.array([1, 1, 2, 2]))})
    cap = _spatiotemporal_caption(ds)
    assert "subset" not in cap and cap.startswith("Computed values")


def test_sensor_catalog_caption_and_legacy_amber(subset_report):
    html = (subset_report / "inventory.html").read_text(encoding="utf-8")
    # The sea045 sensors carry pre-OG1 serial_number / calibration_date, so the cells are amber and
    # the caption names the OG1 attribute and both amber reasons.
    assert "sensor_serial_number" in html  # caption names the OG1 attribute
    assert "from the pre-OG1 attribute serial_number" in html  # legacy hover present


def test_sensor_meta_resolves_og1_legacy_and_missing():
    from glidertest.reports.inventory import _sensor_meta

    # Synthetic because no committed fixture carries the OG1 `sensor_*` form or a missing
    # serial/calibration: sea045's sensors all have both under the pre-OG1 names (that legacy render
    # + amber is covered on the real file by test_sensor_catalog_caption_and_legacy_amber). Here:
    # OG1 name -> not legacy; pre-OG1 name -> legacy (amber); absent -> empty, left unflagged.
    ds = xr.Dataset({"SENSOR_A": ((), 0), "SENSOR_B": ((), 0)})
    ds["SENSOR_A"].attrs["sensor_serial_number"] = "og1"  # current OG1 name
    ds["SENSOR_B"].attrs["serial_number"] = "abc"  # pre-OG1 name only
    a = _sensor_meta(ds, "SENSOR_A")
    b = _sensor_meta(ds, "SENSOR_B")
    assert (a["serial"], a["serial_legacy"]) == ("og1", False)  # OG1 name, not flagged
    assert (b["serial"], b["serial_legacy"]) == ("abc", True)  # pre-OG1 name, flagged
    assert (b["calibration"], b["calibration_legacy"]) == ("", False)  # absent, left unflagged


def test_spatiotemporal_compact_time_parses(subset_path):
    from glidertest.reports.metadata import spatiotemporal_rows

    # OG1 prescribes the compact YYYYmmddTHHMMSS form for time_coverage_*; it must parse and produce
    # a seconds diff, not a dash. Build the compact spelling of the computed start; the diff is a
    # small sub-threshold value (the computed display is truncated to the second, so it is within 1 s
    # of the raw start, not exactly zero).
    with xr.open_dataset(subset_path) as ds:
        iso = {r["attr"]: r for r in spatiotemporal_rows(ds)}["time_coverage_start"]["computed"]
        ds.attrs["time_coverage_start"] = iso.replace("-", "").replace(":", "")  # e.g. 20230604T...
        row = next(r for r in spatiotemporal_rows(ds) if r["attr"] == "time_coverage_start")
        assert row["diff"].endswith(" s") and row["diff"] != "unreadable"  # compact form parsed
        assert row["over"] is False  # within the 60 s threshold
        assert row["note"] == ""  # the OG1-compact form carries no deviation note


def test_spatiotemporal_non_og1_time_format_noted(subset_path):
    from glidertest.reports.metadata import spatiotemporal_rows

    # Readable but non-OG1 spelling (extended ISO) parses and diffs, but carries a format note and
    # is not amber — we surface the deviation, we do not format-check the value.
    with xr.open_dataset(subset_path) as ds:
        iso = {r["attr"]: r for r in spatiotemporal_rows(ds)}["time_coverage_start"]["computed"]
        ds.attrs["time_coverage_start"] = iso  # extended ISO, e.g. 2023-06-04T12:53:04
        row = next(r for r in spatiotemporal_rows(ds) if r["attr"] == "time_coverage_start")
        assert "not OG1" in row["note"]
        assert row["diff"].endswith(" s") and row["over"] is False  # parsed, not flagged amber


def test_spatiotemporal_unreadable_time_flagged(subset_path):
    from glidertest.reports.metadata import spatiotemporal_rows

    # A declared-but-unreadable timestamp is a finding: amber, not a quiet dash.
    with xr.open_dataset(subset_path) as ds:
        ds.attrs["time_coverage_start"] = "not-a-date"
        row = next(r for r in spatiotemporal_rows(ds) if r["attr"] == "time_coverage_start")
        assert row["over"] is True
        assert "unreadable" in row["diff"]


def test_spatiotemporal_vertical_positive_up(subset_path):
    from glidertest.reports.metadata import spatiotemporal_rows

    # positive="up": the file's vertical_min is the most-negative height (deepest sample), its
    # vertical_max is near the surface. The computed bounds must negate AND swap DEPTH's
    # (shallow, deep), so computed_min < computed_max and a correct file is not flagged amber.
    with xr.open_dataset(subset_path) as ds:
        deep = float(np.nanmax(ds["DEPTH"].values))
        shallow = float(np.nanmin(ds["DEPTH"].values))
        ds.attrs["geospatial_vertical_positive"] = "up"
        ds.attrs["geospatial_vertical_min"] = repr(-deep)
        ds.attrs["geospatial_vertical_max"] = repr(-shallow)
        rows = {r["attr"]: r for r in spatiotemporal_rows(ds)}
        cmin = float(rows["geospatial_vertical_min"]["computed"])
        cmax = float(rows["geospatial_vertical_max"]["computed"])
        assert (cmin, cmax) == (-deep, -shallow)  # negated and swapped, not negated in place
        assert cmin < cmax
        assert rows["geospatial_vertical_min"]["over"] is False
        assert rows["geospatial_vertical_max"]["over"] is False


def test_spatiotemporal_nonfinite_file_value_not_diffed(subset_path):
    from glidertest.reports.metadata import spatiotemporal_rows

    # A file attribute of "nan" is not a usable bound: show no diff and never flag amber, rather
    # than render "-nan" and silently pass the threshold check.
    with xr.open_dataset(subset_path) as ds:
        ds.attrs["geospatial_lat_max"] = "nan"
        row = next(r for r in spatiotemporal_rows(ds) if r["attr"] == "geospatial_lat_max")
        assert row["diff"] == ""
        assert row["over"] is False


def test_track_drops_qc_flagged_positions():
    from glidertest.reports.manifest import _track

    # The decimated json track drops only positions the file flags bad (OG1 QC 3/4); good and
    # probably-good (1/2) are kept, so a spurious flagged fix does not stretch the track.
    ds = xr.Dataset(
        {
            "LONGITUDE": ("N_MEASUREMENTS", np.array([10.0, 99.0, 11.0])),
            "LATITUDE": ("N_MEASUREMENTS", np.array([55.0, 0.0, 56.0])),
            "LONGITUDE_QC": ("N_MEASUREMENTS", np.array([1, 4, 2])),  # middle fix flagged bad (4)
            "LATITUDE_QC": ("N_MEASUREMENTS", np.array([1, 4, 2])),
        }
    )
    track = _track(ds)
    assert [p[0] for p in track] == [10.0, 11.0]  # the flag-4 point is dropped, flag-2 kept


def test_track_keeps_unevaluated_qc_zero():
    from glidertest.reports.manifest import _track

    # OG1 flag 0 = "no QC applied" is the common delivered state and must be kept — dropping it
    # would empty the track of every file whose provider ran no position QC.
    ds = xr.Dataset(
        {
            "LONGITUDE": ("N_MEASUREMENTS", np.array([10.0, 11.0, 12.0])),
            "LATITUDE": ("N_MEASUREMENTS", np.array([55.0, 56.0, 57.0])),
            "LONGITUDE_QC": ("N_MEASUREMENTS", np.array([0, 0, 0])),
            "LATITUDE_QC": ("N_MEASUREMENTS", np.array([0, 0, 0])),
        }
    )
    assert [p[0] for p in _track(ds)] == [10.0, 11.0, 12.0]  # all kept, none dropped


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


def test_sensor_meta_prefers_og1_names_and_flags_legacy():
    from glidertest.reports.inventory import _sensor_meta

    ds = xr.Dataset({"SENSOR_NEW": ((), "x"), "SENSOR_OLD": ((), "x")})
    ds["SENSOR_NEW"].attrs = {
        "sensor_model": "SBE",
        "sensor_serial_number": "S1",
        "sensor_calibration_date": "2023-01-01",
    }
    ds["SENSOR_OLD"].attrs = {
        "sensor_model": "SBE",
        "serial_number": "S2",
        "calibration_date": "2022-01-01",
    }
    new = _sensor_meta(ds, "SENSOR_NEW")
    old = _sensor_meta(ds, "SENSOR_OLD")
    # OG1 names: shown, not flagged. Pre-OG1 names: shown, flagged legacy (→ amber).
    assert (new["serial"], new["serial_legacy"]) == ("S1", False)
    assert (new["calibration"], new["calibration_legacy"]) == ("2023-01-01", False)
    assert (old["serial"], old["serial_legacy"]) == ("S2", True)
    assert (old["calibration"], old["calibration_legacy"]) == ("2022-01-01", True)
    # both spellings are consumed, so neither shows up again in the attrs dropdown
    assert "serial_number" not in old["attrs"] and "sensor_serial_number" not in new["attrs"]


def test_sensor_catalog_marks_legacy_names_amber():
    from glidertest.reports._env import get_template
    from glidertest.reports.inventory import inventory_data

    ds = xr.Dataset(
        {"TEMP": ("N_MEASUREMENTS", np.arange(3.0)), "SENSOR_CTD": ((), "x")},
        coords={"TIME": ("N_MEASUREMENTS", np.arange(3))},
    )
    ds["SENSOR_CTD"].attrs = {"sensor_model": "SBE", "serial_number": "123", "calibration_date": "2020"}
    html = get_template("_inventory.html").render(**inventory_data(ds))
    assert "nonconform" in html  # the legacy serial/calibration cells carry the amber class
    assert "123" in html and "2020" in html  # the values are still shown, just flagged


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
    data = inventory.inventory_slice(ds, "scalars")
    scalar = next(g for g in data["groups"] if g.get("scalar"))
    assert scalar["variables"][0]["value"] == "6801673"
    html = get_template("_inventory.html").render(**data)
    # The scalar slice uses a Value column instead of Min/Max + N (valid); its heading is the
    # Section's <h2>, not an <h3> in the panel.
    assert "Value</th>" in html
    assert "6801673" in html  # the scalar value is shown
    assert "N (valid)" not in html  # the scalar table has no N (valid) column


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
