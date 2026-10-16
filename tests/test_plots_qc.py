import re
from pathlib import Path

import numpy as np
import pytest

from glidertest import plots_qc, qc, utilities
from glidertest.reports import qc_section

# conftest sets the Agg backend before any test imports, so no matplotlib.use("agg") is needed here.
# These run on the committed sea045 subset (CTD + oxygen + optics), opened read-only per session.

_BASE_HTML = Path(__file__).resolve().parents[1] / "glidertest" / "reports" / "templates" / "base.html"


# --- colour / marker vocabulary ------------------------------------------------------------------


def test_qc_colors_match_css_tokens():
    # The matplotlib flag colours must equal base.html's --qc-* custom properties, so the delivered-QC
    # markers in a figure read the same as the QC bars and labels rendered in HTML.
    css = _BASE_HTML.read_text()
    tokens = dict(re.findall(r"--qc-(good|suspect|fail):\s*(#[0-9a-fA-F]{6})", css))
    assert plots_qc.QC_COLORS[1] == tokens["good"]
    assert plots_qc.QC_COLORS[3] == tokens["suspect"]
    assert plots_qc.QC_COLORS[4] == tokens["fail"]


def test_test_colors_are_distinct_okabe_ito():
    # One colour per test (gross/spike/flat), all different, and distinct from the flag-level colours.
    assert len(set(plots_qc.TEST_COLORS.values())) == 3
    assert set(plots_qc.TEST_MARKER) == set(plots_qc.TEST_COLORS)  # every test has a marker


# --- plot_test_hits ------------------------------------------------------------------------------


def test_plot_test_hits_draws_data_as_markers_not_line(subset_ds):
    # An off-phase sensor (CHLA) has isolated finite samples, so the data must be markers, not a line,
    # or nothing is drawn (the bug this guards against).
    _fig, ax = plots_qc.plot_test_hits(subset_ds, "CHLA")
    data = ax.get_lines()[0]
    assert data.get_linestyle() == "None"
    assert data.get_marker() == "."
    assert ax.get_ylabel() == utilities.plotting_labels("CHLA") + f" ({utilities.plotting_units(subset_ds, 'CHLA')})"


def test_plot_test_hits_overlays_most_frequent_first(subset_ds):
    # Overlays are drawn most-frequent first so a rarer test's symbols land on top. The scatter
    # collections (after the base data markers) therefore have non-increasing point counts.
    _fig, ax = plots_qc.plot_test_hits(subset_ds, "CHLA")
    sizes = [len(c.get_offsets()) for c in ax.collections]
    assert sizes == sorted(sizes, reverse=True)
    assert len(sizes) >= 2  # at least one test overlay beyond nothing


# --- plot_flag_profile ---------------------------------------------------------------------------


def test_plot_flag_profile_is_value_vs_depth(subset_ds):
    _fig, ax = plots_qc.plot_flag_profile(subset_ds, "TEMP")
    # x is the variable, y is depth increasing downward (inverted axis).
    assert "Temperature" in ax.get_xlabel()
    assert ax.get_ylim()[0] > ax.get_ylim()[1]


def test_plot_flag_profile_requires_depth(fresh_subset):
    ds = fresh_subset.drop_vars("DEPTH")
    with pytest.raises(KeyError):
        plots_qc.plot_flag_profile(ds, "TEMP")


def test_plot_flag_profile_overlays_most_frequent_first(subset_ds):
    _fig, ax = plots_qc.plot_flag_profile(subset_ds, "CHLA")
    sizes = [len(c.get_offsets()) for c in ax.collections]
    assert sizes == sorted(sizes, reverse=True)


# --- plot_flat_change ----------------------------------------------------------------------------


def test_plot_flat_change_raises_without_multisample_runs(subset_ds):
    # CHLA is sampled off-phase: every flat flag is a lone sample, so there is no start-to-end change
    # to measure and the plotter raises (the report turns this into an informative stub).
    with pytest.raises(ValueError, match="no multi-sample flat run"):
        plots_qc.plot_flat_change(subset_ds, "CHLA")


def test_plot_flat_change_title_reports_fraction_above_tolerance(subset_ds):
    _fig, ax = plots_qc.plot_flat_change(subset_ds, "TEMP")
    assert "% above tolerance" in ax.get_title()


# --- suggested limits (qc) -----------------------------------------------------------------------


def test_suggest_flat_line_scales_with_interval(subset_ds):
    fl = qc.suggest_flat_line(subset_ds)
    assert fl["suspect"] == round(fl["suspect_factor"] * fl["dt"])
    assert fl["fail"] == round(fl["fail_factor"] * fl["dt"])
    assert fl["fail"] > fl["suspect"]


def test_suggest_gross_range_clamps_nonnegative_and_pads(subset_ds):
    gr = qc.suggest_gross_range(subset_ds, "CHLA")
    assert gr["suspect"][0] == 0.0  # non-negative variable: lower bound clamped
    assert gr["fail"][0] <= gr["suspect"][0] and gr["fail"][1] >= gr["suspect"][1]  # fail span wider
    assert gr["inner_pct"] == 99.9


# --- notes: current above suggested --------------------------------------------------------------


def test_suggested_limits_note_shows_current_then_suggested(subset_ds):
    html = qc_section.suggested_limits_note(subset_ds, "TEMP")
    assert html.index("Current limits (applied)") < html.index("Suggested limits (this deployment)")
    assert "≥ 1 s" in html  # current flat-line suspect window (applied)
    assert "≥ 2 s" in html  # suggested flat-line suspect window


def test_gross_range_ts_note_shows_current_then_suggested(subset_ds):
    html = qc_section.gross_range_ts_note(subset_ds)
    assert html.index("Current gross-range limits (applied)") < html.index("Suggested gross-range limits")
    assert "[0, 30]" in html  # current TEMP gross suspect span, in brackets


# --- shared gross mask: matrix count == plotted count --------------------------------------------


def test_gross_mask_count_matches_matrix_and_plot(subset_ds):
    # One definition of "out of range": the diagnostics-matrix cell and the plotted × must both equal
    # qc.gross_range_mask(...).sum() on the committed subset, so the table and the figure cannot diverge.
    had_hits = False
    for var in ("TEMP", "PSAL", "DOXY", "CHLA"):
        if var not in subset_ds:
            continue
        n = int(qc.gross_range_mask(subset_ds, var).sum())
        cell = qc_section._gross_cell(subset_ds, var)
        matrix_n = 0 if "spans" not in cell else int(cell["spans"][0]["text"].split()[0].replace(",", ""))
        assert matrix_n == n
        _fig, ax = plots_qc.plot_test_hits(subset_ds, var)
        gross = [c for c in ax.collections if c.get_label() == "gross range"]
        plotted = len(gross[0].get_offsets()) if gross else 0
        assert plotted == n
        had_hits = had_hits or n > 0
    assert had_hits  # the subset must exercise at least one gross-range hit, else the test is vacuous


# --- robustness ----------------------------------------------------------------------------------


def test_suggest_flat_line_is_per_variable(subset_ds):
    # An off-phase sensor (CHLA, finite only every Nth cycle) has a coarser cadence than the CTD, so
    # its suggested flat-line window must be based on its own interval, not the whole-record one.
    temp = qc.suggest_flat_line(subset_ds, "TEMP")
    chla = qc.suggest_flat_line(subset_ds, "CHLA")
    assert chla["dt"] > temp["dt"]  # CHLA is sampled less often than TEMP
    assert chla["suspect"] == round(chla["suspect_factor"] * chla["dt"])
    note = qc_section.suggested_limits_note(subset_ds, "CHLA")
    assert f"every {chla['dt']:g} s" in note  # the note names CHLA's own cadence, not the record's


def test_suggested_limits_note_none_for_all_nan_variable(subset_ds):
    # A present-but-all-NaN variable has no mission-descriptive limits: the note must be absent, not a
    # misleading panel (and never a crash — guarded).
    ds = subset_ds.copy()
    ds["TEMP"] = ("N_MEASUREMENTS", np.full(ds.TEMP.shape, np.nan))
    assert qc_section.suggested_limits_note(ds, "TEMP") is None


def test_suggest_flat_line_handles_numeric_time(subset_ds):
    # TIME as numeric seconds (un-decoded) must not crash suggest_flat_line (shared _elapsed_seconds
    # handles both dtypes); the whole-record interval is still recovered.
    ds = subset_ds.copy()
    ds["TIME"] = ("N_MEASUREMENTS", utilities._elapsed_seconds(subset_ds.TIME.values))
    fl = qc.suggest_flat_line(ds)
    assert np.isfinite(fl["dt"]) and fl["suspect"] == round(fl["suspect_factor"] * fl["dt"])


def test_suggest_note_panel_is_guarded_against_a_raising_builder(subset_ds, monkeypatch):
    # A note builder that raises must yield a visible failure block on the page, not abort the report.
    from glidertest.reports import _mission

    def boom(ds, var, config=None):
        raise ValueError("kaboom")

    monkeypatch.setattr(_mission.qc_section, "suggested_limits_note", boom)
    html = _mission.PANELS["qc_TEMP_suggest"].render(_mission.Ctx(ds=subset_ds))
    assert "none-note" in html and "kaboom" in html


# --- config overrides propagate through the seam --------------------------------------------------


def test_override_changes_gross_range_count(subset_ds):
    # A narrower suspect span must flag more samples: the override reaches qc.gross_range_mask.
    from glidertest.config import loader

    default = int(qc.gross_range_mask(subset_ds, "TEMP").sum())
    cfg = loader.resolve_qc(loader.load({"qc": {"TEMP": {"gross_range_test": {"suspect_span": [10, 11]}}}}))
    overridden = int(qc.gross_range_mask(subset_ds, "TEMP", config=cfg).sum())
    assert overridden > default


def test_override_shows_in_diagnostics_thresholds_table(subset_ds):
    # An overridden spike threshold appears in the glidertest-diagnostics thresholds table.
    from glidertest.config import loader

    cfg = loader.resolve_qc(loader.load({"qc": {"TEMP": {"spike_test": {"suspect_threshold": 99.0}}}}))
    rows = qc_section.diagnostics_data(subset_ds, cfg)["diagnostics"]["thresholds"]
    spike = next(r for r in rows if r["var"] == "TEMP" and r["test"] == "spike")
    assert "99" in spike["suspect"]


def test_no_config_matches_defaults(subset_ds):
    # Passing the resolved defaults is identical to passing nothing (the byte-identical guarantee).
    from glidertest.config import loader

    default = qc.sample_tests(subset_ds, "TEMP")
    resolved = qc.sample_tests(subset_ds, "TEMP", config=loader.resolve_qc(None))
    for a, b in zip(default, resolved):
        assert np.array_equal(np.asarray(a), np.asarray(b))


# --- region test (mission-level) ------------------------------------------------------------------


def test_region_test_flags_inside_outside_and_missing():
    import xarray as xr

    ds = xr.Dataset({
        "LONGITUDE": ("N_MEASUREMENTS", np.array([15.0, 5.0, np.nan, 20.0])),
        "LATITUDE": ("N_MEASUREMENTS", np.array([60.0, 60.0, 60.0, 80.0])),
    })
    flags = qc.region_test(ds, [10, 50, 25, 70])  # [west, south, east, north]
    # inside -> 1, west of box -> 4, NaN lon -> 9, north of box -> 4
    assert flags.tolist() == [1, 4, 9, 4]


def test_region_row_absent_without_bbox(subset_ds):
    assert "region" not in qc_section.diagnostics_data(subset_ds)["diagnostics"]


def test_region_row_present_and_evaluated_with_bbox(subset_ds):
    # A box around the mission's own track evaluates with no samples outside.
    lon, lat = subset_ds["LONGITUDE"].values, subset_ds["LATITUDE"].values
    bbox = [float(np.nanmin(lon)) - 1, float(np.nanmin(lat)) - 1, float(np.nanmax(lon)) + 1, float(np.nanmax(lat)) + 1]
    region = qc_section.diagnostics_data(subset_ds, None, bbox)["diagnostics"]["region"]
    assert region["evaluated"] is True
    assert region["n_outside"] == 0 and region["n_evaluated"] > 0 and region["n_unknown"] == 0


def test_region_not_evaluated_without_position(fresh_subset):
    ds = fresh_subset.drop_vars([v for v in ("LONGITUDE", "LATITUDE") if v in fresh_subset])
    region = qc_section.diagnostics_data(ds, None, [10, 50, 25, 70])["diagnostics"]["region"]
    assert region["evaluated"] is False


def test_region_runs_without_standard_qc_variable(fresh_subset):
    # Review finding 3 (+ Copilot): the region test is mission-level and position-only, so it must run
    # for a file with position but no TEMP/PSAL/DOXY/CHLA — both the diagnostics data and the page gate.
    from glidertest.reports import _mission

    ds = fresh_subset.drop_vars([v for v in ("TEMP", "PSAL", "DOXY", "CHLA") if v in fresh_subset])
    lon, lat = ds["LONGITUDE"].values, ds["LATITUDE"].values
    bbox = [float(np.nanmin(lon)) - 1, float(np.nanmin(lat)) - 1, float(np.nanmax(lon)) + 1, float(np.nanmax(lat)) + 1]
    diag = qc_section.diagnostics_data(ds, None, bbox)["diagnostics"]
    assert diag is not None and diag["present"] == [] and "region" in diag  # region row, empty var table
    ctx = _mission.Ctx(ds=ds, bbox=bbox)
    assert _mission._qc_or_region_applies(ctx) is True  # the page/section gate admits it
    assert "Region test" in _mission.PANELS["qc_glidertest"].render(ctx)


def test_spike_method_from_config_is_applied(fresh_subset, monkeypatch):
    # Copilot finding: a configured spike_test.method must reach qartod.spike_test, not be ignored
    # while the thresholds table still marks the row "config".
    from glidertest.config import loader

    captured = {}

    def spy(*args, **kwargs):
        captured["method"] = kwargs.get("method")
        return np.zeros(fresh_subset.sizes["N_MEASUREMENTS"], dtype="uint8")

    monkeypatch.setattr(qc.qartod, "spike_test", spy)
    cfg = loader.resolve_qc(loader.load({"qc": {"TEMP": {"spike_test": {"method": "differential"}}}}))
    qc.sample_tests(fresh_subset, var="TEMP", config=cfg)
    assert captured["method"] == "differential"


def test_region_sentence_renders_in_qc_panel(subset_ds):
    # The region row reaches the HTML (guards against a template typo in _qc_glidertest.html).
    from glidertest.reports import _mission

    lon, lat = subset_ds["LONGITUDE"].values, subset_ds["LATITUDE"].values
    bbox = [float(np.nanmin(lon)) - 1, float(np.nanmin(lat)) - 1, float(np.nanmax(lon)) + 1, float(np.nanmax(lat)) + 1]
    html = _mission.PANELS["qc_glidertest"].render(_mission.Ctx(ds=subset_ds, bbox=bbox))
    assert "Region test" in html and "fall outside the region" in html
    # No region row when no box is configured.
    assert "Region test" not in _mission.PANELS["qc_glidertest"].render(_mission.Ctx(ds=subset_ds))


# --- provenance: default vs config marking --------------------------------------------------------


def test_thresholds_table_marks_config_vs_default(subset_ds):
    from glidertest.config import loader

    cfg = loader.resolve_qc(loader.load({"qc": {"TEMP": {"spike_test": {"suspect_threshold": 99.0}}}}))
    rows = qc_section.diagnostics_data(subset_ds, cfg)["diagnostics"]["thresholds"]
    spike = next(r for r in rows if r["var"] == "TEMP" and r["test"] == "spike")
    gross = next(r for r in rows if r["var"] == "TEMP" and r["test"] == "gross-range")
    assert spike["src"] == "config"  # the overridden test
    assert gross["src"] == "default"  # an untouched test on the same variable


def test_thresholds_table_all_default_without_config(subset_ds):
    rows = qc_section.diagnostics_data(subset_ds)["diagnostics"]["thresholds"]
    assert all(r["src"] == "default" for r in rows)


def test_region_excludes_unknown_position_from_denominator(fresh_subset):
    # Finding 3: samples with no position (NaN lon/lat → flag 9) must not pad the denominator; the
    # outside fraction is over positioned samples, and unknown positions are surfaced separately.
    ds = fresh_subset
    lon = np.asarray(ds["LONGITUDE"].values, dtype=float)
    lat = np.asarray(ds["LATITUDE"].values, dtype=float)
    half = lon.size // 2
    lon[:half] = np.nan
    lat[:half] = np.nan
    ds["LONGITUDE"] = ("N_MEASUREMENTS", lon)
    ds["LATITUDE"] = ("N_MEASUREMENTS", lat)
    pad = 1.0
    bbox = [float(np.nanmin(lon)) - pad, float(np.nanmin(lat)) - pad, float(np.nanmax(lon)) + pad, float(np.nanmax(lat)) + pad]
    region = qc_section.diagnostics_data(ds, None, bbox)["diagnostics"]["region"]
    assert region["n_unknown"] == half
    assert region["n_evaluated"] == lon.size - half
    assert region["n_outside"] == 0  # the box contains the whole positioned track
