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

    def boom(ds, var):
        raise ValueError("kaboom")

    monkeypatch.setattr(_mission.qc_section, "suggested_limits_note", boom)
    html = _mission.PANELS["qc_TEMP_suggest"].render(_mission.Ctx(ds=subset_ds))
    assert "none-note" in html and "kaboom" in html
