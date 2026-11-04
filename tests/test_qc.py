import numpy as np
import xarray as xr

from glidertest import qc

# Pure-logic tests for sample_tests and its piece-splitting helpers, on synthetic OG1-shaped datasets
# (one TIME coordinate on N_MEASUREMENTS, a variable finite on a subset of rows). The interleaved and
# sub-second cases mirror the seagliderOG1 shape where every sensor shares one merged TIME axis; the
# real sg271 file is used for local end-to-end verification but is not committed to the suite.

_BASE = np.datetime64("2024-01-01T00:00:00")


def _ds(values, times, var="TEMP"):
    """Build a minimal OG1-shaped dataset: *var* on N_MEASUREMENTS with a TIME coordinate."""
    return xr.Dataset(
        {var: ("N_MEASUREMENTS", np.asarray(values, dtype=float))},
        coords={"TIME": ("N_MEASUREMENTS", np.asarray(times))},
    )


def _times(step_seconds, n, gaps_at=()):
    """Cumulative datetime64 axis with *step_seconds* spacing and a 600 s jump before each gaps_at row."""
    steps = np.full(n - 1, float(step_seconds))
    for g in gaps_at:
        steps[g] = 600.0
    secs = np.concatenate([[0.0], np.cumsum(steps)])
    return _BASE + (secs * 1000).astype("timedelta64[ms]")


def _interleaved(n=400, finite_every=4, union_step=0.33, gaps_at=(100, 200)):
    """A variable finite on every finite_every-th row of a sub-second merged axis, with dive gaps."""
    times = _times(union_step, n, gaps_at=gaps_at)
    values = np.full(n, np.nan)
    idx = np.arange(0, n, finite_every)
    values[idx] = 12.0 + 0.5 * np.sin(np.arange(idx.size))
    return _ds(values, times)


def test_interleaved_runs_per_variable_and_preserves_length():
    ds = _interleaved()
    n = ds.sizes["N_MEASUREMENTS"]
    finite = np.isfinite(ds["TEMP"].values)
    _gr, spike, flat = qc.sample_tests(ds, "TEMP")

    assert spike.size == flat.size == n
    # Non-sampled rows carry MISSING (9): the datum is absent, not merely un-QC'd.
    assert (spike[~finite] == 9).all()
    assert (flat[~finite] == 9).all()
    # Every finite row is evaluated (not left MISSING): the per-variable cadence is >= 1 s here.
    assert (spike[finite] != 9).all()
    assert (flat[finite] != 9).all()


def test_interleaved_moves_spike_off_not_evaluated():
    # The headline fix: on the merged axis a finite value's neighbours are NaN rows from other
    # sensors, so spike flags nearly everything UNKNOWN (2). Running on the variable's own rows moves
    # those to real verdicts, leaving only the per-dive endpoints at 2.
    ds = _interleaved()
    finite = np.isfinite(ds["TEMP"].values)
    _gr, spike, _flat = qc.sample_tests(ds, "TEMP")

    naive = np.asarray(
        qc.qartod.spike_test(ds["TEMP"].values, suspect_threshold=2.0, fail_threshold=6.0, method="average")
    )
    assert (naive[finite] == 2).mean() > 0.9  # merged axis: almost all not-evaluated
    assert (spike[finite] == 2).mean() < 0.2  # per-variable: only piece endpoints remain


def test_subsecond_single_sensor_does_not_crash_and_skips_flat():
    # A genuine > 1 Hz single sensor: every row finite, 0.3 s cadence. flat_line_test would divide by
    # zero (the step truncates to 0 s), so flat is UNKNOWN throughout; spike still runs.
    n = 100
    values = 12.0 + 0.5 * np.sin(np.arange(n))
    ds = _ds(values, _times(0.3, n))
    _gr, spike, flat = qc.sample_tests(ds, "TEMP")

    assert (flat == 2).all()
    assert (spike != 9).all()
    assert set(np.unique(spike)).issubset({1, 2, 3, 4})


def test_flat_line_or_unknown_guards():
    # Explicit short windows so the guards are exercised independently of the (longer) built-in default.
    fl = {"suspect_threshold": 2, "fail_threshold": 4, "tolerance": 0.0005}
    # Fewer than three points: data present but untestable -> UNKNOWN (2), not GOOD.
    short = qc._flat_line_or_unknown(np.array([12.0, 12.1]), _times(2.0, 2), fl)
    assert (short == 2).all()
    # Sub-second cadence (step truncates to 0 s): UNKNOWN, matching ioos_qc's own interval truncation.
    sub = qc._flat_line_or_unknown(12.0 + np.zeros(10), _times(0.3, 10), fl)
    assert (sub == 2).all()
    # Suspect window shorter than two samples (2 s window at 3 s spacing -> count 0): UNKNOWN, not a
    # record-wide flat verdict taken over single points.
    short_window = qc._flat_line_or_unknown(12.0 + np.zeros(10), _times(3.0, 10), fl)
    assert (short_window == 2).all()
    # A window that holds >= 2 samples runs the real test (near-constant values flag as flat).
    flat = qc._flat_line_or_unknown(12.0 + np.zeros(10), _times(2.0, 10), fl)
    assert set(np.unique(flat)).issubset({1, 3, 4})


def test_uniform_single_sensor_matches_direct_qartod():
    # With all rows finite and no gaps, sample_tests is a single piece, so it must reproduce a direct
    # qartod call on the full array (the pre-change behaviour) exactly.
    n = 200
    values = 12.0 + 0.5 * np.sin(np.arange(n))
    times = _times(2.0, n)
    ds = _ds(values, times)
    _gr, spike, flat = qc.sample_tests(ds, "TEMP")

    direct_spike = np.asarray(
        qc.qartod.spike_test(values, suspect_threshold=2.0, fail_threshold=6.0, method="average")
    )
    fl = qc.configs["TEMP"]["flat_line_test"]
    direct_flat = np.asarray(
        qc.qartod.flat_line_test(values, times, fl["suspect_threshold"], fl["fail_threshold"], fl["tolerance"])
    )
    assert np.array_equal(spike, direct_spike)
    assert np.array_equal(flat, direct_flat)


def test_flag_counts_sum_and_missing_bucket():
    ds = _interleaved()
    finite = np.isfinite(ds["TEMP"].values)
    _gr, spike, _flat = qc.sample_tests(ds, "TEMP")
    counts = qc.flag_counts(spike)
    assert sum(counts.values()) == spike.size
    assert counts["missing"] == int((~finite).sum()) == int((spike == 9).sum())


# --- gross-range grading (fail span applied) ------------------------------------------------------


def test_gross_range_flags_grade_good_suspect_fail_missing():
    # Default TEMP spans: suspect [0, 30], fail [-2.5, 40]. A value inside suspect is good (1), between
    # suspect and fail is suspect (3), outside fail is fail (4), NaN is missing (9).
    vals = np.array([10.0, -1.0, 50.0, np.nan, 20.0])
    flags = qc.gross_range_flags(_ds(vals, _times(1.0, 5)), "TEMP")
    assert flags.tolist() == [1, 3, 4, 9, 1]


def test_gross_range_flags_no_fail_span_is_all_good():
    # Config validation requires both spans; with only a suspect span there is nothing to grade
    # against, so finite samples are good and NaN is missing (never a spurious flag).
    cfg = {"TEMP": {"gross_range_test": {"suspect_span": [0, 30]}}}
    flags = qc.gross_range_flags(_ds(np.array([10.0, 999.0, np.nan]), _times(1.0, 3)), "TEMP", config=cfg)
    assert flags.tolist() == [1, 1, 9]


def test_gross_range_mask_wraps_flags_suspect_or_fail():
    ds = _ds(np.array([10.0, -1.0, 50.0, np.nan]), _times(1.0, 4))
    flags = qc.gross_range_flags(ds, "TEMP")
    mask = qc.gross_range_mask(ds, "TEMP")
    assert mask.tolist() == np.isin(flags, (3, 4)).tolist()
    assert mask.tolist() == [False, True, True, False]  # good, suspect, fail, missing


def test_sample_tests_gross_is_graded_flag_array():
    # gr is now a full-length QARTOD flag array (not a filtered DataArray), consistent with spike/flat.
    ds = _interleaved()
    n = ds.sizes["N_MEASUREMENTS"]
    gr, _spike, _flat = qc.sample_tests(ds, "TEMP")
    assert gr.size == n
    assert set(np.unique(gr)).issubset({1, 3, 4, 9})
    assert sum(qc.flag_counts(gr).values()) == n


def test_gross_cell_shows_suspect_and_fail():
    from glidertest.reports import qc_section

    # good / suspect / fail / missing, repeated, so the matrix cell must show both susp and fail spans.
    vals = np.array([10.0, -1.0, 50.0, np.nan] * 5)
    cell = qc_section._gross_cell(_ds(vals, _times(1.0, 20)), "TEMP")
    texts = " ".join(s["text"] for s in cell.get("spans", []))
    assert "susp" in texts and "fail" in texts


def test_fill_str_marks_gross_only_on_suspect_or_fail(monkeypatch):
    # Regression for the flag-array switch: gr is now length N_MEASUREMENTS, so `len(gr) > 0` would mark
    # every variable failed. The gross row must be 'X' only when a sample is suspect/fail, not merely
    # because the array is non-empty.
    ds = _ds(np.ones(10), _times(1.0, 10))
    good = np.ones(10, dtype="uint8")
    for gr, expected in ((good, "✓"), (np.where(np.arange(10) == 3, 4, 1).astype("uint8"), "X")):
        monkeypatch.setattr(qc, "qc_checks", lambda ds, var, config=None, _gr=gr: (_gr, good, good, np.zeros(1), np.zeros(1)))
        rows = [["Gross", "✓", "✓", "✓", "✓"] for _ in range(5)]
        out = qc.fill_str(*rows, ds, var="TEMP")
        assert out[0][1] == expected
