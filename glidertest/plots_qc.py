"""QC diagnostic plots: where glidertest's QC tests fire, and whether a test fits the sampling.

These answer the two questions a flag count cannot: *where* the flagged samples fall (on the data,
so a reader can tell outliers from a water mass or a dark count) and *why* a test fired — whether
its thresholds suit the platform's sampling or the data are genuinely bad. They back the per-test
sections of the ``qc.html`` report page; each is also callable in a notebook.

Figures are born at the report slot width (``width_in``) under the active style, via
:func:`glidertest.plots._ax`, so the report shows them at the width they were drawn at with no
rescaling — see ``feat-qc-page-plan.md`` "Sizing". The flag arrays come from
:func:`glidertest.qc.sample_tests` (one computation, the same the delivered-vs-glidertest matrix
uses); pass them in via ``flags`` to avoid recomputing across the panels of one page.

Original Author: Eleanor Frajka-Williams (glidertest).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from . import plots, qc, utilities

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .config import report_tokens

# Flag-level colours — the matplotlib twin of base.html's --qc-good / --qc-suspect / --qc-fail, same
# hex as the sibling packages use for QARTOD 1/3/4 (a test asserts they match the CSS).
QC_COLORS: dict[int, str] = {1: "#27ae60", 3: "#f39c12", 4: "#e74c3c"}

#: Delivered-flag overlay markers (suspect/fail only), keyed by flag value — small "+" in the flag
#: colour, rasterized so the overlay is cheap even at 10^6 samples.
QC_MARKER: dict[int, dict] = {
    3: {"marker": "+", "color": QC_COLORS[3], "s": 15, "linewidths": 0.8, "zorder": 3, "rasterized": True},
    4: {"marker": "+", "color": QC_COLORS[4], "s": 15, "linewidths": 0.8, "zorder": 4, "rasterized": True},
}

# Per-*test* colours — Okabe–Ito (colour-blind safe), distinct from the flag-level QC_COLORS because
# here the question is which test fired, not at what level.
TEST_COLORS: dict[str, str] = {"gross_range": "#0072B2", "spike": "#E69F00", "flat_line": "#009E73"}

#: Per-test hit markers for the "where it fires" overlay. A multi-sample flat run is drawn as a
#: shaded span (an interval); a lone flat sample has no width, so it uses this square marker.
TEST_MARKER: dict[str, dict] = {
    "gross_range": {"marker": "x", "s": 20, "linewidths": 1.0, "zorder": 5, "rasterized": True},
    "spike": {"marker": "^", "s": 18, "linewidths": 0.0, "zorder": 5, "rasterized": True},
    "flat_line": {"marker": "s", "s": 10, "linewidths": 0.0, "zorder": 5, "rasterized": True},
}

#: Flat-line thresholds used when a variable has no ``flat_line_test`` entry in ``qc.configs`` (a
#: safety fallback; the four report variables carry one). Seconds, seconds, tolerance.
_FLAT_FALLBACK: dict[str, float] = {"suspect_threshold": 1.0, "fail_threshold": 3.0, "tolerance": 0.001}


def _flat_config(var: str) -> dict[str, float]:
    """Return the flat-line thresholds for *var* from ``qc.configs``, or the fallback."""
    return qc.configs.get(var, {}).get("flat_line_test", _FLAT_FALLBACK)


def _axis_label(ds: xr.Dataset, var: str) -> str:
    """Return ``"Label (units)"`` via :mod:`glidertest.utilities`, or just the label when no units."""
    unit = utilities.plotting_units(ds, var)
    return f"{utilities.plotting_labels(var)} ({unit})" if unit else utilities.plotting_labels(var)


def _time_xaxis(ax: Axes) -> None:
    """Give a datetime x-axis compact, non-overlapping labels (concise auto date formatter)."""
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _flags(ds: xr.Dataset, var: str, flags: tuple | None) -> tuple:
    """Return the ``(gross, spike, flat)`` flag tuple — *flags* if given, else computed once."""
    return flags if flags is not None else qc.sample_tests(ds, var)


def _runs(flags: np.ndarray, time_values: np.ndarray) -> list[tuple[int, int, float]]:
    """Return ``[(i_start, i_end, duration_s), ...]`` for each maximal run of flagged samples.

    Flagged means suspect (3) or fail (4) only — flag 9 (missing) and 2 (not evaluated) are not flat.
    Vectorised (run edges from one ``diff``), so it is cheap at glider record sizes.
    """
    bad = np.isin(np.asarray(flags), (3, 4))
    if not bad.any():
        return []
    s = utilities._elapsed_seconds(time_values)
    edges = np.diff(bad.astype(np.int8))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0]
    if bad[0]:
        starts = np.r_[0, starts]
    if bad[-1]:
        ends = np.r_[ends, bad.size - 1]
    return [(int(i0), int(i1), float(s[i1] - s[i0])) for i0, i1 in zip(starts, ends)]


def plot_test_hits(
    ds: xr.Dataset, var: str, *, flags: tuple | None = None,
    width_in: float = report_tokens.W_FULL, ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot *var* over the full record with glidertest's QC test hits overlaid — "where it fires".

    Gross-range hits as ``×``, spike hits as ``▲`` (one colour per test), and flat-line runs as
    shaded spans (a run is an interval). When the file carries ``<var>_QC``, the delivered suspect
    and fail samples are overlaid as ``+`` in the flag colours. Only hits are drawn, so the overlay
    is cheap even at 10^6 samples.

    Parameters
    ----------
    ds : xarray.Dataset
        An OG1 glider dataset.
    var : str
        The variable to plot (e.g. ``"TEMP"``).
    flags : tuple, optional
        The ``(gross, spike, flat)`` flag arrays from :func:`glidertest.qc.sample_tests`; computed on
        demand when ``None`` (so a notebook call needs nothing extra; the report passes them in once).
    width_in : float, default :data:`glidertest.config.report_tokens.W_FULL`
        Figure width in inches; the height is derived from it.
    ax : matplotlib.axes.Axes, optional
        Draw into this axis instead of creating a figure.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axis drawn into.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams (glidertest).
    """
    with plt.style.context(plots._style()):
        fig, ax = plots._ax(width_in, 0.45 * width_in, ax)
        t = np.asarray(ds.TIME.values)
        v = np.asarray(ds[var].values, dtype=float)
        # Points, not a line: on OG1 files a non-CTD sensor (CHLA, DOXY, BBP700) is sampled on a
        # different phase than the CTD, so its finite samples are isolated among NaNs and a line plot
        # draws no segments at all. Markers show every finite value; dense TEMP/PSAL still read as a
        # trace. Rasterized so the whole-record base layer stays one image, not 10^6 vector glyphs.
        ax.plot(t, v, marker=".", ls="none", ms=1.5, color="0.55", zorder=1, rasterized=True)

        _gr, spike, flat = _flags(ds, var, flags)
        # Gross range: the shared per-sample mask (qc.gross_range_mask) — the same one the diagnostics
        # matrix counts — so the plotted × and the table count cannot diverge. Spike/flat are QARTOD flags.
        gross_hit = qc.gross_range_mask(ds, var)
        # Flat-line hits: a multi-sample run is an interval (shaded span, the background layer); a lone
        # flagged sample (common when a sensor is sampled off-phase, so every finite value is isolated)
        # has zero width, so it joins the marker overlays below instead of being an invisible span.
        runs = _runs(flat, t)
        spans = [(i0, i1) for i0, i1, _d in runs if i1 > i0]
        for k, (i0, i1) in enumerate(spans):
            ax.axvspan(t[i0], t[i1], color=TEST_COLORS["flat_line"], alpha=0.25, lw=0, zorder=2,
                       label="flat line (run)" if k == 0 else None)
        lone = np.zeros(v.shape, dtype=bool)
        lone[[i0 for i0, i1, _d in runs if i1 == i0]] = True
        # Marker overlays, most-frequent first so a rarer test's symbols land on top and stay visible
        # (e.g. a few spikes over a flood of flat-line hits).
        overlays = [("gross range", gross_hit, "gross_range"),
                    ("spike", np.isin(np.asarray(spike), (3, 4)), "spike"),
                    ("flat line (sample)", lone, "flat_line")]
        for label, mask, key in sorted(overlays, key=lambda o: o[1].sum(), reverse=True):
            if mask.any():
                ax.scatter(t[mask], v[mask], label=label,
                           **{**TEST_MARKER[key], "color": TEST_COLORS[key]})

        qc_name = f"{var}_QC"
        if qc_name in ds:
            dq = np.asarray(ds[qc_name].values)
            for level in (3, 4):
                hit = dq == level
                if hit.any():
                    ax.scatter(t[hit], v[hit], label=f"delivered: {'suspect' if level == 3 else 'fail'}",
                               **QC_MARKER[level])

        ax.set_xlabel("Time")
        _time_xaxis(ax)
        ax.set_ylabel(_axis_label(ds, var))
        if ax.get_legend_handles_labels()[0]:  # no legend box when nothing was flagged
            ax.legend(loc="best", markerscale=1.5, framealpha=0.85)
        return fig, ax


def plot_flag_profile(
    ds: xr.Dataset, var: str, *, flags: tuple | None = None,
    width_in: float = report_tokens.W_HALF, ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot *var* against depth — a profile — with glidertest's QC hits overlaid, "where it fires".

    Every finite sample is a small grey point at its (value, depth); gross-range, spike and flat-line
    hits are overlaid in the per-test colours. The y-axis is depth (deepest at the bottom), so a reader
    sees at what depth each flag falls — surface, a water mass, or the deep — which is what matters for
    a depth-structured sensor (optics, oxygen). The base is grey, not value-coloured. Requires DEPTH.

    Parameters
    ----------
    ds : xarray.Dataset
        An OG1 glider dataset carrying DEPTH and *var*.
    var : str
        The variable whose QC hits to place against depth (e.g. ``"CHLA"``).
    flags : tuple, optional
        The ``(gross, spike, flat)`` flag arrays from :func:`glidertest.qc.sample_tests`; computed on
        demand when ``None``.
    width_in : float, default :data:`glidertest.config.report_tokens.W_HALF`
        Figure width in inches; the height is derived from it.
    ax : matplotlib.axes.Axes, optional
        Draw into this axis instead of creating a figure.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axis drawn into.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams (glidertest).
    """
    utilities._check_necessary_variables(ds, [var, "DEPTH"])
    with plt.style.context(plots._style()):
        fig, ax = plots._ax(width_in, 1.1 * width_in, ax)
        z = np.asarray(ds.DEPTH.values, dtype=float)
        v = np.asarray(ds[var].values, dtype=float)
        fin = np.isfinite(v) & np.isfinite(z)
        ax.scatter(v[fin], z[fin], s=2, color="0.7", linewidths=0, zorder=1, rasterized=True)

        _gr, spike, flat = _flags(ds, var, flags)
        # Shared gross-range mask (same one the diagnostics matrix counts), so the plotted × and the
        # table count cannot diverge.
        gross_hit = qc.gross_range_mask(ds, var)
        hits = [
            ("gross_range", gross_hit & fin),
            ("spike", np.isin(np.asarray(spike), (3, 4)) & fin),
            ("flat_line", np.isin(np.asarray(flat), (3, 4)) & fin),
        ]
        # Draw the most-frequent test first so a rarer test's symbols land on top and stay visible
        # (e.g. a handful of spikes over a flood of flat-line hits).
        for name, m in sorted(hits, key=lambda nm: nm[1].sum(), reverse=True):
            if m.any():
                ax.scatter(v[m], z[m], label=name.replace("_", " "),
                           **{**TEST_MARKER[name], "color": TEST_COLORS[name]})

        ax.set_xlabel(_axis_label(ds, var))
        ax.set_ylabel(_axis_label(ds, "DEPTH"))
        ax.invert_yaxis()
        if ax.get_legend_handles_labels()[0]:  # no legend box when nothing was flagged
            ax.legend(loc="best", markerscale=1.5, framealpha=0.85)
        return fig, ax


def plot_ts_gross_range(
    ds: xr.Dataset, *, width_in: float = report_tokens.W_HALF, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    """Plot the T–S cloud with the TEMP×PSAL gross-range suspect and fail spans as nested rectangles.

    One glance says whether the limits are regional (the whole cloud hugs an edge) or the points
    outside are genuine outliers. Requires TEMP and PSAL.

    Parameters
    ----------
    ds : xarray.Dataset
        An OG1 glider dataset carrying TEMP and PSAL.
    width_in : float, default :data:`glidertest.config.report_tokens.W_HALF`
        Figure width in inches.
    ax : matplotlib.axes.Axes, optional
        Draw into this axis instead of creating a figure.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axis drawn into.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams (glidertest).
    """
    with plt.style.context(plots._style()):
        fig, ax = plots._ax(width_in, 0.9 * width_in, ax)
        s = np.asarray(ds["PSAL"].values, dtype=float)
        t = np.asarray(ds["TEMP"].values, dtype=float)
        ax.scatter(s, t, s=2, color="0.6", rasterized=True, zorder=1)
        for span_key, flag in (("fail_span", 4), ("suspect_span", 3)):
            sx = qc.configs["PSAL"]["gross_range_test"][span_key]
            ty = qc.configs["TEMP"]["gross_range_test"][span_key]
            ax.add_patch(Rectangle((sx[0], ty[0]), sx[1] - sx[0], ty[1] - ty[0],
                                   fill=False, edgecolor=QC_COLORS[flag], lw=1.3, zorder=3,
                                   label=f"{'fail' if flag == 4 else 'suspect'} span"))
        ax.set_xlabel(_axis_label(ds, "PSAL"))
        ax.set_ylabel(_axis_label(ds, "TEMP"))
        ax.legend(loc="best", framealpha=0.85)
        return fig, ax


def plot_spike_stat(
    ds: xr.Dataset, var: str, *, width_in: float = report_tokens.W_HALF, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    """Plot the spike statistic (|value − neighbour average|) as a histogram with the thresholds.

    The suspect and fail thresholds are drawn as vertical lines; if the bulk of the distribution sits
    past a line the threshold is wrong for this platform, if only a tail does the test is doing its
    job. The median sampling interval is printed, since a spike threshold assumes a sampling interval.

    Parameters
    ----------
    ds : xarray.Dataset
        An OG1 glider dataset.
    var : str
        The variable to assess.
    width_in : float, default :data:`glidertest.config.report_tokens.W_HALF`
        Figure width in inches.
    ax : matplotlib.axes.Axes, optional
        Draw into this axis instead of creating a figure.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axis drawn into.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams (glidertest).
    """
    with plt.style.context(plots._style()):
        fig, ax = plots._ax(width_in, 0.6 * width_in, ax)
        v = np.asarray(ds[var].values, dtype=float)
        stat = np.abs(v[1:-1] - 0.5 * (v[:-2] + v[2:]))
        stat = stat[np.isfinite(stat)]
        if stat.size:
            ax.hist(stat, bins=60, color="0.6")
        cfg = qc.configs[var]["spike_test"]
        for key, flag in (("suspect_threshold", 3), ("fail_threshold", 4)):
            ax.axvline(cfg[key], color=QC_COLORS[flag], lw=1.3,
                       label=f"{'suspect' if flag == 3 else 'fail'} ({cfg[key]:g})")
        ax.set_yscale("log")
        unit = utilities.plotting_units(ds, var)
        ax.set_xlabel(f"spike statistic |Δ{var}| ({unit})" if unit else f"spike statistic |Δ{var}|")
        ax.set_ylabel("count")
        # The variable's own sampling interval (finite samples), falling back to the whole record if
        # the variable has none; a spike threshold is read against the sensor's real cadence.
        finite = np.isfinite(v)
        dt = utilities._median_sampling_interval(ds.TIME.values[finite])
        if not np.isfinite(dt):
            dt = utilities._median_sampling_interval(ds.TIME.values)
        ax.set_title(f"median sampling interval {dt:.1f} s", loc="right")
        ax.legend(loc="best", framealpha=0.85)
        return fig, ax


def plot_flat_runs(
    ds: xr.Dataset, var: str, *, flags: tuple | None = None,
    width_in: float = report_tokens.W_HALF, ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot flat-run durations as a histogram with the test windows and the median sampling interval.

    This is the plot that shows the flat-line test is misconfigured: if the suspect and fail windows
    sit *left* of the median sampling interval, every sample is a "flat run" by construction and the
    counts are meaningless.

    Parameters
    ----------
    ds : xarray.Dataset
        An OG1 glider dataset.
    var : str
        The variable to assess.
    flags : tuple, optional
        The ``(gross, spike, flat)`` flag arrays; computed on demand when ``None``.
    width_in : float, default :data:`glidertest.config.report_tokens.W_HALF`
        Figure width in inches.
    ax : matplotlib.axes.Axes, optional
        Draw into this axis instead of creating a figure.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axis drawn into.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams (glidertest).
    """
    with plt.style.context(plots._style()):
        fig, ax = plots._ax(width_in, 0.6 * width_in, ax)
        _gr, _spike, flat = _flags(ds, var, flags)
        # The variable's own sampling interval (finite samples), not the whole record's: an off-phase
        # sensor's windows must be judged against its real cadence (CHLA finite only every ~Nth cycle).
        finite = np.isfinite(np.asarray(ds[var].values, dtype=float))
        dt = utilities._median_sampling_interval(ds.TIME.values[finite])
        if not np.isfinite(dt) or dt <= 0:
            # Degenerate interval (duplicate or missing timestamps): fall back to 1 s so the floored
            # durations and the log-axis bounds stay positive (log10(0) would break the axis).
            dt = 1.0
        # Durations span seconds to hours and a single-sample run is 0 s; floor at one sampling interval and
        # use a log axis so the bulk and the 1–3 s window lines are both legible.
        durations = np.array([max(dur, dt) for _i0, _i1, dur in _runs(flat, ds.TIME.values)])
        fl = _flat_config(var)
        # Range spans every mark on the axis — the two config windows, the sampling interval and the
        # run durations — so a window line never falls off the edge when the interval is coarse.
        marks = [dt, float(fl["suspect_threshold"]), float(fl["fail_threshold"])]
        if durations.size:
            marks += [float(durations.min()), float(durations.max())]
        lo, hi = max(min(marks) * 0.8, 1e-6), max(marks) * 1.2
        ax.set_xscale("log")
        if durations.size:
            ax.hist(durations, bins=np.logspace(np.log10(lo), np.log10(hi), 40), color="0.6")
        ax.set_xlim(lo, hi)
        ax.axvline(fl["suspect_threshold"], color=QC_COLORS[3], lw=1.3,
                   label=f"suspect window ({fl['suspect_threshold']:g} s)")
        ax.axvline(fl["fail_threshold"], color=QC_COLORS[4], lw=1.3,
                   label=f"fail window ({fl['fail_threshold']:g} s)")
        ax.axvline(dt, color="k", lw=1.3, ls="--", label=f"median sampling interval ({dt:.1f} s)")
        ax.set_xlabel("flat-run duration (s)")
        ax.set_ylabel("count")
        ax.legend(loc="best", framealpha=0.85)
        return fig, ax


def plot_flat_change(
    ds: xr.Dataset, var: str, *, flags: tuple | None = None,
    width_in: float = report_tokens.W_HALF, ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram the start-to-end value change of each multi-sample flat run, against the tolerance.

    The flat-line test is meant to catch a *stuck* sensor — a run whose values are near-identical, so
    its start-to-end change falls within the test tolerance. A run whose total change far exceeds the
    tolerance is a slowly-varying real signal the test caught only because its short *window* (not the
    whole run) stayed within tolerance — a sign the window is too short for the sampling, not a stuck
    sensor. The count split at the tolerance line is the guess at how many flagged runs are genuinely
    flat: left of the line, stuck-like; right of it, the data moved.

    Parameters
    ----------
    ds : xarray.Dataset
        An OG1 glider dataset.
    var : str
        The variable to assess.
    flags : tuple, optional
        The ``(gross, spike, flat)`` flag arrays; computed on demand when ``None``.
    width_in : float, default :data:`glidertest.config.report_tokens.W_HALF`
        Figure width in inches.
    ax : matplotlib.axes.Axes, optional
        Draw into this axis instead of creating a figure.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axis drawn into.

    Raises
    ------
    ValueError
        When *var* has no multi-sample flat run (e.g. an off-phase sensor whose flat flags are all
        isolated single samples), so there is no start-to-end change to measure.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams (glidertest).
    """
    with plt.style.context(plots._style()):
        fig, ax = plots._ax(width_in, 0.6 * width_in, ax)
        _gr, _spike, flat = _flags(ds, var, flags)
        v = np.asarray(ds[var].values, dtype=float)
        runs = [(i0, i1) for i0, i1, _d in _runs(flat, ds.TIME.values) if i1 > i0]
        changes = np.array([abs(v[i1] - v[i0]) for i0, i1 in runs])
        changes = changes[np.isfinite(changes)]
        if not changes.size:
            raise ValueError(f"{var}: no multi-sample flat run (flat flags are all isolated samples)")
        tol = float(_flat_config(var)["tolerance"])
        n_moved = int((changes > tol).sum())
        pct = 100.0 * n_moved / changes.size
        ax.hist(changes, bins=40, color="0.6", label=f"flat runs (n={changes.size})")
        ax.axvline(tol, color=QC_COLORS[3], lw=1.3, label=f"tolerance ({tol:g})")
        unit = utilities.plotting_units(ds, var)
        ax.set_xlabel(f"flat-run start-to-end change ({unit})" if unit else "flat-run start-to-end change")
        ax.set_ylabel("count")
        # The fraction of flagged runs whose data moved more than the tolerance — the "stuck sensor vs
        # slow signal" guess: a high value means the flat test is firing on data that is not flat.
        ax.set_title(f"{pct:.0f}% above tolerance")
        ax.legend(loc="best", framealpha=0.85)
        return fig, ax


def plot_flat_excerpt(
    ds: xr.Dataset, var: str, *, flags: tuple | None = None,
    width_in: float = report_tokens.W_HALF, ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a zoom around the longest flagged flat run: samples as dots and a ±tolerance band.

    Shows what the test called "flat" — often two samples a tolerance apart in genuinely stable
    water or low signal, not a stuck sensor.

    Parameters
    ----------
    ds : xarray.Dataset
        An OG1 glider dataset.
    var : str
        The variable to assess.
    flags : tuple, optional
        The ``(gross, spike, flat)`` flag arrays; computed on demand when ``None``.
    width_in : float, default :data:`glidertest.config.report_tokens.W_HALF`
        Figure width in inches.
    ax : matplotlib.axes.Axes, optional
        Draw into this axis instead of creating a figure.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axis drawn into.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams (glidertest).
    """
    with plt.style.context(plots._style()):
        fig, ax = plots._ax(width_in, 0.45 * width_in, ax)
        _gr, _spike, flat = _flags(ds, var, flags)
        runs = _runs(flat, ds.TIME.values)
        t = np.asarray(ds.TIME.values)
        v = np.asarray(ds[var].values, dtype=float)
        if runs:
            i0, i1, _dur = max(runs, key=lambda r: r[2])
            pad = max(5, (i1 - i0))
            lo, hi = max(0, i0 - pad), min(v.size, i1 + pad + 1)
            ax.plot(t[lo:hi], v[lo:hi], color="0.5", lw=0.6, marker=".", ms=3)
            ax.scatter(t[i0:i1 + 1], v[i0:i1 + 1], color=QC_COLORS[4], s=10, zorder=3, label="flagged run")
            tol = _flat_config(var)["tolerance"]
            ax.axhspan(v[i0] - tol, v[i0] + tol, color=QC_COLORS[3], alpha=0.2, lw=0,
                       label=f"±tolerance ({tol:g})")
            ax.legend(loc="best", framealpha=0.85)
        ax.set_xlabel("Time")
        _time_xaxis(ax)
        ax.set_ylabel(_axis_label(ds, var))
        return fig, ax
