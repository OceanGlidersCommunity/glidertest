"""Quality-control checks and thresholds, independent of any output format.

Holds the QARTOD threshold ``configs``, the on-the-fly check runner ``qc_checks``, the verdict
helpers (``fill_str``, ``hysteresis_verdict``), the profile sentences (``phrase_*_check``), and the
OceanSITES/QARTOD flag vocabulary (``QC_FLAG_CATEGORIES`` / ``flag_counts``). Kept free of the
RST/PDF stack (``rstcloth``/``pypandoc``) so the HTML report can use it without that toolchain;
``summary_sheet`` imports these names back for the RST summary sheet.

Original Author: Chiara Monforte.
"""

import numpy as np
from ioos_qc import qartod

from glidertest import tools, utilities

#: Built-in QARTOD thresholds per variable (the layer-1 defaults). A report config's ``qc:`` block is
#: deep-merged over this by :func:`glidertest.config.loader.resolve_qc`; the resolved mapping, not this
#: dict, is what the checks and plots read when a config is in play. The location test is driven by a
#: configured ``region.bbox`` (no built-in default), so no ``location_test`` entry lives here.
configs = {
    "TEMP": {"gross_range_test": {
        "suspect_span": [0, 30],
        "fail_span": [-2.5, 40],
    },
        "spike_test": {"suspect_threshold": 2.0, "fail_threshold": 6.0},
        "flat_line_test": {"suspect_threshold": 1, "fail_threshold": 3, "tolerance": 0.001},
    },
    "PSAL": {"gross_range_test": {"suspect_span": [5, 38], "fail_span": [2, 41]},
             "spike_test": {"suspect_threshold": 0.3, "fail_threshold": 0.9},
             "flat_line_test": {"suspect_threshold": 1, "fail_threshold": 3, "tolerance": 0.001},
             },
    "DOXY": {"gross_range_test": {
        "suspect_span": [0, 350],
        "fail_span": [0, 500],
    },
        "spike_test": {"suspect_threshold": 10, "fail_threshold": 50},
        "flat_line_test": {"suspect_threshold": 1, "fail_threshold": 3, "tolerance": 0.001},
    },
    "CHLA": {"gross_range_test": {
        "suspect_span": [0, 15],
        "fail_span": [-1, 20],
    },
        "spike_test": {"suspect_threshold": 1, "fail_threshold": 5},
        "flat_line_test": {"suspect_threshold": 1, "fail_threshold": 3, "tolerance": 0.001},
    },
}

#: OceanSITES/QARTOD flag value -> (category key, label), in display order (good, suspect, fail,
#: then the non-graded categories). Covers every flag value so counts sum to the array size.
QC_FLAG_CATEGORIES = (
    (1, "good", "Good"),
    (3, "suspect", "Suspect"),
    (4, "fail", "Fail"),
    (2, "not_eval", "Not evaluated"),
    (9, "missing", "Missing"),
)


def flag_counts(flags):
    """Return a ``{category_key: count}`` dict for a QARTOD/OceanSITES flag array.

    Counts the five graded/non-graded categories (good=1, suspect=3, fail=4, not-evaluated=2,
    missing=9) and adds an ``"other"`` bucket holding every remaining value — for example OG1's
    ``0`` ("no QC applied") or any non-standard flag — so the returned counts always sum to the
    array size and no flagged sample is silently dropped.

    Parameters
    ----------
    flags : array-like
        A QARTOD/OceanSITES integer flag array.

    Returns
    -------
    dict of str to int
        Count per category key in :data:`QC_FLAG_CATEGORIES` order, plus ``"other"``; the values
        sum to ``size``.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams.
    """
    arr = np.asarray(flags)
    counts = {key: int((arr == value).sum()) for value, key, _ in QC_FLAG_CATEGORIES}
    counts["other"] = int(arr.size) - sum(counts.values())
    return counts


def flag_scale_mismatch(da):
    """Return ``(n_values, n_meanings)`` when a QC variable's declared flag scale is inconsistent.

    An OG1 ``*_QC`` variable should declare ``flag_values`` and ``flag_meanings`` of equal length.
    When they differ the file's flag scale cannot be trusted — a finding a diagnose tool should
    surface rather than silently use a partial scale.

    Parameters
    ----------
    da : xarray.DataArray
        A QC flag variable. Only its ``attrs`` are read.

    Returns
    -------
    tuple of (int, int) or None
        ``(len(flag_values), len(flag_meanings))`` when the two differ, else ``None`` (consistent,
        or one/both attributes absent).

    Notes
    -----
    Original Author: Eleanor Frajka-Williams.
    """
    values = da.attrs.get("flag_values")
    meanings = da.attrs.get("flag_meanings")
    if values is None or meanings is None:
        return None
    n_values = int(np.asarray(values).size)
    n_meanings = len(meanings.split() if isinstance(meanings, str) else list(meanings))
    return (n_values, n_meanings) if n_values != n_meanings else None


def flag_labels(da):
    """Return a ``{flag_value: label}`` map read from a QC variable's ``flag_values``/``flag_meanings``.

    OG1 ``*_QC`` variables carry their own flag scale in attributes (for example
    ``flag_values = [1, 2, 3, 4, 9]`` with ``flag_meanings = "GOOD UNKNOWN SUSPECT FAIL MISSING"``),
    so the label of each flag is read from the file rather than assumed. Flag values the file does
    not label fall back to the default labels in :data:`QC_FLAG_CATEGORIES`. Only the label is
    file-specific; the colour category of a flag stays fixed by its numeric value (1 good, 3
    suspect, 4 fail).

    Parameters
    ----------
    da : xarray.DataArray
        A QC flag variable. Only its ``attrs`` are read, not its values.

    Returns
    -------
    dict of int to str
        Label per flag value, covering at least the values in :data:`QC_FLAG_CATEGORIES`.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams.
    """
    labels = {value: label for value, _key, label in QC_FLAG_CATEGORIES}
    values = da.attrs.get("flag_values")
    meanings = da.attrs.get("flag_meanings")
    if values is not None and meanings is not None:
        names = meanings.split() if isinstance(meanings, str) else [str(m) for m in meanings]
        vals = np.asarray(values).ravel().tolist()
        if len(vals) == len(names):  # on a length mismatch keep defaults; flag_scale_mismatch reports it
            for value, name in zip(vals, names):
                labels[int(value)] = name.replace("_", " ").capitalize()
    return labels


#: Hysteresis verdict thresholds, named so the report table and the verdict rule cannot drift (the
#: table reads these constants, not ``hysteresis_verdict.__defaults__`` by position): a depth bin
#: counts as exceeded above ``HYSTERESIS_PCT_THRESHOLD`` percent dive–climb error, and a variable is
#: flagged when more than ``HYSTERESIS_BIN_THRESHOLD`` bins exceed it.
HYSTERESIS_PCT_THRESHOLD = 5
HYSTERESIS_BIN_THRESHOLD = 5


def hysteresis_verdict(err, pct_threshold=HYSTERESIS_PCT_THRESHOLD, bin_threshold=HYSTERESIS_BIN_THRESHOLD):
    """Return ``(n_over, flagged)`` for a hysteresis dive-climb error array.

    The single verdict rule shared by the RST summary (:func:`fill_str`) and the HTML report, so the
    two never disagree. ``err`` is indexed by depth bin, not profile.

    Parameters
    ----------
    err : array-like
        Per-depth-bin percentage dive-climb error (``err_mean`` or ``err_range`` from
        :func:`glidertest.tools.compute_hyst_stat`).
    pct_threshold : float, optional
        Percentage error above which a depth bin counts as exceeded. Default 5.
    bin_threshold : int, optional
        A variable is flagged when more than this many depth bins exceed *pct_threshold*. Default 5.

    Returns
    -------
    tuple of (int, bool)
        ``(n_over, flagged)`` — bins exceeding the threshold, and the flag verdict.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams.
    """
    arr = np.asarray(err)
    n_over = int((arr > pct_threshold).sum())
    return n_over, n_over > bin_threshold


def qc_checks(ds, var='TEMP', *, config=None):
    """
    Run a series of basic quality control (global range, spike test, drift and hysteresis) checks on a selected variable in the dataset.
    We use functions from glidertest.tools as well as ioos qartod.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in **OG1 format**, containing the selected variable and time information.
    var : str, optional, default='TEMP'
        Variable to perform QC checks on.
    config : Mapping, optional
        Resolved per-variable thresholds (see :func:`sample_tests`); ``None`` uses the built-in
        :data:`configs`.

    Returns
    -------
    gr : xarray.DataArray
        Result of the **gross range test**, identifying values outside the expected physical range.
    spike : xarray.DataArray
        Result of the **spike test**, detecting sudden changes in data based on threshold configuration.
    flat : xarray.DataArray
        Result of the **flat line test**, identifying segments with little or no variation.
    err_mean : array-like
        Percentage error of the dive-climb difference based on the mean values, from hysteresis analysis.
    err_range : array-like
        Percentage error of the dive-climb difference based on the value range, from hysteresis analysis.

    Notes
    ------
    - Thresholds for QC tests are taken from *config* when given, else the global `configs` dictionary.
    Original Author: Chiara  Monforte
    """
    gr, spike, flat = sample_tests(ds, var, config=config)
    __, __, err_mean, err_range, __ = tools.compute_hyst_stat(ds, variable=var, v_res=1)

    return gr, spike, flat, err_mean, err_range


def sample_tests(ds, var='TEMP', *, config=None):
    """Run the three sample-wise QARTOD tests (gross range, spike, flat line) on *var*.

    Split out of :func:`qc_checks` (which adds the hysteresis statistics) so the report's QC page
    plots and the delivered-vs-glidertest matrix share one computation and one set of thresholds.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in **OG1 format**, containing *var* and ``TIME``.
    var : str, optional, default='TEMP'
        Variable to test.
    config : Mapping, optional
        Resolved per-variable thresholds (as from :func:`glidertest.config.loader.resolve_qc`); when
        ``None`` the built-in :data:`configs` are used, so a bare call is unchanged.

    Returns
    -------
    gr : numpy.ndarray
        Gross-range test flags.
    spike : numpy.ndarray
        Spike test flags.
    flat : numpy.ndarray
        Flat-line test flags.

    Notes
    -----
    Original Author: Chiara Monforte
    """
    utilities._check_necessary_variables(ds, [var, 'TIME'])
    cfg = config if config is not None else configs
    gr_cfg = cfg[var]['gross_range_test']
    gr = tools.compute_global_range(ds, variable=var, min_val=gr_cfg['suspect_span'][0],
                                    max_val=gr_cfg['suspect_span'][1])
    sp_cfg = cfg[var]['spike_test']
    spike = qartod.spike_test(ds[var], suspect_threshold=sp_cfg['suspect_threshold'],
                              fail_threshold=sp_cfg['fail_threshold'], method=sp_cfg.get('method', "average"))
    fl = cfg[var]['flat_line_test']
    flat = qartod.flat_line_test(ds[var], ds.TIME, fl['suspect_threshold'],
                                 fl['fail_threshold'], fl['tolerance'])
    return gr, spike, flat


def gross_range_mask(ds, var='TEMP', *, config=None):
    """Return a boolean per-sample mask of *var* values outside the gross-range suspect span.

    The one definition of "out of range" shared by the diagnostics-matrix count and the QC plots, so
    the table and the plotted hits cannot diverge. NaN samples are not flagged (the comparison is
    False for NaN). Returns an all-False mask when *var* has no ``gross_range_test`` suspect span in
    the thresholds.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in **OG1 format**, containing *var*.
    var : str, optional, default='TEMP'
        Variable to test.
    config : Mapping, optional
        Resolved per-variable thresholds (see :func:`sample_tests`); ``None`` uses the built-in
        :data:`configs`.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True where the value is below the span minimum or above its maximum.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams
    """
    cfg = config if config is not None else configs
    span = cfg.get(var, {}).get('gross_range_test', {}).get('suspect_span')
    v = np.asarray(ds[var].values, dtype=float)
    if not span:
        return np.zeros(v.shape, dtype=bool)
    return (v < span[0]) | (v > span[1])


def region_test(ds, bbox):
    """Flag each sample by whether its position lies inside the configured region box.

    glidertest's own region check: ``ioos_qc`` 3.0.0's ``location_test`` has no bounding box (it
    checks only the global sane bounds lon [-180, 180], lat [-90, 90]), so the region test lives
    here. A glider outside the region you declared is a **fail**, not a suspect — the point of the
    test is to catch a mission that strayed from, or was mislabelled against, its expected area.
    Points on the boundary count as inside. Run only when a report config supplies ``region.bbox``;
    there is no built-in box (a box drawn from the data cannot fail its own test).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in **OG1 format**; reads ``LONGITUDE`` and ``LATITUDE``.
    bbox : sequence of float
        ``[lon_min, lat_min, lon_max, lat_max]`` = ``[west, south, east, north]`` (GeoJSON / ioos_qc
        2.x order).

    Returns
    -------
    numpy.ndarray
        QARTOD flags per sample: ``1`` inside or on the box, ``4`` outside, ``9`` where ``LONGITUDE``
        or ``LATITUDE`` is NaN (position unknown).

    Notes
    -----
    Original Author: Eleanor Frajka-Williams
    """
    utilities._check_necessary_variables(ds, ["LONGITUDE", "LATITUDE"])
    lon = np.asarray(ds["LONGITUDE"].values, dtype=float)
    lat = np.asarray(ds["LATITUDE"].values, dtype=float)
    lon_min, lat_min, lon_max, lat_max = bbox
    flags = np.full(lon.shape, 1, dtype=np.uint8)  # 1 = good (inside or on the box)
    # NaN comparisons are False, so a NaN position is not caught here; it is set to 9 afterwards.
    flags[(lon < lon_min) | (lon > lon_max) | (lat < lat_min) | (lat > lat_max)] = 4  # outside
    flags[~np.isfinite(lon) | ~np.isfinite(lat)] = 9  # position unknown (overrides the above)
    return flags


#: Variables that cannot be negative — their suggested gross-range lower bound is clamped at 0.
_NONNEGATIVE_VARS = ("DOXY", "CHLA", "BBP700")


def suggest_gross_range(ds, var='TEMP', inner_pct=99.9, suspect_pad=0.01, fail_pad=0.05):
    """Suggest data-driven gross-range spans for *var* from this mission's value distribution.

    A diagnostic suggestion, not a fix: it describes where *this* deployment's values lie, as a
    starting point for a mission config — not a universal physical range. The inner *inner_pct* of
    the finite values is taken (default the central 99.9%, i.e. the 0.05th–99.95th percentiles), then
    padded outward by a fraction of that span — *suspect_pad* and *fail_pad* — so the margin scales
    with each variable's own range rather than with the value (1% of 4 °C is not 1% of 35 psu). The
    lower bound is clamped at 0 for non-negative variables (:data:`_NONNEGATIVE_VARS`).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in **OG1 format**, containing *var*.
    var : str, optional, default='TEMP'
        Variable to suggest limits for.
    inner_pct : float, optional, default=99.9
        Central percentile of the data taken as the core range before padding.
    suspect_pad, fail_pad : float, optional
        Fractions of the core span added on each side for the suspect (0.01) and fail (0.05) spans.

    Returns
    -------
    dict or None
        ``{lo, hi, inner_pct, suspect, fail, suspect_pad, fail_pad}`` (spans as ``[min, max]``, all
        rounded), or ``None`` when *var* has no finite values.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams
    """
    x = np.asarray(ds[var].values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    half = (100.0 - inner_pct) / 2.0
    lo, hi = (float(p) for p in np.percentile(x, [half, 100.0 - half]))
    span = hi - lo
    clamp = var in _NONNEGATIVE_VARS

    def _pad(frac):
        a, b = lo - frac * span, hi + frac * span
        return [round(max(a, 0.0) if clamp else a, 2), round(b, 2)]

    return {
        "lo": round(lo, 2), "hi": round(hi, 2), "inner_pct": inner_pct,
        "suspect": _pad(suspect_pad), "fail": _pad(fail_pad),
        "suspect_pad": suspect_pad, "fail_pad": fail_pad,
    }


def suggest_flat_line(ds, var=None, suspect_factor=2, fail_factor=3):
    """Suggest flat-line windows (seconds) from the sampling interval of *var* (or the whole record).

    A flat-line window set in seconds must span at least a couple of the sensor's *own* samples to say
    anything about flatness. For a sensor sampled off-phase (optics, oxygen — finite only every Nth
    cycle) the per-variable interval is coarser than the record's, so basing the window on the record
    would suggest one too short. Pass *var* to use that variable's finite-sample interval; omit it for
    the whole-record interval. The suggestion is ``suspect_factor``× (default 2) and ``fail_factor``×
    (default 3) that interval.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in **OG1 format**, with ``TIME`` (and *var* when given).
    var : str, optional
        Variable whose finite-sample interval to use; when ``None`` the whole-record interval is used.
    suspect_factor, fail_factor : int, optional
        Multiples of the sampling interval for the suspect (2) and fail (3) windows.

    Returns
    -------
    dict
        ``{dt, suspect, fail, suspect_factor, fail_factor}`` — the interval and the suggested windows in
        seconds (``dt`` NaN when there are fewer than two finite samples).

    Notes
    -----
    Original Author: Eleanor Frajka-Williams
    """
    t = np.asarray(ds.TIME.values)
    if var is not None:
        t = t[np.isfinite(np.asarray(ds[var].values, dtype=float))]
    dt = utilities._median_sampling_interval(t)
    if not np.isfinite(dt):
        return {"dt": float("nan"), "suspect": float("nan"), "fail": float("nan"),
                "suspect_factor": suspect_factor, "fail_factor": fail_factor}
    dt = round(dt, 1)
    return {"dt": dt, "suspect": round(suspect_factor * dt), "fail": round(fail_factor * dt),
            "suspect_factor": suspect_factor, "fail_factor": fail_factor}


def phrase_duration_check(ds):
    """
    Check for anomalies in profile duration and return a human-readable message summarizing the result.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in **OG1 format**, containing time information required for computing profile durations.

    Returns
    -------
    duration_check : str
        A message indicating whether abnormal profile durations have been detected:
        - If outliers are found: `"X profiles have abnormal duration"`
        - Otherwise: `"No issues with profile duration have been detected"`

    Notes
    ------
    Original Author: Chiara  Monforte.
    """
    duration = tools.compute_prof_duration(ds)
    _rolling_mean, overtime = tools.find_outlier_duration(duration, rolling=20, std=2)
    if len(overtime) > 0:
        duration_check = f'{len(overtime)} profiles have abnormal duration'
    else:
        duration_check = 'No issues with profile duration have been detected'
    return duration_check


def phrase_numberprof_check(ds):
    """
    Check the monotonicity of profile numbers and return a human-readable summary of the result.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in **OG1 format**, containing the **PROFILE_NUMBER** variable.

    Returns
    -------
    prof_check : str
        A message indicating whether profile numbers increase monotonically:
        - If monotonic: `"No issues detected"`
        - If not monotonic: `"Issues with profile number have been detected"`

    Notes
    ------
    Original Author: Chiara  Monforte.
    """
    check = tools.check_monotony(ds.PROFILE_NUMBER)
    if check:
        prof_check = 'No issues detected'
    else:
        prof_check = 'Issues with profile number have been detected'
    return prof_check


def fill_str(strgr, strst, strft, strhy, strdr, ds, var='TEMP'):
    """
    Populate QC summary strings for a selected variable based on a series of QC checks.

    Parameters
    ----------
    strgr : list of str
        Row for **Global range test** results with X or ✓, structured as `[label/test name, TEMP, PSAL, DOXY, CHLA]`.
    strst : list of str
        Row for **Spike test** results with X or ✓, structured as `[label/test name, TEMP, PSAL, DOXY, CHLA]`.
    strft : list of str
        Row for **Flat line test** results with X or ✓, structured as `[label/test name, TEMP, PSAL, DOXY, CHLA]`.
    strhy : list of str
        Row for **Hysteresis error (mean)** results with X or ✓, structured as `[label/test name, TEMP, PSAL, DOXY, CHLA]`.
    strdr : list of str
        Row for **Hysteresis error (range)** results with X or ✓, structured as `[label/test name, TEMP, PSAL, DOXY, CHLA]`.
    ds : xarray.Dataset
        Dataset in **OG1 format**, containing the selected variable.
    var : str, optional, default='TEMP'
        Selected variable to evaluate. Must be one of: `'TEMP'`, `'PSAL'`, `'DOXY'`, `'CHLA'`.

    Returns
    -------
    strgr : list of str
        Updated Global range test row.
    strst : list of str
        Updated Spike test row.
    strft : list of str
        Updated Flat line test row.
    strhy : list of str
        Updated Hysteresis error (mean) row.
    strdr : list of str
        Updated Hysteresis error (range) row.

    Notes
    ------
    - Input string is filled with ✓. Each list entry is updated with an `'X'` if a QC issue is detected for the corresponding variable.
    - If the variable is missing in the dataset, all tests for that variable are marked as `"No data"`.
    - Column indices are mapped as: `TEMP=1`, `PSAL=2`, `DOXY=3`, `CHLA=4`.
    Original Author: Chiara  Monforte.
    """
    i = {"TEMP": 1, "PSAL": 2, "DOXY": 3, "CHLA": 4}[var]
    if var not in ds.variables:
        strgr[i] = 'No data'
        strft[i] = 'No data'
        strst[i] = 'No data'
        strhy[i] = 'No data'
        strdr[i] = 'No data'
    else:
        gr, spike, flat, err_mean, err_range = qc_checks(ds, var=var)
        if len(gr) > 0:
            strgr[i] = 'X'
        if len(np.where((spike == 3) | (spike == 4))[0]) > 0:
            strst[i] = 'X'
        if len(np.where((flat == 3) | (flat == 4))[0]) > 0:
            strft[i] = 'X'
        if hysteresis_verdict(err_mean)[1]:
            strhy[i] = 'X'
        if hysteresis_verdict(err_range)[1]:
            strdr[i] = 'X'
    return strgr, strst, strft, strhy, strdr
