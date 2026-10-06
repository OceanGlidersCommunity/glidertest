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

configs = {
    "TEMP": {"gross_range_test": {
        "suspect_span": [0, 30],
        "fail_span": [-2.5, 40],
    },
        "spike_test": {"suspect_threshold": 2.0, "fail_threshold": 6.0},
        "location_test": {"bbox": [10, 50, 25, 70]},
    },
    "PSAL": {"gross_range_test": {"suspect_span": [5, 38], "fail_span": [2, 41]},
             "spike_test": {"suspect_threshold": 0.3, "fail_threshold": 0.9},
             "location_test": {"bbox": [10, 50, 25, 70]},
             },
    "DOXY": {"gross_range_test": {
        "suspect_span": [0, 350],
        "fail_span": [0, 500],
    },
        "spike_test": {"suspect_threshold": 10, "fail_threshold": 50},
        "location_test": {"bbox": [10, 50, 25, 70]},
    },
    "CHLA": {"gross_range_test": {
        "suspect_span": [0, 15],
        "fail_span": [-1, 20],
    },
        "spike_test": {"suspect_threshold": 1, "fail_threshold": 5},
        "location_test": {"bbox": [10, 50, 25, 70]},
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

    Covers all five flag values (good=1, suspect=3, fail=4, not-evaluated=2, missing=9), so the
    returned counts sum to the array size.

    Parameters
    ----------
    flags : array-like
        A QARTOD/OceanSITES integer flag array.

    Returns
    -------
    dict of str to int
        Count per category key, in :data:`QC_FLAG_CATEGORIES` order.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams.
    """
    arr = np.asarray(flags)
    return {key: int((arr == value).sum()) for value, key, _ in QC_FLAG_CATEGORIES}


def hysteresis_verdict(err, pct_threshold=5, bin_threshold=5):
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


def qc_checks(ds, var='TEMP'):
    """
    Run a series of basic quality control (global range, spike test, drift and hysteresis) checks on a selected variable in the dataset.
    We use functions from glidertest.tools as well as ioos qartod.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in **OG1 format**, containing the selected variable and time information.
    var : str, optional, default='TEMP'
        Variable to perform QC checks on.

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
    - Thresholds for QC tests are taken from the global `configs` dictionary.
    Original Author: Chiara  Monforte
    """
    utilities._check_necessary_variables(ds, [var, 'TIME'])
    gr = tools.compute_global_range(ds, var=var, min_val=configs[var]['gross_range_test']['suspect_span'][0],
                                    max_val=configs[var]['gross_range_test']['suspect_span'][1])
    spike = qartod.spike_test(ds[var], suspect_threshold=configs[var]['spike_test']['suspect_threshold'],
                              fail_threshold=configs[var]['spike_test']['fail_threshold'], method="average")
    flat = qartod.flat_line_test(ds[var], ds.TIME, 1, 3, 0.001)
    __, __, err_mean, err_range, __ = tools.compute_hyst_stat(ds, var=var, v_res=1)

    return gr, spike, flat, err_mean, err_range


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
    rolling_mean, overtime = tools.find_outlier_duration(duration, rolling=20, std=2)
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
