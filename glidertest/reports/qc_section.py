"""QC section data for the two QC templates.

The two QC parts render as separate, jumpable sections and are never merged: :func:`delivered_data`
(``templates/_qc_delivered.html``) is the "as delivered" per-variable flag census the provider's
pipeline wrote; :func:`diagnostics_data` (``templates/_qc_glidertest.html``) is glidertest's own
second opinion — the basic-checks sentences plus the on-the-fly diagnostics (thresholds and a
test x variable matrix). :func:`qc_section_data` returns both combined. All plain dicts and numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .. import qc

if TYPE_CHECKING:
    import xarray as xr

_QC_VARS = ("TEMP", "PSAL", "DOXY", "CHLA")

#: Fixed colour class per flag category (by numeric value), independent of the file's labels.
_DELIVERED_CELL_CLASS = {"good": "qc-good", "suspect": "qc-susp", "fail": "qc-fail"}

#: Diagnostics matrix rows: (row label, cell kind, index into the qc_checks tuple).
_MATRIX_ROWS = (
    ("Gross range", "gross", 0),
    ("Spike", "sample", 1),
    ("Flat line", "sample", 2),
    ("Hysteresis (mean)", "hyst", 3),
    ("Hysteresis (range)", "hyst", 4),
)


def _pct(count: int, n: int) -> str:
    """Format *count* of *n* as a percent: ``–`` when zero, ``<0.1% (count)`` when tiny, else ``12.3``."""
    if count == 0 or n == 0:
        return "–"
    p = 100 * count / n
    return f"<0.1% ({count})" if p < 0.1 else f"{p:.1f}"


def _bar_segments(counts: dict[str, int], n: int, labels: dict[int, str] | None = None) -> list[dict[str, str]]:
    """Return the stacked distribution-bar segments (``{key, width, title}``) for *counts* over *n*."""
    if n == 0:
        return []
    segs = []
    for value, key, default_label in qc.QC_FLAG_CATEGORIES:
        count = counts.get(key, 0)
        if not count:
            continue
        label = labels.get(value, default_label) if labels else default_label
        pct = 100 * count / n
        segs.append({"key": key, "width": f"{pct:.1f}", "title": f"{label}: {pct:.1f}%"})
    other = counts.get("other", 0)
    if other:
        pct = 100 * other / n
        segs.append({"key": "other", "width": f"{pct:.1f}", "title": f"Other (non-standard flag): {pct:.1f}%"})
    return segs


def _basic_checks(ds: xr.Dataset) -> dict[str, str] | None:
    """Return the profile-number and profile-duration phrases, or ``None`` if the inputs are absent."""
    if "PROFILE_NUMBER" not in ds or "TIME" not in ds:
        return None
    return {"profnum": qc.phrase_numberprof_check(ds), "dur": qc.phrase_duration_check(ds)}


def _delivered(ds: xr.Dataset) -> dict[str, Any]:
    """Return the 'as delivered' census: per-variable counts of the file's own ``*_QC`` flags."""
    rtqc = str(ds.attrs.get("rtqc_method", "") or "—")
    qc_vars = sorted(n for n in ds.data_vars if n.endswith("_QC"))
    if not qc_vars:
        return {"rtqc": rtqc, "no_flags": True}

    # Labels come from each variable's flag_meanings; use them for the headers only when every QC
    # variable agrees, otherwise fall back to the default category names and say which differ.
    per_var = {v: qc.flag_labels(ds[v]) for v in qc_vars}
    consistent = len({tuple(sorted(m.items())) for m in per_var.values()}) == 1
    if consistent:
        labels = per_var[qc_vars[0]]
        differ_note = ""
    else:
        labels = {value: default for value, _key, default in qc.QC_FLAG_CATEGORIES}
        names = ", ".join(v[:-3] for v in qc_vars)
        differ_note = (
            f" The QC variables ({names}) declare differing flag scales, so the default category "
            "names are shown."
        )

    counts = {v: qc.flag_counts(np.asarray(ds[v].values)) for v in qc_vars}
    show_other = any(c["other"] for c in counts.values())
    cats = [
        {"key": key, "label": labels[value], "cls": _DELIVERED_CELL_CLASS.get(key, "")}
        for value, key, _ in qc.QC_FLAG_CATEGORIES
    ]
    rows = []
    for qcv in qc_vars:
        c = counts[qcv]
        n = sum(c.values())  # flag_counts sums to size, so no need to re-read the array
        row = {
            "var": qcv[:-3],
            "n": f"{n:,}",
            "cells": [{"cls": cat["cls"], "pct": _pct(c[cat["key"]], n)} for cat in cats],
            "other": _pct(c["other"], n) if show_other else None,
            "bar": _bar_segments(c, n, labels),
        }
        rows.append(row)

    mismatches = [
        f"{qcv[:-3]} (flag_values has {mm[0]}, flag_meanings has {mm[1]})"
        for qcv in qc_vars
        if (mm := qc.flag_scale_mismatch(ds[qcv])) is not None
    ]
    caption = (
        f"As delivered — rtqc_method: {rtqc}. Flag labels are read from each variable's "
        f"flag_meanings.{differ_note} The file does not record the thresholds used."
    )
    return {
        "caption": caption,
        "mismatch": "; ".join(mismatches) if mismatches else None,
        "cats": cats,
        "show_other": show_other,
        "rows": rows,
    }


def _gross_cell(r: tuple, ds: xr.Dataset, v: str) -> dict[str, Any]:
    """Diagnostics matrix cell for the gross-range test: out-of-range count + a small bar."""
    n = ds.sizes["N_MEASUREMENTS"] if "N_MEASUREMENTS" in ds.sizes else int(ds[v].size)
    nv = len(r[0])
    if nv == 0:
        return {"text": "clean", "cls": "qc-good"}
    return {
        "spans": [{"cls": "qc-susp", "text": f"{nv:,} out of range"}],
        "bar": _bar_segments({"good": n - nv, "suspect": nv}, n),
    }


def _sample_cell(flags: np.ndarray) -> dict[str, Any]:
    """Diagnostics matrix cell for a per-sample test (spike, flat): flagged counts + a small bar."""
    c = qc.flag_counts(flags)
    n = int(np.asarray(flags).size)
    if c["suspect"] == 0 and c["fail"] == 0:
        return {"text": "clean", "cls": "qc-good"}
    spans = []
    if c["suspect"]:
        spans.append({"cls": "qc-susp", "text": f"{c['suspect']:,} susp"})
    if c["fail"]:
        spans.append({"cls": "qc-fail", "text": f"{c['fail']:,} fail"})
    return {"spans": spans, "bar": _bar_segments(c, n)}


def _hyst_cell(err: np.ndarray) -> dict[str, Any]:
    """Diagnostics matrix cell for a hysteresis test: depth bins over threshold, graded by the verdict."""
    arr = np.asarray(err)
    n_over, flagged = qc.hysteresis_verdict(arr)
    cls = "qc-fail" if flagged else ("qc-susp" if n_over else "qc-good")
    return {"text": f"{n_over}/{arr.size} bins", "cls": cls}


def _diagnostics(ds: xr.Dataset) -> dict[str, Any] | None:
    """Return the glidertest-diagnostics data (thresholds + matrix), or ``None`` when no QC var applies."""
    config = qc.configs
    present = [v for v in _QC_VARS if v in ds]
    if not present:
        return None

    thr_rows = []
    for v in present:
        g = config[v]["gross_range_test"]
        s = config[v]["spike_test"]
        thr_rows.append(
            {
                "var": v,
                "test": "gross-range",
                "suspect": f"[{g['suspect_span'][0]}, {g['suspect_span'][1]}]",
                "fail": f"[{g['fail_span'][0]}, {g['fail_span'][1]}] (not applied)",
            }
        )
        thr_rows.append(
            {
                "var": v,
                "test": "spike",
                "suspect": f"|Δ| > {s['suspect_threshold']}",
                "fail": f"|Δ| > {s['fail_threshold']}",
            }
        )

    # qc_checks runs QARTOD + hysteresis gridding on real data; degenerate data can raise from numpy
    # or the gridding. Blank that one column rather than drop the matrix (anything else reaches the
    # section-level guard in _mission._guarded).
    results: dict[str, tuple | None] = {}
    for v in present:
        try:
            results[v] = qc.qc_checks(ds, var=v)
        except (ValueError, KeyError, IndexError, ZeroDivisionError, RuntimeError):
            results[v] = None

    matrix_rows = []
    for label, kind, idx in _MATRIX_ROWS:
        cells = []
        for v in present:
            r = results[v]
            if r is None:
                cells.append({"text": "–"})
            elif kind == "gross":
                cells.append(_gross_cell(r, ds, v))
            elif kind == "sample":
                cells.append(_sample_cell(np.asarray(r[idx])))
            elif kind == "hyst":
                cells.append(_hyst_cell(r[idx]))
            else:
                raise ValueError(f"unknown matrix cell kind: {kind!r}")
        matrix_rows.append({"label": label, "cells": cells})

    return {"present": present, "thresholds": thr_rows, "matrix": matrix_rows}


def delivered_data(ds: xr.Dataset) -> dict[str, Any]:
    """Return the file's own QC census ("as delivered") as data for ``_qc_delivered.html``."""
    return {"delivered": _delivered(ds)}


def diagnostics_data(ds: xr.Dataset) -> dict[str, Any]:
    """Return glidertest's on-the-fly QC diagnostics as data for ``_qc_glidertest.html``.

    Includes the basic profile-number/duration checks, which glidertest computes on the fly (they are
    not file flags), so they sit with the glidertest diagnostics rather than the delivered census.
    """
    return {"basic_checks": _basic_checks(ds), "diagnostics": _diagnostics(ds)}


def qc_section_data(ds: xr.Dataset) -> dict[str, Any]:
    """Return the combined QC section data (delivered census + glidertest diagnostics)."""
    return {**delivered_data(ds), **diagnostics_data(ds)}
