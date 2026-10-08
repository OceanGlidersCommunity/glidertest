import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr

# make_subset.py lives under tests/data/ (not an importable package), so load it by path.
_spec = importlib.util.spec_from_file_location(
    "make_subset", Path(__file__).parent / "data" / "make_subset.py"
)
make_subset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_subset)


def test_cut_selects_profiles_drops_and_casts():
    # The committed fixtures are only trustworthy if cut() keeps the right profiles, drops
    # non-keep measurement variables, keeps scalars, and casts geophysics to float32 while keeping
    # time/position float64. A synthetic dataset exercises that contract without a download.
    ds = xr.Dataset(
        {
            "TEMP": ("N_MEASUREMENTS", np.arange(6.0)),      # geophysical float64 -> float32
            "LATITUDE": ("N_MEASUREMENTS", np.arange(6.0)),  # position float64 -> stays float64
            "PROFILE_NUMBER": ("N_MEASUREMENTS", np.array([1.0, 1.0, 2.0, 2.0, 3.0, np.nan])),
            "JUNK": ("N_MEASUREMENTS", np.arange(6.0)),      # not in keep -> dropped
            "SENSOR_X": ((), "model-1"),                     # scalar (no N_MEASUREMENTS) -> kept
        },
        attrs={"id": "mission-1"},
    )
    sub = make_subset.cut(ds, n_profiles=2, keep={"TEMP", "LATITUDE", "PROFILE_NUMBER"})

    # first two finite profiles only (the NaN profile row is excluded)
    assert sorted(np.unique(sub["PROFILE_NUMBER"].values).tolist()) == [1.0, 2.0]
    assert sub.sizes["N_MEASUREMENTS"] == 4
    # a non-keep measurement variable is dropped; a scalar is kept regardless of the keep-list
    assert "JUNK" not in sub.variables
    assert "SENSOR_X" in sub.variables
    # dtype contract: geophysical -> float32, position -> float64
    assert sub["TEMP"].dtype == np.float32
    assert sub["LATITUDE"].dtype == np.float64
    # global attributes carry over
    assert sub.attrs["id"] == "mission-1"
