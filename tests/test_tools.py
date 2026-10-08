import math

import numpy as np
import pytest

from glidertest import fetchers, tools

# These run on the committed subsets (opened fresh per test): sea045 for the CTD/optics tools,
# sg014 for the vertical-velocity tools. Day/night averaging needs night profiles the 12-profile
# subset lacks, so test_daynight stays on the full sample (slow). See tests/data/README.md.


@pytest.mark.slow
def test_updown_bias(v_res=1):
    # The length invariant assumes contiguous integer-depth coverage (one bin per rounded metre),
    # which holds for a full mission but not a 12-profile subset that skips depths; so this runs on
    # the full sample. quant_updown_bias itself is exercised on the subset by test_plots.
    ds = fetchers.load_sample_dataset()
    df = tools.quant_updown_bias(ds, var='PSAL', v_res=v_res)
    bins = np.unique(np.round(ds.DEPTH, 0))
    ncell = math.ceil(len(bins) / v_res)
    assert len(df) == ncell


def test_mean_profile(fresh_subset):
    tools.mean_profile(fresh_subset, var='TEMP', v_res=1)


@pytest.mark.slow
def test_daynight():
    ds = fetchers.load_sample_dataset()
    if "TIME" not in ds.indexes:
        ds = ds.set_xindex('TIME')

    dayT, nightT = tools.compute_daynight_avg(ds, sel_var='TEMP')
    assert len(nightT.dat.dropna()) > 0
    assert len(dayT.dat.dropna()) > 0


def test_check_monotony(fresh_subset):
    ds = fresh_subset
    profile_number_monotony = tools.check_monotony(ds.PROFILE_NUMBER)
    temperature_monotony = tools.check_monotony(ds.TEMP)
    assert profile_number_monotony
    assert not temperature_monotony
    duration = tools.compute_prof_duration(ds)
    rolling_mean, overtime = tools.find_outlier_duration(duration, rolling=20, std=2)


def test_vert_vel(fresh_sg014):
    ds_sg014 = tools.calc_w_meas(fresh_sg014)  # computes DEPTH_Z (the subset carries none)
    ds_sg014 = tools.calc_w_sw(ds_sg014)

    ds_dives = ds_sg014.sel(N_MEASUREMENTS=ds_sg014.PHASE == 2)
    ds_climbs = ds_sg014.sel(N_MEASUREMENTS=ds_sg014.PHASE == 1)
    tools.quant_binavg(ds_dives, var='VERT_CURR_MODEL', dz=10)

    # extra tests for ramsey calculations of DEPTH_Z
    ds_climbs = ds_climbs.drop_vars(['DEPTH_Z'])
    tools.quant_binavg(ds_climbs, var='VERT_CURR_MODEL', dz=10)
    ds_climbs = ds_climbs.drop_vars(['LATITUDE'])
    with pytest.raises(KeyError):
        tools.quant_binavg(ds_climbs, var='VERT_CURR_MODEL', dz=10)


def test_hyst(fresh_subset):
    ds = fresh_subset
    df_h = tools.quant_hysteresis(ds, var='DOXY', v_res=1)
    df, diff, err_mean, err_range, rms = tools.compute_hyst_stat(ds, var='DOXY', v_res=1)
    assert np.array_equal(df_h.dropna(), df.dropna())
    assert len(diff) == len(err_mean)


def test_sop(fresh_subset):
    tools.compute_global_range(fresh_subset, var='DOXY', min_val=-5, max_val=600)


def test_maxdepth(fresh_subset):
    tools.max_depth_per_profile(fresh_subset)


def test_maxdepth_units_unknown_when_missing():
    import xarray as xr

    ds = xr.Dataset(
        {
            "DEPTH": ("N_MEASUREMENTS", np.array([1.0, 5.0, 2.0, 8.0])),  # no units attr
            "PROFILE_NUMBER": ("N_MEASUREMENTS", np.array([1, 1, 2, 2])),
        }
    )
    # Units must not be silently blanked when DEPTH has no units attribute.
    assert tools.max_depth_per_profile(ds).attrs["units"] == "UNK"


def test_mld(fresh_subset):
    ds = fresh_subset
    ds = tools.add_sigma_1(ds)
    ### Test all three MLD methods
    mld_thresh = tools.compute_mld(ds, variable='DENSITY', method='threshold')
    mld_CR = tools.compute_mld(ds, variable='SIGMA_1', method='CR', threshold=-1)

    assert len(np.unique(ds.PROFILE_NUMBER)) == len(mld_thresh)
    assert len(np.unique(ds.PROFILE_NUMBER)) == len(mld_CR)
    # Test if len(df) == 0
    ds['CHLA'] = np.nan
    mld_thresh = tools.compute_mld(ds, 'CHLA', method='threshold', threshold=0.01, ref_depth=10)
    assert np.isnan(np.unique(mld_thresh['MLD']))


def test_add_sigma1(fresh_subset):
    tools.add_sigma_1(fresh_subset)
