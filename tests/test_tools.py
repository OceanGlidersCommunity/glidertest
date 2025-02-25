import pytest
from glidertest import fetchers, tools
import math
import numpy as np
import matplotlib
matplotlib.use('agg')  # use agg backend to prevent creating plot windows during tests


def test_updown_bias(v_res=1):
    ds = fetchers.load_sample_dataset()
    df = tools.quant_updown_bias(ds, var='PSAL', v_res=v_res)
    bins = np.unique(np.round(ds.DEPTH,0))
    ncell = math.ceil(len(bins)/v_res)
    assert len(df) == ncell

def test_daynight():
    ds = fetchers.load_sample_dataset()
    if not "TIME" in ds.indexes.keys():
        ds = ds.set_xindex('TIME')

    dayT, nightT = tools.compute_daynight_avg(ds, sel_var='TEMP')
    assert len(nightT.dat.dropna()) > 0
    assert len(dayT.dat.dropna()) > 0

def test_check_monotony():
    ds = fetchers.load_sample_dataset()
    profile_number_monotony = tools.check_monotony(ds.PROFILE_NUMBER)
    temperature_monotony = tools.check_monotony(ds.TEMP)
    assert profile_number_monotony
    assert not temperature_monotony
    duration = tools.compute_prof_duration(ds)
    rolling_mean, overtime = tools.find_outlier_duration(duration, rolling=20, std=2)


def test_vert_vel():
    ds_sg014 = fetchers.load_sample_dataset(dataset_name="sg014_20040924T182454_delayed_subset.nc")
    ds_sg014 = ds_sg014.drop_vars("DEPTH_Z")
    ds_sg014 = tools.calc_w_meas(ds_sg014)
    ds_sg014 = tools.calc_w_sw(ds_sg014)

    ds_dives = ds_sg014.sel(N_MEASUREMENTS=ds_sg014.PHASE == 2)
    ds_climbs = ds_sg014.sel(N_MEASUREMENTS=ds_sg014.PHASE == 1)
    tools.quant_binavg(ds_dives, var = 'VERT_CURR_MODEL', dz=10)
    
    # extra tests for ramsey calculations of DEPTH_Z
    ds_climbs = ds_climbs.drop_vars(['DEPTH_Z'])
    tools.quant_binavg(ds_climbs, var='VERT_CURR_MODEL', dz=10)
    ds_climbs = ds_climbs.drop_vars(['LATITUDE'])
    with pytest.raises(KeyError) as e:
        tools.quant_binavg(ds_climbs, var='VERT_CURR_MODEL', dz=10)
def test_hyst():
    ds = fetchers.load_sample_dataset()
    df_h = tools.quant_hysteresis(ds, var = 'DOXY', v_res = 1)
    df, diff, err_mean, err_range, rms = tools.compute_hyst_stat(ds, var='DOXY', v_res=1)
    assert np.array_equal(df_h.dropna(), df.dropna())
    assert len(diff) == len(err)

def test_sop():
    ds = fetchers.load_sample_dataset()
    tools.compute_global_range(ds, var='DOXY', min_val=-5, max_val=600)

def test_maxdepth():
    ds = fetchers.load_sample_dataset()
    tools.max_depth_per_profile(ds)