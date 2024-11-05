import pytest
from glidertest import fetchers, tools
import matplotlib.pyplot as plt


def test_plots():
    ds = fetchers.load_sample_dataset()
    tools.plot_basic_vars(ds, end_prof=int(ds.PROFILE_NUMBER.max()))


def test_up_down_bias():
    ds = fetchers.load_sample_dataset()
    fig, ax = plt.subplots()
    df = tools.updown_bias(ds, var='PSAL', v_res=1)
    tools.plot_updown_bias(df, ax,  xlabel='Salinity')


def test_chl():
    ds = fetchers.load_sample_dataset()
    tools.optics_first_check(ds, var='CHLA')
    tools.optics_first_check(ds, var='BBP700')
    with pytest.raises(KeyError) as e:
        tools.optics_first_check(ds, var='nonexistent_variable')


def test_quench_sequence():
    ds = fetchers.load_sample_dataset()
    if not "TIME" in ds.indexes.keys():
        ds = ds.set_xindex('TIME')
    fig, ax = plt.subplots()
    tools.plot_section_with_srss(ds, 'CHLA')
    dayT, nightT = tools.day_night_avg(ds, sel_var='TEMP')
    fig, ax = plt.subplots()
    tools.plot_daynight_avg(dayT, nightT, ax, xlabel='Temperature [C]') 


def test_temporal_drift():
    ds = fetchers.load_sample_dataset()
    fig, ax = plt.subplots(1, 2)
    tools.check_temporal_drift(ds,'DOXY', ax)
    tools.check_temporal_drift(ds,'CHLA')


def test_profile_check():
    ds = fetchers.load_sample_dataset()
    tools.check_monotony(ds.PROFILE_NUMBER)
    tools.plot_profIncrease(ds)


def test_check_monotony():
    ds = fetchers.load_sample_dataset()
    profile_number_monotony = tools.check_monotony(ds.PROFILE_NUMBER)
    temperature_monotony = tools.check_monotony(ds.TEMP)
    assert profile_number_monotony
    assert not temperature_monotony


def test_basic_statistics():
    ds = fetchers.load_sample_dataset()
    tools.plot_glider_track(ds)
    tools.plot_grid_spacing_histograms(ds)
    tools.plot_ts_histograms(ds)


def test_vert_vel():
    ds_sg014 = fetchers.load_sample_dataset(dataset_name="sg014_20040924T182454_delayed_subset.nc")
    ds_sg014 = tools.calc_glider_w_from_depth(ds_sg014)
    ds_sg014 = tools.calc_seawater_w(ds_sg014)
    tools.plot_vertical_speeds_with_histograms(ds_sg014)
    ds_dives = ds_sg014.sel(N_MEASUREMENTS=ds_sg014.PHASE == 2)
    ds_climbs = ds_sg014.sel(N_MEASUREMENTS=ds_sg014.PHASE == 1)
    ds_out_dives = tools.ramsey_binavg(ds_dives, var = 'VERT_CURR_MODEL', dz=10)
    ds_out_climbs = tools.ramsey_binavg(ds_climbs, var = 'VERT_CURR_MODEL', dz=10)
    tools.plot_combined_velocity_profiles(ds_out_dives, ds_out_climbs)

