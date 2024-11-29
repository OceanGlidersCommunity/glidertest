import pytest
from glidertest import fetchers, tools, plots
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from ioos_qc import qartod
matplotlib.use('agg')  # use agg backend to prevent creating plot windows during tests

def test_plots(start_prof=0, end_prof=100):
    ds = fetchers.load_sample_dataset()
    ds = ds.drop_vars(['DENSITY'])
    fig, ax = plots.plot_basic_vars(ds,start_prof=start_prof, end_prof=end_prof)
    assert ax[0].get_ylabel() == 'Depth (m)'
    assert ax[0].get_xlabel() == f'Temperature [C]'


def test_up_down_bias(v_res=1, xlabel='Salinity'):
    ds = fetchers.load_sample_dataset()
    fig, ax = plt.subplots()
    df = tools.quant_updown_bias(ds, var='PSAL', v_res=v_res)
    plots.plot_updown_bias(df, ax,  xlabel=xlabel)
    lims = np.abs(df.dc)
    assert ax.get_xlim() == (-np.nanpercentile(lims, 99.5), np.nanpercentile(lims, 99.5))
    assert ax.get_ylim() == (df.depth.max() + 1, -df.depth.max() / 30)
    assert ax.get_xlabel() == xlabel
    # check without passing axis
    new_fig, new_ax = plots.plot_updown_bias(df, xlabel=xlabel)
    assert new_ax.get_xlim() == (-np.nanpercentile(lims, 99.5), np.nanpercentile(lims, 99.5))
    assert new_ax.get_ylim() == (df.depth.max() + 1, -df.depth.max() / 30)
    assert new_ax.get_xlabel() == xlabel


def test_chl(var1='CHLA', var2='BBP700'):
    ds = fetchers.load_sample_dataset()
    ax = plots.process_optics_assess(ds, var=var1)
    assert ax.get_ylabel() == var1
    ax = plots.process_optics_assess(ds, var=var2)
    assert ax.get_ylabel() == var2
    with pytest.raises(KeyError) as e:
        plots.process_optics_assess(ds, var='nonexistent_variable')


def test_quench_sequence(xlabel='Temperature [C]',ylim=45):
    ds = fetchers.load_sample_dataset()
    if not "TIME" in ds.indexes.keys():
        ds = ds.set_xindex('TIME')
    fig, ax = plt.subplots()
    plots.plot_quench_assess(ds, 'CHLA', ax,ylim=ylim)
    assert ax.get_ylabel() == 'Depth [m]'
    assert ax.get_ylim() == (ylim, -ylim / 30)
    
    dayT, nightT = tools.compute_daynight_avg(ds, sel_var='TEMP')
    fig, ax = plots.plot_daynight_avg(dayT, nightT,xlabel=xlabel) 
    assert ax.get_ylabel() == 'Depth [m]'
    assert ax.get_xlabel() == xlabel


def test_temporal_drift(var='DOXY'):
    ds = fetchers.load_sample_dataset()
    fig, ax = plt.subplots(1, 2)
    plots.check_temporal_drift(ds,var, ax)
    assert ax[1].get_ylabel() == 'Depth (m)'
    assert ax[0].get_ylabel() == var
    assert ax[1].get_xlim() == (np.nanpercentile(ds[var], 0.01), np.nanpercentile(ds[var], 99.99))
    plots.check_temporal_drift(ds,'CHLA')


def test_profile_check():
    ds = fetchers.load_sample_dataset()
    tools.check_monotony(ds.PROFILE_NUMBER)
    fig, ax = plots.plot_prof_monotony(ds)
    assert ax[0].get_ylabel() == 'Profile number'
    assert ax[1].get_ylabel() == 'Depth (m)'
    duration = tools.compute_prof_duration(ds)
    rolling_mean, overtime = tools.find_outlier_duration(duration, rolling=20, std=2)
    fig, ax = plots.plot_outlier_duration(ds, rolling_mean, overtime, std = 2)
    assert ax[0].get_ylabel() == 'Profile duration (min)'
    assert ax[0].get_xlabel() == 'Profile number'
    assert ax[1].get_ylabel() == 'Depth (m)'

def test_basic_statistics():
    ds = fetchers.load_sample_dataset()
    plots.plot_glider_track(ds)
    plots.plot_grid_spacing(ds)
    plots.plot_ts(ds)


def test_vert_vel():
    ds_sg014 = fetchers.load_sample_dataset(dataset_name="sg014_20040924T182454_delayed_subset.nc")
    ds_sg014 = tools.calc_w_meas(ds_sg014)
    ds_sg014 = tools.calc_w_sw(ds_sg014)
    plots.plot_vertical_speeds_with_histograms(ds_sg014)
    ds_dives = ds_sg014.sel(N_MEASUREMENTS=ds_sg014.PHASE == 2)
    ds_climbs = ds_sg014.sel(N_MEASUREMENTS=ds_sg014.PHASE == 1)
    ds_out_dives = tools.quant_binavg(ds_dives, var = 'VERT_CURR_MODEL', dz=10)
    ds_out_climbs = tools.quant_binavg(ds_climbs, var = 'VERT_CURR_MODEL', dz=10)
    plots.plot_combined_velocity_profiles(ds_out_dives, ds_out_climbs)
    # extra tests for ramsey calculations of DEPTH_Z
    ds_climbs = ds_climbs.drop_vars(['DEPTH_Z'])
    tools.quant_binavg(ds_climbs, var='VERT_CURR_MODEL', dz=10)
    ds_climbs = ds_climbs.drop_vars(['LATITUDE'])
    with pytest.raises(KeyError) as e:
        tools.quant_binavg(ds_climbs, var='VERT_CURR_MODEL', dz=10)
def test_hyst_plot(var='DOXY'):
    ds = fetchers.load_sample_dataset()
    fig, ax = plots.plot_hysteresis(ds, var=var, v_res=1, perct_err=2, ax=None)
    assert ax[3].get_ylabel() == 'Depth (m)'
    assert ax[0].get_ylabel() == 'Depth (m)'

def test_sop():
    ds = fetchers.load_sample_dataset()
    plots.plot_global_range(ds, var='DOXY', min_val=-5, max_val=600, ax=None)
    spike = qartod.spike_test(ds.DOXY,suspect_threshold=25,fail_threshold=50, method="average", )
    plots.plot_ioosqc(spike, suspect_threshold=[25], fail_threshold=[50], title='Spike test DOXY')
    flat = qartod.flat_line_test(ds.DOXY, ds.TIME, 1, 2, 0.001)
    plots.plot_ioosqc(flat, suspect_threshold=[0.01], fail_threshold=[0.1], title='Flat test DOXY')

def test_plot_sampling_period_all():
    ds = fetchers.load_sample_dataset()
    plots.plot_sampling_period_all(ds)
    plots.plot_sampling_period(ds,variable='CHLA')
