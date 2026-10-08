import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from ioos_qc import qartod

from glidertest import fetchers, plots, tools, utilities

# These run on the committed subsets (opened fresh per test): sea045 for the CTD/optics plots,
# sg014 for the vertical-velocity plots. The day/night average plot needs night profiles the
# 12-profile subset lacks, so it stays on the full sample (slow). See tests/data/README.md.


def test_plots(fresh_subset, start_prof=0, end_prof=100):
    ds = fresh_subset
    ds = ds.drop_vars(['DENSITY'])
    fig, ax = plots.plot_basic_vars(ds, start_prof=start_prof, end_prof=end_prof)
    assert ax[0].get_ylabel() == 'Depth (m)'
    assert ax[0].get_xlabel() == f'{utilities.plotting_labels("TEMP")} \n({utilities.plotting_units(ds,"TEMP")})'


def test_up_down_bias(fresh_subset, v_res=1):
    ds = fresh_subset
    fig, ax = plt.subplots()
    plots.plot_updown_bias(ds, var='PSAL', v_res=1, ax=ax)
    df = tools.quant_updown_bias(ds, var='PSAL', v_res=v_res)
    lims = np.abs(df.dc)
    assert ax.get_xlim() == (-np.nanpercentile(lims, 99.5), np.nanpercentile(lims, 99.5))
    assert ax.get_ylim() == (df.depth.max() + 1, -df.depth.max() / 30)
    assert ax.get_xlabel() == f'{utilities.plotting_labels("PSAL")} ({utilities.plotting_units(ds,"PSAL")})'
    # check without passing axis
    new_fig, new_ax = plots.plot_updown_bias(ds, var='PSAL', v_res=1)
    assert new_ax.get_xlim() == (-np.nanpercentile(lims, 99.5), np.nanpercentile(lims, 99.5))
    assert new_ax.get_ylim() == (df.depth.max() + 1, -df.depth.max() / 30)
    assert new_ax.get_xlabel() == f'{utilities.plotting_labels("PSAL")} ({utilities.plotting_units(ds,"PSAL")})'


def test_chl(fresh_subset, var1='CHLA', var2='BBP700'):
    ds = fresh_subset
    fig, ax = plots.process_optics_assess(ds, var=var1)
    assert ax.get_ylabel() == f'{utilities.plotting_labels(var1)} ({utilities.plotting_units(ds,var1)})'
    fig, ax = plots.process_optics_assess(ds, var=var2)
    assert ax.get_ylabel() == f'{utilities.plotting_labels(var2)} ({utilities.plotting_units(ds,var2)})'


def test_quench_sequence(fresh_subset, ylim=45):
    ds = fresh_subset
    if "TIME" not in ds.indexes:
        ds = ds.set_xindex('TIME')
    fig, ax = plt.subplots()
    plots.plot_quench_assess(ds, 'CHLA', ax, ylim=ylim)
    assert ax.get_ylabel() == 'Depth (m)'
    assert ax.get_ylim() == (ylim, -ylim / 30)


@pytest.mark.slow
def test_daynight_avg_plot():
    # The day/night average plot needs night profiles, so it runs on the multi-day full sample.
    ds = fetchers.load_sample_dataset()
    if "TIME" not in ds.indexes:
        ds = ds.set_xindex('TIME')
    fig, ax = plots.plot_daynight_avg(ds, var='TEMP')
    assert ax.get_ylabel() == 'Depth (m)'
    assert ax.get_xlabel() == f'{utilities.plotting_labels("TEMP")} ({utilities.plotting_units(ds,"TEMP")})'


def test_temporal_drift(fresh_subset, var='DOXY'):
    ds = fresh_subset
    fig, ax = plt.subplots(1, 2)
    plots.check_temporal_drift(ds, var, ax)
    assert ax[1].get_ylabel() == 'Depth (m)'
    assert ax[0].get_ylabel() == f'{utilities.plotting_labels(var)} ({utilities.plotting_units(ds, var)})'
    assert ax[1].get_xlim() == (np.nanpercentile(ds[var], 0.01), np.nanpercentile(ds[var], 99.99))
    plots.check_temporal_drift(ds, 'CHLA')


def test_profile_check(fresh_subset):
    ds = fresh_subset
    tools.check_monotony(ds.PROFILE_NUMBER)
    fig, ax = plots.plot_prof_monotony(ds)
    assert ax[0].get_ylabel() == 'Profile number'
    assert ax[1].get_ylabel() == 'Depth (m)'
    duration = tools.compute_prof_duration(ds)
    rolling_mean, overtime = tools.find_outlier_duration(duration, rolling=20, std=2)
    fig, ax = plots.plot_outlier_duration(ds, rolling_mean, overtime, std=2)
    assert ax[0].get_ylabel() == 'Profile duration (min)'
    assert ax[0].get_xlabel() == 'Profile number'
    assert ax[1].get_ylabel() == 'Depth (m)'


def test_basic_statistics(fresh_subset):
    ds = fresh_subset
    plots.plot_glider_track(ds)
    plots.plot_grid_spacing(ds)
    plots.plot_ts(ds)


def test_vert_vel(fresh_sg014):
    ds_sg014 = tools.calc_w_meas(fresh_sg014)  # computes DEPTH_Z (the subset carries none)
    ds_sg014 = tools.calc_w_sw(ds_sg014)
    plots.plot_vertical_speeds_with_histograms(ds_sg014)
    ds_dives = ds_sg014.sel(N_MEASUREMENTS=ds_sg014.PHASE == 2)
    ds_climbs = ds_sg014.sel(N_MEASUREMENTS=ds_sg014.PHASE == 1)
    ds_out_dives = tools.quant_binavg(ds_dives, var='VERT_CURR_MODEL', dz=10)
    ds_out_climbs = tools.quant_binavg(ds_climbs, var='VERT_CURR_MODEL', dz=10)
    plots.plot_combined_velocity_profiles(ds_out_dives, ds_out_climbs)
    # extra tests for ramsey calculations of DEPTH_Z
    ds_climbs = ds_climbs.drop_vars(['DEPTH_Z'])
    tools.quant_binavg(ds_climbs, var='VERT_CURR_MODEL', dz=10)
    ds_climbs = ds_climbs.drop_vars(['LATITUDE'])
    with pytest.raises(KeyError):
        tools.quant_binavg(ds_climbs, var='VERT_CURR_MODEL', dz=10)


def test_hyst_plot(fresh_subset, var='DOXY'):
    fig, ax = plots.plot_hysteresis(fresh_subset, var=var, v_res=1, threshold=2, ax=None)
    assert ax[4].get_ylabel() == 'Depth (m)'
    assert ax[0].get_ylabel() == 'Depth (m)'


def test_sop(fresh_subset):
    ds = fresh_subset
    plots.plot_global_range(ds, var='DOXY', min_val=-5, max_val=600, ax=None)
    spike = qartod.spike_test(ds.DOXY, suspect_threshold=25, fail_threshold=50, method="average")
    plots.plot_ioosqc(spike, suspect_threshold=[25], fail_threshold=[50], title='Spike test DOXY')
    flat = qartod.flat_line_test(ds.DOXY, ds.TIME, 1, 2, 0.001)
    plots.plot_ioosqc(flat, suspect_threshold=[0.01], fail_threshold=[0.1], title='Flat test DOXY')


def test_plot_sampling_period_all(fresh_subset):
    ds = fresh_subset
    plots.plot_sampling_period_all(ds)
    plots.plot_sampling_period(ds, variable='CHLA')


def test_plot_max_depth(fresh_subset):
    plots.plot_max_depth_per_profile(fresh_subset)


def test_plot_profile(fresh_subset):
    ds = fresh_subset
    prof_num = ds.PROFILE_NUMBER[0].values
    plots.plot_profile(ds, profile_num=prof_num)


def test_plot_CR(fresh_subset):
    ds = tools.add_sigma_1(fresh_subset)
    prof_num = ds.PROFILE_NUMBER[0].values
    plots.plot_CR(ds, profile_num=prof_num)


def test_plot_section(fresh_subset):
    ds = fresh_subset
    plots.plot_section(ds, vars=['TEMP'], start=475, end=500, method='pcolormesh')
    plots.plot_section(ds, vars=['PSAL'], start=None, end=475, method='contourf')


def test_style_override(fresh_subset):
    # Setting _ACTIVE_STYLE (including the list form) restyles the figures the
    # wrappers produce; restoring it returns to the default. figure.facecolor is
    # not set in the package style, so a distinct value proves the override reached
    # the figure.
    ds = fresh_subset
    distinct = matplotlib.colors.to_rgba('#123456')
    override = [plots.glidertest_style_file, {'figure.facecolor': '#123456'}]
    original = plots._ACTIVE_STYLE
    try:
        plots._ACTIVE_STYLE = override
        fig, ax = plots.plot_updown_bias(ds, var='PSAL', v_res=1)
        assert fig.get_facecolor() == distinct
    finally:
        plots._ACTIVE_STYLE = original
    # Default restored: a fresh figure no longer carries the override.
    fig, ax = plots.plot_updown_bias(ds, var='PSAL', v_res=1)
    assert fig.get_facecolor() != distinct
