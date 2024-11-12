import pytest
from glidertest import fetchers, tools
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib
matplotlib.use('agg') # use agg backend to prevent creating plot windows during tests

def test_plots(start_prof=0, end_prof=100):
    ds = fetchers.load_sample_dataset()
    fig, ax = tools.plot_basic_vars(ds,start_prof=start_prof, end_prof=end_prof)
    assert ax[0].get_ylabel() == 'Depth (m)'
    assert ax[0].get_xlabel() == f'Average Temperature [C] \nbetween profile {start_prof} and {end_prof}'


def test_up_down_bias(v_res=1, xlabel='Salinity'):
    ds = fetchers.load_sample_dataset()
    fig, ax = plt.subplots()
    df = tools.updown_bias(ds, var='PSAL', v_res=v_res)
    bins = np.unique(np.round(ds.DEPTH,0))
    ncell = math.ceil(len(bins)/v_res)
    assert len(df) == ncell
    tools.plot_updown_bias(df, ax,  xlabel=xlabel)
    lims = np.abs(df.dc)
    assert ax.get_xlim() == (-np.nanpercentile(lims, 99.5), np.nanpercentile(lims, 99.5))
    assert ax.get_ylim() == (df.depth.max() + 1, -df.depth.max() / 30)
    assert ax.get_xlabel() == xlabel
 

def test_chl(var1='CHLA', var2='BBP700'):
    ds = fetchers.load_sample_dataset()
    ax = tools.optics_first_check(ds, var=var1)
    assert ax.get_ylabel() == var1
    ax = tools.optics_first_check(ds, var=var2)
    assert ax.get_ylabel() == var2
    with pytest.raises(KeyError) as e:
        tools.optics_first_check(ds, var='nonexistent_variable')


def test_quench_sequence(xlabel='Temperature [C]',ylim=45):
    ds = fetchers.load_sample_dataset()
    if not "TIME" in ds.indexes.keys():
        ds = ds.set_xindex('TIME')
    fig, ax = plt.subplots()
    tools.plot_section_with_srss(ds, 'CHLA', ax,ylim=ylim)
    assert ax.get_ylabel() == 'Depth [m]'
    assert ax.get_ylim() == (ylim, -ylim / 30)
    
    dayT, nightT = tools.day_night_avg(ds, sel_var='TEMP')
    assert len(nightT.dat.dropna()) > 0
    assert len(dayT.dat.dropna()) > 0
    
    fig, ax = tools.plot_daynight_avg(dayT, nightT,xlabel=xlabel) 
    assert ax.get_ylabel() == 'Depth [m]'
    assert ax.get_xlabel() == xlabel

def test_temporal_drift(var='DOXY'):
    ds = fetchers.load_sample_dataset()
    fig, ax = plt.subplots(1, 2)
    tools.check_temporal_drift(ds,var, ax)
    assert ax[1].get_ylabel() == 'Depth (m)'
    assert ax[0].get_ylabel() == var
    assert ax[1].get_xlim() == (np.nanpercentile(ds[var], 0.01), np.nanpercentile(ds[var], 99.99))
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


