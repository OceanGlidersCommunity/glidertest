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
    tools.plot_section_with_srss(ds, 'CHLA', ax,start_time = '2023-06-04', end_time = '2023-06-06', ylim=35)
    dayT, nightT = tools.day_night_avg(ds, sel_var='TEMP',start_time = '2023-06-04', end_time = '2023-06-06')
    fig, ax = plt.subplots()
    tools.plot_daynight_avg( dayT, nightT,ax,sel_day='2023-06-04', xlabel='Temperature [C]') 

def test_temporal_drift():
    ds = fetchers.load_sample_dataset()
    fig, ax = plt.subplots(1, 2)
    tools.check_temporal_drift(ds,'DOXY', ax)
    tools.check_temporal_drift(ds,'CHLA')
        
def test_profile_check():
    ds = fetchers.load_sample_dataset()
    tools.check_monotony(ds.PROFILE_NUMBER)
    tools.plot_profIncrease(ds)
