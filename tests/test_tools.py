from glidertest import fetchers, tools
import matplotlib.pyplot as plt


def test_plots():
    ds = fetchers.load_sample_dataset()
    tools.plot_basic_vars(ds, end_prof=int(ds.PROFILE_NUMBER.max()))


def test_up_down_bias():
    ds = fetchers.load_sample_dataset()
    fig, ax = plt.subplots()
    tools.updown_bias(ds, ax, var='PSAL', v_res=0.1)


def test_chl():
    ds = fetchers.load_sample_dataset()
    tools.chl_first_check(ds)


def test_quench():
    ds = fetchers.load_sample_dataset()
    if not "TIME" in ds.indexes.keys():
        ds = ds.set_xindex('TIME')
    tools.check_npq(ds, start_time='2023-09-06', end_time='2023-09-08', sel_day=0)