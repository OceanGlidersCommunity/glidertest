from glidertest import fetchers, interactive
import matplotlib
matplotlib.use('agg')  # use agg backend to prevent creating plot windows during tests

ds = fetchers.load_sample_dataset()


def test_mission_map():
    interactive.mission_map(ds)

def test_interactive_profiles():
    interactive.profile(ds)
    interactive.ts_plot(ds)
    interactive.daynight_avg(ds)
    interactive.up_down_bias(ds)
    