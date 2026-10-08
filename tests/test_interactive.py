import pytest

from glidertest import fetchers, interactive

# Jupyter-only outputs (folium maps, ipywidgets) run on the committed sea045 subset. The day/night
# average needs night profiles the subset lacks, so it stays on the full sample (slow).


def test_mission_map(fresh_subset):
    interactive.mission_map(fresh_subset)


def test_interactive_profiles(fresh_subset):
    interactive.profile(fresh_subset)
    interactive.ts_plot(fresh_subset)
    interactive.up_down_bias(fresh_subset)


@pytest.mark.slow
def test_interactive_daynight():
    # daynight_avg averages day vs night, so it needs night profiles: the full sample.
    ds = fetchers.load_sample_dataset()
    interactive.daynight_avg(ds)
