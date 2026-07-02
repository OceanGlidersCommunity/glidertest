from glidertest import fetchers, interactive
import pytest

def test_mission_map():
    ds = fetchers.load_sample_dataset()
    interactive.mission_map(ds)
    dsa = fetchers.load_sample_dataset(dataset_name="sg638_20191205T121839_delayed.nc")
    dsa['DIVE_NUM'] = dsa['divenum'].copy()
    interactive.mission_map([ds, dsa])
