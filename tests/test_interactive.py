from glidertest import fetchers, interactive
import pytest

def test_mission_map():
    ds = fetchers.load_sample_dataset()
    interactive.mission_map(ds)