from glidertest import fetchers
import pytest


def test_source():
    source = fetchers.data_source_og
    assert len(source.registry.keys()) > 4


@pytest.mark.slow
def test_demo_dataset():
    # Exercises the real pooch download, so it needs the network: a slow, non-default test.
    fetchers.load_sample_dataset()


def test_missing_dataset():
    with pytest.raises(KeyError):
        fetchers.load_sample_dataset(dataset_name="non-existent dataset")
