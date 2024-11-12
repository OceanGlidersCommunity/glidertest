from glidertest import fetchers
import pytest


def test_source():
    source = fetchers.data_source_og
    assert len(source.registry.keys()) > 4


def test_demo_dataset():
    fetchers.load_sample_dataset()


def test_missing_dataset():
    with pytest.raises(KeyError) as e:
        fetchers.load_sample_dataset(dataset_name="non-existent dataset")
