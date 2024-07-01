from glidertest import fetchers


def test_demo_datasets():
    for remote_ds_name in fetchers.data_source_og.registry.keys():
        ds = fetchers.load_sample_dataset(dataset_name=remote_ds_name)
