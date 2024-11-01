from glidertest import fetchers


def test_demo_dataset():
    for remote_ds_name in fetchers.data_source_og.registry.keys():
        if not "sea055" in remote_ds_name:
            ds = fetchers.load_sample_dataset(dataset_name=remote_ds_name)