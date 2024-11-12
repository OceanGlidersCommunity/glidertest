import pooch
import xarray as xr

server = "https://callumrollo.com/files"
data_source_og = pooch.create(
    path=pooch.os_cache("glidertest"),
    base_url=server,
    registry={
        "sea055_20220104T1536_delayed.nc": "sha256:7f72f8a0398c3d339687d7b7dcf0311036997f6855ed80cae5bbf877e09975a6",
        "sea045_20230530T0832_delayed.nc": "sha256:9a97b5b212e9ad751909e175bc01fdc75fd7b7fd784cc59d8e9f29e8bdfb829f",
        "sg015_20050213T230253_delayed.nc": "sha256:ca64e33e9f317e1fc3442e74485a9bf5bb1b4a81b5728e9978847b436e0586ab",
        "sg014_20040924T182454_delayed.nc": "sha256:c9fca08ce676573224c04512f4d5bfe251d0419478ee433dfefa03aa70e2eb9a",
        "sg014_20040924T182454_delayed_subset.nc": "sha256:0e97a4107364d27364d076ed8007f08c891b2015b439cf524a44612de0a1a2ea",
    },
)


def load_sample_dataset(dataset_name="sea045_20230530T0832_delayed.nc"):
    """Download sample datasets for use with glidertest

    Args:
        dataset_name (str, optional): _description_. Defaults to "sea045_20230530T0832_delayed.nc".

    Raises:
        ValueError: If the requests dataset is not known, raises a value error

    Returns:
        xarray.Dataset: Requested sample dataset
    """
    if dataset_name in data_source_og.registry.keys():
        file_path = data_source_og.fetch(dataset_name)
        return xr.open_dataset(file_path)
    else:
        msg = f"Requested sample dataset {dataset_name} not known. Specify one of the following available datasets: {list(data_source_og.registry.keys())}"
        raise KeyError(msg)
