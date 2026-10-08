import pooch
import xarray as xr

server = "https://erddap.observations.voiceoftheocean.org/examples/glidertest"
data_source_og = pooch.create(
    path=pooch.os_cache("glidertest"),
    base_url=server,
    registry={
        "sea045_20230530T0832_delayed.nc": "sha256:6d3075280d2e4a075d011b59486d7fa6bf616459dc92a7ece14a9d9a749f5ce4",
        "sea045_20230604T1253_delayed.nc": "sha256:9d71b5fb580842e8296b447041cd5b544c8b4336df40a8f3c2e09279f436f502",
        "sea055_20220104T1536_delayed.nc": "sha256:7f72f8a0398c3d339687d7b7dcf0311036997f6855ed80cae5bbf877e09975a6",
        "sea055_20240628T0856_delayed.nc": "sha256:7756a0034dba7241dd626b7e21cfb1d0bce634865a5f3dceea391faaa3195546",
        "sea055_20241009T1345_delayed.nc": "sha256:cd411c326b2734efd73a02b97e43c019dccbe1825447542b1d7cbfa4130639e4",
        'sea063_20230811T1657_delayed.nc': 'sha256:8d4bf2555f226a35078290961c49749389fb0f9075917b263b3231823f87fc35',
        'sea067_20221113T0853_delayed.nc': 'sha256:9f7bcbf516357d122bc73285dfdcd2cf747cead92a9dfc4438fbf0fc1c6af5d2',
        'sea069_20230726T0628_delayed.nc': 'sha256:2459c0e2d6fef1241eba0eef08da038329c20d770b0476195830d514a0ccfaea',
        'sea076_20230620T1201_delayed.nc': 'sha256:fffcf3ed2d0761d88fd9caf2e4be708ab626e66b3c3c9e1b3558b2b4e2806413',
        'sea077_20220906T0748_delayed.nc': 'sha256:3c2e9def777a2e230cc1b2114ca069dcbdb3c58ada2794a8305a781f3f76da01',
        'sea077_20230316T1019_delayed.nc': 'sha256:a146247dd33d38ab7b1d379481f66c749fdf16db31aacad1fe70287c55b359a6',
        'sg014_20040924T182454_delayed.nc': 'sha256:c9fca08ce676573224c04512f4d5bfe251d0419478ee433dfefa03aa70e2eb9a',
        'sg014_20040924T182454_delayed_subset.nc': 'sha256:0e97a4107364d27364d076ed8007f08c891b2015b439cf524a44612de0a1a2ea',
        'sg015_20050213T230253_delayed.nc': 'sha256:ca64e33e9f317e1fc3442e74485a9bf5bb1b4a81b5728e9978847b436e0586ab',
        "sg638_20191205T121839_delayed.nc": "sha256:d67021237bfe707cc0cd11f09a0d009f15504eefbaf1bd45dfbba8a2700a229a",
    },
)


def load_sample_dataset(dataset_name="sea045_20230604T1253_delayed.nc"):
    """Download sample datasets for use with glidertest

    Parameters
    ----------
    dataset_name: str, optional
        Default is "sea045_20230604T1253_delayed.nc".

    Raises
    ------
    ValueError: 
        If the requests dataset is not known, raises a value error

    Returns
    -------
    xarray.Dataset: 
        Requested sample dataset
    """
    if dataset_name in data_source_og.registry.keys():
        file_path = data_source_og.fetch(dataset_name)
        return xr.open_dataset(file_path)
    else:
        msg = f"Requested sample dataset {dataset_name} not known. Specify one of the following available datasets: {list(data_source_og.registry.keys())}"
        raise KeyError(msg)
