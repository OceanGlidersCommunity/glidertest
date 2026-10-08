"""Build the committed test subsets from the pooch sample registry.

Reproducible: ``python tests/data/make_subset.py``. Each full mission is cut to its first N
profiles, reduced to a keep-list of variables (the ones the report uses, plus a few it does not
plot so the inventory page has something to list), downcast to float32 except time and position,
and written compressed. The profile count and keep-list are tuned against the panel-parity check so
every diagnostic still renders a real panel except day/night (needs the multi-day full sample).
Real VOTO / Seaglider sample data; provenance and credit in the README.
"""

from pathlib import Path

import numpy as np

from glidertest import fetchers

HERE = Path(__file__).parent
N_SEA045 = 12
N_SG014 = 6

# float64 is kept only for time and position; geophysical data goes to float32. TIME is datetime64
# and keeps its own encoding (CF units), so it is never touched here.
FLOAT64_KEEP = {"TIME_GPS", "LATITUDE", "LONGITUDE", "LATITUDE_GPS", "LONGITUDE_GPS"}

# Measurement variables to keep. Everything carrying N_MEASUREMENTS and not listed is dropped;
# scalars (SENSOR_*, PLATFORM_*, …) are always kept.
KEEP_SEA045 = {
    "TEMP", "PSAL", "DOXY", "CHLA", "CNDC", "PRES", "DEPTH",
    "TIME", "LATITUDE", "LONGITUDE", "PROFILE_NUMBER", "PROFILE_DIRECTION", "DIVE_NUM",
    "LATITUDE_GPS", "LONGITUDE_GPS", "TIME_GPS",
    "TEMP_QC", "PSAL_QC", "DOXY_QC", "CHLA_QC", "CNDC_QC",
    "DENSITY", "BBP700", "VOLTAGE",  # not plotted — kept so the inventory lists some extra variables
}
KEEP_SG014 = {
    "TEMP", "PSAL", "PRES", "DEPTH", "TIME", "LATITUDE", "LONGITUDE", "PROFILE_NUMBER",
    "LATITUDE_GPS", "LONGITUDE_GPS",
    "GLIDER_VERT_VELO_MODEL", "GLIDER_HORZ_VELO_MODEL", "GLIDE_ANGLE", "GLIDE_SPEED", "PHASE",
    "TEMP_QC", "PSAL_QC",
}


def cut(ds, n_profiles, keep):
    """Reduce *ds* to its first *n_profiles*, the *keep* measurement vars, and float32 geophysics."""
    pn = np.asarray(ds["PROFILE_NUMBER"].values)
    finite = np.isfinite(pn)
    keep_profs = np.unique(pn[finite])[:n_profiles]
    sub = ds.isel(N_MEASUREMENTS=np.isin(pn, keep_profs))  # scalars have no N_MEASUREMENTS dim
    drop = [v for v in sub.data_vars if "N_MEASUREMENTS" in sub[v].dims and v not in keep]
    sub = sub.drop_vars(drop)
    for v in sub.data_vars:
        if sub[v].dtype == np.float64 and v not in FLOAT64_KEEP:
            sub[v] = sub[v].astype("float32")
            sub[v].encoding.pop("dtype", None)  # so the cast lands on disk, not re-upcast to float64
    sub.attrs = dict(ds.attrs)
    return sub


def write(ds, path):
    """Write *ds* to *path* (h5netcdf) with gzip on the measurement variables."""
    enc = {
        v: {"compression": "gzip", "compression_opts": 4}
        for v in ds.data_vars
        if ds[v].ndim > 0
    }
    ds.to_netcdf(path, engine="h5netcdf", encoding=enc)


def main():
    """Build sea045_subset.nc (CTD+oxygen+optics) and sg014_subset.nc (flight)."""
    sea = fetchers.load_sample_dataset("sea045_20230604T1253_delayed.nc")
    write(cut(sea, N_SEA045, KEEP_SEA045), HERE / "sea045_subset.nc")
    sg = fetchers.load_sample_dataset("sg014_20040924T182454_delayed_subset.nc")
    write(cut(sg, N_SG014, KEEP_SG014), HERE / "sg014_subset.nc")


if __name__ == "__main__":
    main()
