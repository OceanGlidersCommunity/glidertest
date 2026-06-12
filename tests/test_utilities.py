import pytest
from glidertest import fetchers, utilities
import matplotlib
matplotlib.use('agg')  # use agg backend to prevent creating plot windows during tests

def test_utilitiesmix():
    ds = fetchers.load_sample_dataset()
    utilities._check_necessary_variables(ds, ['PROFILE_NUMBER', 'DEPTH', 'TEMP', 'PSAL', 'LATITUDE', 'LONGITUDE'])
    ds = utilities._calc_teos10_variables(ds)
    p = 1
    z = 1
    tempG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds.TEMP, p, z)
    denG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds.DENSITY, p, z)
    groups = utilities.group_by_profiles(ds, ['DENSITY', "DEPTH"])

def test_sunset_sunrise():
    ds = fetchers.load_sample_dataset()
    sunrise, sunset = utilities.compute_sunset_sunrise(ds.TIME, ds.LATITUDE, ds.LONGITUDE)


def test_depth_z():
    ds = fetchers.load_sample_dataset()
    assert 'DEPTH_Z' not in ds.variables
    ds = utilities.calc_DEPTH_Z(ds)
    assert 'DEPTH_Z' in ds.variables
    assert ds.DEPTH_Z.min() < -50

def test_labels():
    ds = fetchers.load_sample_dataset()
    var = 'PITCH'
    label = utilities.plotting_labels(var)
    assert label == 'PITCH'
    var = 'TEMP'
    label = utilities.plotting_labels(var)
    assert label == 'Temperature'
    unit=utilities.plotting_units(ds, var)
    #assert unit == "°C"

def test_bin_profile():
    ds = fetchers.load_sample_dataset()
    prof_num = ds.PROFILE_NUMBER[0].values
    ds_profile = ds.where(ds.PROFILE_NUMBER == prof_num, drop=True)
    utilities.bin_profile(ds_profile, vars=['TEMP'], binning=5)