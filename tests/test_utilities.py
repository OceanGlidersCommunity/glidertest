import cmocean.cm as cmo

from glidertest import utilities

# These run on the committed sea045 subset (opened fresh per test, since several add derived
# variables to the dataset). See tests/data/README.md.


def test_utilitiesmix(fresh_subset):
    ds = fresh_subset
    utilities._check_necessary_variables(ds, ['PROFILE_NUMBER', 'DEPTH', 'TEMP', 'PSAL', 'LATITUDE', 'LONGITUDE'])
    ds = utilities._calc_teos10_variables(ds)
    p = 1
    z = 1
    tempG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds.TEMP, p, z)
    denG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds.DENSITY, p, z)
    utilities.group_by_profiles(ds, ['DENSITY', "DEPTH"])


def test_sunset_sunrise(fresh_subset):
    ds = fresh_subset
    sunrise, sunset = utilities.compute_sunset_sunrise(ds.TIME, ds.LATITUDE, ds.LONGITUDE)


def test_depth_z(fresh_subset):
    ds = fresh_subset
    assert 'DEPTH_Z' not in ds.variables
    ds = utilities.calc_DEPTH_Z(ds)
    assert 'DEPTH_Z' in ds.variables
    assert ds.DEPTH_Z.min() < -50


def test_labels(fresh_subset):
    ds = fresh_subset
    var = 'PITCH'
    label = utilities.plotting_labels(var)
    assert label == 'PITCH'
    colormap = utilities.plotting_colormap(var)
    assert colormap == cmo.delta
    var = 'TEMP'
    label = utilities.plotting_labels(var)
    assert label == 'Temperature'
    unit = utilities.plotting_units(ds, var)
    assert unit == 'Celsius'
    colormap = utilities.plotting_colormap(var)
    assert colormap == cmo.thermal


def test_bin_profile(fresh_subset):
    ds = fresh_subset
    prof_num = ds.PROFILE_NUMBER[0].values
    ds_profile = ds.where(ds.PROFILE_NUMBER == prof_num, drop=True)
    utilities.bin_profile(ds_profile, vars=['TEMP'], binning=5)
