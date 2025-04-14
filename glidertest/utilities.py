import numpy as np
import pandas as pd
import xarray as xr
from pandas import DataFrame
from skyfield import almanac
from skyfield import api
import gsw
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates


def _check_necessary_variables(ds: xr.Dataset, vars: list):
    """
    Checks that all of a list of variables are present in a dataset.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset that should be checked
    vars: list
        List of variables

    Raises
    ------
    KeyError:
        Raises an error if all vars not present in ds

    Notes
    -----
    Original Author: Callum Rollo
    """
    missing_vars = set(vars).difference(set(ds.variables))
    if missing_vars:
        msg = f"Required variables {list(missing_vars)} do not exist in the supplied dataset."
        raise KeyError(msg)


def _calc_teos10_variables(ds):
    """
    Calculates TEOS 10 variables not present in the dataset
    :param ds:
    :return:
    """
    _check_necessary_variables(ds, ['DEPTH', 'LONGITUDE', 'LATITUDE', 'TEMP', 'PSAL'])
    if 'DENSITY' not in ds.variables:
        SA = gsw.SA_from_SP(ds.PSAL, ds.DEPTH, ds.LONGITUDE, ds.LATITUDE)
        CT = gsw.CT_from_t(SA, ds.TEMP, ds.DEPTH)
        ds['DENSITY'] = ('N_MEASUREMENTS', gsw.rho(SA, CT, ds.DEPTH).values)
    return ds

def _time_axis_formatter(ax, ds, format_x_axis=True):
    """
    Apply time formatter and locator to an axis.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to format.
    ds : xarray.Dataset
        Dataset containing TIME coordinate.
    format_x_axis : bool, default=True
        Whether to format the x-axis (True) or y-axis (False).
    """
    formatter, locator, xlabel = _select_time_formatter_and_locator(ds)

    axis = ax.xaxis if format_x_axis else ax.yaxis
    axis.set_major_formatter(formatter)
    if locator:
        axis.set_major_locator(locator)

    if format_x_axis:
        ax.xaxis.set_major_formatter(formatter)
        if locator:
            ax.xaxis.set_major_locator(locator)
        ax.set_xlabel(xlabel)
    else:
        ax.yaxis.set_major_formatter(formatter)
        if locator:
            ax.yaxis.set_major_locator(locator)
        ax.set_ylabel(xlabel)
        

def _select_time_formatter_and_locator(ds):
    """
    Select appropriate time formatter and locator based on time range.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing TIME coordinate.

    Returns
    -------
    formatter : matplotlib.dates.DateFormatter
        Formatter for time axis or colorbar.
    locator : matplotlib.dates.Locator or None
        Locator for axis ticks (None if not needed).
    xlabel : str
        Label for the axis (optional for colorbars).
    """
    start_time = ds.TIME.min().values
    end_time = ds.TIME.max().values
    time_range_hours = (end_time - start_time) / np.timedelta64(1, 'h')

    if time_range_hours < 24:
        formatter = DateFormatter('%H:%M')
        # Dynamic interval based on time range
        if time_range_hours <= 6:
            interval = 1
        elif time_range_hours <= 12:
            interval = 2
        else:
            interval = 3
        locator = mdates.HourLocator(byhour=range(0, 24, 2))
        start_date = pd.to_datetime(start_time).strftime('%Y-%b-%d')
        end_date = pd.to_datetime(end_time).strftime('%Y-%b-%d')
        xlabel = f'Time [UTC] ({start_date})' if start_date == end_date else f'Time [UTC] ({start_date} to {end_date})'
    elif time_range_hours < 24 * 7:
        formatter = DateFormatter('%d-%b')
        locator = mdates.DayLocator(interval=1)
        start_date = pd.to_datetime(start_time).strftime('%Y-%b-%d')
        end_date = pd.to_datetime(end_time).strftime('%Y-%b-%d')
        xlabel = f'Time [UTC] ({start_date})' if start_date == end_date else f'Time [UTC] ({start_date} to {end_date})'
    else:
        formatter = DateFormatter('%d-%b')
        locator = None
        xlabel = 'Time [UTC]'

    return formatter, locator, xlabel


def construct_2dgrid(x, y, v, xi=1, yi=1):
    """
    Constructs a 2D gridded representation of input data based on specified resolutions.

    Parameters
    ----------
    x : array-like  
        Input data representing the x-dimension.  
    y : array-like  
        Input data representing the y-dimension.  
    v : array-like  
        Input data representing the z-dimension (values to be gridded).  
    xi : int or float, optional, default=1  
        Resolution for the x-dimension grid spacing.  
    yi : int or float, optional, default=1  
        Resolution for the y-dimension grid spacing.  

    Returns
    -------
    grid : numpy.ndarray  
        Gridded representation of the z-values over the x and y space.  
    XI : numpy.ndarray  
        Gridded x-coordinates corresponding to the specified resolution.  
    YI : numpy.ndarray  
        Gridded y-coordinates corresponding to the specified resolution. 

    Notes
    -----
    Original Author: Bastien Queste  
    [Source Code](https://github.com/bastienqueste/gliderad2cp/blob/de0652f70f4768c228f83480fa7d1d71c00f9449/gliderad2cp/process_adcp.py#L140)  
    """

    if np.size(xi) == 1:
        xi = np.arange(np.nanmin(x), np.nanmax(x) + xi, xi)
    if np.size(yi) == 1:
        yi = np.arange(np.nanmin(y), np.nanmax(y) + yi, yi)
    raw = pd.DataFrame({'x': x, 'y': y, 'v': v}).dropna()
    grid = np.full([np.size(xi), np.size(yi)], np.nan)
    raw['xbins'], xbin_iter = pd.cut(raw.x, xi, retbins=True, labels=False)
    raw['ybins'], ybin_iter = pd.cut(raw.y, yi, retbins=True, labels=False)
    _tmp = raw.groupby(['xbins', 'ybins'])['v'].agg('median')
    grid[_tmp.index.get_level_values(0).astype(int), _tmp.index.get_level_values(1).astype(int)] = _tmp.values
    YI, XI = np.meshgrid(yi, xi)
    return grid, XI, YI

def compute_sunset_sunrise(time, lat, lon):
    """
    Calculates the local sunrise/sunset of the glider location from GliderTools.

    The function uses the Skyfield package to calculate the sunrise and sunset
    times using the date, latitude and longitude. The times are returned
    rather than day or night indices, as it is more flexible for the quenching
    correction.

    Parameters
    ----------
    time: numpy.ndarray or pandas.Series
        The date & time array in a numpy.datetime64 format.
    lat: numpy.ndarray or pandas.Series
        The latitude of the glider position.
    lon: numpy.ndarray or pandas.Series
        The longitude of the glider position.

    Returns
    -------
    sunrise: numpy.ndarray
        An array of the sunrise times.
    sunset: numpy.ndarray
        An array of the sunset times.

    Notes
    -----
    Original Author: Function from GliderTools 
    [Source Code](https://github.com/GliderToolsCommunity/GliderTools/blob/master/glidertools/optics.py)
    """

    ts = api.load.timescale()
    eph = api.load("de421.bsp")

    df = DataFrame.from_dict(dict([("time", time), ("lat", lat), ("lon", lon)]))

    # set days as index
    df = df.set_index(df.time.values.astype("datetime64[D]"))

    # groupby days and find sunrise/sunset for unique days
    grp_avg = df.groupby(df.index).mean(numeric_only=False)
    date = grp_avg.index

    time_utc = ts.utc(date.year, date.month, date.day, date.hour)
    time_utc_offset = ts.utc(
        date.year, date.month, date.day + 1, date.hour
    )  # add one day for each unique day to compute sunrise and sunset pairs

    bluffton = []
    for i in range(len(grp_avg.lat)):
        bluffton.append(api.wgs84.latlon(grp_avg.lat.iloc[i], grp_avg.lon.iloc[i]))
    bluffton = np.array(bluffton)

    sunrise = []
    sunset = []
    for n in range(len(bluffton)):

        f = almanac.sunrise_sunset(eph, bluffton[n])
        t, y = almanac.find_discrete(time_utc[n], time_utc_offset[n], f)

        if not t:
            if f(time_utc[n]):  # polar day
                sunrise.append(
                    pd.Timestamp(
                        date[n].year, date[n].month, date[n].day, 0, 1
                    ).to_datetime64()
                )
                sunset.append(
                    pd.Timestamp(
                        date[n].year, date[n].month, date[n].day, 23, 59
                    ).to_datetime64()
                )
            else:  # polar night
                sunrise.append(
                    pd.Timestamp(
                        date[n].year, date[n].month, date[n].day, 11, 59
                    ).to_datetime64()
                )
                sunset.append(
                    pd.Timestamp(
                        date[n].year, date[n].month, date[n].day, 12, 1
                    ).to_datetime64()
                )

        else:
            sr = t[y == 1]  # y=1 sunrise
            sn = t[y == 0]  # y=0 sunset

            sunup = pd.to_datetime(sr.utc_iso()).tz_localize(None)
            sundown = pd.to_datetime(sn.utc_iso()).tz_localize(None)

            # this doesn't look very efficient at the moment, but I was having issues with getting the datetime64
            # to be compatible with the above code to handle polar day and polar night

            su = pd.Timestamp(
                sunup.year[0],
                sunup.month[0],
                sunup.day[0],
                sunup.hour[0],
                sunup.minute[0],
            ).to_datetime64()

            sd = pd.Timestamp(
                sundown.year[0],
                sundown.month[0],
                sundown.day[0],
                sundown.hour[0],
                sundown.minute[0],
            ).to_datetime64()

            sunrise.append(su)
            sunset.append(sd)

    sunrise = np.array(sunrise).squeeze()
    sunset = np.array(sunset).squeeze()

    grp_avg["sunrise"] = sunrise
    grp_avg["sunset"] = sunset

    # reindex days to original dataframe as night
    df_reidx = grp_avg.reindex(df.index)
    sunrise, sunset = df_reidx[["sunrise", "sunset"]].values.T

    return sunrise, sunset

def calc_DEPTH_Z(ds):
    """
    Calculate the depth (Z position) of the glider using the gsw library to convert pressure to depth.
    
    Parameters
    ----------
    ds: xarray.Dataset
        The input dataset containing 'PRES', 'LATITUDE', and 'LONGITUDE' variables.
    
    Returns
    -------
    xarray.Dataset: 
        The dataset with an additional 'DEPTH_Z' variable.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams
    """
    _check_necessary_variables(ds, ['PRES', 'LONGITUDE', 'LATITUDE'])

    # Initialize the new variable with the same dimensions as dive_num
    ds['DEPTH_Z'] = (['N_MEASUREMENTS'], np.full(ds.dims['N_MEASUREMENTS'], np.nan))

    # Calculate depth using gsw
    depth = gsw.z_from_p(ds['PRES'], ds['LATITUDE'])
    ds['DEPTH_Z'] = depth

    # Assign the calculated depth to a new variable in the dataset
    ds['DEPTH_Z'].attrs = {
        "units": "meters",
        "positive": "up",
        "standard_name": "depth",
        "comment": "Depth calculated from pressure using gsw library, positive up.",
    }
    
    return ds

label_dict={
    "PSAL": {
        "label": "Practical salinity",
        "units": "PSU"},
    "TEMP": {
        "label": "Temperature",
        "units": "°C"},
    "DENSITY":{
        "label": "In situ density",
        "units": "kg m⁻³"
    },
    "SIGMA": {
        "label": "Sigma-t",
        "units": "kg m⁻³"
    },
    "DOXY": {
        "label": "Dissolved oxygen",
        "units": "mmol m⁻³"
    },
    "SA":{
        "label": "Absolute salinity",
        "units": "g kg⁻¹"
    },
    "CHLA":{
        "label": "Chlorophyll",
        "units": "mg m⁻³"
    },
    "CNDC":{
        "label": "Conductivity",
        "units": "mS cm⁻¹"
    },
    "DPAR":{
        "label": "Irradiance PAR",
        "units": "μE cm⁻² s⁻¹"
    },
    "BBP700":{
        "label": "Red backscatter, b${bp}$(700)",
        "units": "m⁻¹"
    }
}

def plotting_labels(var: str):
    """
    Retrieves the label associated with a variable from a predefined dictionary.

    This function checks if the given variable `var` exists as a key in the `label_dict` dictionary.
    If found, it returns the associated label from `label_dict`. If not, it returns the variable name itself as the label.

    Parameters
    ----------
    var: str
        The variable (key) whose label is to be retrieved.

    Returns
    -------
    str: 
        The label corresponding to the variable `var`. If the variable is not found in `label_dict`,
        the function returns the variable name as the label.

    Notes
    -----
    Original Author: Chiara Monforte
    """
    if var in label_dict:
        label = f'{label_dict[var]["label"]}'
    else:
        label= f'{var}'
    return label
def plotting_units(ds: xr.Dataset,var: str):
    """
    Retrieves the units associated with a variable from a dataset or a predefined dictionary.

    This function checks if the given variable `var` exists as a key in the `label_dict` dictionary.
    If found, it returns the associated units from `label_dict`. If not, it returns the units of the variable
    from the dataset `ds` using the `var` key.

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset containing the variable `var`.
    var: str 
        The variable (key) whose units are to be retrieved.

    Returns
    -------
    str: 
        The units corresponding to the variable `var`. If the variable is found in `label_dict`,
        the associated units will be returned. If not, the function returns the units from `ds[var]`.

    Notes
    -----
    Original Author: Chiara Monforte
    """
    if var in label_dict:
        return f'{label_dict[var]["units"]}'
    elif 'units' in ds[var].attrs:
        return f'{ds[var].units}'
    else:
        return ""
def group_by_profiles(ds, variables=None):
    """
    Group glider dataset by the dive profile number.

    This function groups the dataset by the `PROFILE_NUMBER` column, where each group corresponds to a single profile.
    The resulting groups can be evaluated statistically, for example using methods like `pandas.DataFrame.mean` or
    other aggregation functions. To filter a specific profile, use `xarray.Dataset.where` instead.

    Parameters
    ----------
    ds : xarray.Dataset
        A 1-dimensional glider dataset containing profile information.
    variables : list of str, optional
        A list of variable names to group by, if only a subset of the dataset should be included in the grouping.
        Grouping by a subset is more memory-efficient and faster.

    Returns
    -------
    profiles : pandas.core.groupby.DataFrameGroupBy
        A pandas `GroupBy` object, grouped by the `PROFILE_NUMBER` of the glider dataset.
        This can be further aggregated using methods like `mean` or `sum`.

    Notes
    -----
    This function is based on the original GliderTools implementation and was modified by
    Chiara Monforte to ensure compliance with the OG1 standards.
    [Source Code](https://github.com/GliderToolsCommunity/GliderTools/blob/master/glidertools/utils.py)
    """
    ds = ds.reset_coords().to_pandas().reset_index().set_index("PROFILE_NUMBER")
    if variables:
        return ds[variables].groupby("PROFILE_NUMBER")
    else:
        return ds.groupby("PROFILE_NUMBER")