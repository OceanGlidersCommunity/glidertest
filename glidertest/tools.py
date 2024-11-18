import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import gsw
import warnings
from glidertest import utilities


def quant_updown_bias(ds, var='PSAL', v_res=1):
    """
    This function computes up and downcast averages for a specific variable and outputs the difference between them

    Parameters
    ----------
    ds: xarray on OG1 format containing at least depth, profile number and the selected variable.
        Data should not be gridded.
    var: Selected variable
    v_res: Vertical resolution for the gridding
                
    Returns
    -------
    df: pandas dataframe containing dc (Dive - Climb average), cd (Climb - Dive average) and depth

    Original author
    ----------------
    Chiara Monforte
    """
    utilities._check_necessary_variables(ds, ['PROFILE_NUMBER', 'DEPTH', var])
    p = 1  # Horizontal resolution
    z = v_res  # Vertical resolution

    if var in ds.variables:
        varG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds[var], p, z)

        grad = np.diff(varG, axis=0)  # Horizontal gradients
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dc = np.nanmean(grad[0::2, :], axis=0)  # Dive - CLimb
            cd = np.nanmean(grad[1::2, :], axis=0)  # Climb - Dive

        df = pd.DataFrame(data={'dc': dc, 'cd': cd, 'depth': depthG[0, :]})
    else:
        print(f'{var} is not in the dataset')
        df = pd.DataFrame()
    return df

def compute_daynight_avg(ds, sel_var='CHLA', start_time=None, end_time=None, start_prof=None, end_prof=None):
    """
    This function computes night and day averages for a selected variable over a specific period of time or a specific series of dives
    Data in divided into day and night using the sunset and sunrise time as described in the above function sunset_sunrise from GliderTools
    Parameters
    ----------
    ds: xarray on OG1 format containing at least time, depth, latitude, longitude and the selected variable. 
        Data should not be gridded.
    sel_var: variable to use to compute the day night averages
    start_time: Start date of the data selection. As missions can be long and can make it hard to visualise NPQ effect,
                we recommend selecting small section of few days to a few weeks. Defaults to the central week of the deployment
    end_time: End date of the data selection. As missions can be long and can make it hard to visualise NPQ effect,
                we recommend selecting small section of few days to a few weeks. Defaults to the central week of the deployment
    start_prof: Start profile of the data selection. If no profile is specified, the specified time selection will be used
                or the central week of the deployment.
                It is important to have a large enough number of dives to have some day and night data otherwise the function will not run
    end_prof:  End profile of the data selection. If no profile is specified, the specified time selection will be used
            or the central week of the deployment.
    It is important to have a large enough number of dives to have some day and night data otherwise the function will not run
                
    Returns
    -------
    day_av: pandas.Dataframe
        A dataframe with the day averages of the selected variable with the following columns:
            batch: Number representing the grouping for each day. This number can represent the number of the day in chronological order
            depth: Depth values for the average
            dat: Average value for the selected variable
            day: Actual date for the batch 
    night_av: pandas.Dataframe
        A dataframe with the night averages of the selected variable with the following columns:
            batch: Number representing the grouping for each day. This number can represent the number of the day in chronological order
            depth: Depth values for the average
            dat: Average value for the selected variable
            day: Actual date for the batch 
    
    Original author
    ----------------
    Chiara Monforte

    """
    utilities._check_necessary_variables(ds, ['TIME', sel_var, 'DEPTH'])
    if "TIME" not in ds.indexes.keys():
        ds = ds.set_xindex('TIME')

    if not start_time:
        start_time = ds.TIME.mean() - np.timedelta64(3, 'D')
    if not end_time:
        end_time = ds.TIME.mean() + np.timedelta64(3, 'D')

    if start_prof and end_prof:
        t1 = ds.TIME.where(ds.PROFILE_NUMBER==start_prof).dropna(dim='N_MEASUREMENTS')[0]
        t2 = ds.TIME.where(ds.PROFILE_NUMBER==end_prof).dropna(dim='N_MEASUREMENTS')[-1]
        ds_sel = ds.sel(TIME=slice(t1,t2))
    else:
        ds_sel = ds.sel(TIME=slice(start_time, end_time))
    sunrise, sunset = utilities.compute_sunset_sunrise(ds_sel.TIME, ds_sel.LATITUDE, ds_sel.LONGITUDE)

    # creating batches where one batch is a night and the following day
    day = (ds_sel.TIME > sunrise) & (ds_sel.TIME < sunset)
    # find day and night transitions
    daynight_transitions = np.abs(np.diff(day.astype(int)))
    # get the cumulative sum of daynight to generate separate batches for day and night
    daynight_batches = daynight_transitions.cumsum()
    batch = np.r_[0, daynight_batches // 2]

    # Create day and night averages to then have easy to plot
    df = pd.DataFrame(np.c_[ds_sel[sel_var], day, batch, ds_sel['DEPTH']], columns=['dat', 'day', 'batch', 'depth'])
    ave = df.dat.groupby([df.day, df.batch, np.around(df.depth)]).mean()
    day_av = ave[1].to_frame().reset_index()
    night_av = ave[0].to_frame().reset_index()
    #Assign date value

    for i in np.unique(day_av.batch):
        date_val = str(ds_sel.TIME.where(batch == i).dropna(dim='N_MEASUREMENTS')[-1].values)[:10]
        day_av.loc[np.where(day_av.batch == i)[0], 'date'] = date_val
        night_av.loc[np.where(night_av.batch == i)[0], 'date'] = date_val
    return day_av, night_av


def check_monotony(da):
    """
    This function check weather the selected variable over the mission is monotonically increasing or not. This is developed in particular for profile number.
    If the profile number is not monotonically increasing, this may mean that whatever function was used to assign the profile number may have misassigned at some points.
    
    Parameters
    ----------
    da: xarray.DataArray on OG1 format. Data should not be gridded.

    Returns
    -------
    It will print a sentence stating whether data is

    Original author
    ----------------
    Chiara Monforte

    """
    if not pd.Series(da).is_monotonic_increasing:
        print(f'{da.name} is not always monotonically increasing')
        return False
    else:
        print(f'{da.name} is always monotonically increasing')
        return True

def calc_w_meas(ds):
    """
    Calculate the vertical velocity of a glider using changes in pressure with time.

    Parameters
    ----------
    ds (xarray.Dataset): Dataset containing 'DEPTH' and 'TIME'.
    - DEPTH (array-like): Array of depth measurements
    - TIME (array-like): Array of time stamps
    
    Returns
    -------
    ds (xarray.Dataset): Containing the new variable
    - GLIDER_VERT_VELO_DZDT (array-like): with vertical velocities calculated from dz/dt

    Original author
    ----------------
    Eleanor Frajka-Williams
    """
    utilities._check_necessary_variables(ds, ['TIME'])
    # Ensure inputs are numpy arrays
    time = ds.TIME.values
    if 'DEPTH_Z' not in ds.variables and all(var in ds.variables for var in ['PRES', 'LATITUDE', 'LONGITUDE']):
        ds = utilities.calc_DEPTH_Z(ds)
    depth = ds.DEPTH_Z.values

    # Calculate the centered differences in pressure and time, i.e. instead of using neighboring points, 
    # use the points two steps away.  This has a couple of advantages: one being a slight smoothing of the
    # differences, and the other that the calculated speed will be the speed at the midpoint of the two 
    # points.
    # For data which are evenly spaced in time, this will be equivalent to a centered difference.
    # For data which are not evenly spaced in time, i.e. when a Seaglider sample rate changes from 5 
    # seconds to 10 seconds, there may be some uneven weighting of the differences.
    delta_z_meters = (depth[2:] - depth[:-2]) 
    delta_time_datetime64ns = (time[2:] - time[:-2]) 
    delta_time_sec = delta_time_datetime64ns / np.timedelta64(1, 's')  # Convert to seconds

    # Calculate vertical velocity (rate of change of pressure with time)
    vertical_velocity = delta_z_meters / delta_time_sec

    # Pad the result to match the original array length
    vertical_velocity = np.pad(vertical_velocity, (1, 1), 'edge') 

    # No - Convert vertical velocity from m/s to cm/s
    vertical_velocity = vertical_velocity 

    # Add vertical velocity to the dataset
    ds = ds.assign(GLIDER_VERT_VELO_DZDT=(('N_MEASUREMENTS'), vertical_velocity,  {'long_name': 'glider_vertical_speed_from_pressure', 'units': 'm s-1'}))

    return ds

def calc_w_sw(ds):
    """
    Calculate the vertical seawater velocity and add it to the dataset.

    Parameters
    ----------
    ds (xarray.Dataset): Dataset containing 'VERT_GLIDER_SPEED' and 'VERT_SPEED_DZDT'.

    Returns
    -------
    ds (xarray.Dataset): Dataset with the new variable 'VERT_SW_SPEED', which is the inferred vertical seawater velocity.

    Note
    -----
    This could be bundled with calc_glider_w_from_depth, but keeping them separate allows for some extra testing/flexibility for the user. 

    Original author
    ----------------
    Eleanor Frajka-Williams
    """
    # Eleanor's note: This could be bundled with calc_glider_w_from_depth, but keeping them separate allows for some extra testing/flexibility for the user. 
    utilities._check_necessary_variables(ds, ['GLIDER_VERT_VELO_MODEL', 'GLIDER_VERT_VELO_DZDT'])
    
    # Calculate the vertical seawater velocity
    vert_sw_speed = ds['GLIDER_VERT_VELO_DZDT'].values - ds['GLIDER_VERT_VELO_MODEL'].values 

    # Add vertical seawater velocity to the dataset as a data variable
    ds = ds.assign(VERT_CURR_MODEL=(('N_MEASUREMENTS'), vert_sw_speed, {'long_name': 'vertical_current_of_seawater_derived_from_glider_flight_model', 'units': 'm s-1'}))
    return ds

def quant_binavg(ds, var='VERT_CURR', zgrid=None, dz=None):
    """
    Calculate the bin average of vertical velocities within specified depth ranges.
    This function computes the bin average of all vertical velocities within depth ranges,
    accounting for the uneven vertical spacing of seaglider data in depth (but regular in time).
    It uses the pressure data to calculate depth and then averages the vertical velocities
    within each depth bin.

    Parameters
    ----------
    - ds using the variables PRES and VERT_SW_SPEED
    - zgrid (array-like, optional): Depth grid for binning. If None, a default grid is created.
    - dz (float, optional): Interval for creating depth grid if zgrid is not provided.

    Returns
    -------
    - meanw (array-like): Bin-averaged vertical velocities for each depth bin.

    Note
    ----
    I know this is a non-sensical name.  We should re-name, but is based on advice from Ramsey Harcourt.

    Original author
    ----------------
    Eleanor Frajka-Williams
    """
    utilities._check_necessary_variables(ds, [var, 'PRES'])
    press = ds.PRES.values
    ww = ds[var].values

    # Calculate depth from pressure using gsw
    if 'DEPTH_Z' in ds:
        depth = ds.DEPTH_Z.values
    elif 'LATITUDE' in ds:
        latmean = np.nanmean(ds.LATITUDE)
        depth = gsw.z_from_p(press, lat=latmean)  # Assuming latitude is 0, adjust as necessary
    else: 
        msg = f"DEPTH_Z and LATITUDE are missing. At least one of the two variables is needed."
        raise KeyError(msg)

    if zgrid is None:
        if dz is None:
            dz = 5  # Default interval if neither zgrid nor dz is provided
        zgrid = np.arange(np.floor(np.nanmin(depth)/10)*10, np.ceil(np.nanmax(depth) / 10) * 10 + 1, dz)

    def findbetw(arr, bounds):
        return np.where((arr > bounds[0]) & (arr <= bounds[1]))[0]

    # Calculate bin edges from zgrid centers
    bin_edges = np.zeros(len(zgrid) + 1)
    bin_edges[1:-1] = (zgrid[:-1] + zgrid[1:]) / 2
    bin_edges[0] = zgrid[0] - (zgrid[1] - zgrid[0]) / 2
    bin_edges[-1] = zgrid[-1] + (zgrid[-1] - zgrid[-2]) / 2

    meanw = np.zeros(len(zgrid))
    NNz = np.zeros(len(zgrid))
    w_lower = np.zeros(len(zgrid))
    w_upper = np.zeros(len(zgrid))

    # Cycle through the bins and calculate the mean vertical velocity
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for zdo in range(len(zgrid)):
            z1 = bin_edges[zdo]
            z2 = bin_edges[zdo + 1]
            ifind = findbetw(depth, [z1, z2])

            CIlimits = .95 # Could be passed as a variable. 0.95 for 95% confidence intervals
            if len(ifind):
                meanw[zdo] = np.nanmean(ww[ifind])

                # Confidence intervals
                # Number of data points used in the mean at this depth (zgrid[zdo])
                NNz[zdo] = len(ifind)
                if NNz[zdo] > 1:
                    se = np.nanstd(ww[ifind]) / np.sqrt(NNz[zdo])  # Standard error
                    ci = se * stats.t.ppf((1 + CIlimits) / 2, NNz[zdo] - 1)  # Confidence interval based on CIlimits
                    w_lower[zdo] = meanw[zdo] - ci
                    w_upper[zdo] = meanw[zdo] + ci
                else:
                    w_lower[zdo] = np.nan
                    w_upper[zdo] = np.nan

            else:
                meanw[zdo] = np.nan


    # Package the outputs into an xarray dataset
    ds_out = xr.Dataset(
        {
            "meanw": (["zgrid"], meanw),
            "w_lower": (["zgrid"], w_lower),
            "w_upper": (["zgrid"], w_upper),
            "NNz": (["zgrid"], NNz),
        },
        coords={
            "zgrid": zgrid
        },
        attrs={
            "CIlimits": CIlimits
        }
    )
    return ds_out


def quant_hysteresis(ds: xr.Dataset, var='DOXY', v_res=1):
    """
    This function computes up and downcast averages for a specific variable

    Parameters
    ----------
    ds: xarray on OG1 format containing at least depth, profile_number and the selected variable.
        Data should not be gridded.
    var: Selected variable
    v_res: Vertical resolution for the gridding

    Returns
    -------
   df: pandas dataframe containing dive and climb average over depth for the selected variable. A third column contains the depth values

    Original author
    ----------------
    Chiara Monforte
    """
    # utilities._check_necessary_variables(ds, ['PROFILE_NUMBER', 'DEPTH', var])
    p = 1  # Horizontal resolution
    z = v_res  # Vertical resolution
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        varG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds[var], p, z)
        dive = np.nanmedian(varG[0::2, :], axis=0)  # Dive
        climb = np.nanmedian(varG[1::2, :], axis=0)  # Climb
        df = pd.DataFrame(data={'dive': dive, 'climb': climb, 'depth': depthG[0, :]})
    return df


def compute_hyst_stat(ds, var='DOXY', v_res=1):
    """
    This function computes some basic statistics for the differences between climb and dive data

    Parameters
    ----------
    ds: xarray on OG1 format containing at least depth, profile_number and the selected variable. Data should not be gridded.
    var: Selected variable
    v_res: Vertical resolution for the gridding

    Returns
    -------
    diff: difference between upcast and downcast
    err: Percentage error of the dive-climb difference. One value per depth step
    rms: Root Mean Square of the difference in values between dive and climb
    df: pandas dataframe containing dive and climb average over depth for the selected variable. A third column contains the depth values

    Original author
    ----------------
    Chiara  Monforte
    """
    df = quant_hysteresis(ds, var=var, v_res=v_res)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        diff = abs(df.dive - df.climb)
        err = (diff * 100) / np.nanmedian([df.dive, df.climb], axis=0)
        rms = np.sqrt(np.nanmean(abs(df.dive - df.climb) ** 2))
    return df, diff, err, rms