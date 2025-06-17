from typing import Any

import numpy as np
from tqdm import tqdm
import pandas as pd
import xarray as xr
from scipy import stats
import gsw
import warnings
from glidertest import utilities
from scipy.integrate import cumulative_trapezoid

def quant_updown_bias(ds, var='PSAL', v_res=1):
    """
    This function computes up and downcast averages for a specific variable

    Parameters
    ----------
    ds: Dataset.xarray 
        Dataset in **OG1 format**, containing at least **TIME, DEPTH, LATITUDE, LONGITUDE,** and the selected variable.  
        Data should **not** be gridded.
    var: str, optional, default='PSAL'
        Selected variable.
    v_res: float
        Vertical resolution for the gridding in meters.
                
    Returns
    -------
    df: pandas.DataFrame 
        Dataframe containing dc (Dive - Climb average), cd (Climb - Dive average) and depth

    Notes
    -----
    Original Author: Chiara Monforte
    """
    utilities._check_necessary_variables(ds, ['PROFILE_NUMBER', 'DEPTH', var])
    p = 1  # Horizontal resolution
    z = v_res  # Vertical resolution

    if var in ds.variables:
        varG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds[var], p, z, x_bin_center = False) ## included x_bin_center = False to take left edge of the bin (profile number i.e. 1,2 ....)

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

def mean_profile(ds, var='TEMP', v_res=1):
    """
    This function computes the mean vertical profile for a specific variable.

    Parameters
    ----------
    ds: xarray.Dataset 
        Dataset in **OG1 format**, containing at least **PROFILE_NUMBER, DEPTH**, and the selected variable.  
        Data should **not** be gridded.
    var: str, optional, default='PSAL'
        Selected variable to average.
    v_res: float
        Vertical resolution in meters for binning the profile.

    Returns
    -------
    df: pandas.DataFrame 
        DataFrame containing the mean profile and corresponding depth bins.

    Notes
    -----
    Original Author: Till Moritz
    """
    # Ensure required variables are in the dataset
    utilities._check_necessary_variables(ds, ['PROFILE_NUMBER', 'DEPTH', var])

    p = 1       # Horizontal resolution (not used here)
    z = v_res   # Vertical resolution

    if var in ds.variables:
        # 2D gridding by profile and depth
        varG, profG, depthG = utilities.construct_2dgrid(
            ds.PROFILE_NUMBER, ds.DEPTH, ds[var], p, z, x_bin_center=False)

        # Compute mean across profiles for each depth level
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_var = np.nanmean(varG, axis=0)

        df = pd.DataFrame(data={'mean': mean_var, 'depth': depthG[0, :]})
    else:
        print(f'{var} is not in the dataset')
        df = pd.DataFrame()

    return df

def compute_daynight_avg(ds, sel_var='CHLA', start_time=None, end_time=None, start_prof=None, end_prof=None):
    """
    Computes day and night averages for a selected variable over a specified time period or range of dives.  
    Day and night are determined based on **sunrise and sunset times** from the `compute_sunset_sunrise` function in GliderTools.  
    A sufficient number of dives is required to ensure both **day and night data** are present; otherwise, the function will not run.  

    Parameters
    ----------
    ds : xarray.Dataset  
        Dataset in **OG1 format**, containing at least **TIME, DEPTH, LATITUDE, LONGITUDE,** and the selected variable.  
        Data should **not** be gridded.  
    sel_var : str, optional, default='CHLA'  
        The variable for which day and night averages will be computed.  
    start_time : str or datetime-like, optional  
        The **start date** for data selection.  
        - If not specified, defaults to the **central week of the deployment**.  
        - Selecting a smaller section (a few days to weeks) is recommended for better visualization of NPQ effects.  
    end_time : str or datetime-like, optional  
        The **end date** for data selection.  
        - If not specified, defaults to the **central week of the deployment**.  
    start_prof : int, optional  
        The **starting profile number** for data selection.  
        - If not specified, the function uses `start_time` or defaults to the **central week of deployment**.  
        - A sufficient number of dives is required for valid day and night separation.  
    end_prof : int, optional  
        The **ending profile number** for data selection.  
        - If not specified, the function uses `end_time` or defaults to the **central week of deployment**.  

    Returns
    -------
    day_av : pandas.DataFrame  
        DataFrame containing **daytime averages** of the selected variable with the following columns:  
        - `batch` : Integer representing the grouped day-night cycles (e.g., chronological day number).  
        - `depth` : Depth values for the average calculation.  
        - `dat` : Average value of the selected variable.  
        - `date` : Actual date corresponding to the batch.  
    night_av : pandas.DataFrame  
        DataFrame containing **nighttime averages** of the selected variable with the same columns as for day_av

    Notes
    ------
    Original Author: Chiara Monforte  
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
    Checks whether the selected variable is **monotonically increasing** throughout the mission.  
    This function is particularly designed to check **profile numbers** to ensure they are assigned correctly.  
    If the profile number is not monotonically increasing, it may indicate misassignment due to an error in data processing.  

    Parameters
    ----------
    da: xarray.DataArray   
        DataArray in **OG1 format**. Data should **not** be gridded. 

    Returns
    -------
    bool:
        **True** if the variable is monotonically increasing, else **False**. 
        Additionally, a message is printed indicating the result.  

    Notes
    ------
    Original Author: Chiara Monforte
    """
    if not pd.Series(da).is_monotonic_increasing:
        print(f'{da.name} is not always monotonically increasing')
        return False
    else:
        print(f'{da.name} is always monotonically increasing')
        return True

def calc_w_meas(ds):
    """
    Calculates the vertical velocity of a glider using changes in pressure with time.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing **DEPTH** and **TIME**.
    
    Returns
    -------
    ds: xarray.Dataset 
        Containing the new variable **GLIDER_VERT_VELO_DZDT** (array-like), with vertical velocities calculated from dz/dt

    Notes
    ------
    Original Author: Eleanor Frajka-Williams
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
    ds: xarray.Dataset 
        Dataset containing **VERT_GLIDER_SPEED** and **VERT_SPEED_DZDT**.

    Returns
    -------
    ds: xarray.Dataset 
        Dataset with the new variable **VERT_SW_SPEED**, which is the inferred vertical seawater velocity.

    Notes
    -----
    - This could be bundled with calc_glider_w_from_depth, but keeping them separate allows for some extra testing/flexibility for the user.
    Original Author: Eleanor Frajka-Williams
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
    ds: xarray.Dataset 
        Dataset containing the variables **PRES** and **VERT_SW_SPEED**
    zgrid:  array-like, optional 
        Depth grid for binning. If None, a default grid is created.
    dz: float, optional 
        Interval for creating depth grid if zgrid is not provided.

    Returns
    -------
    meanw: array-like 
        Bin-averaged vertical velocities for each depth bin.

    Notes
    ----
    Original Author: Eleanor Frajka-Williams
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
    ds: xarray.Dataset 
        Dataset in **OG1 format**, containing at least **DEPTH, PROFILE_NUMBER,** and the selected variable.  
        Data should **not** be gridded.
    var: str, optional, default='DOXY' 
        Selected variable
    v_res: float
        Vertical resolution for the gridding in meters.

    Returns
    -------
    df: pandas.DataFrame 
        Dataframe containing dive and climb average over depth for the selected variable. A third column contains the depth values

    Notes
    ------
    Original Author: Chiara Monforte
    """
    utilities._check_necessary_variables(ds, ['PROFILE_NUMBER', 'DEPTH', var])
    p = 1  # Horizontal resolution
    z = v_res  # Vertical resolution
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        varG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds[var], p, z, x_bin_center = False) ## included x_bin_center = False to take left edge of the bin (profile number i.e. 1,2 ....)

        dive = np.nanmedian(varG[0::2, :], axis=0)  # Dive
        climb = np.nanmedian(varG[1::2, :], axis=0)  # Climb
        df = pd.DataFrame(data={'dive': dive, 'climb': climb, 'depth': depthG[0, :]})
    return df


def compute_hyst_stat(ds: xr.Dataset, var='DOXY', v_res=1):
    """
    This function computes some basic statistics for the differences between climb and dive data

    Parameters
    ----------
    ds: xarray.Dataset 
        Dataset in **OG1 format**, containing at least **DEPTH, PROFILE_NUMBER,** and the selected variable.
        Data should not be gridded.
    var: str, optional, default='DOXY' 
        Selected variable
    v_res: float
        Vertical resolution for the gridding

    Returns
    -------
    diff : array-like  
        Difference between upcast and downcast values.  
    err_range : array-like
        Percentage error of the dive-climb difference based on the range of values, computed for each depth step.
    err_mean : array-like
        Percentage error of the dive-climb difference based on the mean values, computed for each depth step.
    rms : float  
        Root Mean Square (RMS) of the difference in values between dive and climb.  
    df : pandas.DataFrame  
        DataFrame containing:  
        - **Dive and climb averages** over depth for the selected variable.  
        - **Depth values** in a separate column.  

    Notes
    ------
    Original Author: Chiara  Monforte
    """
    df = quant_hysteresis(ds, var=var, v_res=v_res)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        diff = abs(df.dive - df.climb)
        err_range = (diff * 100) / (np.nanmax(np.nanmedian([df.dive, df.climb], axis=0)) - np.nanmin(
            np.nanmedian([df.dive, df.climb], axis=0)))
        err_mean = (diff * 100) / np.nanmedian([df.dive, df.climb], axis=0)
        rms = np.sqrt(np.nanmean(abs(df.dive - df.climb) ** 2))
    return df, diff, err_mean, err_range, rms


def compute_prof_duration(ds:xr.Dataset):
    """
    This function computes some basic statistics for the differences between climb and dive data.

    Parameters
    ----------
    ds: xarray.Dataset 
        Dataset in **OG1 format**, containing at least **DEPTH, PROFILE_NUMBER,** and the selected variable.
        Data should not be gridded.

    Returns
    -------
    df: pandas.DataFrame 
        Dataframe containing the profile number and the duration of that profile in minutes

    Notes
    ------
    Original Author: Chiara  Monforte
    """
    utilities._check_necessary_variables(ds, ['PROFILE_NUMBER', 'TIME'])
    deltat = ds.TIME.groupby(ds.PROFILE_NUMBER).max() - ds.TIME.groupby(ds.PROFILE_NUMBER).min()
    deltat_min = (pd.to_timedelta(deltat).astype('timedelta64[s]') / 60).astype('int64')
    df = pd.DataFrame(data={'profile_num': deltat.PROFILE_NUMBER, 'profile_duration': deltat_min})
    return df

def find_outlier_duration(df: pd.DataFrame, rolling=20, std=2):
    """
    Identifies **outlier profile durations** based on a rolling mean and standard deviation threshold.  
    This helps detect profiles with significantly longer or shorter durations compared to surrounding profiles. 

    Parameters
    ----------
    df: pandas.DataFrame 
        Dataframr containing the profile number and the duration of that profile in minutes
    rolling: int, default = 20
        Window size for the rolling mean
    std: int, default = 2
        Number of standard deviations to use for 'odd' profile duration

    Returns
    -------
    rolling_mean : pandas.Series  
        Rolling mean of **profile duration** computed using the specified window size.  
    overt_prof : numpy.ndarray  
        Array of **profile numbers** where the duration exceeds the rolling mean by more than the set **standard deviation threshold**.  
        - If outliers are found, a message is printed recommending further investigation. 

    Notes
    ------
    Original Author: Chiara  Monforte
    """
    rolling_mean = df['profile_duration'].rolling(window=rolling, min_periods=1).mean()
    overtime = np.where((df['profile_duration'] > rolling_mean + (np.std(rolling_mean) * std)) | (
                df['profile_duration'] < rolling_mean - (np.std(rolling_mean) * std)))
    overt_prof = df['profile_num'][overtime[0]].values
    if len(overtime[0]) > 0:
        print(
            f'There are {len(overtime[0])} profiles where the duration differs by {std} standard deviations of the nearby {rolling} profiles. Further checks are recommended')
    return rolling_mean, overt_prof

def compute_global_range(ds: xr.Dataset, var='DOXY', min_val=-5, max_val=600):
    """
    Applies a gross filter to the dataset by removing observations outside the specified global range.

    This function checks if any values of the specified variable (`var`) fall outside the provided
    range `[min_val, max_val]`. If a value is out of the specified range, it is excluded from the
    output. The function returns the filtered dataset with the out-of-range values removed.

    Parameters
    ----------
    ds : xarray.Dataset 
        Dataset containing the variable to be filtered.
    var : str, optional, default='DOXY' 
        The name of the variable to apply the range filter on.
    min_val : float, default = -5
        The minimum allowable value for the variable.
    max_val : float, default = 600
        The maximum allowable value for the variable.

    Returns
    -------
    xarray.DataArray : 
        A filtered DataArray containing only the values of the specified variable
        within the range `[min_val, max_val]`. Values outside this range are dropped.

    Notes
    ------
    Original Author: Chiara Monforte
    """
    utilities._check_necessary_variables(ds, [var])
    out_range = ds[var].where((ds[var]<min_val )| (ds[var]>max_val ))
    return out_range.dropna(dim='N_MEASUREMENTS')

def max_depth_per_profile(ds: xr.Dataset):
    """
    This function computes the maximum depth for each profile in the dataset

    Parameters
    ----------
    ds: xarray.Dataset 
        Dataset in **OG1 format**, containing at least **DEPTH, PROFILE_NUMBER,** and the selected variable.
        Data should not be gridded.

    Returns
    -------
    max_depths: pandas dataframe containing the profile number and the maximum depth of that profile

    Notes
    ------
    Original Author: Till Moritz
    """
    max_depths = ds.groupby('PROFILE_NUMBER').apply(lambda x: x['DEPTH'].max())
    ### add the unit to the dataarray
    max_depths.attrs['units'] = ds['DEPTH'].attrs['units']
    return max_depths

def add_sigma_1(ds: xr.Dataset, var_sigma_1: str = "SIGMA_1") -> xr.Dataset:
    """
    Computes the potential density anomaly (sigma_1) referenced to 1000 dbar 
    and adds it to the dataset if not already present.

    Parameters
    ----------
    ds : xr.Dataset
        OG1-format dataset with required variables: DEPTH, TEMP, PSAL, LATITUDE, LONGITUDE.
    var_sigma_1 : str, optional
        Name of the variable to be added to the dataset. Default is "SIGMA_1".

    Returns
    -------
    xr.Dataset
        Dataset with the additional variable `var_sigma_1`.

    Notes
    -----
    Original author: Till Moritz
    """
    required_vars = ['DEPTH', 'TEMP', 'PSAL', 'LATITUDE', 'LONGITUDE']
    utilities._check_necessary_variables(ds, required_vars)

    if var_sigma_1 in ds:
        print(f"Variable '{var_sigma_1}' already exists in the dataset. Skipping calculation.")
        return ds

    # Extract required variables
    TEMP = ds['TEMP'].values
    PSAL = ds['PSAL'].values
    PRES = ds['DEPTH'].values
    LAT = ds['LATITUDE'].values
    LON = ds['LONGITUDE'].values

    # Filter valid entries
    valid = ~np.isnan(TEMP) & ~np.isnan(PSAL) & ~np.isnan(PRES) & ~np.isnan(LAT) & ~np.isnan(LON)

    if not np.any(valid):
        print(f"All values are invalid for {var_sigma_1}; output will contain only NaNs.")
        ds[var_sigma_1] = xr.DataArray(
            np.full_like(PRES, np.nan), 
            dims=('N_MEASUREMENTS',),
            attrs={'units': 'kg/m^3', 'long_name': 'potential density anomaly with respect to 1000 dbar'}
        )
        return ds

    # Compute absolute salinity and conservative temperature
    SA = gsw.SA_from_SP(PSAL[valid], PRES[valid], LON[valid], LAT[valid])
    CT = gsw.CT_from_t(PSAL[valid], TEMP[valid], PRES[valid])
    SIGMA_1 = gsw.sigma1(SA, CT)

    # Create result array with NaNs where invalid
    full_sigma_1 = np.full_like(PRES, np.nan, dtype=np.float64)
    full_sigma_1[valid] = SIGMA_1

    ds[var_sigma_1] = xr.DataArray(full_sigma_1, dims=('N_MEASUREMENTS',),
        attrs={'units': 'kg/m^3', 'long_name': 'potential density anomaly with respect to 1000 dbar'}
    )

    return ds

def compute_mld(ds: xr.Dataset, variable, method: str = 'threshold', threshold = 0.03, ref_depth = 10,
                 use_bins: bool = True, binning: float = 10):
    """
    Computes the mixed layer depth (MLD) for each profile in the dataset. Two methods are available:
    1. **Threshold Method**: Computes MLD based on a density threshold (default is 0.03 kg/m³).
    2. **Convective Resistance (CR) Method**: Computes MLD based on the CR method. Values close to
    0 indicate a well-mixed layer, while values below 0 indicate a stratified layer. For the threshold,
    a value of -2 is recommended.
    
    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the profiles.
    variable : str
        Variable used for the MLD calculation. For the CR method use density anomaly with reference to 1000 dbar ('SIGMA_1').
    method : str, optional
        The method to use for MLD calculation. Options are 'threshold' or 'CR'. Default is 'threshold'.
    threshold : float, optional
        If using 'threshold', this is the density threshold for MLD calculation. Default is 0.03 kg/m³.
        If using 'CR', this is the CR threshold for the convective resistance. A value of -2 is recommended.
    use_bins : bool, optional
        Whether to use binned data for MLD calculation. Default is True.
    binning : float, optional
        The binning resolution in meters. Default is 10m.

    Returns
    -------
    mld_values : pd.DataFrame
        A DataFrame containing the MLD values for each profile, along with the mean time for each profile.
        The DataFrame has columns: 'PROFILE_NUMBER', 'MLD' and 'TIME'.

    Notes
    -----
    Original Author: Till Moritz
    Based on the convective resistance caluclation, following the method described in FW2014 (https://doi.org/10.1175/JPO-D-13-069.1))
    """
    utilities._check_necessary_variables(ds, [variable,"DEPTH", "PROFILE_NUMBER","TIME"])
    if method == 'threshold':
        groups = utilities.group_by_profiles(ds, [variable, "DEPTH","TIME"])
        mld = groups.apply(mld_profile_treshhold, variable=variable, threshold=threshold,
                            ref_depth=ref_depth, use_bins=use_bins, binning=binning)
    elif method == 'CR':
        if variable != 'SIGMA_1':
            print(f"Warning: {variable} can not be used for convective resistance calulation. Instead use SIGMA_1 for CR calculation.")
            variable = 'SIGMA_1'
        groups = utilities.group_by_profiles(ds, [variable, "DEPTH","TIME"])
        if threshold > 0:
            print("Warning: CR threshold should be negative. Using -2 as default.")
            threshold = -2
        mld = groups.apply(mld_profile_CR, threshold=threshold, use_bins=use_bins, binning=binning)
    else:
        raise ValueError("Invalid MLD calculation method. Use 'threshold' or 'CR'.")
    # Convert the result to a DataFrame and name the MLD column
    mld = mld.reset_index(name='MLD')
    # Add mean time for each profile to the DataFrame
    mld['TIME'] = groups.TIME.mean().values
    mld['TIME'] = mld['TIME'].dt.round('1min')
    return mld

def linear_interpolation(x, y, x_new):
    """Linearly interpolates y over x to estimate y at x_new."""
    return np.interp(x_new, x, y)

def mld_profile_treshhold(profile, variable: str = 'SIGMA_T', threshold: float = 0.03, ref_depth: float = 10,
                          use_bins: bool = False, binning: float = 10) -> float:
    """
    Computes the mixed layer depth (MLD) from a profile dataset based on the density profile, 
    using a threshold of 0.03 kg/m³

    Parameters
    ----------
    profile : pd.DataFrame or xr.Dataset
        Dataset or DataFrame containing depth and density columns.
    variable : str
        The name of the variable to use for the threshold calculation (default is 'SIGMA_T').
    threshold : float
        Density threshold for MLD estimation (default is 0.03 kg/m³).
    ref_depth : float
        Reference depth for MLD estimation (default is 10m).
    use_bins : bool
        Whether to bin the profile data before computing MLD.
    binning : float
        Bin resolution in meters if use_bins is True.

    Returns
    -------
    float
        Estimated mixed layer depth, or NaN if it cannot be determined.

    Notes
    -----
    Original Author: Till Moritz
    """
    
    if use_bins:
        profile = utilities.bin_profile(profile, [variable], binning=binning)
        depth = profile['DEPTH'].to_numpy()
        density = profile[variable].to_numpy()
    else:
        if isinstance(profile, pd.DataFrame):
            depth = profile['DEPTH'].to_numpy()
            density = profile[variable].to_numpy()
        elif isinstance(profile, xr.Dataset):
            depth = profile['DEPTH'].values
            density = profile[variable].values
        else:
            raise TypeError("Input must be a pandas.DataFrame or xarray.Dataset")

    # Convert to float arrays to avoid isfinite errors
    depth = np.asarray(depth, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)

    # Remove NaNs and check if valid data remains
    valid = np.isfinite(depth) & np.isfinite(density)
    if not np.any(valid):
        print("No valid depth or density data for MLD calculation.")
        return np.nan

    depth, density = depth[valid], density[valid]
    
    if np.nanmean(np.diff(depth)) < 0:
        depth = -1 * depth

    # Sort by depth
    sort_idx = np.argsort(depth)
    depth, density = depth[sort_idx], density[sort_idx]

    # Estimate density at reference depth
    if ref_depth in depth:
        idx_ref = np.nanargmin(np.abs(depth - ref_depth))
        density_ref = density[idx_ref]
    else:
        density_ref = linear_interpolation(depth, density, ref_depth)

    # Focus on depths below reference depth
    mask_below = depth > ref_depth
    depth_below = depth[mask_below]
    density_below = density[mask_below]

    if depth_below.size == 0:
        print(f"No data below reference depth {ref_depth} m")
        return np.nan

    if np.nanmax(density_below) < density_ref + threshold:
        print(f"No density values below reference depth {ref_depth} m exceed the threshold.")
        return np.nan

    # Find first crossing of the threshold
    for i in range(1, len(density_below)):
        if density_below[i] > density_ref + threshold:
            mld = (depth_below[i] + depth_below[i - 1]) / 2
            return round(mld, 1)

    return np.nan


def mld_profile_CR(profile, threshold: float = -2, use_bins: bool = True, binning: float = 10) -> float:
    """
    Calculate the mixed layer depth (MLD) using the Convective Resistance (CR) method.
    Returns NaN if no valid depth data below 10m is available or no CR values meet the threshold.

    Parameters
    ----------
    profile : xarray.Dataset or pandas.DataFrame
        Profile data containing 'DEPTH' and 'SIGMA_1'.
    threshold : float, optional
        CR threshold for determining MLD. Default is -2.
    use_bins : bool, optional
        Whether to apply depth binning. Default is False.
    binning : float, optional
        Bin size for depth binning, in meters. Default is 10.

    Returns
    -------
    float
        Computed MLD in meters, or NaN if criteria are not met.

    Notes
    -----
    Original author: Till Moritz
    """
    if use_bins:
        profile = utilities.bin_profile(profile, ['SIGMA_1'], binning=binning)

    CR_df = calculate_CR_for_all_depth(profile)
    depth = CR_df['DEPTH'].to_numpy()
    CR_values = CR_df['CR'].to_numpy()

    # Filter out NaNs
    valid = ~np.isnan(CR_values)
    depth = depth[valid]
    CR_values = CR_values[valid]

    if len(depth) == 0 or np.nanmin(depth) > 10:
        return np.nan

    # Identify where CR is below threshold
    below_threshold = CR_values < threshold
    if not np.any(below_threshold):
        return np.nan
    
    mld = np.nanmin(depth[below_threshold])

    return round(mld, 1)


def calculate_CR_for_all_depth(profile):
    """
    Calculate CR for all depths in the profile.

    Parameters
    ----------
    profile : xarray.Dataset or pandas.DataFrame
        Profile data containing depth and SIGMA_1.
    use_bins : bool, optional
        If True, bins the data before computation. Default is False.
    binning : float, optional
        Bin size for binning. Default is 10.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: DEPTH, CR.

    Notes
    -----
    Original Author: Till Moritz
    """
    required_vars = ['DEPTH', 'SIGMA_1']

    if isinstance(profile, xr.Dataset):
        df_profile = profile[required_vars].to_dataframe().reset_index()
    elif isinstance(profile, pd.DataFrame):
        df_profile = profile[required_vars].copy()
    else:
        raise TypeError("Input must be an xarray.Dataset or pandas.DataFrame.")

    # Drop NaNs in SIGMA_1 to avoid problems
    df_profile = df_profile.dropna(subset=['SIGMA_1'])

    # Prepare CR values
    CR_values = []
    depths = df_profile['DEPTH'].to_numpy()

    for h in depths:
        CR_h = compute_CR(df_profile, h)
        CR_values.append(CR_h)
    
    #return CR_values, depths
    return pd.DataFrame({'DEPTH': depths, 'CR': CR_values})

def compute_CR(profile, h: float) -> float:
    """
    Compute the CR (density anomaly integral) up to the reference depth h.

    Parameters
    ----------
    profile : xarray.Dataset or pandas.DataFrame
        Profile data containing 'DEPTH' and 'SIGMA_1'.
    h : float
        Reference depth up to which CR is computed.

    Returns
    -------
    float
        Computed CR up to the depth h, or NaN if insufficient data.

    Notes
    -----
    Original Author: Till Moritz
    """
    # Extract depth and sigma_1 depending on input type
    if isinstance(profile, xr.Dataset):
        depth = profile['DEPTH'].values
        sigma1 = profile['SIGMA_1'].values
    elif isinstance(profile, pd.DataFrame):
        depth = profile['DEPTH'].to_numpy()
        sigma1 = profile['SIGMA_1'].to_numpy()
    else:
        raise TypeError("Input must be an xarray.Dataset or pandas.DataFrame.")

    # Filter out NaNs
    valid = ~np.isnan(depth) & ~np.isnan(sigma1)
    depth = depth[valid]
    sigma1 = sigma1[valid]

    if len(depth) < 2 or h > np.nanmax(depth):
        return np.nan

    # Sort by depth
    idx = np.argsort(depth)
    depth = depth[idx]
    sigma1 = sigma1[idx]

    # Select depths up to h
    mask = (depth <= h) & (depth >= 0)
    if np.sum(mask) < 1:
        print(f"Not enough data points for depth {h} m")
        return np.nan

    depth_masked = depth[mask]
    sigma1_masked = sigma1[mask]

    # Fill in missing top layer if needed
    min_depth = depth_masked[0]
    
    if min_depth > 0 and min_depth <= 15:
        new_depth = np.arange(0, min_depth, 0.25)
        top_mask = depth_masked <= 15
        sigma1_top_mean = np.nanmean(sigma1_masked[top_mask])
        new_sigma1 = np.full_like(new_depth, sigma1_top_mean, dtype=float)

        depth_filled = np.concatenate([new_depth, depth_masked])
        sigma1_filled = np.concatenate([new_sigma1, sigma1_masked])
    else:
        depth_filled = depth_masked
        sigma1_filled = sigma1_masked
    # Integration and CR computation
    integral = cumulative_trapezoid(sigma1_filled, depth_filled, initial=0)[-1]
    sigma1_h = sigma1_filled[-1]
    CR_h = integral - np.nanmax(depth_filled) * sigma1_h

    return CR_h