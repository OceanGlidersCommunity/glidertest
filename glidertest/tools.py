import numpy as np
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
from scipy import stats
from cmocean import cm as cmo
import matplotlib.pyplot as plt
from pandas import DataFrame
from skyfield import api
from skyfield import almanac
from tqdm import tqdm


def grid2d(x, y, v, xi=1, yi=1):
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


def updown_bias(ds, axis, var='PSAL', v_res=0, return_val=False):
    p = 1  # Horizontal resolution
    z = v_res  # Vertical resolution
    varG, profG, depthG = grid2d(ds.PROFILE_NUMBER, ds.DEPTH, ds[var], p, z)

    grad = np.diff(varG, axis=0)  # Horizontal gradients

    dc = np.nanmean(grad[0::2, :], axis=0)  # Dive - CLimb
    cd = np.nanmean(grad[1::2, :], axis=0)  # Climb - Dive
    axis.plot(dc, depthG[0, :], label='Dive-Climb')
    axis.plot(cd, depthG[0, :], label='Climb-Dive')
    axis.legend(loc=3)
    lims = np.abs(dc)
    axis.set_xlim(-np.nanpercentile(lims, 99.5), np.nanpercentile(lims, 99.5))
    axis.set_xlabel(ds[var].attrs['long_name'])
    axis.set_ylim(depthG.max() + 10, -5)
    axis.grid()
    if return_val:
        return dc, cd


def find_cline(var, depth_array):
    clin = np.where(np.abs(np.diff(np.nanmean(var, axis=0))) == np.nanmax(np.abs(np.diff(np.nanmean(var, axis=0)))))
    return np.round( depth_array[0, clin[0]], 1)


def plot_basic_vars(ds, v_res=1, start_prof=0, end_prof=-1):
    p = 1
    z = v_res
    tempG, profG, depthG = grid2d(ds.PROFILE_NUMBER, ds.DEPTH, ds.TEMP, p, z)
    salG, profG, depthG = grid2d(ds.PROFILE_NUMBER, ds.DEPTH, ds.PSAL, p, z)
    denG, profG, depthG = grid2d(ds.PROFILE_NUMBER, ds.DEPTH, ds.DENSITY, p, z)

    tempG = tempG[start_prof:end_prof, :]
    salG = salG[start_prof:end_prof, :]
    denG = denG[start_prof:end_prof, :]
    depthG = depthG[start_prof:end_prof, :]

    halo = find_cline(salG, depthG)
    thermo = find_cline(tempG, depthG)
    pycno = find_cline(denG, depthG)
    print(
        f'The thermocline, halocline and pycnocline are located at respectively {thermo}, {halo} and {pycno}m as shown in the plots as well')

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax1 = ax[0].twiny()
    ax2 = ax[0].twiny()
    ax2.spines["top"].set_position(("axes", 1.2))
    ax[0].plot(np.nanmean(tempG, axis=0), depthG[0, :], c='blue')
    ax1.plot(np.nanmean(salG, axis=0), depthG[0, :], c='red')
    ax2.plot(np.nanmean(denG, axis=0), depthG[0, :], c='black')
    ax[0].axhline(thermo, linestyle='dashed', c='blue')
    ax1.axhline(halo, linestyle='dashed', c='red')
    ax2.axhline(pycno, linestyle='dashed', c='black')

    ax[0].set(xlabel=f'Average Temperature [C] \nbetween profile {start_prof} and {end_prof}', ylabel='Depth (m)')
    ax[0].tick_params(axis='x', colors='blue')
    ax[0].xaxis.label.set_color('blue')
    ax1.spines['bottom'].set_color('blue')
    ax1.set(xlabel=f'Average Salinity [PSU] \nbetween profile {start_prof} and {end_prof}')
    ax1.xaxis.label.set_color('red')
    ax1.spines['top'].set_color('red')
    ax1.tick_params(axis='x', colors='red')
    ax2.spines['bottom'].set_color('black')
    ax2.set(xlabel=f'Average Density [kg m-3] \nbetween profile {start_prof} and {end_prof}')
    ax2.xaxis.label.set_color('black')
    ax2.spines['top'].set_color('black')
    ax2.tick_params(axis='x', colors='black')

    if 'CHLA' in ds.variables:
        chlaG, profG, depthG = grid2d(ds.PROFILE_NUMBER, ds.DEPTH, ds.CHLA, p, z)
        chlaG = chlaG[start_prof:end_prof, :]
        ax2_1 = ax[1].twiny()
        ax2_1.plot(np.nanmean(chlaG, axis=0), depthG[0, :], c='green')
        ax2_1.set(xlabel=f'Average Chlorophyll-a [mg m-3] \nbetween profile {start_prof} and {end_prof}')
        ax2_1.xaxis.label.set_color('green')
        ax2_1.spines['top'].set_color('green')
        ax2_1.tick_params(axis='x', colors='green')
    else:
        ax[1].text(0.3, 0.7, 'Chlorophyll data unavailable', va='top', transform=ax[1].transAxes)

    if 'DOXY' in ds.variables:
        oxyG, profG, depthG = grid2d(ds.PROFILE_NUMBER, ds.DEPTH, ds.DOXY, p, z)
        oxyG = oxyG[start_prof:end_prof, :]
        ax[1].plot(np.nanmean(oxyG, axis=0), depthG[0, :], c='orange')
        ax[1].set(xlabel=f'Average Oxygen [mmol m-3] \nbetween profile {start_prof} and {end_prof}')
        ax[1].xaxis.label.set_color('orange')
        ax[1].spines['top'].set_color('orange')
        ax[1].tick_params(axis='x', colors='orange')
        ax[1].spines['bottom'].set_color('orange')
    else:
        ax[1].text(0.3, 0.5, 'Oxygen data unavailable', va='top', transform=ax[1].transAxes)

    [a.set_ylim(depthG.max() + 10, -5) for a in ax]
    [a.grid() for a in ax]


# Check if there is any negative scaled data and/or raw data
def chl_first_check(ds):
    # Check how much negative data there is
    neg_chl = np.round((len(np.where(ds.CHLA < 0)[0]) * 100) / len(ds.CHLA), 1)
    if neg_chl > 0:
        print(f'{neg_chl}% of scaled chlorophyll data is negative, consider recalibrating data')
        # Check where the negative values occur and if we just see them at specific time of the mission or not
        start = ds.TIME[np.where(ds.CHLA < 0)][0]
        end = ds.TIME[np.where(ds.CHLA < 0)][-1]
        min_z = (np.round(ds.DEPTH[np.where(ds.CHLA < 0)].min().values, 1))
        max_z = (np.round(ds.DEPTH[np.where(ds.CHLA < 0)].max().values, 1))
        print(f'Negative data in present from {str(start.values)[:16]} to {str(end.values)[:16]}')
        print(f'Negative data is present between {min_z} and {max_z} ')
    else:
        print('There is no negative scaled chlorophyll data, recalibration and further checks are still recommended ')
    # Check if there is any missing data throughout the mission
    if len(ds.TIME) != len(ds.CHLA.dropna(dim='N_MEASUREMENTS').TIME):
        print('Chlorophyll data is missing for part of the mission')  # Add to specify where the gaps are
    else:
        print('Chlorophyll data is present for the entire mission duration')
    # Check bottom dark count and any drift there
    bottom_chl_data = ds.CHLA.where(ds.CHLA.DEPTH > ds.DEPTH.max() - (ds.DEPTH.max() * 0.1)).dropna(
        dim='N_MEASUREMENTS')
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0, len(bottom_chl_data)), bottom_chl_data)
    ax = sns.regplot(data=ds, x=np.arange(0, len(bottom_chl_data)), y=bottom_chl_data,
                     scatter_kws={"color": "grey"},
                     line_kws={"color": "red", "label": "y={0:.6f}x+{1:.3f}".format(slope, intercept)},
                     )
    ax.legend(loc=2)
    ax.grid()
    ax.set(ylim=(np.nanpercentile(bottom_chl_data, 0.1), np.nanpercentile(bottom_chl_data, 99.9)),
           xlabel='Measurements',
           ylabel='Chla')
    plt.show()
    if slope >= 0.00001:
        print(
            'Data from the deepest 10% of data has been analysed and data does not seem stable. An alternative solution for dark counts has to be considered. \nMoreover, it is recommended to check the sensor has this may suggest issues with the sensor (i.e water inside the sensor, temporal drift etc)')
    else:
        print(
            'Data from the deepest 10% of data has been analysed and data seems stable. These deep values can be used to re-assess the dark count if the no chlorophyll at depth assumption is valid in this site and this depth')
def sunset_sunrise(time, lat, lon):
    """
    Calculates the local sunrise/sunset of the glider location from GliderTools.
    [https://github.com/GliderToolsCommunity/GliderTools/blob/master/glidertools/optics.py]

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
        bluffton.append(api.wgs84.latlon(grp_avg.lat[i], grp_avg.lon[i]))
    bluffton = np.array(bluffton)

    sunrise = []
    sunset = []
    for n in tqdm(range(len(bluffton))):

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

            # this doesnt look very efficient at the moment, but I was having issues with getting the datetime64
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


def check_npq(ds, offset=np.timedelta64(1, "h"), start_time='2024-04-18', end_time='2024-04-20', sel_day=6):
    """
    Calculates day and night chlorophyll averages to check if data is affected by NPQ.

    We separate day and night using the GliderTools sunset/sunrise function. 
    We plot a section of chlorophyll for a selected slice of data to observe any NPQ effetcs.
    We then plot the day and night average for a specific day for a more detailed view.


    Parameters
    ----------
    ds: xarray on OG1 format containing at least time, depth, latitude, longitude and chlorophyll. 
        Data should not be gridded.
    offset: The delayed onset and recovery of quenching in hours.
    start_time: Start date of the data selection. As missions can be long and came make it hard to visualise NPQ effetc, 
                we reccomend selecting small section of few days to few weeks.
    end_time: End date of the data selection. As missions can be long and came make it hard to visualise NPQ effetc, 
                we reccomend selecting small section of few days to few weeks.
    sel_day: Selected day (int) for the detailed plot comparing day and night averages. 
             The integer value refers to the first, second, third day etc. of the slected time period.

    Returns
    -------
    Two plots: a chlorphyll section and a comparison of day and night average chlorphyll over depth for the selcted day

    """
    
    
    ds_sel = ds.sel(TIME=slice(start_time, end_time))
    sunrise, sunset = sunset_sunrise(ds_sel.TIME, ds_sel.LATITUDE, ds_sel.LONGITUDE)

    # creating quenching correction batches, where a batch is a night and the following day
    day = (ds_sel.TIME > (sunrise + offset)) & (ds_sel.TIME < (sunset + offset))
    # find day and night transitions
    daynight_transitions = np.abs(np.diff(day.astype(int)))
    # get the cumulative sum of daynight to generate separate batches for day and night
    daynight_batches = daynight_transitions.cumsum()
    batch = np.r_[0, (daynight_batches) // 2]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), sharex='all')
    c = ax.scatter(ds_sel.TIME, ds_sel.DEPTH, c=ds_sel.CHLA, s=10, vmin=np.nanpercentile(ds_sel.CHLA, 0.5),
                   vmax=np.nanpercentile(ds_sel.CHLA, 99.5))
    ax.set_ylim(30, -2)
    for n in np.unique(sunset):
        ax.axvline(np.unique(n), c='blue')
    for m in np.unique(sunrise):
        ax.axvline(np.unique(m), c='orange')
    ax.set_ylabel('Depth [m]')
    plt.colorbar(c, label='Chlorophyll [mg m-3]')
    
    # Create day and night avergaes to then have easy to plot
    df = pd.DataFrame(np.c_[ds_sel['CHLA'], day, batch, ds_sel['DEPTH']], columns=['flr', 'day', 'batch', 'depth'])
    ave = df.flr.groupby([df.day, df.batch, np.around(df.depth)]).mean()
    day_av = ave[1]
    night_av = ave[0]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex='all')

    ax.plot(night_av[sel_day], night_av[sel_day].index, label='Night time average')
    ax.plot(day_av[sel_day], day_av[sel_day].index, label='Daytime average')
    ax.legend()
    ax.invert_yaxis()
    ax.grid()
    ax.set(xlabel='Chlorophyll [mg m-3]', ylabel='Depth [m]')
    ax.set_title(str(ds_sel.TIME.where(batch == sel_day).dropna(dim='N_MEASUREMENTS')[-1].values)[:10])
