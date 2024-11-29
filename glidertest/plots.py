import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.dates import DateFormatter
from scipy import stats
import matplotlib.colors as mcolors
import gsw
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
from glidertest import utilities, tools
import os

dir = os.path.dirname(os.path.realpath(__file__))
glidertest_style_file = f"{dir}/glidertest.mplstyle"



def plot_updown_bias(df: pd.DataFrame, ax: plt.Axes = None, xlabel='Temperature [C]', **kw: dict, ) -> tuple({plt.Figure, plt.Axes}):
    """
    This function can be used to plot the up and downcast differences computed with the updown_bias function
    
    Parameters
    ----------
    df: pandas dataframe containing dc (Dive - Climb average), cd (Climb - Dive average) and depth
    ax: axis to plot the data
    xlabel: label for the x-axis
    
    Returns
    -------
    A line plot comparing the day and night average over depth for the selected day

    Original author
    ----------------
    Chiara Monforte
    """
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig, ax = plt.subplots()
            third_width = fig.get_size_inches()[0] / 3.11
            fig.set_size_inches(third_width, third_width *1.1)
            force_plot = True
        else:
            fig = plt.gcf()
            force_plot = False

        if not all(hasattr(df, attr) for attr in ['dc', 'depth']):
            ax.text(0.5, 0.55, xlabel, va='center', ha='center', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
            ax.text(0.5, 0.45, 'data unavailable', va='center', ha='center', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        else:
            ax.plot(df.dc, df.depth, label='Dive-Climb')
            ax.plot(df.cd, df.depth, label='Climb-Dive')
            ax.legend(loc=3)
            lims = np.abs(df.dc)
            ax.set_xlim(-np.nanpercentile(lims, 99.5), np.nanpercentile(lims, 99.5))
            ax.set_ylim(df.depth.max() + 1, -df.depth.max() / 30)
        ax.set_xlabel(xlabel)
        ax.grid()
        if force_plot:
            plt.show()
    return fig, ax

def plot_basic_vars(ds: xr.Dataset, v_res=1, start_prof=0, end_prof=-1):
    """
    This function plots the basic oceanographic variables temperature, salinity and density. A second plot is created and filled with oxygen and 
    chlorophyll data if available.
    
    Parameters
    ----------
    ds: xarray in OG1 format containing at least temperature, salinity and density and depth
    v_res: vertical resolution for the gridding. Horizontal resolution (by profile) is assumed to be 1
    start_prof: start profile used to compute the means that will be plotted. This can vary in case the dataset spread over a large timescale
                or region and subsections want to be plotted-1     
    end_prof: end profile used to compute the means that will be plotted. This can vary in case the dataset spread over a large timescale
              or region and subsections want to be plotted-1          
    
    Returns
    -------
    Line plots for the averages of the different variables.
    
    Original author
    ----------------
    Chiara Monforte
    """
    utilities._check_necessary_variables(ds, ['PROFILE_NUMBER', 'DEPTH', 'TEMP', 'PSAL', 'LATITUDE', 'LONGITUDE'])
    ds = utilities._calc_teos10_variables(ds)
    p = 1
    z = v_res
    tempG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds.TEMP, p, z)
    salG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds.PSAL, p, z)
    denG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds.DENSITY, p, z)


    tempG = tempG[start_prof:end_prof, :]
    salG = salG[start_prof:end_prof, :]
    denG = denG[start_prof:end_prof, :]
    depthG = depthG[start_prof:end_prof, :]

    with plt.style.context(glidertest_style_file):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fig, ax = plt.subplots(1, 2)
            # Resize to half-width
            half_width = fig.get_size_inches()[0] / 2.07
            fig.set_size_inches(half_width, half_width * 0.85)
            ax1 = ax[0].twiny()
            ax2 = ax[0].twiny()
            ax2.spines["top"].set_position(("axes", 1.2))
            ax[0].plot(np.nanmean(tempG, axis=0), depthG[0, :], c='blue')
            ax1.plot(np.nanmean(salG, axis=0), depthG[0, :], c='red')
            ax2.plot(np.nanmean(denG, axis=0), depthG[0, :], c='black')

            ax[0].set(xlabel=f'Temperature [C]', ylabel='Depth (m)')
            ax[0].tick_params(axis='x', colors='blue')
            ax[0].xaxis.label.set_color('blue')
            ax1.spines['bottom'].set_color('blue')
            ax1.set(xlabel=f'Salinity [PSU]')
            ax1.xaxis.label.set_color('red')
            ax1.spines['top'].set_color('red')
            ax1.tick_params(axis='x', colors='red')
            ax2.spines['bottom'].set_color('black')
            ax2.set(xlabel=f'Density [kg m-3]')
            ax2.xaxis.label.set_color('black')
            ax2.spines['top'].set_color('black')
            ax2.tick_params(axis='x', colors='black')

            # Add text annotation to the right, outside of the plot
            ax2.text(1.2, 1.25, f'Averaged profiles {start_prof}-{end_prof}', transform=ax2.transAxes, 
                     verticalalignment='center', horizontalalignment='left', rotation=0, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

            if 'CHLA' in ds.variables:
                chlaG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds.CHLA, p, z)
                chlaG = chlaG[start_prof:end_prof, :]
                ax2_1 = ax[1].twiny()
                ax2_1.plot(np.nanmean(chlaG, axis=0), depthG[0, :], c='green')
                ax2_1.set(xlabel=f'Chlorophyll-a [mg m-3]')
                ax2_1.xaxis.label.set_color('green')
                ax2_1.spines['top'].set_color('green')
                ax2_1.tick_params(axis='x', colors='green')
            else:
                ax[1].text(0.3, 0.7, 'Chlorophyll data unavailable', va='top', transform=ax[1].transAxes)

            if 'DOXY' in ds.variables:
                oxyG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds.DOXY, p, z)
                oxyG = oxyG[start_prof:end_prof, :]
                ax[1].plot(np.nanmean(oxyG, axis=0), depthG[0, :], c='orange')
                ax[1].set(xlabel=f'Oxygen [mmol m-3]')
                ax[1].xaxis.label.set_color('orange')
                ax[1].spines['top'].set_color('orange')
                ax[1].tick_params(axis='x', colors='orange')
                ax[1].spines['bottom'].set_color('orange')
            else:
                ax[1].text(0.3, 0.5, 'Oxygen data unavailable', va='top', transform=ax[1].transAxes)
            [a.set_ylim(depthG.max(), 0) for a in ax]
            [a.grid() for a in ax]
            plt.show()
    return fig, ax


def process_optics_assess(ds, var='CHLA'):
    """
    Function to assess visually any drift in deep optics data and the presence of any possible negative data. This function returns  both plots and text
    
    Parameters
    ----------
    ds: xarray dataset in OG1 format containing at least time, depth and the selected optical variable
    var: name of the selected variable         
    
    Returns
    -------
    Text giving info on where and when negative data was observed
    Plot showing bottom data with a linear regression line to highlight any drift 

    Original author
    ----------------
    Chiara Monforte
    """
    utilities._check_necessary_variables(ds, [var, 'TIME', 'DEPTH'])
    # Check how much negative data there is
    neg_chl = np.round((len(np.where(ds[var] < 0)[0]) * 100) / len(ds[var]), 1)
    if neg_chl > 0:
        print(f'{neg_chl}% of scaled {var} data is negative, consider recalibrating data')
        # Check where the negative values occur and if we just see them at specific time of the mission or not
        start = ds.TIME[np.where(ds[var] < 0)][0]
        end = ds.TIME[np.where(ds[var] < 0)][-1]
        min_z = ds.DEPTH[np.where(ds[var] < 0)].min().values
        max_z = ds.DEPTH[np.where(ds[var] < 0)].max().values
        print(f'Negative data in present from {str(start.values)[:16]} to {str(end.values)[:16]}')
        print(f'Negative data is present between {"%.1f" % np.round(min_z, 1)} and {"%.1f" % np.round(max_z, 1)} ')
    else:
        print(f'There is no negative scaled {var} data, recalibration and further checks are still recommended ')
    # Check if there is any missing data throughout the mission
    if len(ds.TIME) != len(ds[var].dropna(dim='N_MEASUREMENTS').TIME):
        print(f'{var} data is missing for part of the mission')  # Add to specify where the gaps are
    else:
        print(f'{var} data is present for the entire mission duration')
    # Check bottom dark count and any drift there
    bottom_opt_data = ds[var].where(ds[var].DEPTH > ds.DEPTH.max() - (ds.DEPTH.max() * 0.1)).dropna(
        dim='N_MEASUREMENTS')
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0, len(bottom_opt_data)), bottom_opt_data)

    # Generate the plot
    with plt.style.context(glidertest_style_file):
        ax = sns.regplot(data=ds, x=np.arange(0, len(bottom_opt_data)), y=bottom_opt_data,
                         scatter_kws={"color": "grey"},
                         line_kws={"color": "red", "label": "y={0:.8f} x+{1:.5f}".format(slope, intercept)},
                         )
        ax.legend(loc=2)
        ax.grid()
        ax.set(ylim=(np.nanpercentile(bottom_opt_data, 0.5), np.nanpercentile(bottom_opt_data, 99.5)),
               xlabel='Measurements',
               ylabel=var)
        plt.show()
    percentage_change = (((slope * len(bottom_opt_data) + intercept) - intercept) / abs(intercept)) * 100

    if abs(percentage_change) >= 1:
        print(
            'Data from the deepest 10% of data has been analysed and data does not seem perfectly stable. An alternative solution for dark counts has to be considered. \nMoreover, it is recommended to check the sensor has this may suggest issues with the sensor (i.e water inside the sensor, temporal drift etc)')
        print(
            f'Data changed (increased or decreased) by {"%.1f" % np.round(percentage_change, 1)}% from the beginning to the end of the mission')
    else:
        print(
            f'Data from the deepest 10% of data has been analysed and data seems stable. These deep values can be used to re-assess the dark count if the no {var} at depth assumption is valid in this site and this depth')
    return ax


def plot_daynight_avg(day: pd.DataFrame, night: pd.DataFrame, ax: plt.Axes = None, sel_day=None,
                      xlabel='Chlorophyll [mg m-3]', **kw: dict, ) -> tuple({plt.Figure, plt.Axes}):
    """
    This function can be used to plot the day and night averages computed with the day_night_avg function
    
    Parameters
    ----------
    day: pandas dataframe containing the day averages
    night: pandas dataframe containing the night averages
    ax: axis to plot the data
    sel_day: selected day to plot. Defaults to the median day
    xlabel: label for the x-axis
    
    Returns
    -------
    A line plot comparing the day and night average over depth for the selected day

    Original author
    ----------------
    Chiara Monforte

    """
    if not sel_day:
        dates = list(day.date.dropna().values) + list(night.date.dropna().values)
        dates.sort()
        sel_day = dates[int(len(dates)/2)]
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig, ax = plt.subplots()
            force_plot = True
        else:
            fig = plt.gcf()
            force_plot = False
        ax.plot(night.where(night.date == sel_day).dropna().dat, night.where(night.date == sel_day).dropna().depth,
                label='Night time average')
        ax.plot(day.where(day.date == sel_day).dropna().dat, day.where(day.date == sel_day).dropna().depth,
                label='Daytime average')
        ax.legend()
        ax.invert_yaxis()
        ax.grid()
        ax.set(xlabel=xlabel, ylabel='Depth [m]')
        ax.set_title(sel_day)
        if force_plot:
            plt.show()
    return fig, ax


def plot_quench_assess(ds: xr.Dataset, sel_var: str, ax: plt.Axes = None, start_time=None,
                           end_time=None,start_prof=None, end_prof=None, ylim=45, **kw: dict, ) -> tuple({plt.Figure, plt.Axes}):
    """
    This function can be used to plot sections for any variable with the sunrise and sunset plotted over
    
    Parameters
    ----------
    ds: xarray on OG1 format containing at least time, depth, latitude, longitude and the selected variable. 
        Data should not be gridded.
    sel_var: selected variable to plot
    ax: axis to plot the data
    start_time: Start date of the data selection format 'YYYY-MM-DD'. As missions can be long and came make it hard to visualise NPQ effect. 
                Defaults to mid 4 days
    end_time: End date of the data selection format 'YYYY-MM-DD'. As missions can be long and came make it hard to visualise NPQ effect. 
                Defaults to mid 4 days
    start_prof: Start profile of the data selection. If no profile is specified, the specified time selection will be used or the mid 4 days of the deployment
    end_prof:  End profile of the data selection. If no profile is specified, the specified time selection will be used or the mid 4 days of the deployment
    ylim: specified limit for the maximum y-axis value. The minimum is computed as ylim/30
    
    Returns
    -------
    A section showing the variability of the selected data over time and depth

    Original author
    ----------------
    Chiara Monforte
    """
    utilities._check_necessary_variables(ds, ['TIME', sel_var, 'DEPTH'])
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig, ax = plt.subplots()
            full_width = fig.get_size_inches()[0] 
            fig.set_size_inches(full_width, full_width * 0.5)
        else:
            fig = plt.gcf()

        if "TIME" not in ds.indexes.keys():
            ds = ds.set_xindex('TIME')

        if not start_time:
            start_time = ds.TIME.mean() - np.timedelta64(2, 'D')
        if not end_time:
            end_time = ds.TIME.mean() + np.timedelta64(2, 'D')

        if start_prof and end_prof:
            t1 = ds.TIME.where(ds.PROFILE_NUMBER==start_prof).dropna(dim='N_MEASUREMENTS')[0]
            t2 = ds.TIME.where(ds.PROFILE_NUMBER==end_prof).dropna(dim='N_MEASUREMENTS')[-1]
            ds_sel = ds.sel(TIME=slice(t1,t2))
        else:
            ds_sel = ds.sel(TIME=slice(start_time, end_time))


        if len(ds_sel.TIME) == 0:
            msg = f"supplied limits start_time: {start_time} end_time: {end_time} do not overlap with dataset TIME range {str(ds.TIME.values.min())[:10]} - {str(ds.TIME.values.max())[:10]}"
            raise ValueError(msg)

        sunrise, sunset = utilities.compute_sunset_sunrise(ds_sel.TIME, ds_sel.LATITUDE, ds_sel.LONGITUDE)

        c = ax.scatter(ds_sel.TIME, ds_sel.DEPTH, c=ds_sel[sel_var], s=10, vmin=np.nanpercentile(ds_sel[sel_var], 0.5),
                       vmax=np.nanpercentile(ds_sel[sel_var], 99.5))
        ax.set_ylim(ylim, -ylim / 30)
        for n in np.unique(sunset):
            ax.axvline(np.unique(n), c='blue')
        for m in np.unique(sunrise):
            ax.axvline(np.unique(m), c='orange')
        ax.set_ylabel('Depth [m]')

        # Set x-tick labels based on duration of the selection
        # Could pop out as a utility plotting function?
        utilities._time_axis_formatter(ax, ds_sel, format_x_axis=True)

        plt.colorbar(c, label=f'{sel_var} [{ds[sel_var].units}]')
        plt.show()
    return fig, ax


def check_temporal_drift(ds: xr.Dataset, var: str, ax: plt.Axes = None, **kw: dict, ) -> tuple({plt.Figure, plt.Axes}):
    """
    This function can be used to plot sections for any variable with the sunrise and sunset plotted over
    
    Parameters
    ----------
    ds: xarray on OG1 format containing at least time, depth, latitude, longitude and the selected variable. 
        Data should not be gridded.
    var: selected variable to plot
    ax: axis to plot the data
    
    Returns
    -------
    A figure with two subplots. One is a section containing the data over time and depth. The other one is a scatter of data from the variable
    over depth and colored by date

    Original author
    ----------------
    Chiara Monforte
    """
    utilities._check_necessary_variables(ds, ['TIME', var, 'DEPTH'])
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig, ax = plt.subplots(1, 2)
            force_plot = True
        else:
            fig = plt.gcf()
            force_plot = False

        ax[0].scatter(mdates.date2num(ds.TIME), ds[var], s=10)
        # Set x-tick labels based on duration of the selection
        utilities._time_axis_formatter(ax[0], ds, format_x_axis=True)

        ax[0].set(ylim=(np.nanpercentile(ds[var], 0.01), np.nanpercentile(ds[var], 99.99)), ylabel=var)

        c = ax[1].scatter(ds[var], ds.DEPTH, c=mdates.date2num(ds.TIME), s=10)
        ax[1].set(xlim=(np.nanpercentile(ds[var], 0.01), np.nanpercentile(ds[var], 99.99)), ylabel='Depth (m)', xlabel=var)
        ax[1].invert_yaxis()

        [a.grid() for a in ax]
        plt.colorbar(c, format=DateFormatter('%b %d'))
        if force_plot:
            plt.show()
    return fig, ax


def plot_prof_monotony(ds: xr.Dataset, ax: plt.Axes = None, **kw: dict, ) -> tuple({plt.Figure, plt.Axes}):

    """
    This function can be used to plot the profile number and check for any possible issues with the profile index assigned.

    Parameters
    ----------
    ds: xarray dataset in OG1 format with at least PROFILE_NUMBER, TIME, DEPTH. Data should not be gridded
    ax: axis to plot the data

    Returns 
    -------
    Two plots, one line plot with the profile number over time (expected to be always increasing). A
    second plot which is a scatter plot showing at which depth over time there was a profile index where the
    difference was neither 0 nor 1 (meaning there are possibly issues with how the profile index was assigned).

    Original author
    ----------------
    Chiara Monforte

    """
    utilities._check_necessary_variables(ds, ['TIME', 'PROFILE_NUMBER', 'DEPTH'])
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig, ax = plt.subplots(2, 1, sharex=True)
        else:
            fig = plt.gcf()

        ax[0].plot(ds.TIME, ds.PROFILE_NUMBER)

        ax[0].set(ylabel='Profile number')
        if len(np.where((np.diff(ds.PROFILE_NUMBER) != 0) & (np.diff(ds.PROFILE_NUMBER) != 1))[0]) == 0:
            ax[1].text(0.2, 0.5, 'Data are monotonically increasing - no issues identified',
                       transform=ax[1].transAxes)
        else:
            ax[1].scatter(ds.TIME[np.where((np.diff(ds.PROFILE_NUMBER) != 0) & (np.diff(ds.PROFILE_NUMBER) != 1))],
                          ds.DEPTH[np.where((np.diff(ds.PROFILE_NUMBER) != 0) & (np.diff(ds.PROFILE_NUMBER) != 1))],
                          s=10, c='red', label='Depth at which we have issues \n with the profile number assigned')
            ax[1].legend()
        ax[1].set(ylabel='Depth (m)')
        ax[1].invert_yaxis()
        ax[1].xaxis.set_major_locator(plt.MaxNLocator(8))
        utilities._time_axis_formatter(ax[1], ds, format_x_axis=True)
        [a.grid() for a in ax]
        plt.show()
    return fig, ax


def plot_glider_track(ds: xr.Dataset, ax: plt.Axes = None, **kw: dict) -> tuple({plt.Figure, plt.Axes}):
    """
    This function plots the glider track on a map, with latitude and longitude colored by time.

    Parameters
    ----------
    ds: xarray in OG1 format with at least TIME, LATITUDE, and LONGITUDE.
    ax: Optional; axis to plot the data.
    kw: Optional; additional keyword arguments for the scatter plot.

    Returns
    -------
    One plot with the map of the glider track.
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes._subplots.AxesSubplot

    Original author
    ----------------
    Eleanor Frajka-Williams
    """
    utilities._check_necessary_variables(ds, ['TIME', 'LONGITUDE', 'LATITUDE'])
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        else:
            fig = plt.gcf()

        latitudes = ds.LATITUDE.values
        longitudes = ds.LONGITUDE.values
        times = ds.TIME.values

        # Reduce the number of latitudes, longitudes, and times to no more than 2000
        if len(latitudes) > 2000:
            indices = np.linspace(0, len(latitudes) - 1, 2000).astype(int)
            latitudes = latitudes[indices]
            longitudes = longitudes[indices]
            times = times[indices]

        # Plot latitude and longitude colored by time
        sc = ax.scatter(longitudes, latitudes, c=times, cmap='viridis', **kw)

        # Add colorbar with formatted time labels
        cbar = plt.colorbar(sc, ax=ax) #, label='Time')
        cbar.ax.set_yticklabels([pd.to_datetime(t).strftime('%Y-%b-%d') for t in cbar.get_ticks()])

        ax.set_extent([np.min(longitudes)-1, np.max(longitudes)+1, np.min(latitudes)-1, np.max(latitudes)+1], crs=ccrs.PlateCarree())

        # Add features to the map
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Glider Track')
        gl = ax.gridlines(draw_labels=True, color='black', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        plt.show()

    return fig, ax

def plot_grid_spacing(ds: xr.Dataset, ax: plt.Axes = None, **kw: dict) -> tuple({plt.Figure, plt.Axes}):
    """
    This function plots histograms of the grid spacing (diff(ds.DEPTH) and diff(ds.TIME)) where only the inner 99% of values are plotted.

    Parameters
    ----------
    ds: xarray in OG1 format with at least TIME and DEPTH.
    ax: Optional; axis to plot the data.
    kw: Optional; additional keyword arguments for the histograms.

    Returns
    -------
    Two histograms showing the distribution of grid spacing for depth and time.
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes._subplots.AxesSubplot

    Original author
    ----------------
    Eleanor Frajka-Williams
    """
    utilities._check_necessary_variables(ds, ['TIME', 'DEPTH'])
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig, ax = plt.subplots(1, 2)
            # Set aspect ration of plot to be 2:1
            fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[0] / 2)
        else:
            fig = plt.gcf()
        # Set font sizes for all annotations
        #def_font_size = 14

        # Calculate the depth and time differences
        depth_diff = np.diff(ds.DEPTH)
        orig_time_diff = np.diff(ds.TIME) / np.timedelta64(1, 's')  # Convert to seconds

        # Remove NaN values
        depth_diff = depth_diff[np.isfinite(depth_diff)]
        time_diff = orig_time_diff[np.isfinite(orig_time_diff)]

        # Calculate some statistics (using original data)
        median_neg_depth_diff = np.median(depth_diff[depth_diff < 0])
        median_pos_depth_diff = np.median(depth_diff[depth_diff > 0])
        max_depth_diff = np.max(depth_diff)
        min_depth_diff = np.min(depth_diff)

        median_time_diff = np.median(orig_time_diff)
        mean_time_diff = np.mean(orig_time_diff)
        max_time_diff = np.max(orig_time_diff)
        min_time_diff = np.min(orig_time_diff)
        max_time_diff_hrs = max_time_diff/3600

        # Remove the top and bottom 0.5% of values to get a better histogram
        # This is hiding some data from the user
        depth_diff = depth_diff[(depth_diff >= np.nanpercentile(depth_diff, 0.5)) & (depth_diff <= np.nanpercentile(depth_diff, 99.5))]
        time_diff = time_diff[(time_diff >= np.nanpercentile(time_diff, 0.5)) & (time_diff <= np.nanpercentile(time_diff, 99.5))]
        print('Depth and time differences have been filtered to the middle 99% of values.')
        print('Numeric median/mean/max/min values are based on the original data.')

        # Histogram of depth spacing
        ax[0].hist(depth_diff, bins=50, **kw)
        ax[0].set_xlabel('Depth Spacing (m)')
        ax[0].set_ylabel('Frequency')
        ax[0].set_title('Histogram of Depth Spacing')

        annotation_text_left = (
            f'Median Negative: {median_neg_depth_diff:.2f} m\n'
            f'Median Positive: {median_pos_depth_diff:.2f} m\n'
            f'Max: {max_depth_diff:.2f} m\n'
            f'Min: {min_depth_diff:.2f} m'
        )
        # Determine the best location for the annotation based on the x-axis limits
        x_upper_limit = ax[0].get_xlim()[1]
        x_lower_limit = ax[0].get_xlim()[0]
        if abs(x_lower_limit) > abs(x_upper_limit):
            annotation_loc = (0.04, 0.96)  # Top left
            ha = 'left'
        else:
            annotation_loc = (0.96, 0.96)  # Top right
            ha = 'right'
        ax[0].annotate(annotation_text_left, xy=annotation_loc, xycoords='axes fraction',
                       ha=ha, va='top',
                       bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', alpha=.5))

        # Histogram of time spacing
        ax[1].hist(time_diff, bins=50, **kw)
        ax[1].set_xlabel('Time Spacing (s)')
        ax[1].set_ylabel('Frequency')
        ax[1].set_title('Histogram of Time Spacing')

        annotation_text = (
            f'Median: {median_time_diff:.2f} s\n'
            f'Mean: {mean_time_diff:.2f} s\n'
            f'Max: {max_time_diff:.2f} s ({max_time_diff_hrs:.2f} hr)\n'
            f'Min: {min_time_diff:.2f} s'
        )
        ax[1].annotate(annotation_text, xy=(0.96, 0.96), xycoords='axes fraction'
                       , ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', alpha=.5))

        # Set font sizes for all annotations
        # Font size 14 looks roughly like fontsize 8 when I drop this figure in Word - a bit small
        # Font size 14 looks like fontsize 13 when I drop the top *half* of this figure in powerpoint - acceptable
        for axes in ax:
            axes.tick_params(axis='both', which='major')
            # More subtle grid lines
            axes.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')
        plt.show()

    return fig, ax

def plot_sampling_period_all(ds: xr.Dataset):
    """
    This function plots several histograms of the sampling period for several variables (TEMP/PSAL by default and DOXY/CHLA if present) after removing their nans.

    Parameters
    ----------
    ds: xarray in OG1 format.

    Returns
    -------
    One histogram for each variable showing the distribution of the sampling period.

    Original author
    ----------------
    Louis Clement
    """

    count_vars=2
    vars=['TEMP', 'PSAL']
    if 'DOXY' in set(ds.variables): 
        count_vars+=1
        vars.append('DOXY')
    if 'CHLA' in set(ds.variables): 
        count_vars+=1
        vars.append('CHLA')

    fig, ax = plt.subplots(1, count_vars, figsize=(5*count_vars, 6))
    for i in range(len(vars)):
        ax[i] = plot_sampling_period(ds, ax[i], vars[i])
    plt.show()

def plot_sampling_period(ds: xr.Dataset, ax: plt.Axes = None, vars='TEMP'):
    """
    Similar to plot_grid_spacing, this function plots histograms but of the sampling period for one variable after removing its nans.
    

    Parameters
    ----------
    ds: xarray in OG1 format.
    ax: Optional, axis to plot the data.
    vars: the variable 

    Returns
    -------
    One histogram showing the distribution of the sampling period.
    ax: matplotlib.axes._subplots.AxesSubplot

    Original author
    ----------------
    Louis Clement
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6))

    nonan = ~np.isnan(ds[vars].values)
    time_diff = np.diff(ds.TIME.values[nonan]) / np.timedelta64(1, 's')  # Convert to seconds

    median_time_diff = np.median(time_diff)
    mean_time_diff = np.mean(time_diff)
    max_time_diff = np.max(time_diff)
    min_time_diff = np.min(time_diff)
    max_time_diff_hrs = max_time_diff/3600

    # Remove the top and bottom 0.5% of values to get a better histogram
    # This is hiding some data from the user
    time_diff = time_diff[(time_diff >= np.nanpercentile(time_diff, 0.5)) & (time_diff <= np.nanpercentile(time_diff, 99.5))]
    if vars=='TEMP': 
        print('Depth and time differences have been filtered to the middle 99% of values.')
        print('Numeric median/mean/max/min values are based on the original data.')

    ax.hist(time_diff, bins=50)
    ax.set_xlabel('Time Spacing (s)')
    if vars=='TEMP': ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Sampling Period' + '\n' + 
                 'for ' + vars + ', \n' + 
                 'valid values: {:.1f}'.format(100*(np.sum(nonan)/ds.TIME.values.shape[0]))+'%')

    annotation_text = (
        f'Median: {median_time_diff:.2f} s\n'
        f'Mean: {mean_time_diff:.2f} s\n'
        f'Max: {max_time_diff:.2f} s ({max_time_diff_hrs:.2f} hr)\n'
        f'Min: {min_time_diff:.2f} s'
    )
    ax.annotate(annotation_text, xy=(0.96, 0.96), xycoords='axes fraction'
                    , ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', alpha=.5))

    ax.tick_params(axis='both', which='major')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')

    return ax

def plot_ts(ds: xr.Dataset, axs: plt.Axes = None, **kw: dict) -> tuple({plt.Figure, plt.Axes}):
    """
    This function plots histograms of temperature and salinity values (middle 95%), and a 2D histogram of salinity and temperature with density contours.

    Parameters
    ----------
    ds: xarray in OG1 format with at least TEMP and PSAL.
    ax: Optional; axis to plot the data.
    kw: Optional; additional keyword arguments for the histograms.

    Returns
    -------
    Three plots: histogram of temperature, histogram of salinity, and 2D histogram of salinity and temperature with density contours.
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes._subplots.AxesSubplot

    Original author
    ----------------
    Eleanor Frajka-Williams
    """
    utilities._check_necessary_variables(ds, ['DEPTH', 'LONGITUDE', 'LATITUDE', 'PSAL', 'TEMP'])
    with plt.style.context(glidertest_style_file):
        if axs is None:
            fig, ax = plt.subplots(2, 3)
            plt.subplots_adjust(wspace=0.03, hspace=0.03)
            force_plot = True
            axs = ax.flatten()
        else:
            fig = plt.gcf()
            plt.subplots_adjust(wspace=0.03, hspace=0.03)
            force_plot = False
        axs[3].set_visible(False)
        axs[5].set_visible(False)
        num_bins = 30

        temp_orig = ds.TEMP.values
        sal_orig = ds.PSAL.values

        # Reduce both to where both are finite
        temp = temp_orig[np.isfinite(temp_orig) & np.isfinite(sal_orig)]
        sal = sal_orig[np.isfinite(sal_orig) & np.isfinite(temp_orig)]
        depth = ds.DEPTH[np.isfinite(sal_orig) & np.isfinite(temp_orig)]
        long = ds.LONGITUDE[np.isfinite(sal_orig) & np.isfinite(temp_orig)]
        lat = ds.LATITUDE[np.isfinite(sal_orig) & np.isfinite(temp_orig)]

        SA = gsw.SA_from_SP(sal, depth, long, lat)
        CT = gsw.CT_from_t(SA, temp, depth)

        # Reduce to middle 99% of values
        # This helps a lot for plotting, but is also hiding some of the data (not great for a test)
        CT_filtered = CT[(CT >= np.nanpercentile(CT, .5)) & (CT <= np.nanpercentile(CT, 99.5))]
        SA_filtered = SA[(SA >= np.nanpercentile(SA, .5)) & (SA <= np.nanpercentile(SA, 99.5))]
        print('Temperature and Salinity values have been filtered to the middle 99% of values.')

        # Calculate density to add contours
        xi = np.linspace(SA_filtered.values.min() - .2, SA_filtered.values.max() + .2, 100)
        yi = np.linspace(CT_filtered.values.min() - .2, CT_filtered.values.max() + .2, 100)
        xi, yi = np.meshgrid(xi, yi)
        zi = gsw.sigma0(xi, yi)

        # Temperature histogram
        axs[0].hist(CT_filtered, bins=num_bins, orientation="horizontal", **kw)
        axs[0].set_ylabel('Conservative Temperature (°C)')
        axs[0].set_xlabel('Frequency', rotation="horizontal")
        axs[0].invert_xaxis()

        # Salinity histogram
        axs[4].hist(SA_filtered, bins=num_bins, **kw)
        axs[4].set_xlabel('Absolute Salinity ( )')
        axs[4].set_ylabel('Frequency', rotation="vertical")
        axs[4].yaxis.set_label_position("right")
        axs[4].yaxis.tick_right()
        axs[4].invert_yaxis()

        for tick in axs[1].xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        for tick in axs[1].yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

        # 2-d T-S histogram
        h = axs[1].hist2d(SA_filtered, CT_filtered, bins=num_bins, cmap='viridis', norm=mcolors.LogNorm(), **kw)
        axs[1].contour(xi, yi, zi, colors='black', alpha=0.5, linewidths=0.5)
        axs[1].clabel(axs[1].contour(xi, yi, zi, colors='black', alpha=0.5, linewidths=0.5), inline=True)
        cbar = fig.colorbar(h[3], orientation='vertical',cax=axs[2])
        cbar.set_label('Log Counts')
        axs[1].set_title('2D Histogram \n (Log Scale)')
        #Resize axs[2] as colorbar
        box2 = axs[2].get_position()
        axs[2].set_position([box2.x0, box2.y0, box2.width / 6, box2.height])
        # Set x-limits based on salinity plot and y-limits based on temperature plot
        axs[1].set_xlim(axs[4].get_xlim())
        axs[1].set_ylim(axs[0].get_ylim())

        # Set font sizes for all annotations
        for axes in [axs[0],axs[1],axs[4]]:
            axes.tick_params(axis='both', which='major')
            axes.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')
        if force_plot:
            plt.show()
        all_ax = axs
        return fig, all_ax
def plot_vertical_speeds_with_histograms(ds, start_prof=None, end_prof=None):
    """
    Plot vertical speeds with histograms for diagnostic purposes.
    This function generates a diagnostic plot for the calculation of vertical seawater velocity.
    It plots the modelled and computed (from dz/dt) vertical velocities as line plots, where these
    should be similar. The difference between these velocities is the implied seawater velocity,
    which should be closer to zero than the vehicle velocities. The histogram provides a visual
    representation to identify any biases. The final calculation of the median should be close to
    zero if a large enough sample of dives is input and if the glider flight model is well-tuned.

    Parameters
    ----------
    ds (xarray.Dataset): The dataset containing the vertical speed data where
    - VERT_GLIDER_SPEED is the modelled glider speed
    - VERT_SPEED_DZDT is the computed glider speed from the pressure sensor
    - VERT_SW_SPEED is the implied seawater velocity.
    start_prof (int, optional): The starting profile number for subsetting the dataset. Defaults to first profile number.
    end_prof (int, optional): The ending profile number for subsetting the dataset. Defaults to last profile number.

    Returns
    -------
    fig, axs (tuple): The figure and axes objects for the plot.

    Original author
    ----------------
    Eleanor Frajka-Williams
    """
    utilities._check_necessary_variables(ds, ['GLIDER_VERT_VELO_MODEL', 'GLIDER_VERT_VELO_DZDT', 'VERT_CURR_MODEL','PROFILE_NUMBER'])
    with plt.style.context(glidertest_style_file):
        if start_prof is None:
            start_prof = int(ds['PROFILE_NUMBER'].values.mean())-10

        if end_prof is None:
            end_prof = int(ds['PROFILE_NUMBER'].values.mean())+10

        ds = ds.where((ds['PROFILE_NUMBER'] >= start_prof) & (ds['PROFILE_NUMBER'] <= end_prof), drop=True)
        vert_curr = ds.VERT_CURR_MODEL.values * 100  # Convert to cm/s
        vert_dzdt = ds.GLIDER_VERT_VELO_DZDT.values * 100  # Convert to cm/s
        vert_model = ds.GLIDER_VERT_VELO_MODEL.values * 100  # Convert to cm/s

        # Calculate the median line for the lower right histogram
        median_vert_sw_speed = np.nanmedian(vert_curr)

        # Create a dictionary to map the variable names to their labels for legends
        labels_dict = {
            'vert_dzdt': 'w$_{meas}$ (from dz/dt)',
            'vert_model': 'w$_{model}$ (flight model)',
            'vert_curr': 'w$_{sw}$ (calculated)'
        }

        fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [3, 1]})

        # Upper left subplot for vertical velocity and glider speed
        ax1 = axs[0, 0]
        ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5)  # Add zero horizontal line
        ax1.plot(ds['TIME'], vert_dzdt, label=labels_dict['vert_dzdt'])
        ax1.plot(ds['TIME'], vert_model, color='r', label=labels_dict['vert_model'])
        ax1.plot(ds['TIME'], vert_curr, color='g', label=labels_dict['vert_curr'])
        # Annotations
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Vertical Velocity (cm/s)')
        ax1.legend(loc='lower left')
        ax1.legend(loc='lower right')
        utilities._time_axis_formatter(ax1, ds, format_x_axis=True)

        # Upper right subplot for histogram of vertical velocity
        ax1_hist = axs[0, 1]
        ax1_hist.hist(vert_dzdt, bins=50, orientation='horizontal', alpha=0.5, color='blue', label=labels_dict['vert_dzdt'])
        ax1_hist.hist(vert_model, bins=50, orientation='horizontal', alpha=0.5, color='red', label=labels_dict['vert_model'])
        ax1_hist.hist(vert_curr, bins=50, orientation='horizontal', alpha=0.5, color='green', label=labels_dict['vert_curr'])
        ax1_hist.set_xlabel('Frequency')

        # Determine the best location for the legend based on the y-axis limits and zero
        y_upper_limit = ax1_hist.get_ylim()[1]
        y_lower_limit = ax1_hist.get_ylim()[0]
        if abs(y_upper_limit) > abs(y_lower_limit):
            legend_loc = 'upper right'
        else:
            legend_loc = 'lower right'
        plt.rcParams['legend.fontsize'] = 12
        ax1_hist.legend(loc=legend_loc)
        plt.rcParams['legend.fontsize'] = 15
        # Lower left subplot for vertical water speed
        ax2 = axs[1, 0]
        ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)  # Add zero horizontal line
        ax2.plot(ds['TIME'], vert_curr, 'g', label=labels_dict['vert_curr'])
        # Annotations
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Vertical Water Speed (cm/s)')
        ax2.legend(loc='upper left')
        utilities._time_axis_formatter(ax2, ds, format_x_axis=True)

        # Lower right subplot for histogram of vertical water speed
        ax2_hist = axs[1, 1]
        ax2_hist.hist(vert_curr, bins=50, orientation='horizontal', alpha=0.5, color='green', label=labels_dict['vert_curr'])
        ax2_hist.axhline(median_vert_sw_speed, color='red', linestyle='dashed', linewidth=1, label=f'Median: {median_vert_sw_speed:.2f} cm/s')
        ax2_hist.set_xlabel('Frequency')

        # Determine the best location for the legend based on the y-axis limits and median
        y_upper_limit = ax2_hist.get_ylim()[1]
        y_lower_limit = ax2_hist.get_ylim()[0]
        if abs(y_upper_limit - median_vert_sw_speed) > abs(y_lower_limit - median_vert_sw_speed):
            legend_loc = 'upper right'
        else:
            legend_loc = 'lower right'
        ax2_hist.legend(loc=legend_loc)

        # Set font sizes for all annotations
        # Font size 14 looks roughly like fontsize 8 when I drop this figure in Word - a bit small
        # Font size 14 looks like fontsize 13 when I drop the top *half* of this figure in powerpoint - acceptable

        for ax in [ax1, ax2, ax1_hist, ax2_hist]:
            ax.tick_params(axis='both', which='major')

        # Adjust the axes so that the distance between y-ticks on the top and lower panel is the same
        # Get the y-axis range of the top left plot
        y1_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
        # Get the y-axis limits of the lower left plot
        y2_range = ax2.get_ylim()[1] - ax2.get_ylim()[0]
        # Get the height in inches of the top left plot
        box1 = ax1.get_position()
        height1 = box1.height
        # Get the height in inches of the lower left plot
        box2 = ax2.get_position()
        height2 = box2.height
        # Set a scaled height for the lower left plot
        new_height = height1 * y2_range / y1_range
        # Determine the change in height
        height_change = height2 - new_height
        # Shift the y-position of the lower left plot by the change in height
        ax2.set_position([box2.x0, box2.y0 + height_change, box2.width, new_height])

        # Get the position of the lower right plot
        box2_hist = ax2_hist.get_position()
        # Adjust the position of the lower right plot to match the height of the lower left plot
        ax2_hist.set_position([box2_hist.x0, box2_hist.y0 + height_change, box2_hist.width, new_height])

        # Find the distance between the right edge of the top left plot and the left edge of the top right plot
        box1_hist = ax1_hist.get_position()
        distance =  box1_hist.x0 - (box1.x0 + box1.width)
        shift_dist = distance/3 # Not sure this will always work; it may depend on the def_fault_size
        # Adjust the width of the top right plot to extend left by half the distance
        ax1_hist.set_position([box1_hist.x0 - shift_dist, box1_hist.y0, box1_hist.width + shift_dist, box1_hist.height])
        # Adjust the width of the bottom right plot to extend left by half the distance
        box2_hist = ax2_hist.get_position()
        ax2_hist.set_position([box2_hist.x0 - shift_dist, box2_hist.y0, box2_hist.width + shift_dist, box2_hist.height])

    plt.show()

    return fig, axs

def plot_combined_velocity_profiles(ds_out_dives: xr.Dataset, ds_out_climbs: xr.Dataset):
    """
    Plots combined vertical velocity profiles for dives and climbs.

    Replicates Fig 3 in Frajka-Williams et al. 2011, but using an updated dataset from Jim Bennett (2013), 
    now in OG1 format as sg014_20040924T182454_delayed.nc.  Note that flight model parameters may differ from those in the paper.

    Parameters
    ----------
    ds_out_dives (xarray.Dataset): Dataset containing dive profiles with variables 'zgrid', 'meanw', 'w_lower', and 'w_upper'.
    ds_out_climbs (xarray.Dataset): Dataset containing climb profiles with variables 'zgrid', 'meanw', 'w_lower', and 'w_upper'.

    The function converts vertical velocities from m/s to cm/s, plots the mean vertical velocities and their ranges for both dives and climbs, and customizes the plot with labels, legends, and axis settings.

    Note
    ----
    Assumes that the vertical velocities are in m/s and the depth grid is in meters.

    Original author
    ----------------
    Eleanor Frajka-Williams
    """
    conv_factor = 100  # Convert m/s to cm/s
    depth_negative = ds_out_dives.zgrid.values * -1
    meanw_dives = ds_out_dives.meanw.values * conv_factor
    zgrid_dives = depth_negative
    w_lower_dives = ds_out_dives.w_lower.values * conv_factor
    w_upper_dives = ds_out_dives.w_upper.values * conv_factor

    meanw_climbs = ds_out_climbs.meanw.values * conv_factor
    zgrid_climbs = ds_out_climbs.zgrid.values * -1
    w_lower_climbs = ds_out_climbs.w_lower.values * conv_factor
    w_upper_climbs = ds_out_climbs.w_upper.values * conv_factor
    with plt.style.context(glidertest_style_file):
        fig, ax = plt.subplots(1, 1)
        # Resize to half-width
        half_width = fig.get_size_inches()[0] / 2.07
        fig.set_size_inches(half_width, half_width * 0.9)
        ax.tick_params(axis='both', which='major')

        # Plot dives
        ax.fill_betweenx(zgrid_dives, w_lower_dives, w_upper_dives, color='black', alpha=0.3)
        ax.plot(meanw_dives, zgrid_dives, color='black', label='w$_{dive}$')

        # Plot climbs
        ax.fill_betweenx(zgrid_climbs, w_lower_climbs, w_upper_climbs, color='red', alpha=0.3)
        ax.plot(meanw_climbs, zgrid_climbs, color='red', label='w$_{climb}$')

        ax.invert_yaxis()  # Invert y-axis to show depth increasing downwards
        ax.axvline(x=0, color='darkgray', linestyle='-', linewidth=0.5)  # Add vertical line through 0
        ax.set_xlabel('Vertical Velocity w$_{sw}$ (cm s$^{-1}$)')
        ax.set_ylabel('Depth (m)')
        ax.set_ylim(top=0, bottom=1000)  # Set y-limit maximum to zero
        #ax.set_title('Combined Vertical Velocity Profiles')

        ax.set_xlim(-1, 1.5)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1.0, 1.5])
        plt.tight_layout()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='major')
        ax.legend()
        plt.show()
        return fig, ax


def plot_hysteresis(ds, var='DOXY', v_res=1, perct_err=2, ax=None):
    """
    This function creates 4 plots which can help the user visualize any possible hysteresis
    present in their dataset for a specific variable

    Parameters
    ----------
    ds: xarray on OG1 format containing at least depth, profile_number and the selected variable.
        Data should not be gridded.
    var: Selected variable
    v_res: Vertical resolution for the gridding
    perct_err: Percentage error threshold used to plot a vertical line in one of the subplots
    ax: Specific axis in can the user wants to add this plot to an existing figure

    Returns
    -------
   df: pandas dataframe containing dive and climb average over depth for the selected variable. A third column contains the depth values

    Original author
    ----------------
    Chiara Monforte
    """
    varG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds[var], 1, v_res)
    df, diff, err, rms = tools.compute_hyst_stat(ds, var=var, v_res=v_res)
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig = plt.figure()
            ax = [plt.subplot(3, 3, 1), plt.subplot(3, 3, 2), plt.subplot(3, 3, 3), plt.subplot(3, 1, 2)]
            force_plot = True
        else:
            if len(ax) == 1:
                fig = plt.gcf()
                ax = [plt.subplot(3, 3, 1), plt.subplot(3, 3, 2), plt.subplot(3, 3, 3), plt.subplot(3, 1, 2)]
                force_plot = False
            else:
                fig = plt.gcf()
                force_plot = False

        ax[0].plot(df.climb, df.depth, label='Dive')
        ax[0].plot(df.dive, df.depth, label='Climb')
        ax[0].legend()
        ax[1].plot(diff, df.depth)
        ax[2].plot(err, df.depth)
        ax[2].axvline(perct_err, c='red')
        [a.grid() for a in ax]
        [a.invert_yaxis() for a in ax]
        ax[0].set_ylabel('Depth (m)')
        ax[0].set_xlabel(f'{var} concentration $=mean$ \n({ds[var].units})')
        ax[1].set_xlabel(f'Absolute difference = |$\Delta$| \n({ds[var].units})')
        ax[2].set_xlabel('Percent error = |$\Delta$|/$mean$ \n(%)')
        for ax1 in ax[:-1]:
            ax1.xaxis.set_label_position('top')
        c = ax[3].pcolor(profG[:-1, :], depthG[:-1, :], np.diff(varG, axis=0),
                         vmin=np.nanpercentile(np.diff(varG, axis=0), 0.5),
                         vmax=np.nanpercentile(np.diff(varG, axis=0), 99.5), cmap='seismic')
        plt.colorbar(c, ax=ax[3], label=f'Difference dive-climb \n({ds[var].units})', fraction=0.05)
        ax[3].set(ylabel='Depth (m)', xlabel='Profile number')
        fig.suptitle(var, y=.98)
        if force_plot:
            plt.show()
    return fig, ax


def plot_outlier_duration(ds: xr.Dataset, rolling_mean: pd.Series, overtime, std=2, ax=None):
    """
       Generates two plots to visualize profile durations and highlight outliers.
       This helps identify profiles with abnormal durations by comparing the actual profile durations
       to a rolling mean, and by visualizing the shape and depth of the selected outlier profiles.

       Parameters
       ----------
       ds : An xarray object containing at least the variables 'TIME', 'DEPTH', and 'PROFILE_NUMBER'.
       These are used to compute the profile durations and plot depth profiles.

       rolling_mean : A series representing the rolling mean of the profile durations,
       which is used to highlight outliers based on standard deviation.

       overtime :  A list of profile numbers identified as having unusual durations.
       These profiles are marked on the plot to highlight the outliers.

       std : float, optional, default 2
           The number of standard deviations above and below the rolling mean that will be used to define the range
           of "normal" durations. Profiles outside this range are considered outliers.

       ax :The axes object on which to plot the results. If not provided, a new figure with two subplots is created.

       Returns
       -------
       fig : The figure containing the generated plots.
           1. A plot showing the profile durations with the rolling mean and the range defined by the rolling mean ± `std`
              (standard deviation). The range is highlighted in orange.
           2. A scatter plot of the profile depths, with outlier profiles marked in red. These outliers are determined based
              on the duration exceeding the threshold defined by the rolling mean ± `std`.

       ax :  A 1x2 array of axes used for the two subplots.

       Original author
       ------
       Chiara Monforte
       """

    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig,ax = plt.subplots(1,2)
            force_plot = True
        else:
            fig = plt.gcf()
            force_plot = False

        data = tools.compute_prof_duration(ds)
        ax[0].plot(data['profile_num'], data['profile_duration'], label='Profile duration')
        ax[0].plot(data['profile_num'], rolling_mean, label='Rolling mean')
        ax[0].fill_between(data['profile_num'], rolling_mean - (np.std(rolling_mean) * std),
                           rolling_mean + (np.std(rolling_mean) * std), color='orange',
                           edgecolor="orange", alpha=0.5,
                           label=f'Rolling mean ± {std} std')
        ax[0].legend()
        ax[0].grid()
        ax[0].set(xlabel='Profile number', ylabel='Profile duration (min)')

        ax[1].scatter(ds.TIME, ds.DEPTH, s=0.1)
        for i in range(len(overtime)):
            profile = ds.TIME.where(ds.PROFILE_NUMBER == overtime[i]).dropna(dim='N_MEASUREMENTS')
            ax[1].scatter(profile.TIME, profile.DEPTH, s=0.1, c='red', label='Profiles with odd duration')
        ax[1].invert_yaxis()
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[1].legend(by_label.values(), by_label.keys(), markerscale=8., loc='lower right')
        ax[1].set(ylabel='Depth (m)')
        ax[1].grid()
        every_nth = 2
        for n, label in enumerate(ax[1].xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
        if force_plot:
            plt.show()
    return fig, ax


def plot_global_range(ds, var='DOXY', min_val=-5, max_val=600, ax=None):
    """
   This function generates a histogram of the specified variable (`var`) from the dataset (`ds`) and
    overlays vertical lines at the specified minimum (`min_val`) and maximum (`max_val`) values to
    visually represent the global range. The function is useful for visually inspecting whether
    values of the specified variable fall within the expected global range.

    Parameters
    ----------
    ds : The xarray dataset containing the variable (`var`) to be plotted.

    var : The name of the variable to plot. Default is 'DOXY'.

    min_val : The minimum value of the global range to highlight on the plot. Default is -5.

    max_val : The maximum value of the global range to highlight on the plot. Default is 600.

    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the histogram. If `None`, a new figure and axes are created.
        Default is `None`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.

    ax : matplotlib.axes.Axes
        The axes object containing the histogram plot.

    Original author
    ----------------
    Chiara Monforte
    """
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig, ax = plt.subplots()
            force_plot = True
        else:
            fig = plt.gcf()
            force_plot = False

        ax.hist(ds[var], bins=50)
        ax.axvline(min_val, c='r')
        ax.axvline(max_val, c='r')
        ax.set(xlabel=f'{ds[var].long_name} ({ds[var].units})', ylabel='Frequency')
        ax.set_title('Global range check')
        ax.grid()
        if force_plot:
            plt.show()
    return fig, ax


def plot_ioosqc(data, suspect_threshold=[25], fail_threshold=[50], title='', ax=None):
    """
    Plots a scatter plot of the the results of IOOS qQC tests with quality control labels (GOOD, UNKNOWN, SUSPECT, FAIL, MISSING) on the y-axis,
    and overlays threshold-based markers for suspect and fail values. This function is useful for visualizing the status of
    data points according to the quality control thresholds.

    Parameters:
    -----------
    data : The result from the IOOS_QC test.
            A sequence of numerical values representing the data points to be plotted.

    suspect_threshold : A list containing one or two numerical values indicating the thresholds for suspect values. If one value is provided,
        it applies to both lower and upper bounds for suspect data points. If two values are provided, they define the
        lower and upper bounds for suspect values.

    fail_threshold A list containing one or two numerical values indicating the thresholds for fail values. Similar to `suspect_threshold`,
        it can have one or two values to define the bounds for fail data points.

    title : str, optional, default = ''
        The title to display at the top of the plot.

    ax : matplotlib Axes object, optional, default = None
        If provided, the plot will be drawn on this existing Axes object. If None, a new figure and axis will be created.

    Returns:
    --------
    fig : matplotlib figure
        The figure object containing the plot.

    ax : matplotlib Axes object
        The axes object used for plotting.

    Notes:
    ------
    - The plot uses two y-axes: one for labeling data points as 'GOOD', 'UNKNOWN', 'SUSPECT', 'FAIL', or 'MISSING' based
      on thresholds, and another for marking specific suspect and fail ranges.
    Original author
    ----------------
    Chiara Monforte
    """
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig, ax = plt.subplots()
            force_plot = True
        else:
            fig = plt.gcf()
            force_plot = False

        ax.scatter(np.arange(len(data)), data, s=4)
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        a = ax.get_yticks().tolist()
        a[:] = ['', 'GOOD', 'UNKNOWN', 'SUSPECT', 'FAIL', '', '', '', '', 'MISSING', '']
        ax.set_yticklabels(a)
        ax2 = ax.twinx()
        ax2.scatter(np.arange(len(data)), data, s=4)
        ax2.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        a_2 = ax2.get_yticks().tolist()
        a_2[:] = ['', '', '', '', '', '', '', '', '', '', '']
        if len(suspect_threshold) > 1:
            a_2[1] = f'x>{suspect_threshold[0]} or \nx<{suspect_threshold[1]}'
        else:
            a_2[1] = f'x<{suspect_threshold[0]}'
        if len(fail_threshold) > 1:
            a_2[3] = f'{suspect_threshold[1]}<x<{fail_threshold[1]} &\n {fail_threshold[0]}<x<{suspect_threshold[0].values}'
            a_2[4] = f'x<{fail_threshold[0]} or \nx>{fail_threshold[1]}'
        else:
            a_2[3] = f'x>{suspect_threshold[0]} and \nx<{fail_threshold[0]}'
            a_2[4] = f'x>{fail_threshold[0]}'
        a_2[9] = 'Nan'

        ax2.set_yticklabels(a_2, fontsize=12)

        ax.set_xlabel('Data Index')
        ax.grid()
        ax.set_title(title)
        if force_plot:
            plt.show()
    return fig, ax
