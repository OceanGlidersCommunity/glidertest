import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from glidertest import tools, plots, utilities
import glidertest
from ioos_qc import qartod
import cartopy.crs as ccrs
from matplotlib.font_manager import FontProperties
from datetime import datetime
import os

configs = {
    "TEMP": {"gross_range_test": {
        "suspect_span": [0, 30],
        "fail_span": [-2.5, 40],
    },
        "spike_test": {"suspect_threshold": 2.0, "fail_threshold": 6.0},
        "location_test": {"bbox": [10, 50, 25, 70]},
    },
    "PSAL": {"gross_range_test": {"suspect_span": [5, 38], "fail_span": [2, 41]},
             "spike_test": {"suspect_threshold": 0.3, "fail_threshold": 0.9},
             "location_test": {"bbox": [10, 50, 25, 70]},
             },
    "DOXY": {"gross_range_test": {
        "suspect_span": [0, 350],
        "fail_span": [0, 500],
    },
        "spike_test": {"suspect_threshold": 10, "fail_threshold": 50},
        "location_test": {"bbox": [10, 50, 25, 70]},
    },
    "CHLA": {"gross_range_test": {
        "suspect_span": [0, 15],
        "fail_span": [-1, 20],
    },
        "spike_test": {"suspect_threshold": 1, "fail_threshold": 5},
        "location_test": {"bbox": [10, 50, 25, 70]},
    },
}


def qc_checks(ds, var='TEMP'):
    gr = tools.compute_global_range(ds, var=var, min_val=configs[var]['gross_range_test']['suspect_span'][0],
                                    max_val=configs[var]['gross_range_test']['suspect_span'][1])
    spike = qartod.spike_test(ds[var], suspect_threshold=configs[var]['spike_test']['suspect_threshold'],
                              fail_threshold=configs[var]['spike_test']['fail_threshold'], method="average")
    flat = qartod.flat_line_test(ds[var], ds.TIME, 1, 3, 0.001)
    __, __,__,err_range, __ = tools.compute_hyst_stat(ds, var=var, v_res=1)

    return gr, spike, flat, err_range


def fill_tableqc(df,ds, var='TEMP'):
    if var == 'TEMP':
        variable = 'Temperature'
    if var == 'PSAL':
        variable = 'Salinity'
    if var == 'DOXY':
        variable = 'Oxygen'
    if var == 'CHLA':
        variable = 'Chlorophyll'
    if var not in ds.variables:
        df.loc[:, variable] = 'No data'
    else:
        gr, spike, flat, err = qc_checks(ds, var=var)
        if len(gr) > 0:
            df.loc[0, variable] = 'X'
        if len(np.where((spike == 3) | (spike == 4))[0]) > 0:
            df.loc[1, variable] = 'X'
        if len(np.where((flat == 3) | (flat == 4))[0]) > 0:
            df.loc[2, variable] = 'X'
        if len(np.where(err > 5)[0]) > 5:
            df.loc[3, variable] = 'X'
    return df


def phrase_duration_check(ds):
    duration = tools.compute_prof_duration(ds)
    rolling_mean, overtime = tools.find_outlier_duration(duration, rolling=20, std=2)
    if len(overtime) > 0:
        duration_check = f'{len(overtime)} profiles have abnormal duration'
    else:
        duration_check = 'No issues with profile duration have been detected'
    return duration_check


def phrase_numberprof_check(ds):
    check = tools.check_monotony(ds.PROFILE_NUMBER)
    if check:
        prof_check = 'No issues detected'
    else:
        prof_check = 'Issues with profile number have been detected'
    return prof_check

# Page 1 with general info and location of the mission
def summary_plot(ds, save_dir='.', test=True):
    fig, ax = plt.subplots(figsize=(8.3, 11.7))
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(10)

    fig.subplots_adjust(hspace=0.35, wspace=0.15)
    font_size = 10

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('off')

    ### Create the text boxes
    if 'PLATFORM_SERIAL_NUMBER' in ds.variables:
        gserial = ds['PLATFORM_SERIAL_NUMBER'].values
    else:
        gserial = 'Unknown'
    if 'deployment_id' in ds.attrs:
        mission = ds.attrs['deployment_id']
    else:
        mission = 'Unknown'
    if 'contributor_name' in ds.attrs:
        cname = ds.attrs['contributor_name']
    else:
        cname = 'Unknown'

    ### Create the text boxes
    textstr = '\n'.join((
        f"Glider serial:  {gserial}",
        f"Mission number:  {mission}",
        f"Deployment date:  {ds.TIME[0].values.astype('datetime64[D]')}",
        f"Recovery date:  {ds.TIME[-1].values.astype('datetime64[D]')}",
        f"Mission duration:  {np.round(((ds.TIME.max() - ds.TIME.min()) / np.timedelta64(1, 'D')).astype(float).values, 1)} days",
        f"Diving depth range:  {int(tools.max_depth_per_profile(ds).min())} - {int(tools.max_depth_per_profile(ds).max())} m",
        f"Dataset creator:  {cname} \n({ds.contributing_institutions})",
    ))

    # Add text titles
    ax.text(-0.05, 1.05, 'Glidertest summary mission sheet', transform=ax.transAxes, fontsize=font_size + 10,
            weight='bold')
    ax.text(-0.05, 1.01, 'Main mission info', transform=ax.transAxes, fontsize=font_size, weight='bold')
    ax.text(-0.05, 0.64, 'Basic payload configuration', transform=ax.transAxes, fontsize=font_size, weight='bold')
    ax.text(0.45, 0.64, 'Basic water column structure', transform=ax.transAxes, fontsize=font_size, weight='bold')
    ax.text(-0.05, 0.99, textstr, transform=ax.transAxes, fontsize=font_size,
            verticalalignment='top')

    # Payload config table
    df = pd.DataFrame({'': ['Temperature', 'Salinity', 'Oxygen', 'Chlorophyll', 'Backscatter', 'Altimeter', 'ADCP'],
                       'Data exist': ['X', 'X', 'X', 'X', 'X', 'X', 'X']})

    if 'TEMP' in ds.variables:
        df.loc[0, 'Data exist'] = "✓"
    if 'PSAL' in ds.variables:
        df.loc[1, 'Data exist'] = "✓"
    if 'ALTITUDE' in ds.variables:
        df.loc[5, 'Data exist'] = "✓"
    if 'CHLA' in ds.variables:
        df.loc[3, 'Data exist'] = "✓"
    if 'BBP700' in ds.variables:
        df.loc[4, 'Data exist'] = "✓"
    if 'PRES_ADCP' in ds.variables:
        df.loc[6, 'Data exist'] = "✓"
    if 'DOXY' in ds.variables:
        df.loc[2, 'Data exist'] = "✓"
    tab_ax = fig.add_axes([0.16, 0.58, 0.15, 0.25], anchor='NE')
    fig.patch.set_visible(False)
    tab_ax.axis('off')
    tab_ax.axis('tight')
    table = tab_ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center')
    table.scale(2, 2)

    for (row, col), cell in table.get_celld().items():
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
    for (row, col), cell in table.get_celld().items():
        if (col == 0):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    # Basic checks
    ax.text(-0.05, 0.22, 'Basic checks', transform=ax.transAxes, fontsize=font_size, weight='bold')
    basictext = '\n'.join((
        f"Issues with profile number: {phrase_numberprof_check(ds)} ",
        f"Issues with profile duration: {phrase_duration_check(ds)}",))
    ax.text(-0.05, 0.21, basictext, transform=ax.transAxes, fontsize=font_size,
            verticalalignment='top')
    ## Basic checks table
    df2 = pd.DataFrame({'':['Global range', 'Spike test', 'Flat test', 'Hysteresis (mean)', 'Hysteresis (range)', 'Drift'],
                    'Temperature':['✓', '✓', '✓', '✓', '✓', '✓'], 
                    'Salinity':['✓', '✓', '✓', '✓', '✓', '✓'], 
                    'Oxygen':['✓', '✓', '✓', '✓', '✓', '✓'], 
                    'Chlorophyll':['✓', '✓', '✓', '✓', '✓', '✓']})
    tableT = fill_tableqc(df2,ds, var='TEMP')
    tableS = fill_tableqc(tableT,ds, var='PSAL')
    tableO = fill_tableqc(tableS,ds, var='DOXY')
    tableC = fill_tableqc(tableO,ds, var='CHLA')

    colrs = np.copy(tableC)
    colrs[np.where(colrs != 'X')] = 'w'
    colrs[np.where(colrs == 'X')] = 'lightcoral'

    tab2_ax = fig.add_axes([0.34, 0.24, 0.27, 0.55], anchor='NE')
    fig.patch.set_visible(False)
    tab2_ax.axis('off')
    tab2_ax.axis('tight')
    table2 = tab2_ax.table(cellText=df2.values, colLabels=df2.columns, cellColours=colrs, cellLoc='center')
    table2.scale(3, 2)


    for (row, col), cell in table2.get_celld().items():
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight='bold', size=55))
        if col == 0:
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    ## Add logo at the bottom
    im = plt.imread('img/logos.png')
    newax = fig.add_axes([0.88, -0.03, 0.1, 0.1], anchor='NE')
    newax.imshow(im)
    newax.axis('off')
    ax.text(0.78, -0.11, 'Created with Glidertest', transform=ax.transAxes, fontsize=font_size - 3,
            verticalalignment='top')

    # Add glider track plot
    gt_ax = fig.add_axes([0.47, 0.64, 0.38, 0.25], projection=ccrs.PlateCarree(), anchor='SE', zorder=-1, )
    plots.plot_glider_track(ds, ax=gt_ax)
    # Add basic variables profiles plot
    bv1_ax = fig.add_axes([0.5, 0.31, 0.21, 0.21], anchor='SE', zorder=-1)
    bv2_ax = fig.add_axes([0.78, 0.31, 0.21, 0.21], anchor='SE', zorder=-1)
    plots.plot_basic_vars(ds, v_res=1, start_prof=0, end_prof=-1, ax=[bv1_ax, bv2_ax])
    todays_date = datetime.today().strftime('%Y%m%d')
    if test:
        fig.savefig(f'{save_dir}/{gserial}_{mission}_report{todays_date}.pdf')
    return fig, ax


def summary_plot_template(ds, var='CHLA', save_dir=''):
    fig, ax = plt.subplots(figsize=(8.3, 11.7))
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(10)

    fig.subplots_adjust(hspace=0.35, wspace=0.15)
    font_size = 10

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('off')

    ### Basic info
    textstr = '\n'.join((
        f"Sensor name:  ",
        f"Sensor serial: ",
        f"Calibration date:  ",
        f"Calibration parameters:  ",
        f"Data availabiliy time period:{str(ds[var].dropna(dim='N_MEASUREMENTS').TIME[0].values.astype('datetime64[D]'))} - {str(ds[var].dropna(dim='N_MEASUREMENTS').TIME[-1].values.astype('datetime64[D]'))}",
        f"Data availabiliy depth range: {int(ds[var].dropna(dim='N_MEASUREMENTS').DEPTH.min())} - {int(ds[var].dropna(dim='N_MEASUREMENTS').DEPTH.max())} m",
    ))

    # Add text titles
    ax.text(-0.05, 1.05, f'Glidertest summary mission sheet - {var}', transform=ax.transAxes, fontsize=font_size + 10,
            weight='bold')
    ax.text(-0.05, 1.01, 'Main mission info', transform=ax.transAxes, fontsize=font_size, weight='bold')
    ax.text(-0.05, 0.64, 'Sampling resolution', transform=ax.transAxes, fontsize=font_size, weight='bold')
    ax.text(0.45, 0.64, 'Basic checks', transform=ax.transAxes, fontsize=font_size, weight='bold')
    ax.text(-0.05, 0.99, textstr, transform=ax.transAxes, fontsize=font_size,
            verticalalignment='top')
    # Section plots
    s_ax = fig.add_axes([0.58, 0.65, 0.45, 0.25], anchor='SE', zorder=-1, )
    c = s_ax.scatter(ds.TIME, ds.DEPTH, c=ds[var], s=10, vmin=np.nanpercentile(ds[var], 0.5),
                     vmax=np.nanpercentile(ds[var], 99.5))
    plt.colorbar(c, ax=s_ax, label=f'{utilities.plotting_labels(var)} \n({utilities.plotting_units(ds, var)})')
    s_ax.invert_yaxis()
    s_ax.set_ylabel('Depth (m)')
    utilities._time_axis_formatter(s_ax, ds, format_x_axis=True)
    # Sampling period plot
    sp_ax = fig.add_axes([0.08, 0.32, 0.35, 0.22], anchor='SE', zorder=-1, )
    plots.plot_sampling_period(ds, sp_ax, variable=var)

    # Basic check tests or other plots
    bc_ax1 = fig.add_axes([0.58, 0.48, 0.20, 0.10], anchor='SE', zorder=-1, )
    bc_ax2 = fig.add_axes([0.58, 0.34, 0.20, 0.10], anchor='SE', zorder=-1, )
    bc_ax3 = fig.add_axes([0.58, 0.23, 0.20, 0.10], anchor='SE', zorder=-1, )
    bc_ax4 = fig.add_axes([0.80, 0.48, 0.20, 0.10], anchor='SE', zorder=-1, )
    bc_ax5 = fig.add_axes([0.80, 0.34, 0.20, 0.10], anchor='SE', zorder=-1, )
    bc_ax6 = fig.add_axes([0.80, 0.23, 0.20, 0.10], anchor='SE', zorder=-1, )
    return fig, ax