import numpy as np
import folium
from glidertest.utilities import _check_necessary_variables
from glidertest import plots
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt


def mission_map(ds_list):
    """
    Makes an interactive folium map of a glider mission. Needs to run in a jupyter notebook

    Parameters
    ----------
    ds_list: xarray.DataSet or list of xarray.DataSets
        OG1 dataset of a glider mission. can pass a list of missions

    Returns
    -------
    m: folium.Map
        A folium map with a point for each glider dive and a sub-sampled line of glider track

    Notes
    -----
    Original author: Callum Rollo
    """
    if type(ds_list) is not list:
        ds_list = [ds_list]
    for ds in ds_list:
        _check_necessary_variables(ds, ["LONGITUDE", "LATITUDE", "DIVE_NUM"])
    df = ds.to_pandas()[["LONGITUDE", "LATITUDE", "DIVE_NUM"]].groupby("DIVE_NUM").median()
    m = folium.Map(location=[df.LATITUDE.mean(), df.LONGITUDE.mean()], zoom_start=8, tiles="cartodb positron")
    folium.WmsTileLayer(
        url="https://ows.emodnet-bathymetry.eu/wms",
        layers='mean_atlas_land',
        attr='EMODnet bathymetry'
    ).add_to(m)
    color_cycle = ["red", "yellow", "green", "black", "pink"]
    for i, ds in enumerate(ds_list):
        df = ds.to_pandas()[["LONGITUDE", "LATITUDE", "DIVE_NUM"]].groupby("DIVE_NUM").median()
        df = df.dropna()
        df_points = ds.to_pandas()[["LONGITUDE", "LATITUDE"]].dropna()
        coordinates = [[lat, lon] for lat, lon in zip(df_points['LATITUDE'], df_points['LONGITUDE'])]
        if len(coordinates) > 5000:
            coordinates = coordinates[::int(np.ceil(len(coordinates) / 5000))]
        df['DIVE_NUM'] = df.index

        folium.PolyLine(
            locations=coordinates,
            color=color_cycle[i],
            weight=5,
            tooltip=f"{ds.attrs['id']}",
        ).add_to(m)

        for j, row in df.iterrows():
            folium.CircleMarker(
                location=[row['LATITUDE'], row['LONGITUDE']],
                tooltip=f"Dive {int(row['DIVE_NUM'])}",
                color= 'black',
                fillOpacity= 1,
                fillColor= color_cycle[i],
                weight = 2,
            ).add_to(m)
    return m

def profile(ds):
    """
    Creates an interactive profile viewer with external slider inputs.

    Parameters
    ----------
        ds : xarray.Dataset
            The dataset containing profile information.

    The function provides the following interactive widgets:
        - Profile Number: Selects the profile number from available profiles.
        - Variable 1, Variable 2, Variable 3: Dropdowns to choose up to three variables to plot.
        - Use Binned Data: Checkbox to toggle between raw and binned data.
        - Binning Resolution: Slider to adjust the binning resolution if binned data is used.

    Notes
    -----
    Original author: Till Moritz
    """

    def plot_profile(profile_num, variable1, variable2, variable3, use_bins, binning):
        variables = [variable1, variable2, variable3]
        fig, ax = plots.plot_profile(ds, profile_num, variables, use_bins, binning)
        plt.close(fig)
        del fig, ax

    # Profile selection slider
    profile_slider = widgets.SelectionSlider(options=ds.PROFILE_NUMBER.values,value=ds.PROFILE_NUMBER.values[0],description='Profile: ',continuous_update=False)

    # Variable selection dropdowns (with an empty option)
    # take only into account the variables with float or int data type
    var_options = [''] + [variable for variable in ds.data_vars if ds[variable].dtype.kind in {'i', 'f'}]


    variable1_dropdown = widgets.Dropdown(options=var_options, value=var_options[0], description="Variable 1:")
    variable2_dropdown = widgets.Dropdown(options=var_options, value=var_options[0], description="Variable 2:")
    variable3_dropdown = widgets.Dropdown(options=var_options, value=var_options[0], description="Variable 3:")

    # Checkbox for using binned data
    use_bins_button = widgets.Checkbox(value=False, description='Bin Data')

    # Binning resolution slider
    binning_slider = widgets.FloatSlider(value=2, min=1, max=20, step=1,description='Res (m):',continuous_update=False)


    # Use a VBox to show/hide the binning slider based on the checkbox state
    binning_box = widgets.VBox([binning_slider])
    def toggle_binning_visibility(change):
        binning_box.layout.display = 'flex' if change['new'] else 'none'

    # Attach observer to toggle visibility
    use_bins_button.observe(toggle_binning_visibility, names='value')

    # Set initial visibility
    binning_box.layout.display = 'none' if not use_bins_button.value else 'flex'

    # Arrange variable dropdowns in a horizontal row
    var_selection_box = widgets.HBox([variable1_dropdown, variable2_dropdown, variable3_dropdown])

    # Arrange all widgets in a vertical layout
    ui = widgets.VBox([widgets.Label("Select the profile number to visualize:"),profile_slider,
                       widgets.Label("Choose up to three variables to plot:"),var_selection_box,
                       widgets.Label("Additional settings:"),use_bins_button,binning_box])

    # Create interactive plot
    out = widgets.interactive_output(plot_profile, {
        'profile_num': profile_slider,
        'variable1': variable1_dropdown,
        'variable2': variable2_dropdown,
        'variable3': variable3_dropdown,
        'use_bins': use_bins_button,
        'binning': binning_slider
    })

    display(ui, out)


def up_down_bias(ds):
    """
    Creates an interactive up/down bias plot with external slider inputs.

    Parameters
    ----------
        ds : xarray.Dataset
            The dataset containing profile information.

    The function provides the following interactive widgets:
        - Variable: Dropdown to choose the variable to plot.
        - Vertical Resolution: Slider to adjust the vertical resolution for binning.

    Notes
    -----
    Original author: Till Moritz
    """

    def plot_up_down_bias(variable, v_res):
        fig, ax = plots.plot_updown_bias(ds, variable, v_res)
        plt.close(fig)
        del fig, ax

    # Variable selection dropdown (with an empty option)
    var_options = [variable for variable in ds.data_vars if ds[variable].dtype.kind in {'i', 'f'}]
    var_dropdown = widgets.Dropdown(options=var_options, value=var_options[0], description="Variable:")
    res_slider = widgets.FloatSlider(value=1, min=0.1, max=10, step=0.1, description='Vertical Resolution (m):', continuous_update=False)

    # Arrange all widgets in a vertical layout
    ui = widgets.VBox([widgets.Label("Choose a variable to plot:"), var_dropdown, res_slider])

    # Create interactive plot
    out = widgets.interactive_output(plot_up_down_bias, {
        'variable': var_dropdown,
        'v_res': res_slider
    })

    display(ui, out)

def daynight_avg(ds):
    """
    Creates an interactive day/night average plot with external slider inputs.

    Parameters
    ----------
        ds : xarray.Dataset
            The dataset containing profile information.

    The function provides the following interactive widgets:
        - Variable: Dropdown to choose the variable to plot.
        - Select the specific day to plot.

    Notes
    -----
    Original author: Till Moritz
    """

    def plot_daynight_avg(variable, sel_day):
        fig, ax = plots.plot_daynight_avg(ds, variable= variable, sel_day=None)
        plt.close(fig)
        del fig, ax

    # Variable selection dropdown (with an empty option)
    var_options = [variable for variable in ds.data_vars if ds[variable].dtype.kind in {'i', 'f'}]
    var_dropdown = widgets.Dropdown(options=var_options, value=var_options[0], description="Variable:")
    day_options = np.unique(ds.TIME.values.astype('datetime64[D]'))
    day_options = [str(i) for i in day_options]
    #day_slider = widgets.SelectionSlider(options=day_options, value=day_options[0], description="Day:")
    day_slider = widgets.Dropdown(options=day_options, value=day_options[0], description="Day:")

    # Arrange all widgets in a vertical layout
    ui = widgets.VBox([widgets.Label("Choose a variable to plot:"), var_dropdown, day_slider])

    # Create interactive plot
    out = widgets.interactive_output(plot_daynight_avg, {
        'variable': var_dropdown,
        'sel_day': day_slider
    })

    display(ui, out)

def ts_plot(ds):
    """
    Creates an interactive temperature-salinity plot with external slider inputs.

    Parameters
    ----------
        ds : xarray.Dataset
            The dataset containing profile information.

    The function provides the following interactive widgets:
        - Percentile: Slider to adjust the percentiles for filtering the data.

    Notes
    -----
    Original author: Till Moritz
    """

    def plot_ts_plot(percentile):
        fig, ax = plots.plot_ts(ds, percentile=percentile)
        plt.close(fig)
        del fig, ax

    # Percentile slider
    percentile_slider = widgets.FloatRangeSlider(value=[0.5, 99.5], min=0, max=100, step=0.5, description='Percentile:', continuous_update=False)

    # Arrange all widgets in a vertical layout
    ui = widgets.VBox([percentile_slider])

    # Create interactive plot
    out = widgets.interactive_output(plot_ts_plot, {'percentile': percentile_slider})

    display(ui, out)
