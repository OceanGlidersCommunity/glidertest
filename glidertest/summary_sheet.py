import numpy as np
from rstcloth import RstCloth
from glidertest import tools, plots, utilities, __version__
from ioos_qc import qartod
from datetime import datetime
import pypandoc
import matplotlib
from pathlib import Path
import matplotlib.pyplot as plt

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
    """
    Run a series of basic quality control (global range, spike test, drift and hysteresis) checks on a selected variable in the dataset.
    We use functions from glidertest.tools as well as ioos qartod.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in **OG1 format**, containing the selected variable and time information.
    var : str, optional, default='TEMP'
        Variable to perform QC checks on.

    Returns
    -------
    gr : xarray.DataArray
        Result of the **gross range test**, identifying values outside the expected physical range.
    spike : xarray.DataArray
        Result of the **spike test**, detecting sudden changes in data based on threshold configuration.
    flat : xarray.DataArray
        Result of the **flat line test**, identifying segments with little or no variation.
    err_mean : array-like
        Percentage error of the dive-climb difference based on the mean values, from hysteresis analysis.
    err_range : array-like
        Percentage error of the dive-climb difference based on the value range, from hysteresis analysis.

    Notes
    ------
    - Thresholds for QC tests are taken from the global `configs` dictionary.
    Original Author: Chiara  Monforte
    """
    utilities._check_necessary_variables(ds, [var, 'TIME'])
    gr = tools.compute_global_range(ds, var=var, min_val=configs[var]['gross_range_test']['suspect_span'][0],
                                    max_val=configs[var]['gross_range_test']['suspect_span'][1])
    spike = qartod.spike_test(ds[var], suspect_threshold=configs[var]['spike_test']['suspect_threshold'],
                              fail_threshold=configs[var]['spike_test']['fail_threshold'], method="average")
    flat = qartod.flat_line_test(ds[var], ds.TIME, 1, 3, 0.001)
    __, __,err_mean,err_range, __ = tools.compute_hyst_stat(ds, var=var, v_res=1)

    return gr, spike, flat, err_mean,err_range

def phrase_duration_check(ds):
    """
    Check for anomalies in profile duration and return a human-readable message summarizing the result.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in **OG1 format**, containing time information required for computing profile durations.

    Returns
    -------
    duration_check : str
        A message indicating whether abnormal profile durations have been detected:
        - If outliers are found: `"X profiles have abnormal duration"`
        - Otherwise: `"No issues with profile duration have been detected"`

    Notes
    ------
    Original Author: Chiara  Monforte.
    """
    duration = tools.compute_prof_duration(ds)
    rolling_mean, overtime = tools.find_outlier_duration(duration, rolling=20, std=2)
    if len(overtime) > 0:
        duration_check = f'{len(overtime)} profiles have abnormal duration'
    else:
        duration_check = 'No issues with profile duration have been detected'
    return duration_check

def phrase_numberprof_check(ds):
    """
   Check the monotonicity of profile numbers and return a human-readable summary of the result.

   Parameters
   ----------
   ds : xarray.Dataset
       Dataset in **OG1 format**, containing the **PROFILE_NUMBER** variable.

   Returns
   -------
   prof_check : str
       A message indicating whether profile numbers increase monotonically:
       - If monotonic: `"No issues detected"`
       - If not monotonic: `"Issues with profile number have been detected"`

   Notes
   ------
   Original Author: Chiara  Monforte.
   """
    check = tools.check_monotony(ds.PROFILE_NUMBER)
    if check:
        prof_check = 'No issues detected'
    else:
        prof_check = 'Issues with profile number have been detected'
    return prof_check

def fill_str(strgr, strst, strft, strhy, strdr,ds, var='TEMP'):
    """
    Populate QC summary strings for a selected variable based on a series of QC checks.

    Parameters
   ----------
    strgr : list of str
        Row for **Global range test** results with X or ✓, structured as `[label/test name, TEMP, PSAL, DOXY, CHLA]`.
    strst : list of str
        Row for **Spike test** results with X or ✓, structured as `[label/test name, TEMP, PSAL, DOXY, CHLA]`.
    strft : list of str
        Row for **Flat line test** results with X or ✓, structured as `[label/test name, TEMP, PSAL, DOXY, CHLA]`.
    strhy : list of str
        Row for **Hysteresis error (mean)** results with X or ✓, structured as `[label/test name, TEMP, PSAL, DOXY, CHLA]`.
    strdr : list of str
        Row for **Hysteresis error (range)** results with X or ✓, structured as `[label/test name, TEMP, PSAL, DOXY, CHLA]`.
    ds : xarray.Dataset
        Dataset in **OG1 format**, containing the selected variable.
    var : str, optional, default='TEMP'
        Selected variable to evaluate. Must be one of: `'TEMP'`, `'PSAL'`, `'DOXY'`, `'CHLA'`.

    Returns
    -------
    strgr : list of str
        Updated Global range test row.
    strst : list of str
        Updated Spike test row.
    strft : list of str
        Updated Flat line test row.
    strhy : list of str
        Updated Hysteresis error (mean) row.
    strdr : list of str
        Updated Hysteresis error (range) row.

    Notes
    ------
    - Input string is filled with ✓. Each list entry is updated with an `'X'` if a QC issue is detected for the corresponding variable.
    - If the variable is missing in the dataset, all tests for that variable are marked as `"No data"`.
    - Column indices are mapped as: `TEMP=1`, `PSAL=2`, `DOXY=3`, `CHLA=4`.
    Original Author: Chiara  Monforte.
    """
    if var == 'TEMP':
        i = 1
    if var == 'PSAL':
        i = 2
    if var == 'DOXY':
        i = 3
    if var == 'CHLA':
        i = 4
    if var not in ds.variables:
        strgr[i] = 'No data'
        strft[i] = 'No data'
        strst[i] = 'No data'
        strhy[i] = 'No data'
        strdr[i] = 'No data'
    else:
        gr, spike, flat, err_mean,err_range = qc_checks(ds, var=var)
        if len(gr) > 0:
            strgr[i] = 'X'
        if len(np.where((spike == 3) | (spike == 4))[0]) > 0:
            strst[i] = 'X'
        if len(np.where((flat == 3) | (flat == 4))[0]) > 0:
            strft[i] = 'X'
        if len(np.where(err_mean > 5)[0]) > 5:
            strhy[i] = 'X'
        if len(np.where(err_range > 5)[0]) > 5:
            strdr[i] = 'X'
    return strgr, strst, strft, strhy, strdr

def create_docfile(ds, path):
    """
    Generate a reStructuredText (.rst) mission summary report for a specific glider deployment.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in **OG1 format**, containing metadata and profile data for the mission.
        Should include variables like **TIME**, **PLATFORM_SERIAL_NUMBER**, and measurement variables
        (e.g., TEMP, PSAL, DOXY, CHLA, etc.).
    path : Path()
        Output file path for the generated `.rst` document.

    Returns
    -------
    doc : RstCloth
        RstCloth document object with the mission summary content written to disk.

    Notes
    ------
    - The document includes:
        - Main mission info (serial, deployment ID, duration, depth range)
        - Instrument payload summary (✓ if data is available)
        - Water column structure image (`bv.png`)
        - Basic quality checks:
            - Profile number monotonicity
            - Duration anomalies
            - Summary of global range, spike, flat line, hysteresis, and drift checks
        - Glider track image (`gt.png`)
    - The function assumes supporting images (`gt.png`, `bv.png`) are available in the same directory.
    Original Author: Chiara  Monforte.
    """

    '''
    if not (path / 'gt.png').exists():
        raise ValueError(f"Did not find required file gt.png in supplied path {path}. Aborting")
    if not (path / 'bv.png').exists():
        raise ValueError(f"Did not find required file bv.png in supplied path {path}. Aborting")
    '''
    with open(f'{path}/summary.rst', 'w', encoding='utf-8') as output_file:
        doc = RstCloth(output_file)
        doc.title('Glidertest summary mission sheet')
        doc.newline()
        doc.h1('Main mission info')
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
        doc.content(f'Glider serial:  {gserial}')
        doc.newline()
        doc.content(f"Mission number:  {mission}")
        doc.newline()
        doc.content(f"Deployment date:  {ds.TIME[0].values.astype('datetime64[D]')}")
        doc.newline()
        doc.content(f"Recovery date:  {ds.TIME[-1].values.astype('datetime64[D]')}")
        doc.newline()
        doc.content(
            f"Mission duration:  {np.round(((ds.TIME.max() - ds.TIME.min()) / np.timedelta64(1, 'D')).astype(float).values, 1)} days")
        doc.newline()
        doc.content(
            f"Diving depth range:  {int(tools.max_depth_per_profile(ds).min())} - {int(tools.max_depth_per_profile(ds).max())} m")
        doc.newline()
        doc.content(f"Dataset creator:  {cname} \n({ds.contributing_institutions})")

        # Add glider track image

        doc.newline()
        doc.directive('image', './gt.png', fields=[
            ('alt', ''),
            ('width', '600px')
        ])

        doc.newline()
        doc.h1('Basic payload configuration')
        temp_str = ['Temperature', 'X']
        sal_str = ['Salinity', 'X']
        oxy_str = ['Oxygen', 'X']
        chla_str = ['Chlorophyll', 'X']
        back_str = ['Backscatter', 'X']
        alt_str = ['Altimeter', 'X']
        adcp_str = ['ADCP', 'X']

        if 'TEMP' in ds.variables:
            temp_str[-1] = "✓"
        if 'PSAL' in ds.variables:
            sal_str[-1] = "✓"
        if 'ALTITUDE' in ds.variables:
            alt_str[-1] = "✓"
        if 'CHLA' in ds.variables:
            chla_str[-1] = "✓"
        if 'BBP700' in ds.variables:
            back_str[-1] = "✓"
        if 'PRES_ADCP' in ds.variables:
            adcp_str[-1] = "✓"
        if 'DOXY' in ds.variables:
            oxy_str[-1] = "✓"
        doc.table(
            ['', 'Data exists'],
            data=[
                temp_str,
                sal_str,
                oxy_str,
                chla_str,
                back_str,
                alt_str,
                adcp_str
            ])

        # Add basic water column structure
        doc.h1(text='Basic water column structure', char='-')
        doc.newline()
        doc.directive('image', './bv.png', fields=[
            ('alt', ''),
            ('width', '600px')
        ])
        doc.newline()
        doc.h1('Basic checks')
        profnum = phrase_numberprof_check(ds)
        doc.newline()
        dur = phrase_duration_check(ds)
        doc.content(f"Issues with profile number: {profnum} ")
        doc.newline()
        doc.content(f"Issues with profile duration: {dur}", )
        gr_strb = ['Global range', '✓', '✓', '✓', '✓']
        st_strb = ['Spike test', '✓', '✓', '✓', '✓']
        ft_strb = ['Flat test', '✓', '✓', '✓', '✓']
        hy_strb = ['Hysteresis', '✓', '✓', '✓', '✓']
        dr_strb = ['Drift', '✓', '✓', '✓', '✓']
        gr_strb, st_strb, ft_strb, hy_strb, dr_strb = fill_str(gr_strb, st_strb, ft_strb, hy_strb, dr_strb, ds,
                                                               var='TEMP')
        gr_strb, st_strb, ft_strb, hy_strb, dr_strb = fill_str(gr_strb, st_strb, ft_strb, hy_strb, dr_strb, ds,
                                                               var='PSAL')
        gr_strb, st_strb, ft_strb, hy_strb, dr_strb = fill_str(gr_strb, st_strb, ft_strb, hy_strb, dr_strb, ds,
                                                               var='DOXY')
        gr_strb, st_strb, ft_strb, hy_strb, dr_strb = fill_str(gr_strb, st_strb, ft_strb, hy_strb, dr_strb, ds,
                                                               var='CHLA')
        doc.table(
            ['', 'Temperature', 'Salinity', 'Oxygen', 'Chlorophyll'],
            data=[
                gr_strb,
                st_strb,
                ft_strb,
                hy_strb,
                dr_strb,
            ])
        todays_date = datetime.today().strftime('%Y-%m-%d')
        doc.content(f'Created with glidertest version {__version__} on {todays_date}')

    return doc


def optics_avaialble_data(ds):
    """
    Identify available bio-optical sensors and their variables in the dataset.

    This function scans the dataset for bio-optical sensor variables and irradiance sensor variables,
    then creates lists of sensor descriptions and recognized optical variable names present in the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing measurement variables and metadata.

    Returns
    -------
    tuple of three lists:
        available_sensor : list of str
            Descriptions of available bio-optical and irradiance sensors found in the dataset.
            If none are found, placeholders indicating missing data are returned.
        available_optics : list of str
            Human-readable names of recognized optical variables present in the dataset (e.g., 'Chlorophyll').
        vars_optics : list of str
            Corresponding variable names (dataset keys) for the available optical variables (e.g., 'CHLA').
    Notes
    -----
    Original Author: Chiara  Monforte.
    """
    biooptics_var = [name for name in ds.data_vars if 'SENSOR_FLUOROMETERS' in name]
    irradiance_var = [name for name in ds.data_vars if 'SENSOR_RADIOMETERS' in name]
    available_sensor = []
    if biooptics_var:
        available_sensor.append(ds[biooptics_var[0]].long_name)
    else:
        available_sensor.append('No  bio optical data available')
    if irradiance_var:
        available_sensor.append(ds[irradiance_var[0]].long_name)
    else:
        available_sensor.append('No irradiance data available')
    available_optics = []
    vars_optics = []
    if 'BBP700' in ds.data_vars:
        available_optics.append('Backscatter at 700 nm')
        vars_optics.append('BBP700')
    if 'CHLA' in ds.data_vars:
        available_optics.append('Chlorophyll')
        vars_optics.append('CHLA')
    if 'DPAR' in ds.data_vars:
        available_optics.append('Downwelling PAR')
        vars_optics.append('DPAR')
    if 'CDOM' in ds.data_vars:
        available_optics.append('CDOM')
        vars_optics.append('CDOM')
    if 'FDOM' in ds.data_vars:
        available_optics.append('fDOM')
        vars_optics.append('FDOM')
    if 'TURBIDITY' in ds.data_vars:
        available_optics.append('Turbidity')
        vars_optics.append('TURBIDITY')
    if 'PHYCOCYANIN' in ds.data_vars:
        available_optics.append('Phycocyanin')
        vars_optics.append('PHYCOCYANIN')
    if 'PHYC' in ds.data_vars:
        available_optics.append('Phycoerythrin')
        vars_optics.append('PHYC')
    return available_sensor, available_optics, vars_optics


def optics_negative_check(ds):
    """
    Calculate the percentage of negative values for each available optical variable in the dataset.

    This function identifies all available optical variables in the dataset, computes the proportion of
    negative data points for each variable, and returns a list of these percentages rounded to one decimal place.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing optical sensor data in OG1 format.

    Returns
    -------
    numpy.ndarray
        An array of percentages representing the proportion of negative values for each optical variable,
        rounded to one decimal place. The order corresponds to the list of available optical variables.

    Notes
    -----
    Original Author: Chiara  Monforte.
    """
    available_sensor, available_optics, vars_optics = optics_avaialble_data(ds)
    neg_perc = []
    for var in vars_optics:
        neg_val = len(ds[var].where(ds[var] < 0).dropna(dim='N_MEASUREMENTS'))
        neg_perc.append((100 * neg_val) / len(ds[var]))
    return np.round(neg_perc, 1)


def optics_negative_string(ds):
    """
    Generate a summary string reporting the percentage of negative values in calibrated optical data variables.

    This function retrieves available optical variables from the dataset, checks the percentage of negative
    values in each variable using `optics_negative_check`, and returns a formatted string summarizing the results.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing optical sensor data in OG1 format.

    Returns
    -------
    str
        A human-readable string summarizing the percentage of negative values for each optical variable,
        formatted as "<var> has <percentage>% negative data", joined by commas and 'and' before the last item.
    Notes
    -----
    Original Author: Chiara  Monforte.
    """
    available_sensor, available_optics, vars_optics = optics_avaialble_data(ds)
    neg_check = optics_negative_check(ds)

    parts = [f"{var} has {neg:.1f}% negative data" for var, neg in zip(vars_optics, neg_check)]

    if len(parts) > 1:
        result_str = ", ".join(parts[:-1]) + ", and " + parts[-1]
    else:
        result_str = parts[0]
    return result_str


def create_hyst_plots(ds, path):
    """
    Generate and save hysteresis diagnostic plots for each available optical variable in the dataset.

    For each optical variable detected in the dataset, this function computes hysteresis statistics
    using `tools.compute_hyst_stat` and creates a hysteresis plot using `plots.plot_hysteresis`.
    The resulting plot is saved as `<variable>_hyst.png` in the specified directory.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing optical sensor data in OG1 format.

    path : str or Path
        Directory path where the generated hysteresis plot images will be saved.

    Returns
    -------
    None
        Saves PNG images for hysteresis plots for each optical variable found in the dataset.

    Notes
    -----
    Original Author: Chiara  Monforte.
    """
    available_sensor, available_optics, vars_optics = optics_avaialble_data(ds)
    for var in vars_optics:
        df, diff, err_mean, err_range, rms = tools.compute_hyst_stat(ds, var=var, v_res=1)
        fig, ax = plots.plot_hysteresis(ds, var=var)
        fig_name = f'{var}_hyst.png'
        fig.savefig(f'{path}/{fig_name}')


def create_drift_plots(ds, path):
    """
    Generate and save drift diagnostic plots for each available optical variable in the dataset.

    For each optical variable detected in the dataset, this function creates a drift assessment plot
    using the `plots.process_optics_assess` function and saves the figure as `<variable>_drift.png`
    in the specified directory.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing optical sensor data in OG1 format.

    path : str or Path
        Directory path where the generated drift plot images will be saved.

    Returns
    -------
    None
        Saves PNG images for drift plots for each optical variable found in the dataset.
    Notes
    -----
    Original Author: Chiara  Monforte.
    """
    available_sensor, available_optics, vars_optics = optics_avaialble_data(ds)
    for var in vars_optics:
        fig, ax = plots.process_optics_assess(ds, var=var)
        fig_name = f'{var}_drift.png'
        fig.savefig(f'{path}/{fig_name}')

def create_optics_doc(ds, path):
    """
    Generate a reStructuredText (.rst) summary report for bio-optical sensor data.

    The report includes:
    - A list of available optical variables and sensors
    - General quality check notes
    - Diagnostic plots for chlorophyll quenching, hysteresis, and drift (if available)

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing optical sensor data in OG1 format.

    path : str or Path
        Directory where the `optics.rst` file and associated plots are saved.

    Returns
    -------
    RstCloth
        The RstCloth document object used to create the report.

    Notes
    -----
    Original Author: Chiara  Monforte.
    """
    with open(f'{path}/optics.rst', 'w', encoding='utf-8') as output_file:
        doc = RstCloth(output_file)

        doc.title('Mission summary - Bio optical sensors')
        doc.newline()
        doc.h1('Available variables and sensor info')
        available_sensor, available_optics, vars_optics = optics_avaialble_data(ds)
        doc.newline()
        doc.content(f'The following variables are available in this dataset: {str(available_optics)[1:-1]}')
        doc.newline()
        doc.content(f'Data is collected by the following sensors: {str(available_sensor)[1:-1]}')

        doc.newline()
        doc.h1('General data quality checks')
        doc.newline()
        if vars_optics:
            doc.content(
                f'Negative values in the calibrated data: {optics_negative_string(ds)}. Negative values could indicate invalid calibration factors and/or issues with the sensor ')
            if 'CHLA' in vars_optics:
                ### Quenching
                doc.newline()
                doc.h2('Chlorophyll quenching')
                doc.newline()
                name= 'quench.png'
                doc.directive('image', f'./{name}', fields=[
                    ('alt', ''),
                    ('width', '600px')
                ])
            doc.newline()

            doc.h2('Hysteresis')
            for var in vars_optics:
                fig_name = f'{var}_hyst.png'
                doc.directive('image', f'./{fig_name}', fields=[
                    ('alt', ''),
                    ('width', '600px')
                ])
            doc.newline()

            doc.h2('Drift')
            for var in vars_optics:
                fig_name = f'{var}_drift.png'
                doc.directive('image', f'./{fig_name}', fields=[
                    ('alt', ''),
                    ('width', '600px')
                ])
            doc.newline()

        todays_date = datetime.today().strftime('%Y-%m-%d')
        doc.content(f'Created with glidertest {__version__} on {todays_date}')

        return doc


def rst_to_md(path, filename):
    """
    Convert a reStructuredText (.rst) file to Markdown format using Pandoc.

    Parameters
    ----------
    path : Path or str
        Directory containing the input `.rst` file and where the output `.md` file will be saved.
    filename : str
        Base name of the file (without extension). For example, 'summary' will convert 'summary.rst' to 'summary.md'.

    Returns
    -------
    None
        Prints a confirmation message after successful conversion.

    Notes
    ------
    - Requires `pypandoc` and a working Pandoc installation.
    Original Author: Chiara  Monforte.
    """

    pypandoc.convert_file(f'{path}/{filename}.rst', 'md', format='rst', outputfile=f'{path}/{filename}.md')
    print(f"Converted RST to Markdown and saved to: {path}")

def mission_report(ds, report_folder_path, type='General'):
    """
    Generate a full mission report for a glider deployment, including plots and summary documents
    in both reStructuredText (.rst) and Markdown (.md) formats.

    The output is saved in a new subfolder named `summary_sheet<Type><YYYYMMDD>` within the specified report folder path.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in OG1 format, containing metadata and measured variables for the deployment.

    report_folder_path : str or Path
        Path to the parent directory where the report folder will be created.

    type : str, optional
        Type of report to generate. Accepts either 'General' (default) or 'Optics'.
        - 'General' includes a glider track and basic variables.
        - 'Optics' includes additional analyses and plots specific to optical sensors (e.g., CHLA, BBP700).

    Returns
    -------
    None
        Creates a report folder containing:
        - Plots (e.g., `gt.png`, `bv.png`, `quench.png`, hysteresis/drift images)
        - A reStructuredText summary file (`summary.rst` or `optics.rst`)
        - A corresponding Markdown version (`summary.md` or `optics.md`)

    Notes
    -----
    - The `type='Optics'` report is only created if relevant optical sensors are found in the dataset.

    Original Author: Chiara Monforte
    """
    folder_name = 'summary_sheet'+type+str(datetime.today().strftime('%Y%m%d'))
    report_dir = f'{report_folder_path}/{folder_name}'
    if not Path(report_dir).is_dir():
        Path(report_dir).mkdir()

    matplotlib.use('Agg')
    if type=='General':
        fig_gt, ax_gt = plots.plot_glider_track(ds)
        fig_bv, ax_bv = plots.plot_basic_vars(ds, v_res=1, start_prof=0, end_prof=-1)
        fig_gt.savefig(f'{report_dir}/gt.png')
        fig_bv.savefig(f'{report_dir}/bv.png')
        create_docfile(ds, report_dir)
        rst_to_md(Path(report_dir), 'summary')
    if type=='Optics':
        available_sensor, _, _ = optics_avaialble_data(ds)
        if available_sensor:
            fig_quench, ax_quench = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [3, 2]})
            plots.plot_quench_assess(ds, 'CHLA', ax=ax_quench[0], ylim=35);
            plots.plot_daynight_avg(ds, var='CHLA', ax=ax_quench[1])
            create_hyst_plots(ds, report_dir)
            create_drift_plots(ds, report_dir)
            fig_quench.savefig(f'{report_dir}/quench.png')
            create_optics_doc(ds, report_dir)
            rst_to_md(Path(report_dir), 'optics')

    print(f"Your report is saved in {report_dir} in rst and markdown")

def template_docfile(ds, path):
    """
    Create a template reStructuredText (.rst) document with example text, image, and table.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in **OG1 format**, used as a placeholder or reference for generating content.
    path : Path()
        Path to the output `.rst` file.

    Returns
    -------
    doc : RstCloth
        RstCloth document object written to disk with placeholder content.

    Notes
    ------
    Original Author: Chiara  Monforte.
    """
    with open(Path(path/'summary_template.rst'), 'w', encoding='utf-8') as output_file:
        doc = RstCloth(output_file)
        doc.title('Title of the document')
        doc.newline()
        doc.h1('First subtitle')
        ### Create the text boxes
        doc.content('Add text with specific info')
        doc.newline()
        doc.content('Add some more text')

        # Add a plot
        doc.newline()
        doc.directive('image', 'example.png', fields=[
            ('alt', ''),
            ('width', '600px')
        ])

        doc.newline()
        doc.h1('Second subtitle')
        #Add a table
        doc.table(['Row heading', 'Example Table'],
          data=[
              ['Info 1','Detail 1'],
              ['Info 2','Detail 2'],
              ['Info 3','Detail 3'],
          ])
        todays_date = datetime.today().strftime('%Y-%m-%d')
        doc.content(f'Created with glidertest on {todays_date}')
    return doc
