import numpy as np
from rstcloth import RstCloth
from glidertest import tools, plots, utilities
from ioos_qc import qartod
from datetime import datetime
import pypandoc
import matplotlib
import os
from pathlib import Path

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
    with open(path/'summary.rst', 'w', encoding='utf-8') as output_file:
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
        doc.content(f'Created with glidertest on {todays_date}')

    return doc
def rst_to_md(path):
    """
    Convert a reStructuredText (.rst) file to Markdown format using Pandoc.

    Parameters
    ----------
    input_rst_path : Path()
        Path to the input `.rst` file.
    output_md_path : Path()
        Path where the converted `.md` file will be saved.

    Returns
    -------
    None
        Prints a confirmation message after successful conversion.

    Notes
    ------
    - Requires `pypandoc` and a working Pandoc installation.
    Original Author: Chiara  Monforte.
    """
    pypandoc.convert_file(path / 'summary.rst', 'md', format='rst', outputfile=path/'summary.rst')
    print(f"Converted RST to Markdown and saved to: {path}")

def mission_report(ds, report_folder_path):
    """
    Generate a full mission report for a glider deployment, including figures and summary documents in .rst and .md format.
    All is saved in a new folder called summary_sheet_date.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in **OG1 format**, containing metadata and measured variables for the deployment.
    report_name : str
        Base name for the report folder. The current date will be appended (format: YYYYMMDD).

    Returns
    -------
    None
        Saves the following in a new report folder:
        - Glider track plot (`gt.png`)
        - Basic variable structure plot (`bv.png`)
        - Mission summary in `.rst` and `.md` formats

    Notes
    ------
    Original Author: Chiara  Monforte.
    """
    folder_name = 'summary_sheet'+str(datetime.today().strftime('%Y%m%d'))
    report_dir = report_folder_path / folder_name
    if not Path(report_dir).is_dir():
        Path(report_dir).mkdir()

    matplotlib.use('Agg')
    fig_gt, ax_gt = plots.plot_glider_track(ds)
    fig_bv, ax_bv = plots.plot_basic_vars(ds, v_res=1, start_prof=0, end_prof=-1)
    fig_gt.savefig(report_dir / 'gt.png')
    fig_bv.savefig(report_dir / 'bv.png')

    create_docfile(ds, report_dir)
    rst_to_md(Path(report_dir))
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
