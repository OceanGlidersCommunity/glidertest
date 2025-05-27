from glidertest import fetchers, summary_sheet
import matplotlib
from pathlib import Path

matplotlib.use('agg')  # use agg backend to prevent creating plot windows during tests

def test_qc_checks():
    ds = fetchers.load_sample_dataset()
    gr, spike, flat, err_mean,err_range = summary_sheet.qc_checks(ds, var='PSAL')
def test_tableqc():
    ds = fetchers.load_sample_dataset()
    strgr = ['Global range', '✓', '✓', '✓', '✓']
    strst = ['Spike test', '✓', '✓', '✓', '✓']
    strft = ['Flat test', '✓', '✓', '✓', '✓']
    strhy = ['Hysteresis', '✓', '✓', '✓', '✓']
    strdr = ['Drift', '✓', '✓', '✓', '✓']
    summary_sheet.fill_str(strgr, strst, strft, strhy, strdr,ds, var='TEMP')
def test_phrase_duration_check():
    ds = fetchers.load_sample_dataset()
    summary_sheet.phrase_numberprof_check(ds)
    summary_sheet.phrase_duration_check(ds)
def test_docs():
    ds = fetchers.load_sample_dataset()
    library_dir = Path(__file__).parent.parent.absolute()
    example_dir = library_dir / 'tests/example-summarysheet'
    if not Path(example_dir).is_dir():
        Path(example_dir).mkdir()
    summary_sheet.create_docfile(ds, example_dir)
    summary_sheet.rst_to_md(example_dir, 'summary')
    summary_sheet.mission_report(ds, example_dir,type='General')
    summary_sheet.template_docfile(ds, example_dir)
    summary_sheet.create_hyst_plots(ds, example_dir)
    summary_sheet.create_drift_plots(ds, example_dir)
    summary_sheet.create_optics_doc(ds, example_dir)
    summary_sheet.mission_report(ds, example_dir, type='Optics')

def test_optics_func():
    ds = fetchers.load_sample_dataset()
    summary_sheet.optics_avaialble_data(ds)
    summary_sheet.optics_negative_check(ds)
    summary_sheet.optics_negative_string(ds)
