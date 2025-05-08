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
def test_summary_plot():
    library_dir = Path(__file__).parent.parent.absolute()
    example_dir = f'{library_dir}/tests/example-summarysheet/'
    ds = fetchers.load_sample_dataset()
    summary_sheet.create_docfile(ds, path=f'{example_dir}/ex_rst.rst')
    summary_sheet.rst_to_md(input_rst_path = f'{example_dir}/ex_rst.rst', output_md_path = f'{example_dir}/ex_md.md')
    summary_sheet.mission_report(ds, example_dir)
    summary_sheet.template_docfile(ds, path=f'{example_dir}/template.rst')