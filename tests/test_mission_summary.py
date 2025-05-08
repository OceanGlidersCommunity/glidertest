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
    ds = fetchers.load_sample_dataset()
    library_dir = Path(__file__).parent
    example_dir = Path(library_dir) / 'example-summarysheet'
    if not Path(example_dir).is_dir():
        Path(example_dir).mkdir()
    summary_sheet.create_docfile(ds,example_dir)
    summary_sheet.rst_to_md(example_dir / 'ex_rst.rst',
                            example_dir / 'ex_md.md')
    summary_sheet.mission_report(ds, example_dir)
    summary_sheet.template_docfile(ds, example_dir)

