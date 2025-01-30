from glidertest import fetchers, summary_sheet
import pandas as pd
import matplotlib

matplotlib.use('agg')  # use agg backend to prevent creating plot windows during tests

def test_qc_checks():
    ds = fetchers.load_sample_dataset()
    gr, spike, flat, err = summary_sheet.qc_checks(ds, var='PSAL')
def test_tableqc():
    ds = fetchers.load_sample_dataset()
    df_test = pd.DataFrame({'': ['Global range', 'Spike test', 'Flat test', 'Hysteresis', 'Drift'],
                        'Temperature': ['✓', '✓', '✓', '✓', '✓'],
                        'Salinity': ['✓', '✓', '✓', '✓', '✓']})
    tableT = summary_sheet.fill_tableqc(df_test,ds, var='TEMP')
def test_phrase_duration_check():
    ds = fetchers.load_sample_dataset()
    summary_sheet.phrase_numberprof_check(ds)
    summary_sheet.phrase_duration_check(ds)
def test_summary_plot():
    ds = fetchers.load_sample_dataset()
    summary_sheet.summary_plot(ds, test=True)