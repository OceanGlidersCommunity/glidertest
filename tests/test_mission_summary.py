import pytest

from glidertest import fetchers, summary_sheet

# Smoke tests for the RST/PDF summary sheet. They run on the committed sea045 subset (opened fresh
# per test, since the summary builders add derived variables to the dataset) and write documents
# into tmp_path. The optics mission report averages by day/night, so it needs the multi-day full
# sample and stays a slow test.


def test_qc_checks(fresh_subset):
    summary_sheet.qc_checks(fresh_subset, var="PSAL")


def test_tableqc(fresh_subset):
    strgr = ["Global range", "✓", "✓", "✓", "✓"]
    strst = ["Spike test", "✓", "✓", "✓", "✓"]
    strft = ["Flat test", "✓", "✓", "✓", "✓"]
    strhy = ["Hysteresis", "✓", "✓", "✓", "✓"]
    strdr = ["Drift", "✓", "✓", "✓", "✓"]
    summary_sheet.fill_str(strgr, strst, strft, strhy, strdr, fresh_subset, var="TEMP")


def test_phrase_duration_check(fresh_subset):
    summary_sheet.phrase_numberprof_check(fresh_subset)
    summary_sheet.phrase_duration_check(fresh_subset)


def test_docs(tmp_path, fresh_subset):
    ds = fresh_subset
    summary_sheet.create_docfile(ds, tmp_path)
    summary_sheet.rst_to_md(tmp_path, "summary")
    summary_sheet.mission_report(ds, tmp_path, report_type="General")
    summary_sheet.template_docfile(ds, tmp_path)
    summary_sheet.create_hyst_plots(ds, tmp_path)
    summary_sheet.create_drift_plots(ds, tmp_path)
    summary_sheet.create_optics_doc(ds, tmp_path)


def test_optics_func(fresh_subset):
    summary_sheet.optics_available_data(fresh_subset)
    summary_sheet.optics_negative_check(fresh_subset)
    summary_sheet.optics_negative_string(fresh_subset)


@pytest.mark.slow
def test_optics_mission_report(tmp_path):
    # The optics report averages day vs night, so it needs night profiles: the full sample.
    ds = fetchers.load_sample_dataset()
    summary_sheet.mission_report(ds, tmp_path, report_type="Optics")
