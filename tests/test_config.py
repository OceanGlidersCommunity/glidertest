"""Tests for the report config loader: load, validate, resolve_qc, and the check convenience.

All offline — the loader reads YAML/dicts and the ds-aware checks use a tiny synthetic dataset, so
no sample download is needed.
"""

import numpy as np
import pytest
import xarray as xr

from glidertest import qc
from glidertest.config import loader


def _ds(*names):
    """Return a tiny OG1-shaped dataset carrying *names* as data variables."""
    return xr.Dataset({n: ("N_MEASUREMENTS", np.arange(5.0)) for n in names})


# --- load ----------------------------------------------------------------------------------------


def test_load_from_dict():
    cfg = loader.load(
        {"source": "pass 1", "region": {"bbox": [10, 50, 25, 70]}, "qc": {"TEMP": {"spike_test": {"suspect_threshold": 1.0}}}}
    )
    assert cfg.source == "pass 1"
    assert cfg.bbox == [10, 50, 25, 70]
    assert cfg.qc["TEMP"]["spike_test"]["suspect_threshold"] == 1.0
    assert cfg.path is None


def test_load_from_yaml_file(tmp_path):
    path = tmp_path / "report.yaml"
    path.write_text("config_version: 1\nsource: eleanor\nqc:\n  TEMP:\n    gross_range_test: {suspect_span: [0, 30]}\n")
    cfg = loader.load(path)
    assert cfg.source == "eleanor"
    assert cfg.qc["TEMP"]["gross_range_test"]["suspect_span"] == [0, 30]
    assert cfg.path == path


def test_load_descends_into_glidertest_block():
    cfg = loader.load({"glidertest": {"source": "nested"}, "other_tool": {"ignored": True}})
    assert cfg.source == "nested"
    # The sibling tool's block is not read, so it is not an unknown key.
    assert cfg.extra_keys == ()


def test_load_rejects_non_mapping_top_level(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(loader.ConfigError, match="must be a mapping"):
        loader.load(path)


def test_load_records_unknown_top_level_keys():
    cfg = loader.load({"qc": {}, "bogus": 1})
    assert cfg.extra_keys == ("bogus",)


# --- resolve_qc ----------------------------------------------------------------------------------


def test_resolve_qc_none_is_defaults_copy():
    resolved = loader.resolve_qc(None)
    assert resolved == {k: dict(v) for k, v in qc.configs.items()}
    resolved["TEMP"]["spike_test"]["suspect_threshold"] = 999
    assert qc.configs["TEMP"]["spike_test"]["suspect_threshold"] != 999  # defaults untouched


def test_resolve_qc_deep_merges_partial_override():
    cfg = loader.load({"qc": {"PSAL": {"gross_range_test": {"suspect_span": [6, 37]}}}})
    resolved = loader.resolve_qc(cfg)
    # The overridden field changes...
    assert resolved["PSAL"]["gross_range_test"]["suspect_span"] == [6, 37]
    # ...but the sibling field and sibling tests fall back to the defaults.
    assert resolved["PSAL"]["gross_range_test"]["fail_span"] == qc.configs["PSAL"]["gross_range_test"]["fail_span"]
    assert resolved["PSAL"]["spike_test"] == qc.configs["PSAL"]["spike_test"]
    assert resolved["TEMP"] == qc.configs["TEMP"]


# --- validate: clean ------------------------------------------------------------------------------


def test_validate_clean_config_has_no_issues():
    cfg = loader.load(
        {"qc": {"TEMP": {"gross_range_test": {"suspect_span": [0, 30], "fail_span": [-2.5, 40]}}}, "region": {"bbox": [10, 50, 25, 70]}}
    )
    assert loader.validate(cfg) == []


# --- validate: errors -----------------------------------------------------------------------------


def test_validate_unknown_top_level_key():
    issues = loader.validate(loader.load({"bogus": 1}))
    assert any(i.severity == loader.ERROR and "unknown top-level key 'bogus'" in i.message for i in issues)


def test_validate_unknown_variable_names_allowed_set():
    issues = loader.validate(loader.load({"qc": {"NOPE": {"spike_test": {"suspect_threshold": 1}}}}))
    err = next(i for i in issues if i.severity == loader.ERROR)
    assert "unknown variable 'NOPE'" in err.message
    assert "TEMP" in err.message  # names the allowed set


def test_validate_unknown_test():
    issues = loader.validate(loader.load({"qc": {"TEMP": {"bogus_test": {}}}}))
    assert any("unknown test 'bogus_test'" in i.message for i in issues)


def test_validate_unknown_field():
    issues = loader.validate(loader.load({"qc": {"TEMP": {"spike_test": {"bogus": 1}}}}))
    assert any("unknown field 'bogus'" in i.message for i in issues)


@pytest.mark.parametrize(
    "span",
    [[1], [1, 2, 3], [2, 1], ["a", "b"], [True, False]],
)
def test_validate_bad_span(span):
    issues = loader.validate(loader.load({"qc": {"TEMP": {"gross_range_test": {"suspect_span": span}}}}))
    assert any(i.severity == loader.ERROR and "suspect_span" in i.path for i in issues)


def test_validate_suspect_must_sit_inside_fail():
    cfg = loader.load({"qc": {"TEMP": {"gross_range_test": {"suspect_span": [-5, 45], "fail_span": [0, 40]}}}})
    issues = loader.validate(cfg)
    assert any("must sit inside fail_span" in i.message for i in issues)


@pytest.mark.parametrize(
    "bbox",
    [[1, 2, 3], [25, 50, 10, 70], [10, 70, 25, 50], ["a", 1, 2, 3]],
)
def test_validate_bad_bbox(bbox):
    issues = loader.validate(loader.load({"region": {"bbox": bbox}}))
    assert any(i.severity == loader.ERROR and i.path == "region.bbox" for i in issues)


def test_validate_unknown_region_key():
    issues = loader.validate(loader.load({"region": {"polygon": []}}))
    assert any("unknown region key 'polygon'" in i.message for i in issues)


def test_validate_bad_config_version():
    issues = loader.validate(loader.load({"config_version": 99}))
    assert any(i.severity == loader.ERROR and i.path == "config_version" for i in issues)


# --- validate: warnings ---------------------------------------------------------------------------


def test_validate_deferred_block_warns():
    issues = loader.validate(loader.load({"display": {"vmin": {"TEMP": 4}}}))
    assert any(i.severity == loader.WARNING and i.path == "display" for i in issues)


def test_validate_with_ds_accepts_og1_variable():
    cfg = loader.load({"qc": {"BBP700": {"spike_test": {"suspect_threshold": 1}}}})
    # BBP700 is not a built-in default variable, but it is in the dataset, so it is allowed.
    assert loader.validate(cfg, _ds("BBP700")) == []
    # Without the dataset, it is unknown.
    assert any(i.severity == loader.ERROR for i in loader.validate(cfg))


def test_validate_with_ds_warns_on_absent_variable():
    cfg = loader.load({"qc": {"DOXY": {"spike_test": {"suspect_threshold": 1}}}})
    issues = loader.validate(cfg, _ds("TEMP"))
    assert any(i.severity == loader.WARNING and "not in this dataset" in i.message for i in issues)


# --- check ----------------------------------------------------------------------------------------


def test_check_raises_on_error_listing_all():
    cfg = loader.load({"bogus": 1, "qc": {"TEMP": {"bogus_test": {}}}})
    with pytest.raises(loader.ConfigError) as exc:
        loader.check(cfg)
    assert "2 error(s)" in str(exc.value)


def test_check_warns_and_returns_when_clean():
    cfg = loader.load({"display": {"vmin": {"TEMP": 4}}})
    with pytest.warns(UserWarning, match="not applied in this release"):
        assert loader.check(cfg) is cfg


# --- init template --------------------------------------------------------------------------------


def test_init_template_round_trips_and_validates(subset_ds):
    import yaml

    text = loader.init_template(subset_ds)
    assert "config_version: 1" in text
    assert "# default:" in text  # built-in default shown beside each seeded value
    assert "# region:" in text and "bbox" in text  # region left commented out
    cfg = loader.load(yaml.safe_load(text))
    assert loader.validate(cfg, subset_ds) == []  # generated template is clean as written
    assert "TEMP" in cfg.qc  # seeds the variables present in the file
    # The not-yet-applied override categories are shown commented out, so the template advertises the
    # whole menu while the round-trip above still parses to zero issues (comments are not parsed).
    for marker in ("# hysteresis:", "# display:", "#   variables:", "# mission —", "# pages —"):
        assert marker in text


# --- review fixes (findings 1, 2, 4) --------------------------------------------------------------


def test_validate_partial_span_override_checked_against_merged_default():
    # Finding 1: overriding only fail_span must be checked against the default suspect_span it merges
    # with, so an effective suspect-outside-fail pair is caught (previously escaped — only-one-span).
    cfg = loader.load({"qc": {"TEMP": {"gross_range_test": {"fail_span": [0, 20]}}}})
    issues = loader.validate(cfg)
    assert any(i.severity == loader.ERROR and "must sit inside fail_span" in i.message for i in issues)
    # The both-present inconsistent case is still caught.
    bad = loader.load({"qc": {"TEMP": {"gross_range_test": {"suspect_span": [-5, 45], "fail_span": [0, 40]}}}})
    assert any("must sit inside fail_span" in i.message for i in loader.validate(bad))


def test_hysteresis_warns_and_is_dropped_from_effective():
    # Finding 4: hysteresis loads/validates but is not applied — it must warn, and must NOT appear in
    # the effective thresholds (so report.json does not claim a threshold that never ran).
    cfg = loader.load({"qc": {"hysteresis": {"threshold": 10}}})
    issues = loader.validate(cfg)
    assert any(i.severity == loader.WARNING and i.path == "qc.hysteresis" for i in issues)
    assert "hysteresis" not in loader.resolve_qc(cfg)


def test_init_template_validates_on_constant_variable():
    # Finding 2: a near-constant variable gives a degenerate [v, v] suggested span; init must fall
    # back to the defaults so the generated template still loads+validates clean (round-trip holds).
    import yaml

    n = 12
    time = np.datetime64("2020-01-01T00:00:00") + np.arange(n) * np.timedelta64(10, "s")
    ds = xr.Dataset(
        {
            "TEMP": ("N_MEASUREMENTS", np.full(n, 10.0)),
            "TIME": ("N_MEASUREMENTS", time),
        }
    )
    text = loader.init_template(ds)
    cfg = loader.load(yaml.safe_load(text))
    assert [i for i in loader.validate(cfg, ds) if i.severity == loader.ERROR] == []
