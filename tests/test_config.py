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


def test_load_missing_file_is_configerror(tmp_path):
    # Review finding 1: a missing -c file must surface as ConfigError (caught by the CLI -> exit 2),
    # not an unhandled FileNotFoundError traceback.
    with pytest.raises(loader.ConfigError, match="cannot read config file"):
        loader.load(tmp_path / "does_not_exist.yaml")


def test_load_malformed_yaml_is_configerror(tmp_path):
    # Review finding 1: a syntactically broken YAML file is a ConfigError, not a raw yaml.YAMLError.
    path = tmp_path / "broken.yaml"
    path.write_text("qc:\n  TEMP: [unclosed\n")
    with pytest.raises(loader.ConfigError, match="invalid YAML"):
        loader.load(path)


@pytest.mark.parametrize(
    ("block", "shape"),
    [
        ({"region": [10, 50, 25, 70]}, "bbox"),  # list where a {bbox: ...} mapping was meant
        ({"qc": "TEMP"}, "mapping"),  # scalar where the variables mapping was meant
    ],
)
def test_wrong_typed_block_is_error(block, shape):
    # Review finding 4: never silently coerce a wrong-typed known block to {} — it is a validate ERROR
    # naming the shape, collected with the config's other problems (one multi-issue report) and raised
    # by check. load itself does not raise (so validate can report everything at once).
    cfg = loader.load(block)
    assert any(i.severity == loader.ERROR and shape in i.message for i in loader.validate(cfg))
    with pytest.raises(loader.ConfigError, match=shape):
        loader.check(cfg)


def test_wrong_typed_block_reported_with_other_issues():
    # Finding 6: a wrong-typed block no longer short-circuits — validate returns it alongside the rest.
    cfg = loader.load({"region": [1, 2, 3, 4], "qc": {"TEMP": {"bogus_test": {}}}})
    msgs = [i.message for i in loader.validate(cfg) if i.severity == loader.ERROR]
    assert any("region: expected" in m for m in msgs) and any("unknown test 'bogus_test'" in m for m in msgs)


def test_validate_rejects_unknown_spike_method():
    # Review round finding 1: an invalid spike_test.method is an ERROR naming the allowed values, not a
    # late ValueError swallowed at render time (which would blank the column yet mark the row 'config').
    issues = loader.validate(loader.load({"qc": {"TEMP": {"spike_test": {"method": "median"}}}}))
    assert any(i.severity == loader.ERROR and i.path.endswith("method") and "average" in i.message for i in issues)
    for m in ("average", "differential"):
        clean = loader.load({"qc": {"TEMP": {"spike_test": {"method": m}}}})
        assert not any(i.severity == loader.ERROR for i in loader.validate(clean))


def test_validate_warns_on_nonbuiltin_variable_without_dataset():
    # Review round finding 2: without a dataset, a variable name outside the built-ins is a WARNING
    # (likely typo), surfaced by the CLI pre-check — not silently clean — but not an error that blocks.
    issues = loader.validate(loader.load({"qc": {"TMEP": {"spike_test": {"suspect_threshold": 1}}}}))
    assert any(i.severity == loader.WARNING and "TMEP" in i.message and "typo" in i.message for i in issues)
    assert not any(i.severity == loader.ERROR for i in issues)


def test_source_with_colon_round_trips(tmp_path):
    # Review finding 8: a source carrying a YAML-special character (a colon) round-trips — load reads
    # it back whole and check raises nothing. init builds with ruamel for the same reason (comments +
    # proper quoting on write).
    path = tmp_path / "c.yaml"
    path.write_text('config_version: 1\nsource: "a: b"\nqc: {}\n')
    cfg = loader.load(path)
    assert cfg.source == "a: b"
    assert loader.check(cfg) is cfg  # no error raised


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


def test_validate_variable_name_not_a_vocabulary():
    # The built-in four are defaults, not an allowed-set: without a dataset the check is shape-only,
    # so any variable name validates (CNDC/BBP700/…). Variable existence is a WARNING, checked only
    # with the file open in report(), never an error that blocks the run.
    issues = loader.validate(loader.load({"qc": {"CNDC": {"spike_test": {"suspect_threshold": 1}}}}))
    assert not any(i.severity == loader.ERROR for i in issues)


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
    # BBP700 is not a built-in default variable; it is still accepted (shape-only), and with the
    # dataset carrying it the check raises nothing.
    assert loader.validate(cfg, _ds("BBP700")) == []
    # Without the dataset the check is shape-only — the unknown name is not an error.
    assert not any(i.severity == loader.ERROR for i in loader.validate(cfg))


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
