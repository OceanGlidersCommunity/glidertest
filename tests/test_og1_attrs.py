import xarray as xr

from glidertest import fetchers, og1_attrs


def _ds_with(attrs):
    return xr.Dataset(attrs=attrs)


def test_check_globals_count():
    rows = og1_attrs.check_globals(_ds_with({}))
    assert len(rows) == len(og1_attrs.MANDATORY_GLOBALS) == 16


def test_all_missing_are_none():
    rows = og1_attrs.check_globals(_ds_with({}))
    assert all(status == "none" for _, status, _ in rows)


def test_present_valid_is_match():
    attrs = dict.fromkeys(og1_attrs.MANDATORY_GLOBALS, "x")
    attrs["start_date"] = "20230604T125304"
    attrs["date_created"] = "20230604T125304"
    attrs["featureType"] = "trajectory"
    status = {a: s for a, s, _ in og1_attrs.check_globals(_ds_with(attrs))}
    assert status["title"] == "match"
    assert status["start_date"] == "match"
    assert status["featureType"] == "match"


def test_empty_string_counts_as_missing():
    status = {a: s for a, s, _ in og1_attrs.check_globals(_ds_with({"title": ""}))}
    assert status["title"] == "none"


def test_present_value_is_match_regardless_of_format():
    # Values are not format-checked: amber means missing only. A seconds-less datetime and a
    # non-"trajectory" featureType are present, so both are "match".
    status = {
        a: s
        for a, s, _ in og1_attrs.check_globals(
            _ds_with({"start_date": "20230604T1253", "featureType": "profile"})
        )
    }
    assert status["start_date"] == "match"
    assert status["featureType"] == "match"


def test_sample_dataset_conformance():
    ds = fetchers.load_sample_dataset()
    rows = og1_attrs.check_globals(ds)
    # the VOTO sample has all 16 mandatory attributes present
    assert sum(1 for _, s, _ in rows if s != "none") == 16
    assert all(s in ("match", "none") for _, s, _ in rows)
