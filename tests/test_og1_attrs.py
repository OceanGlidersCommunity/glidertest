import xarray as xr

from glidertest import og1_attrs


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


def test_sample_dataset_conformance(subset_ds):
    rows = og1_attrs.check_globals(subset_ds)
    # the sea045 subset keeps all 16 mandatory attributes present
    assert sum(1 for _, s, _ in rows if s != "none") == 16
    assert all(s in ("match", "none") for _, s, _ in rows)


def test_registry_is_the_source_of_truth():
    # The manual's four categories cover all 41 global attributes; the derived constants come from
    # the registry so they cannot drift from it.
    flat = [name for _title, names in og1_attrs.ATTR_GROUPS for name, _tier in names]
    assert len(flat) == len(set(flat)) == 41
    assert tuple(flat) == og1_attrs.GLOBAL_ATTR_ORDER
    tiers = {t: sum(1 for _n, tt in og1_attrs._TIER_OF.items() if tt == t) for t in
             ("mandatory", "highly_desirable", "suggested")}
    assert tiers == {"mandatory": 16, "highly_desirable": 14, "suggested": 11}
    assert len(og1_attrs.MANDATORY_GLOBALS) == 16


def test_group_globals_includes_absent_rows_but_hides_absent_suggested():
    # A file with only `title`: mandatory/highly-desirable rows still appear (for the conformance
    # status), but absent *suggested* attributes (e.g. the geospatial bounds) are omitted.
    groups = og1_attrs.group_globals({"title": "t", "zzz_custom": "q"})
    by_title = {g["title"]: g["rows"] for g in groups}
    rows = {r["name"]: r for g in groups for r in g["rows"]}
    assert rows["title"]["present"] is True and rows["title"]["value"] == "t"
    assert rows["id"]["present"] is False and rows["id"]["value"] is None  # mandatory, absent
    assert rows["time_coverage_start"]["present"] is False  # highly desirable, absent -> shown
    assert "geospatial_lat_min" not in rows  # suggested, absent -> hidden
    # Spatiotemporal has only highly-desirable rows left, so it is not dropped; but a group with no
    # surviving rows would be. Non-registry keys land in the "Other" group (shown last), never dropped.
    assert by_title[og1_attrs.OTHER_GROUP] == [
        {"name": "zzz_custom", "value": "q", "tier": None, "present": True}
    ]
    assert og1_attrs.OTHER_GROUP == "Other (not in OG1)"
    assert [g["title"] for g in groups][-1] == og1_attrs.OTHER_GROUP  # Other is last


def test_conformance_summary_counts():
    summary = og1_attrs.conformance_summary({"title": "t"})
    assert summary["mandatory_total"] == 16
    assert summary["mandatory_present"] == 1
    assert summary["highly_desirable_missing"] == 14
