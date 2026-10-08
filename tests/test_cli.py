import shutil
import subprocess
import sys

import pytest

from glidertest.cli import _expand_inputs, main
from glidertest.reports import paths

# The committed subsets stand in for a downloaded sample: sea045 carries an OG1 `id`, sg014 does not
# (so it falls back to its file stem) — two distinct mission ids for the multi-file tests. See
# tests/data/README.md. Most CLI tests stub the figure render (the `fake_report` fixture), so the
# only cost is opening the small subset; the few that assert real HTML/figures are marked `slow`.


def _run(argv):
    """Call ``main(argv)`` and return the process exit code."""
    with pytest.raises(SystemExit) as exc:
        main(argv)
    return 0 if exc.value.code is None else exc.value.code


def _mission_dirs(root):
    """Subdirectories of *root* that hold a report.json (i.e. a written mission)."""
    if not root.is_dir():
        return []
    return [d for d in sorted(root.iterdir()) if d.is_dir() and paths.manifest_in(d).exists()]


# --- parser / help (no data) -------------------------------------------------


def test_help_lists_subcommands(capsys):
    assert _run(["--help"]) == 0
    out = capsys.readouterr().out
    assert "report" in out
    assert "navigator" in out


def test_version(capsys):
    assert _run(["--version"]) == 0
    assert capsys.readouterr().out.strip().startswith("glidertest ")


def test_no_subcommand_lists_commands(capsys):
    assert _run([]) == 2
    err = capsys.readouterr().err  # full help, so the commands are named
    assert "report" in err
    assert "navigator" in err


def test_report_requires_output(capsys):
    assert _run(["report", "dummy.nc"]) == 2
    assert "required" in capsys.readouterr().err.lower()


def test_cli_import_is_backend_free():
    # Importing the CLI must not pull in matplotlib or xarray, so --help stays fast and backend-free.
    code = (
        "import sys, glidertest.cli; "
        "assert 'matplotlib' not in sys.modules, 'cli import pulled in matplotlib'; "
        "assert 'xarray' not in sys.modules, 'cli import pulled in xarray'"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


def test_module_invocation_version():
    r = subprocess.run(
        [sys.executable, "-m", "glidertest", "--version"], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0
    assert r.stdout.strip().startswith("glidertest ")


# --- _expand_inputs (pure logic, no data) ------------------------------------


def test_expand_inputs_dir_and_dedup(tmp_path):
    (tmp_path / "a.nc").touch()
    (tmp_path / "b.nc").touch()
    (tmp_path / "c.txt").touch()
    got = _expand_inputs([str(tmp_path), str(tmp_path / "a.nc")], "*.nc")
    assert [p.name for p in got] == ["a.nc", "b.nc"]  # .txt excluded, a.nc not duplicated


def test_expand_inputs_literal_glob(tmp_path):
    (tmp_path / "x1.nc").touch()
    (tmp_path / "x2.nc").touch()
    got = _expand_inputs([str(tmp_path / "x*.nc")], "*.nc")
    assert sorted(p.name for p in got) == ["x1.nc", "x2.nc"]


# --- report (root layout) ----------------------------------------------------


@pytest.mark.slow
@pytest.mark.e2e
def test_report_writes_mission_and_navigator(tmp_path, subset_path):
    assert _run(["report", str(subset_path), "--report-dir", str(tmp_path)]) == 0
    assert (tmp_path / "index.html").exists()  # fleet navigator
    dirs = _mission_dirs(tmp_path)
    assert len(dirs) == 1
    assert (dirs[0] / "index.html").exists()
    assert (dirs[0] / "report.json").exists()


def test_report_no_navigator(tmp_path, subset_path, fake_report):
    assert _run(["report", str(subset_path), "--report-dir", str(tmp_path), "--no-navigator"]) == 0
    assert not (tmp_path / "index.html").exists()
    assert len(_mission_dirs(tmp_path)) == 1


def test_report_mission_id_override(tmp_path, subset_path, fake_report):
    argv = ["report", str(subset_path), "--report-dir", str(tmp_path), "--mission-id", "custom"]
    assert _run(argv) == 0
    assert (tmp_path / "custom" / "index.html").exists()


def test_report_mission_id_rejects_multiple(tmp_path, subset_path):
    a, b = tmp_path / "a.nc", tmp_path / "b.nc"
    shutil.copy(subset_path, a)
    shutil.copy(subset_path, b)
    out = tmp_path / "out"
    assert _run(["report", str(a), str(b), "--report-dir", str(out), "--mission-id", "x"]) == 2
    assert not out.exists()


def test_report_duplicate_id_refused(tmp_path, subset_path, capsys):
    # Refused in the pre-pass (both copies resolve to one OG1 id), before any report is rendered.
    a, b = tmp_path / "a.nc", tmp_path / "b.nc"
    shutil.copy(subset_path, a)
    shutil.copy(subset_path, b)
    out = tmp_path / "out"
    assert _run(["report", str(a), str(b), "--report-dir", str(out)]) == 1
    assert "map to mission" in capsys.readouterr().err
    assert _mission_dirs(out) == []  # nothing written


def test_report_skip_existing(tmp_path, subset_path, fake_report, capsys):
    p = str(subset_path)
    assert _run(["report", p, "--report-dir", str(tmp_path), "--no-navigator"]) == 0
    manifest = _mission_dirs(tmp_path)[0] / "report.json"
    before = manifest.stat().st_mtime_ns
    capsys.readouterr()  # clear
    argv = ["report", p, "--report-dir", str(tmp_path), "--no-navigator", "--skip-existing"]
    assert _run(argv) == 0
    assert "skipped" in capsys.readouterr().out
    assert manifest.stat().st_mtime_ns == before  # not rewritten


def test_report_dry_run(tmp_path, subset_path, capsys):
    assert _run(["report", str(subset_path), "--report-dir", str(tmp_path), "-n"]) == 0
    out = capsys.readouterr().out
    assert "->" in out
    assert "would rebuild" in out
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_report_bad_file_continues(tmp_path, subset_path, fake_report, capsys):
    missing = str(tmp_path / "nope.nc")
    assert _run(["report", missing, str(subset_path), "--report-dir", str(tmp_path)]) == 1
    assert "nope.nc" in capsys.readouterr().err
    assert (tmp_path / "index.html").exists()  # navigator still built over the one success
    assert len(_mission_dirs(tmp_path)) == 1


def test_report_directory_input(tmp_path, subset_path, sg014_subset_path, fake_report):
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(subset_path, data / "m1.nc")
    shutil.copy(sg014_subset_path, data / "m2.nc")
    out = tmp_path / "out"
    assert _run(["report", str(data), "--report-dir", str(out)]) == 0
    assert len(_mission_dirs(out)) == 2
    assert (out / "index.html").exists()


def test_report_pattern_selects(tmp_path, subset_path, sg014_subset_path, fake_report):
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(sg014_subset_path, data / "sg_one.nc")
    shutil.copy(subset_path, data / "sea_one.nc")
    out = tmp_path / "out"
    assert _run(["report", str(data), "--report-dir", str(out), "--pattern", "sg_*.nc"]) == 0
    assert len(_mission_dirs(out)) == 1


@pytest.mark.slow
def test_report_links_back_to_fleet(tmp_path, subset_path):
    # Each mission's pages are written with navigator=False in the batch, yet still link back to the
    # fleet page that is built once at the end. Needs the real HTML, so it renders.
    assert _run(["report", str(subset_path), "--report-dir", str(tmp_path)]) == 0
    html = (_mission_dirs(tmp_path)[0] / "index.html").read_text(encoding="utf-8")
    assert "All missions" in html
    assert 'href="../index.html"' in html


# --- report (flat layout, -o) ------------------------------------------------


@pytest.mark.slow
@pytest.mark.e2e
def test_report_flat_output(tmp_path, subset_path):
    out = tmp_path / "one"
    assert _run(["report", str(subset_path), "-o", str(out)]) == 0
    assert (out / "index.html").exists()  # landing written directly into DIR
    assert (out / "report.json").exists()
    assert (out / "figures").is_dir()
    assert _mission_dirs(out) == []  # no <mission_id>/ nesting
    assert "All missions" not in (out / "index.html").read_text(encoding="utf-8")  # no fleet page


def test_report_flat_rejects_multiple(tmp_path, subset_path):
    a, b = tmp_path / "a.nc", tmp_path / "b.nc"
    shutil.copy(subset_path, a)
    shutil.copy(subset_path, b)
    assert _run(["report", str(a), str(b), "-o", str(tmp_path / "out")]) == 2


def test_report_flat_rejects_mission_id(tmp_path, capsys):
    assert _run(["report", "dummy.nc", "-o", str(tmp_path / "out"), "--mission-id", "x"]) == 2
    assert "no effect with -o" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_report_flat_skip_existing(tmp_path, subset_path, fake_report, capsys):
    out = tmp_path / "one"
    p = str(subset_path)
    assert _run(["report", p, "-o", str(out)]) == 0
    manifest = out / "report.json"
    before = manifest.stat().st_mtime_ns
    capsys.readouterr()
    assert _run(["report", p, "-o", str(out), "--skip-existing"]) == 0
    assert "skipped" in capsys.readouterr().out
    assert manifest.stat().st_mtime_ns == before  # not regenerated


# --- navigator ---------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.e2e
def test_navigator_rebuilds(tmp_path, subset_path):
    assert _run(["report", str(subset_path), "--report-dir", str(tmp_path)]) == 0
    (tmp_path / "index.html").unlink()
    assert _run(["navigator", str(tmp_path), "--title", "My Fleet"]) == 0
    assert (tmp_path / "index.html").exists()
    assert "My Fleet" in (tmp_path / "index.html").read_text(encoding="utf-8")


def test_navigator_missing_root(tmp_path, capsys):
    assert _run(["navigator", str(tmp_path / "nope")]) == 1
    assert "not found" in capsys.readouterr().err


def test_navigator_rejects_mission_dir_via_cli(tmp_path, subset_path, fake_report, capsys):
    # A flat -o report folder has a top-level report.json; pointing navigator at it would overwrite
    # its landing page with an empty fleet page, so it is refused.
    out = tmp_path / "one"
    assert _run(["report", str(subset_path), "-o", str(out)]) == 0
    landing_before = (out / "index.html").read_bytes()
    assert _run(["navigator", str(out)]) == 1
    assert "mission report directory" in capsys.readouterr().err
    assert (out / "index.html").read_bytes() == landing_before  # landing page untouched


# --- error branches (no data download) ---------------------------------------


def test_report_no_files_matched(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _run(["report", str(empty), "--report-dir", str(tmp_path / "out")]) == 1
    assert "no files matched" in capsys.readouterr().err


def test_report_unreadable_file_continues(tmp_path, capsys):
    bad = tmp_path / "bad.nc"
    bad.write_text("not a netcdf file")  # present but not a dataset: open raises, run continues
    assert _run(["report", str(bad), "--report-dir", str(tmp_path / "out")]) == 1
    assert "bad.nc" in capsys.readouterr().err


def test_report_rejects_bad_layout(tmp_path):
    import xarray as xr

    from glidertest.reports import report

    with pytest.raises(ValueError, match="layout"):
        report(xr.Dataset(), tmp_path, layout="sideways")


# --- library guards that back the CLI (report()/navigator()) -----------------


def test_report_rejects_mission_id_with_flat(tmp_path):
    import xarray as xr

    from glidertest.reports import report

    with pytest.raises(ValueError, match="mission_id"):
        report(xr.Dataset(), tmp_path, layout="flat", mission_id="x")


def test_report_flat_refuses_report_root(tmp_path):
    import xarray as xr

    from glidertest.reports import report

    root = tmp_path / "root"
    (root / "m1").mkdir(parents=True)
    (root / "m1" / "report.json").write_text("{}")  # makes `root` a report root
    with pytest.raises(ValueError, match="report root"):
        report(xr.Dataset(), root, layout="flat")


def test_report_flat_into_report_root_fails_via_cli(tmp_path, subset_path, capsys):
    root = tmp_path / "root"
    (root / "m1").mkdir(parents=True)
    (root / "m1" / "report.json").write_text("{}")
    assert _run(["report", str(subset_path), "-o", str(root)]) == 1
    assert "report root" in capsys.readouterr().err


def test_navigator_missing_root_raises(tmp_path):
    from glidertest.reports import navigator

    with pytest.raises(FileNotFoundError, match="report root not found"):
        navigator(tmp_path / "nope")


def test_navigator_rejects_mission_dir(tmp_path):
    from glidertest.reports import navigator

    (tmp_path / "report.json").write_text("{}")  # a mission dir (top-level manifest), not a root
    with pytest.raises(ValueError, match="mission report directory"):
        navigator(tmp_path)


def test_report_skip_existing_leaves_fleet_page(tmp_path, subset_path, fake_report, capsys):
    p = str(subset_path)
    assert _run(["report", p, "--report-dir", str(tmp_path)]) == 0
    index = tmp_path / "index.html"
    before = index.stat().st_mtime_ns
    capsys.readouterr()
    # An all-skipped re-run writes nothing, so the fleet page is left untouched (cheap re-run).
    assert _run(["report", p, "--report-dir", str(tmp_path), "--skip-existing"]) == 0
    assert index.stat().st_mtime_ns == before
    # --dry-run over the same (all skipped) says the real run would not rebuild.
    capsys.readouterr()
    assert _run(["report", p, "--report-dir", str(tmp_path), "--skip-existing", "-n"]) == 0
    assert "would not be rebuilt" in capsys.readouterr().out
