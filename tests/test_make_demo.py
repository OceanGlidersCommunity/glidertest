import importlib.util
from pathlib import Path

from glidertest import fetchers, reports

# scripts/make_demo.py builds the docs live-demo fleet. Load it by path (scripts/ is not a package)
# and stub the library calls so the tests exercise the orchestration — cleanup, per-mission failure
# continuation, the navigator call, and the exit code — without a download or a figure render.
_spec = importlib.util.spec_from_file_location(
    "make_demo", Path(__file__).parent.parent / "scripts" / "make_demo.py"
)
make_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_demo)


def _stub(monkeypatch, *, fail=()):
    """Stub fetchers/reports; return a dict recording the report and navigator calls."""
    calls = {"reports": [], "navigator": []}

    def fake_load(dataset_name=None):
        return dataset_name  # the mission name stands in for the dataset

    def fake_report(ds, outdir, *, navigator=True):
        if ds in fail:
            raise RuntimeError(f"cannot report {ds}")
        mdir = Path(outdir) / str(ds)
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "index.html").write_text("mission", encoding="utf-8")
        calls["reports"].append(ds)
        return mdir / "index.html"

    def fake_navigator(root, title=None):
        calls["navigator"].append((Path(root), title))
        fleet = Path(root) / "index.html"
        fleet.write_text("fleet", encoding="utf-8")
        return fleet

    monkeypatch.setattr(fetchers, "load_sample_dataset", fake_load)
    monkeypatch.setattr(reports, "report", fake_report)
    monkeypatch.setattr(reports, "navigator", fake_navigator)
    return calls


def test_make_demo_success(tmp_path, monkeypatch):
    calls = _stub(monkeypatch)
    rc = make_demo.main(["--out", str(tmp_path)])
    assert rc == 0
    assert calls["reports"] == list(make_demo.MISSIONS)  # every mission reported, in order
    assert len(calls["navigator"]) == 1  # fleet page built once, at the end
    assert calls["navigator"][0][1] == "glidertest demo"  # with the demo title
    assert (tmp_path / "index.html").exists()


def test_make_demo_failure_continues_and_returns_nonzero(tmp_path, monkeypatch):
    first = make_demo.MISSIONS[0]
    calls = _stub(monkeypatch, fail={first})
    rc = make_demo.main(["--out", str(tmp_path)])
    assert rc == 1  # a failed mission is a non-zero exit
    assert first not in calls["reports"]  # it failed
    assert calls["reports"] == list(make_demo.MISSIONS[1:])  # the rest were still written
    assert len(calls["navigator"]) == 1  # navigator still runs over the successes


def test_make_demo_clean_wipes_first(tmp_path, monkeypatch):
    out = tmp_path / "demo"
    out.mkdir()
    stale = out / "stale.txt"
    stale.write_text("old", encoding="utf-8")
    _stub(monkeypatch)
    rc = make_demo.main(["--out", str(out), "--clean"])
    assert rc == 0
    assert not stale.exists()  # --clean removed the previous contents
    assert (out / "index.html").exists()  # and rebuilt
