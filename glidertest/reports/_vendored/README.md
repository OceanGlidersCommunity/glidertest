# Vendored report layer

The files in this directory are the shared report presentation layer — the
design tokens, the figure encoder, the CSS generator, the page model, the
matplotlib style, the shared templates, the Leaflet map building blocks
(`leaflet.py`, `templates/_map.html`, and the bundled Leaflet library under
`leaflet/`), and the units layer (`units_tables.py`, the canonical spellings and
their aliases; `units.py`, the `resolve_units` / `merge_tables` / `format_units`
machinery over them). They are kept **byte-identical** across ctdcast, oceanarray
and glidertest.

ctdcast `main` is the source of truth. Make an edit here in ctdcast, then
re-vendor it out to the siblings. Never edit a sibling's copy in place — the
gate below will flag it as drift.

The directory describes and checks itself: `MANIFEST.txt` records the sha256 of
every file here, and `check.py` is the gate. A consumer copies the whole
directory and takes nothing else — no manifest or script outside it.

## Re-vendor into a sibling

From the sibling repo, copy ctdcast's directory over its own and refresh the
hash manifest:

    rsync -a --delete ../ctdcast/ctdcast/reports/_vendored/ <pkg>/reports/_vendored/
    python -m <pkg>.reports._vendored.check --update

then commit. `python -m <pkg>.reports._vendored.check ../ctdcast` confirms every
file matches.

## The gate

`check.py` hashes this directory against `MANIFEST.txt` and fails if a file
drifts; a three-line test runs it in CI, so an edit here that skips the
re-vendor is caught before merge. `check.py` and `MANIFEST.txt` live in the set
too, so the gate guards itself.
