"""Shared report layer, vendored byte-identical across sibling packages.

The modules in this directory (``design_tokens``, ``figure_png``, ``css``,
``page_model``, ``leaflet``) and the shared ``templates`` are the report
presentation layer used by ctdcast, oceanarray and glidertest, kept
byte-identical across the three repositories. The bundled Leaflet library
(``leaflet/``) ships here too, so the interactive map works offline.

ctdcast ``main`` is the source of truth. An edit is made here in ctdcast and
re-vendored out to the siblings (see ``README.md`` for the recipe); it is never
edited in a consumer repository. ``check.py`` is the gate that hashes this
directory against ``MANIFEST.txt`` and the siblings; both live here, so the
directory describes and checks itself.

This package re-exports nothing: a consumer imports the specific module it
needs (``from ...reports._vendored import design_tokens``), so each dependency
is visible at its import site.
"""
