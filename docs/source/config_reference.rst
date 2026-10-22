.. _config_reference:

======================
Report config (``-c``)
======================

The QC thresholds glidertest applies — gross range, spike and flat-line per variable — are
built-in defaults tuned for one region. A **report config** overrides them per variable and per
test, and adds a region test, for one mission or a whole run. It is optional: with no config the
report uses the defaults and its output is unchanged.

Pass it to the command line or the Python API::

   glidertest report mission.nc --report-dir reports/ -c report.yaml

.. code-block:: python

   from glidertest import reports
   reports.report(ds, "reports/", config="report.yaml")   # or config=<dict>

The config only changes which thresholds the diagnostics use and what the report records. It
never modifies the data.

Generating a starting template
------------------------------

``glidertest init`` reads a mission file and writes a commented template seeded with glidertest's
suggested thresholds for *that* mission (from its own data distribution), with the built-in
default shown beside each so you see both::

   glidertest init mission.nc -o report.yaml

Edit the values you want, then pass the file back with ``-c``. The region box is left commented
out: a box drawn from the mission's own track cannot fail its own region test, so you set it
deliberately. The categories the loader recognises but does not yet apply (``qc.hysteresis`` and
the top-level ``display``, ``mission`` and ``pages`` blocks) are written out commented, so the
template shows the whole menu without triggering their not-applied warnings.

The file
--------

.. code-block:: yaml

   config_version: 1
   source: "sea045 QC pass, 2026-10"      # free text, recorded in report.json (qc_config_source)

   region:                                # optional; absent → the region test is not run
     bbox: [10, 50, 25, 70]               # [west, south, east, north] = lon_min, lat_min, lon_max, lat_max

   qc:                                    # variable-first; ioos_qc test and field names
     TEMP:
       gross_range_test: {suspect_span: [0, 30], fail_span: [-2.5, 40]}
       spike_test:       {suspect_threshold: 2.0, fail_threshold: 6.0}
       flat_line_test:   {suspect_threshold: 1, fail_threshold: 3, tolerance: 0.001}
     PSAL:
       gross_range_test: {suspect_span: [5, 38]}   # partial: only the fields you set change

Each block is a deep-merge over the defaults: a variable, test or field you do not mention keeps
its built-in value. Thresholds are in each variable's OG1 units. The test and field names are
ioos_qc's own vocabulary (``gross_range_test.suspect_span`` / ``fail_span``,
``spike_test.suspect_threshold`` / ``fail_threshold``, ``flat_line_test`` windows in seconds plus
``tolerance``); the same keys read the same across the OG1 toolchain.

A config may also ride inside another tool's YAML under a top-level ``glidertest:`` key — then
only that block is read.

The region test
---------------

With ``region.bbox`` set, glidertest flags every sample by position: inside or on the box is good,
outside is **fail** (a glider outside the region you declared is a finding, not a "maybe"), and a
sample with no position (``LONGITUDE``/``LATITUDE`` missing or NaN) is "not evaluated". The box is
``[west, south, east, north]``. There is no built-in region — it runs only when you configure one.
This is glidertest's own test, not ioos_qc's ``location_test`` (which has no bounding box).

Validation
----------

The file is checked at load, before any figure is drawn; an error names the allowed set and stops
the run (exit code 2 from the CLI). Errors:

- an unknown top-level key, variable, test, or field name;
- a span that is not a two-element increasing numeric list;
- a suspect span not contained in its fail span;
- a ``region.bbox`` that is not four increasing numbers.

A threshold set for a variable the file does not contain is a **warning**, not an error. The
``display:``, ``mission:`` and ``pages:`` blocks are recognised but not yet applied (a later
release); their presence warns rather than being silently ignored.

Provenance — what the report records
------------------------------------

So a reader can tell a configured threshold from a default and reproduce the run:

- the QC page's threshold table marks each row **default** or **config** (by comparing the applied
  value to the built-in), and the region row shows the box and how many samples fell outside it;
- ``report.json`` carries ``qc_thresholds`` (the effective thresholds that ran), ``qc_config`` (the
  overrides exactly as given, or ``null``), ``qc_config_source`` (the ``source:`` string),
  ``config_file`` (the path you passed, or ``null`` for a dict) and ``region`` (the box, or ``null``);
- the config itself is saved beside ``report.json`` as ``report-config.yaml``, so a re-run can read
  it back: ``report(ds, outdir, config="<report-dir>/report-config.yaml")``. A file is copied
  verbatim (comments and the not-yet-applied ``display:``/``mission:``/``pages:`` blocks are kept); a
  config given as a Python ``dict`` is written out from its ``qc``/``region``/``source`` blocks only,
  so it carries no comments and drops any ``display:``/``mission:``/``pages:`` the dict held.
