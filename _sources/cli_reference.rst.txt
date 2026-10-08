.. _cli_reference:

=============
CLI reference
=============

All commands are available through the ``glidertest`` entry point (or ``python -m glidertest``).
Run ``glidertest <command> --help`` for the full flag list at any time.

Exit status, every command: **0** success; **1** one or more input files failed (the rest were
still written, each failure one line on stderr); **2** usage error.

Typical workflow::

   glidertest report mission.nc --report-dir reports/   one mission -> reports/<id>/, fleet page rebuilt
   glidertest report data/ --report-dir reports/        every *.nc in data/, fleet page rebuilt once
   glidertest report mission.nc -o out/                 one mission's pages directly in out/, no fleet page
   glidertest navigator reports/                        re-index after adding or removing missions

----

glidertest report
-----------------

Write an HTML mission report for each OG1 netCDF file: a landing page, one page per sensor
present, a flight page when the data supports it, an inventory page, and a ``report.json``
manifest. The dataset is read, never modified.

.. argparse::
   :module: glidertest.cli
   :func: build_parser
   :prog: glidertest
   :path: report
   :nodescription:

Examples:

.. code-block:: bash

   # One mission into a report tree; the fleet page is rebuilt afterwards:
   glidertest report sea045_20230604T1253_delayed.nc --report-dir reports/

   # A directory of missions; the fleet page is rebuilt once at the end:
   glidertest report data/ --report-dir reports/

   # Name the subdirectory yourself when the file has no usable OG1 id:
   glidertest report odd_file.nc --report-dir reports/ --mission-id sea045_2023_leg2

   # Re-run over a folder, reporting only missions not yet in the tree:
   glidertest report data/ --report-dir reports/ --skip-existing

   # One mission's pages straight into a folder, no tree and no fleet page:
   glidertest report sea045_20230604T1253_delayed.nc -o sea045_report/

Notes:

- ``FILE`` may be a directory; ``--pattern`` (default ``*.nc``) selects files inside it, not
  recursively. A wildcard the shell did not expand (Windows) is expanded by glidertest.
- Two files with the same OG1 ``id`` would write to the same ``ROOT/<id>/``; the run refuses
  before writing anything and names both files. Use ``--mission-id`` on one of them.
- ``--report-dir`` and ``-o`` carry the same meanings as in oceanarray: the first is a
  portable tree of ``ROOT/<id>/`` with its own ``index.html``; the second puts one mission's
  pages directly into a directory.

----

glidertest navigator
--------------------

Rebuild ``ROOT/index.html``, the fleet page, from the ``ROOT/<mission_id>/report.json``
manifests. Reads no netCDF and touches no mission pages. Run it after adding, removing or
renaming mission directories.

.. argparse::
   :module: glidertest.cli
   :func: build_parser
   :prog: glidertest
   :path: navigator
   :nodescription:

----

The same commands in the sibling packages
-----------------------------------------

Skip this unless you also use ctdcast or oceanarray.

glidertest, `ctdcast <https://github.com/ocean-uhh/ctdcast>`_ (shipboard CTD) and
`oceanarray <https://github.com/ocean-uhh/oceanarray>`_ (moorings) share one report design and
one command-line vocabulary:

.. list-table::
   :widths: 30 23 23 24
   :header-rows: 1

   * -
     - glidertest
     - ctdcast
     - oceanarray
   * - Build the report
     - ``report FILE…``
     - ``report config.yaml``
     - ``report MOORING``
   * - Rebuild the index over existing output
     - ``navigator ROOT``
     - ``report --index``
     - ``report --array``
   * - One unit's pages into a directory
     - ``-o DIR``
     - (``output.dir`` in the config)
     - ``-o DIR``
   * - A portable tree of many units
     - ``--report-dir ROOT``
     - —
     - ``--report-dir ROOT``
   * - See what would be written
     - ``-n`` / ``--dry-run``
     - ``--dry-run``
     - ``-n`` / ``--dry-run``
   * - Skip units already built
     - ``--skip-existing``
     - ``--skip-existing``
     - ``--skip-existing``
