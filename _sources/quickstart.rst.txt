.. _quickstart:

==========
Quickstart
==========

From an OG1 file to an open report in three commands.

Input: an OG1 mission file
--------------------------

glidertest reads one glider mission as an `OceanGliders (OG1)
<https://github.com/OceanGlidersCommunity/OG-format-user-manual>`_ netCDF file — one file per
deployment, with the measurements along an ``N_MEASUREMENTS`` dimension. If you have Seaglider
basestation files, the ``seagliderOG1`` package converts them to OG1; SeaExplorer and Slocum
data reach OG1 through your processing pipeline. No file yet? ``fetchers.load_sample_dataset()``
returns a ready OG1 sample.

Install
-------

.. code-block:: bash

   pip install glidertest

For a development install see the `README <https://github.com/OceanGlidersCommunity/glidertest#install>`_.

.. note::

   The glider-track and fleet maps download Natural Earth coastlines through cartopy the first
   time they are drawn, so the first map needs internet. On an offline machine, draw one map
   while online once (cartopy caches the data) or point ``CARTOPY_DATA_DIR`` at a pre-filled
   directory.

Run it
------

One mission, into a report tree:

.. code-block:: bash

   glidertest report sea045_20230604T1253_delayed.nc --report-dir reports/

This writes

.. code-block:: text

   reports/
     index.html                               the fleet page (one row per mission)
     sea045_20230604T1253_delayed/
       index.html  ctd.html  oxygen.html  optics.html  inventory.html
       figures/    report.json

and prints the path of the landing page. Two variants you will want soon:

.. code-block:: bash

   # every *.nc in a directory; the fleet page is rebuilt once at the end
   glidertest report data/ --report-dir reports/

   # one mission's pages straight into a folder, no tree and no fleet page
   glidertest report sea045_20230604T1253_delayed.nc -o sea045_report/

Every flag is in the :ref:`cli_reference`.

Open it
-------

Find ``reports/index.html`` in your file manager and double-click it, or from the terminal::

   open reports/index.html          # macOS
   xdg-open reports/index.html      # Linux
   start reports/index.html         # Windows

The fleet page links to each mission; a mission's masthead links to its sensor pages and
back to the fleet.

From Python or a notebook
-------------------------

The same thing, as a function call:

.. code-block:: python

   import xarray as xr
   from glidertest import reports

   ds = xr.open_dataset("sea045_20230604T1253_delayed.nc")
   reports.report(ds, "reports/")

No file to hand? The sample missions download on first use:

.. code-block:: python

   from glidertest import fetchers, reports

   ds = fetchers.load_sample_dataset()          # a SeaExplorer mission in the Baltic
   reports.report(ds, "reports/")

Or run a single diagnostic, with no report at all — every figure in the report is one of these
functions:

.. code-block:: python

   from glidertest import plots

   fig, ax = plots.plot_hysteresis(ds, var="DOXY")   # just the dive–climb check for oxygen

For many missions, write each with ``navigator=False`` and rebuild the fleet page once:

.. code-block:: python

   for path in paths:
       with xr.open_dataset(path) as ds:
           reports.report(ds, "reports/", navigator=False)
   reports.navigator("reports/")

What to look at first
---------------------

The masthead of the landing page gives the mission at a glance — profiles, time span,
extent, platform — and a one-line OG1 conformance verdict. The two **QC** sections below it
separate what the file *says* (its own ``*_QC`` flags) from what glidertest *finds*. The
**inventory** page (linked under the masthead) lists every attribute and variable in the
file, with anything mandatory that is missing marked in amber. The verdict — e.g. "OG1: 15 of
16 mandatory attributes" — is about *presence*, not validity: it does not check that the
values are correct (see the inventory page). :doc:`reports` walks through every page.

Where next
----------

* :doc:`reports` — what each page shows and how to read it.
* :ref:`cli_reference` — every command and flag.
* :doc:`demo-output` — the diagnostics one function at a time, in a notebook.
