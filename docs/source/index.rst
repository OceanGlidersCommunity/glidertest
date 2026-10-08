.. glidertest documentation master file

=====================================================================
glidertest: diagnostics and HTML reports for OG1 glider data
=====================================================================

**glidertest** reads a glider mission in `OG1 format <https://github.com/OceanGlidersCommunity/OG-format-user-manual>`_
and tells you what is in it and what looks wrong: sensor drift, dive–climb bias, quenching,
flight-model performance, sampling gaps, and the QC flags the file already carries.
It is a diagnostic tool only — it never modifies your data and never fills in a value it
could not determine.

**As a library.** Every diagnostic is a function: ``plots.plot_quench_assess(ds, 'CHLA')``
draws the chlorophyll quenching check, ``tools.compute_daynight_avg(ds)`` returns the numbers
behind it. Call the ones you want from a notebook, on your own ``xarray.Dataset``, and look at
the figure — see :doc:`glidertest` and the example notebook.

**As a report.** ``glidertest report mission.nc --report-dir reports/`` runs every diagnostic
that applies to the file and writes a **self-contained HTML report**: a landing page about the
mission, one page per sensor present (CTD, oxygen, optics), a flight page for gliders that
report a flight-model velocity, and an inventory page about the file itself. Every figure is
embedded, so the pages work offline and can be mailed or dropped on a share. Many missions in
one directory get a **fleet page** with a map and a table. The same from Python is
``reports.report(ds, "reports/")``.

The report is the library run in a fixed order; nothing in it is unavailable as a function.

.. admonition:: Live demo

   `Open the example report <_static/demo/index.html>`__ — two missions, a SeaExplorer in the
   Baltic and a Seaglider in the Labrador Sea, built from the sample data by this site's own
   documentation build.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Examples

   demo-output.ipynb
   Example HTML report <https://oceangliderscommunity.github.io/glidertest/_static/demo/index.html>

.. toctree::
   :maxdepth: 2
   :caption: Reference

   cli_reference
   reports
   glidertest

.. toctree::
   :maxdepth: 1
   :caption: Links

   GitHub repository <https://github.com/OceanGlidersCommunity/glidertest>
   Report a problem <https://github.com/OceanGlidersCommunity/glidertest/issues/new>
   OG1 format manual <https://github.com/OceanGlidersCommunity/OG-format-user-manual>

Background
----------

We recommend the OceanGliders best-practice guides, starting with the
`Oxygen SOP <https://oceangliderscommunity.github.io/Oxygen_SOP/README.html>`_, and the
`GliderTools <https://glidertools.readthedocs.io/en/latest/>`_ package for processing and for
addressing some of the issues glidertest points out.

The diagnostics build on:

* Bennett, J., Stahr, F., and Eriksen, C. (2019). Determining Seaglider Velocities Automatically. http://hdl.handle.net/1773/44948
* Frajka-Williams, E., Eriksen, C. C., Rhines, P. B., and Harcourt, R. R. (2011). Determining vertical water velocities from Seaglider. *Journal of Atmospheric and Oceanic Technology*, 28(12), 1641–1656. https://doi.org/10.1175/2011JTECHO830.1
* Thomalla, S. J., Moutier, W., Ryan-Keogh, T. J., Gregor, L., and Schütt, J. (2018). An optimized method for correcting fluorescence quenching using optical backscattering on autonomous platforms. *Limnology and Oceanography: Methods*, 16, 132–144. https://doi.org/10.1002/lom3.10234

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
