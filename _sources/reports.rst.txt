.. _reports:

===============
The HTML report
===============

``glidertest report`` turns one OG1 mission file into a set of linked, self-contained HTML
pages, and many missions into a fleet page over them. Everything is embedded — figures, styles,
the lot — so a report folder can be zipped, mailed or put on a share and opened anywhere
without a server or a network connection.

Live example
------------

The pages below are generated from the two sample missions by this site's own build, with the
same code you run.

- `Fleet page <_static/demo/index.html>`__
- `sea045 — SeaExplorer, Baltic: landing <_static/demo/sea045_20230604T1253_delayed/index.html>`__ ·
  `CTD <_static/demo/sea045_20230604T1253_delayed/ctd.html>`__ ·
  `Oxygen <_static/demo/sea045_20230604T1253_delayed/oxygen.html>`__ ·
  `Optics <_static/demo/sea045_20230604T1253_delayed/optics.html>`__ ·
  `Inventory <_static/demo/sea045_20230604T1253_delayed/inventory.html>`__
- `sg014 — Seaglider, Labrador Sea: landing <_static/demo/sg014_20040924T182454_delayed_subset/index.html>`__ ·
  `Flight <_static/demo/sg014_20040924T182454_delayed_subset/flight.html>`__

Output layout
-------------

.. code-block:: text

   ROOT/
     index.html                    fleet page: one row per mission, map of all tracks
     <mission_id>/
       index.html                  landing page — about the mission
       ctd.html                    present when TEMP or PSAL is in the file
       oxygen.html                 present when DOXY is in the file
       optics.html                 present when CHLA or BBP700 is in the file
       flight.html                 present when GLIDER_VERT_VELO_MODEL is in the file
       inventory.html              about the file: attributes, variables, QC coverage
       figures/<page>_<panel>.png  every figure as a file, 1350 px wide
       report.json                 machine-readable summary of the mission

``<mission_id>`` is the file's OG1 ``id`` attribute, or the file stem when the attribute is
missing; ``--mission-id`` overrides it. A page that the data does not support is not written,
and its button does not appear in the masthead.

``report.json`` (``manifest_version`` 1) is what the fleet page is built from and what other
tools can read without opening the netCDF:

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Field
     - Meaning
   * - ``id``, ``platform``, ``platform_serial``
     - Mission identity; the glider's platform type and serial from the file.
   * - ``start``, ``end``, ``duration_s``
     - Time span (ISO 8601 to the minute) and duration in seconds.
   * - ``n_profiles``, ``n_dive``, ``n_climb``, ``n_records``
     - Profile counts (dive/climb from ``PROFILE_DIRECTION``) and the record count.
   * - ``lat_min`` … ``lon_max``, ``max_depth_m``
     - Geographic extent and the deepest depth reached.
   * - ``source_file``, ``source_size_bytes``
     - The input file's name and size.
   * - ``sensors``
     - ``{ctd, oxygen, optics, flight}`` → true when that page was written.
   * - ``og1``
     - Mandatory OG1 attributes present / total, and the list of those missing.
   * - ``qc``
     - Whether the file carries ``*_QC`` flags, and the variable with the largest
       suspect-plus-fail fraction.
   * - ``pages``
     - The pages written, with their labels.
   * - ``track``
     - The glider track, decimated to at most 200 points, for the fleet map.
   * - ``glidertest_version``, ``generated_at``
     - Provenance.

What the diagnostics tell you
-----------------------------

Each figure in the report is one diagnostic; each is also one function in
:mod:`glidertest.plots` you can call yourself in a notebook. A *section* is a time–depth field
coloured by the variable; the rest are summarised here.

.. list-table::
   :widths: 24 40 36
   :header-rows: 1

   * - Diagnostic
     - The question
     - A problem looks like
   * - Dive–climb hysteresis
     - Does the sensor lag going down versus up? (slow sensors — optode, CTD in a thermocline)
     - A systematic offset between dive and climb profiles, growing with depth.
   * - Up/down bias by depth
     - The same, binned by depth.
     - A non-zero mean difference below the surface layer.
   * - Temporal drift
     - Does the sensor's value creep over the mission?
     - A trend in the deep (stable) values.
   * - Day/night offset and quenching
     - Is daytime chlorophyll suppressed by non-photochemical quenching?
     - Daytime surface chlorophyll lower than night for the same water.
   * - Deep-drift / negatives (optics)
     - Is the dark count right?
     - Negative or non-zero deep values.
   * - Global range
     - Are the values physically plausible?
     - Points outside the suspect / fail spans.
   * - Sampling period, grid spacing
     - How often and how densely was each variable sampled?
     - Gaps, or a sampling interval that changed mid-mission.
   * - Max depth per profile, profile monotony
     - Did the glider fly as planned?
     - Profiles that stop short, or depth that is not monotonic within a profile.
   * - Convective resistance
     - How much buoyancy loss would mix the column to a given depth? (a property, used with
       the mixed layer, not a pass/fail check)
     - —
   * - Vertical velocity (flight)
     - Does the flight model match the measured vertical speed?
     - A systematic difference, or spread, between modelled and measured *w*.

The oxygen and optics diagnostics follow the OceanGliders best-practice guides (the
`Oxygen SOP <https://oceangliderscommunity.github.io/Oxygen_SOP/README.html>`_); the flight
diagnostic follows Frajka-Williams et al. (2011).

Report pages
------------

Every page shares the **masthead** (the coloured header): the mission id, a page-type label,
generation time, the
mission facts (profiles with the dive/climb split, time span and duration, sample rate,
extent, platform, source file), the page navigation — **Summary / Reports: CTD · Oxygen ·
Optics / Derived: Flight** — and a **Data inventory** strip linking the inventory page. When
the report sits in a fleet, "← All missions" leads back. Each section heading carries "↑ top".

Landing page — ``index.html``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

About the mission.

- **Track** — the glider's path on a map (``plots.plot_glider_track``).
- **Payload** — the variables the glider carried: a one-line OG1 conformance verdict linking
  to the inventory page, and which variables are present with the ``SENSOR_*`` entry each
  comes from.
- **Hydrography** — the core variables against depth (``plots.plot_basic_vars``), the T–S
  diagram (``plots.plot_ts``), and a time–depth section for each of temperature, salinity,
  oxygen and chlorophyll that the file carries (``plots.plot_section``).
- **Sampling** — grid spacing (``plots.plot_grid_spacing``), sampling period of every variable
  (``plots.plot_sampling_period_all``), maximum depth per profile
  (``plots.plot_max_depth_per_profile``) and profile monotony (``plots.plot_prof_monotony``).
- **QC — as delivered** and **QC — glidertest diagnostics** — see :ref:`qc-section`.

Sensor pages — ``ctd.html``, ``oxygen.html``, ``optics.html``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One page per sensor present, with the same section order so the eye knows where to look:

- **Sensor** — make and model from the file's ``SENSOR_*`` variables.
- **Sections** — time–depth sections of the sensor's variables (``plots.plot_section``); on
  the CTD page temperature, salinity and conductivity, on the oxygen page dissolved oxygen, on
  the optics page chlorophyll and backscatter.
- **T–S** (CTD only) — ``plots.plot_ts``.
- **Drift** — temporal drift per variable (``plots.check_temporal_drift``); on the optics page
  also the deep-drift and negative-value assessment (``plots.process_optics_assess``).
- **Dive–climb bias** — hysteresis between dive and climb (``plots.plot_hysteresis``) and the
  up/down-cast bias by depth (``plots.plot_updown_bias``).
- **Day/night offset** — day versus night averages (``plots.plot_daynight_avg``); on the optics
  page the quenching assessment (``plots.plot_quench_assess``).
- **Mixed layer** (CTD only) — convective resistance for the deepest profile
  (``plots.plot_CR``).
- **QC checks** — the global-range check per variable (``plots.plot_global_range``).
- **Sample rate** — sampling period per variable (``plots.plot_sampling_period``).

A section with nothing to show for this file is omitted, and the in-page jump list only
ever lists sections that are there.

Flight page — ``flight.html``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Written when the file carries a flight-model vertical velocity (``GLIDER_VERT_VELO_MODEL``;
Seagliders do, most SeaExplorer files do not). **Vertical velocity** compares measured and
modelled vertical speeds with their histograms, for dives and climbs separately
(``plots.plot_vertical_speeds_with_histograms``) — the basis for the vertical water velocity
of Frajka-Williams et al. (2011).

Inventory page — ``inventory.html``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

About the file rather than the mission.

- **Global attributes** — every OG1 global attribute by category, with its value when
  present and its status otherwise: a missing *mandatory* attribute in amber, a missing
  *highly desirable* one as a dash, absent *suggested* ones omitted.
- **File contents** — the variables grouped by dimension signature with an expandable
  attribute list per variable, the scalar variables, the ``SENSOR_*`` catalogue, and how many
  data variables carry a ``_QC`` companion.

The inventory reports presence, not validity: it does not check attribute values against
the OG1 vocabulary. For that, use the IOOS compliance checker with the
`cc-plugin-og <https://github.com/OceanGlidersCommunity/cc-plugin-og>`_ plugin.

Fleet page — ``ROOT/index.html``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Built from the ``report.json`` of every mission under the root: one row per mission —
platform, dates, profiles, maximum depth, which sensor pages exist, OG1 conformance, worst
QC — with buttons to its pages, and a map of all tracks. Rebuilt whenever a mission is
reported into the root, or on demand with ``glidertest navigator ROOT`` after missions are
added, removed or renamed by hand. You want a fleet page for a season of deployments, a glider
pool, or a cruise with several gliders at once.

.. _qc-section:

The QC section
--------------

Two sections, deliberately kept apart.

**QC — as delivered** reads the ``*_QC`` flag variables the file already carries and shows,
per variable, the distribution of flag values as a bar (good / suspect / fail / not evaluated
/ missing), with the labels taken from the file's own ``flag_meanings``. This is what the
data provider says about the data. A file with no ``*_QC`` variables shows this section as
absent, not as clean — and a file whose flags are all "not evaluated" (the provider applied no
QC) reads the same way, not as a clean bill of health.

**QC — glidertest diagnostics** runs glidertest's own checks on the data — gross range, spike
and flat-line tests per variable, the dive–climb hysteresis verdict — and shows the result
with the thresholds that were applied printed next to it, so a reader can judge whether a
"suspect" means anything for their region. Today the thresholds are the package defaults;
a per-mission configuration file is planned.

The two are never merged: a value the provider flagged good and glidertest finds suspect is
shown as both.

Colour and status conventions
-----------------------------

- Masthead navigation buttons are coloured by the *kind* of page — the summary, the sensor
  reports, the derived flight page — and the current page's button is muted.
- **Amber** marks a missing mandatory OG1 attribute, on the inventory page and in the
  landing-page verdict. A dash marks a missing highly-desirable attribute.
- Under every figure, ``source: plot_…`` names the ``glidertest.plots`` function that drew
  it, so the figure can be reproduced in a notebook.
- A diagnostic that could not be computed for this file says so in place, in a visible
  block; it is never silently dropped.

What the report does not do
---------------------------

- It never modifies the input file and never writes a corrected variable.
- It never substitutes a value: a fact it cannot determine is shown as ``UNK`` or a dash.
- It is not an OG1 validator (see the inventory page note above).
- It does not process the data; see `GliderTools <https://glidertools.readthedocs.io/>`_ and
  the OceanGliders SOPs for that.

Many missions
-------------

Report into one root and the fleet page follows. From the command line a directory of files
is one command (``glidertest report data/ --report-dir reports/``); from Python, write each
mission with ``navigator=False`` and rebuild the fleet page once at the end (see
:doc:`quickstart`). ``--skip-existing`` makes a re-run over a growing data folder cheap.
