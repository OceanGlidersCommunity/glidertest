"""Shared units data — stored spellings, their accepted aliases, and plot labels.

Package-neutral and vendored byte-identical across the packages that share the report
design layer (see ``MANIFEST.txt``); the file a maintainer opens to add a spelling holds
nothing but tables, so that change touches no code.  The machinery that reads these tables
(:class:`~ctdcast.reports._vendored.units.Tables`, ``merge_tables``, ``resolve_units``,
``format_units``) lives beside them in :mod:`ctdcast.reports._vendored.units`.

The data are Python dicts so the module imports with no third-party dependency (the same
choice as :mod:`design_tokens`); every alias entry names the file or package that evidenced
the spelling.  A package layers its own choices over these through ``merge_tables`` and never
edits this file.

Three keys organise the storage side:

``CANONICAL``
    CF ``standard_name`` → the one stored spelling for that quantity (CF/UDUNITS exponent
    form).  A quantity with no entry is decided by the package per variable.
``ALIASES``
    canonical spelling → ``{reader spelling: factor}``; the canonical is an implicit ``×1``
    member, so only differing spellings are listed.  A factor other than ``1`` is a physical
    conversion and must be verifiable (the only one the family has is ``S/m`` → ``mS cm-1``,
    ``×10``).
``SCOPED_ALIASES``
    CF ``standard_name`` → ``{spelling: factor}`` for a spelling that means one quantity only
    (``PSU`` is salinity, ``NTU`` is turbidity, ``deg`` is a latitude or a longitude), so a
    file using it on the wrong quantity is refused.

and three the display side:

``LABELS``
    whole units string → fixed display symbol (``degree_Celsius`` → ``°C``).
``LABELS_BY_STANDARD_NAME``
    CF ``standard_name`` → fixed scale label, consulted before the units string (turbidity
    ``NTU``; dimensionless practical salinity ``PSU``).
``TOKENS``
    substring substitutions applied before the exponent rule (``umol`` → ``µmol``).
"""

from __future__ import annotations

#: CF ``standard_name`` → the stored spelling for that quantity.  Seeded from the ctdcast
#: registry entries that carry a ``standard_name``; a sibling adds its own quantities (depth,
#: etc.) when it re-vendors, evidenced from its own registry.  The spelling is fixed by the
#: convention named in the comment (CF/UDUNITS; CCHDO and OceanSITES agree on these).
CANONICAL: dict[str, str] = {
    "sea_water_pressure": "dbar",  # UDUNITS; CCHDO/OceanSITES
    "latitude": "degrees_north",  # CF §4.1
    "longitude": "degrees_east",  # CF §4.1
    "sea_water_temperature": "degree_Celsius",  # UDUNITS (not degC); ITS-90 scale
    "sea_water_conservative_temperature": "degree_Celsius",  # UDUNITS; TEOS-10
    "sea_water_electrical_conductivity": "mS cm-1",  # CCHDO; UDUNITS exponent form
    "sea_water_practical_salinity": "1",  # CF: PSS-78 is dimensionless
    "sea_water_absolute_salinity": "g kg-1",  # TEOS-10
    "moles_of_oxygen_per_unit_mass_in_sea_water": "umol kg-1",  # CCHDO
    "mass_concentration_of_chlorophyll_in_sea_water": "mg m-3",  # UDUNITS exponent form
    "sea_water_turbidity": "1",  # CF: NTU is dimensionless
    "altitude_of_sea_floor": "m",  # UDUNITS
    "sea_water_sigma_theta": "kg m-3",  # UDUNITS exponent form
    "eastward_sea_water_velocity": "m s-1",  # UDUNITS exponent form
    "northward_sea_water_velocity": "m s-1",  # UDUNITS exponent form
    "square_of_brunt_vaisala_frequency_in_sea_water": "rad2 s-2",  # UDUNITS exponent form
}

#: Non-canonical spellings a reader may emit, ``{canonical: {spelling: factor}}``.  The
#: canonical is an implicit ``×1`` member; only differing spellings are listed.  The baseline
#: is oceanarray #102 (one shared table across the packages); entries are marked
#: ``oceanarray #102`` with a ctdcast raw-CNV-header citation added where the spelling also
#: appears in a ctdcast fixture.  A package's own engineering-channel spellings (ctdcast's
#: ``volts``/``seconds``) stay in that package's ``UNIT_ALIAS_EXTRAS``, not here.
ALIASES: dict[str, dict[str, float]] = {
    "degree_Celsius": {
        "degC": 1.0,  # oceanarray #102
        "degrees_C": 1.0,  # oceanarray #102
        "Degrees C": 1.0,  # oceanarray #102
        "°C": 1.0,  # oceanarray #102 (display form, recovered if it leaks into units)
        # oceanarray #102; ctdcast cnv/mixsed2_004.cnv:40 t090C: Temperature [ITS-90, deg C]
        "ITS-90, deg C": 1.0,
    },
    "mS cm-1": {
        # oceanarray #102; ctdcast cnv/msm_142_1_056_1sec.cnv c0mS/cm: Conductivity [mS/cm]
        "mS/cm": 1.0,
        # oceanarray #102; ctdcast cnv/mixsed2_004.cnv:33 c0S/m: Conductivity [S/m]
        "S/m": 10.0,  # 1 S/m = 10 mS/cm
    },
    "m": {"meters": 1.0},  # oceanarray #102
    # oceanarray #102; ctdcast cnv/msm_142_1_056_1sec.cnv prDM [db]
    "dbar": {"db": 1.0},
    "m s-1": {"m/s": 1.0},  # oceanarray #102
    "umol kg-1": {
        # oceanarray #102; ctdcast cnv/mixsed2_004.cnv:43 sbox0Mm/Kg: Oxygen, SBE 43 [umol/kg]
        "umol/kg": 1.0,
        "µmol/kg": 1.0,  # oceanarray #102
    },
    "kg m-3": {"kg/m³": 1.0},  # oceanarray #102 (display form)
    "g kg-1": {"g/kg": 1.0},  # oceanarray #102 (display form)
    "mg m-3": {"mg/m^3": 1.0},  # ctdcast cnv/msm_142_1_017_1sec.cnv flECO-AFL [mg/m^3]
    # ctdcast cnv/mixsed2_004.cnv:42 sbeox0PS [% saturation]
    "percent": {"% saturation": 1.0},
    # CF §4.1 latitude/longitude spellings (recommended first, rest acceptable); unambiguous
    # (N vs E in the spelling), so global.  The bare `deg` is scoped below.
    "degrees_north": {
        "degree_north": 1.0,
        "degree_N": 1.0,
        "degrees_N": 1.0,
        "degreeN": 1.0,
        "degreesN": 1.0,
    },
    "degrees_east": {
        "degree_east": 1.0,
        "degree_E": 1.0,
        "degrees_E": 1.0,
        "degreeE": 1.0,
        "degreesE": 1.0,
    },
    # A plane angle (glider heading / pitch / glide-angle): UDUNITS `degree` = pi/180 rad,
    # unambiguous, so global.  `deg` is NOT a member here — it is scoped to latitude/longitude
    # below, where the axis disambiguates it; a plane-angle channel using `deg` is refused.
    "degree": {
        "degrees": 1.0,  # oceanarray (glider attitude channels)
        "°": 1.0,  # display form, recovered if it leaks into units
    },
}

#: Spellings accepted only for one quantity, keyed by CF ``standard_name`` so the table is
#: package-neutral (one entry, not one per variable name).  ``PSU`` / ``NTU`` are oceanarray
#: #102's scoped pair (both canonical ``1``, so a salinity file saying ``NTU`` must raise);
#: ``deg`` is latitude or longitude (same spelling, different canonical per axis, so it cannot
#: be a global alias).
SCOPED_ALIASES: dict[str, dict[str, float]] = {
    # oceanarray #102 (salinity); ctdcast cnv/msm_142_1_017_1sec.cnv sal00: Salinity [PSU]
    "sea_water_practical_salinity": {"PSU": 1.0},
    # oceanarray #102 (turbidity); ctdcast cnv/msm_142_1_056_1sec.cnv turbWETntu0 [NTU]
    "sea_water_turbidity": {"NTU": 1.0},
    # ctdcast cnv/msm_142_1_017_1sec.cnv latitude/longitude [deg]
    "latitude": {"deg": 1.0},
    "longitude": {"deg": 1.0},
}

#: Whole-string unit spellings that map to a fixed display symbol.  Matched exactly (not as a
#: substring) so ``degree`` cannot corrupt ``degrees_north`` / ``degree_Celsius`` — those have
#: their own entries and are matched first.
LABELS: dict[str, str] = {
    "degree_Celsius": "°C",  # origin: oceanarray format_units _DISPLAY_SPECIALS
    "degrees_north": "°N",  # origin: oceanarray format_units _DISPLAY_SPECIALS
    "degrees_east": "°E",  # origin: oceanarray format_units _DISPLAY_SPECIALS
    "percent": "%",  # origin: ctdcast registry (oxygen_saturation)
    "degree": "°",  # origin: ctdcast registry (Turner)
}

#: CF ``standard_name`` → fixed scale label, consulted before the units string.  Turbidity is
#: a scale label on any units; practical salinity labels a dimensionless (``"1"``) quantity.
LABELS_BY_STANDARD_NAME: dict[str, str] = {
    "sea_water_turbidity": "NTU",  # origin: oceanarray format_units
    "sea_water_practical_salinity": "PSU",  # origin: oceanarray format_units (units == "1")
}

#: Substring token substitutions applied before the exponent rule.
TOKENS: dict[str, str] = {
    "umol": "µmol",  # origin: oceanarray format_units
}
