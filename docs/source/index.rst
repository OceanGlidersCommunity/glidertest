.. glidertest documentation master file, created by
   sphinx-quickstart on Tue Oct 29 11:30:19 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to glidertest's documentation!
======================================

Glidertest is a Python package aiming to diagnose possible issues with your glider dataset. The package takes as input data from any glider in `OG1 format <https://github.com/OceanGlidersCommunity/OG-format-user-manual>`_.
At the moment, we have implemented functions ranging from basic visualization of your glider data (location, gridding in the dataset, water mass properties, water column structure, etc.) to more complex analysis such as flight model performance when you work with vertical velocities or non-photochemical quenching when you work with optical data. 
This package serves solely as a diagnostic tool. We occasionally have suggestions for packages or tools that aim to address and solve a particular issue or link to SOP where more processing info can be found.

We recommend consulting best practice guides like the `Oxygen SOP <https://oceangliderscommunity.github.io/Oxygen_SOP/README.html>`_ and the other OceanGliders SOPs. We also recommend the `GliderTools <https://glidertools.readthedocs.io/en/latest/>`_ Python package for processing and to possibly address some common issues with glider data.

We use work from the following papers:
    * Bennett, J., Stahr, F., and Eriksen, C. (2019). Determining Seaglider Velocities Automatically. http://hdl.handle.net/1773/44948
    * Frajka-Williams, E., Eriksen, C. C., Rhines, P. B., and Harcourt, R. R. (2011). Determining vertical water velocities from Seaglider. Journal of Atmospheric and Oceanic Technology, 28(12), 1641-1656. https://doi.org/10.1175/2011JTECHO830.1
    * Thomalla, S.J., Moutier, W., Ryan-Keogh, T.J., Gregor, L. and Schütt, J. (2018), An optimized method for correcting fluorescence quenching using optical backscattering on autonomous platforms. Limnol. Oceanogr. Methods, 16: 132-144. https://doi.org/10.1002/lom3.10234
 
We provide an example notebook to demonstrate the purpose of the various function and test datasets from SeaExplorer data in the Baltic and Seaglider data in the Labrador Sea.

For recommendations or bug reports, please visit https://github.com/callumrollo/glidertest/issues/new

======================================

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   demo-output.ipynb
   glidertest


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
