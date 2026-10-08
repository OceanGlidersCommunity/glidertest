# glidertest

glidertest reads an OceanGliders (OG1) mission file, runs a set of diagnostics on it, and shows you what is in the file. It never modifies, fixes or grids the data: the output is figures and an HTML report, the input file is left untouched. It works one mission at a time; a fleet page links the reports of many missions on one map.

There are two ways to use it: call the diagnostic functions from a notebook (`glidertest.plots`, `glidertest.tools`), or run `glidertest report` on a file and read the HTML.

Diagnostics include:

- Time and depth spacing
- Histograms and T–S diagrams
- Suspect profile durations
- Dive–climb bias (the difference between down and up profiles)
- Sensor drift
- Non-photochemical quenching in chlorophyll
- Vertical velocities from the flight model
- Inventory of the file: variables, sensors, attributes, and QC flags as delivered
- Fleet page: one map and table linking many mission reports

## Install

Install from conda with
```sh
conda install --channel conda-forge glidertest
```

Install from PyPI with

```sh
python -m pip install glidertest
```

## HTML report

glidertest turns an OG1 mission file into a self-contained HTML report — a landing page, one page per sensor, a flight page for gliders that report one, and an inventory of the file. From Python, with a dataset open:

```python
from glidertest import reports
reports.report(ds, "reports/")
```

or from the command line:

```sh
glidertest report mission.nc --report-dir reports/
glidertest navigator reports/      # rebuilds the fleet page from the reports already in reports/
```

See the [report guide](https://oceangliderscommunity.github.io/glidertest/reports.html) and the [live demo](https://oceangliderscommunity.github.io/glidertest/_static/demo/index.html).

## Documentation

Documentation is available at [https://oceangliderscommunity.github.io/glidertest/](https://oceangliderscommunity.github.io/glidertest/)

Check out the demo notebook `notebooks/demo.ipynb` for example functionality.

The demo notebook `notebooks/demo_data_issues.ipynb` uses example datasets to check how glidertest can help identify and visualize problems with data.

glidertest reads [OceanGliders (OG1) format files](https://github.com/OceanGlidersCommunity/OG-format-user-manual), one file per mission. Seaglider basestation files can be converted to OG1 with the `seagliderOG1` package; the sample data (`fetchers.load_sample_dataset()`) is already OG1.

## Contributing

All contributions are welcome! See [contributing](CONTRIBUTING.md) for more details

To install a local, development version of glidertest, clone the repo, open a terminal in the root directory (next to this readme file) and run these commands:

```sh
git clone https://github.com/OceanGlidersCommunity/glidertest.git
cd glidertest
pip install -e ".[dev]"
```
This installs glidertest locally. -e ensures that any edits you make in the files will be picked up by scripts that import functions from glidertest.

You can run the example jupyter notebook by launching jupyterlab with `jupyter-lab` and navigating to the `notebooks` directory.

All new functions should include tests, you can run the tests locally and generate a coverage report with:

```sh
pytest --cov=glidertest --cov-report term-missing  tests/
```

Try to ensure that all the lines of your contribution are covered in the tests.

## Acknowledgements

glidertest is developed under the SEA-CODE project (SeaExplorer–Seaglider Cross-platform Open Diagnostics & Evaluation), funded by Voice of the Ocean (VOTO). SEA-CODE builds platform-independent, open-source diagnostics for glider data, with glidertest and seagliderOG1 as its core packages, and supports exchanges between the University of Hamburg and VOTO. VOTO also provides the sample data.

Till Moritz's contributions were funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) through the PycnMix project (Projektnummer 558671572). The HTML report and command-line interface were first developed in preparation for the DFG research infrastructure Swarm of Ocean Gliders (Projektnummer 544335393).

glidertest is an OceanGliders community package and welcomes contributions from the community. Development was assisted by Claude Code (Anthropic) and GitHub Copilot code review.
