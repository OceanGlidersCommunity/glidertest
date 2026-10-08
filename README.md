# glidertest

Glidertest is a python package for diagnosing potential issues in Ocean Gliders format glider data. Glidertest does not modify, fix or grid glider data. Functionality currently includes:

- Checking time and depth spacing
- Making histograms and TS diagrams
- Checking for suspect time duration of profiles
- Quantifying the bias between dives and climbs (profiles when the glider is going down/up)
- Checking for sensor drift
- Detecting quenching in chlorophyll data
- Plotting vertical velocities

This is a work in progress, all contributions welcome!


### Install

Install from conda with
```sh
conda install --channel conda-forge glidertest
```

Install from PyPI with

```sh
python -m pip install glidertest
```

### HTML report

One command turns an OG1 mission file into a self-contained HTML report — a landing page, one page per sensor, a flight page for gliders that report one, and an inventory of the file — and many missions into a fleet page with a map:

```sh
glidertest report mission.nc --report-dir reports/
```

or from Python:

```python
from glidertest import reports
reports.report(ds, "reports/")
```

See the [report guide](https://oceangliderscommunity.github.io/glidertest/reports.html) and the [live demo](https://oceangliderscommunity.github.io/glidertest/_static/demo/index.html).

### Documentation

Documentation is available at [https://oceangliderscommunity.github.io/glidertest/](https://oceangliderscommunity.github.io/glidertest/)

Check out the demo notebook `notebooks/demo.ipynb` for example functionality. 

The demo notebook `notebooks/demo_data_issues.ipynb` uses example datasets to check how glidertest can help identify and visualize problems with data.

As input, glidertest takes [OceanGliders format files](https://github.com/OceanGlidersCommunity/OG-format-user-manual)

### Contributing

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

### Acknowledgements

Initial development of glidertest was supported by the SEACODE project, funded by Voice of the Ocean (VOTO), and by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) through the PycnMix project (Projektnummer 558671572), which funded Till Moritz's contributions. The HTML report and command-line interface were first developed in preparation for the DFG research infrastructure Swarm of Gliders (Projektnummer 544335393). Sample data are provided by VOTO. glidertest is an OceanGliders community package and welcomes contributions from the community.

Development was assisted by Claude Code (Anthropic) and GitHub Copilot code review.
