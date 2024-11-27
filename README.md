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

### Documentation

Documentation is available at [https://callumrollo.github.io/glidertest/](https://callumrollo.github.io/glidertest/)

Check out the demo notebook `notebooks/demo.ipynb` for example functionality.

As input, glidertest takes [OceanGliders format files](https://github.com/OceanGlidersCommunity/OG-format-user-manual)

### Contributing

All contributions are welcome! See [contributing](CONTRIBUTING.md) for more details

To install a local, development version of glidertest, clone the repo, open a terminal in the root directory (next to this readme file) and run these commands:

```sh
pip install -r requirements-dev.txt
pip install -e . 
```
This installs glidertest locally. -e ensures that any edits you make in the files will be picked up by scripts that import functions from glidertest.

All new functions should include tests, you can run the tests locally and generate a coverage report with:

```sh
pytest --cov=glidertest --cov-report term-missing  tests/
```

Try to ensure that all the lines of your contribution are covered in the tests.
