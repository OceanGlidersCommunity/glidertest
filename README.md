# glidertest

This is a repo to diagnose issues in glider data such as CTD thermal lag.

This is a work in progress, all contributions welcome!

### Instal

Install from PyPI with

```sh
python -m pip install glidertest
```

Install a local, development version of this by cloning the repo, opening a terminal in the home directory (next to this readme file) and running these commands:

```sh
pip install -r requirements.txt
pip install -e . 
```
This installs glidertest locally. `-e` ensures that any edits you make in the files will be picked up by scripts that import functions from glidertest.

### Documentation

Documentation website at [https://callumrollo.github.io/glidertest/](https://callumrollo.github.io/glidertest/)

Check out the example notebook `notebooks/demo.ipynb` for example functionality

As input, glidertest takes [OceanGliders format files](https://github.com/OceanGlidersCommunity/OG-format-user-manual)

