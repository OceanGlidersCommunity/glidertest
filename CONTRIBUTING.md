# Contributing to glidertest

First off, thanks for taking the time to contribute!

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents) for different ways to help and details about how this project handles them.

<!-- omit in toc -->
## Table of Contents

- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Your First Code Contribution](#your-first-code-contribution)
- [Improving The Documentation](#improving-the-documentation)


## I Have a Question

> If you want to ask a question, we assume that you have read the available [Documentation](https://callumrollo.github.io/glidertest/).

Before you ask a question, it is best to search for existing [Issues](https://github.com/callumrollo/glidertest/issues) that might help you. If you don't find an existing Issue:

- Open an [Issue](https://github.com/callumrollo/glidertest/issues/new).
- Provide as much context as you can about what you're running into.
- If possible, try to provide a reproducible example, e.g. a jupyter notebook.

## I Want To Contribute

> ### Legal Notice <!-- omit in toc -->
> When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project licence.

### Reporting Bugs

<!-- omit in toc -->

A good bug report shouldn't leave others needing to chase you up for more information. Please complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Collect information about the bug:
  - Stack trace (Traceback) or screenshot error message
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Input dataset, can you recreate this bug with one of the example datasets provided by `glidertest`? If not, provide your dataset if practical
- Open an [Issue](https://github.com/callumrollo/glidertest/issues) describing the bug

<!-- omit in toc -->
### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for XYZ, **including completely new features and minor improvements to existing functionality**. 

<!-- omit in toc -->
#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation](https://callumrollo.github.io/glidertest) carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong case to convince the project's developers of the merits of this feature. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset.

<!-- omit in toc -->
#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://github.com/callumrollo/glidertest/issues).

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as many details as possible.
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why. At this point you can also tell which alternatives do not work for you.
- **Explain why this enhancement would be useful** to most XYZ users. 

### Your First Code Contribution

Getting started adding your own functionality.

#### glidertest organisation

Code is organised into files within `glidertest/*.py` and demonstrated in jupyter notebooks in `notebooks/*.ipynb`. The *.py* files include primarily functions (with their necessary packages imported) while the notebooks call these functions and display the plots generated. The *.py* files are separated into broad categories of plots, tools and utilities. If you'd like to add a function to calculate some thing WIDGET and then to plot the result of the calculation, you will want to write a function in tools.py and then a plotting function in plots.py. There are a couple exceptions: If it's a very simple calculation (mean, median, difference between two quantities), you might put the entire calculation within the plotting function. See for example `plots.process_optics_assess()`. Or, if the calculation is more complicated, but is easily displayed by an existing function, then you might have a calculation function `tools.calc_foo_bar()` and then use an existing `plots.plot_histogram()` to display it.

#### Best practices for new functions

- Once you've added a function, you can test it against one or two of the sample datasets in `notebooks/demo.ipynb`. Does it have the same behaviour on those sample datasets as you expect?
- Have you followed the conventions for naming your function? Generally, function names should be short, agnostic about the vehicle used, and understandable to Person X. We also loosely follow naming conventions to help the new user understand what a function might do (e.g., plotting functions in `plots.py` typically start with the name `plot_blahblah()` while calculations are` calc_blahblah()` and calculations with special outputs  are `compute_blahblah()`. Functions not inteded for use by the end user (e.g. sub-calculations) should be added to `utilities.py`
- Unless otherwise required, we suggest to pass an xarray dataset (as you get from loading an OG1 dataset) as the input. There are some parameters that can be additionally passed to carry out subsets on the data or select the variable of interest.
- Did you write a docstring? We use the [numpy standard for docstings](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard). We also suggest including your name or GitHub handle under the original author heading. Look at some existing docstrings in `glidertest` if you are unsure of the format.
- There are also some basic error checking behaviours you should consider including. If you're plotting a particular variable, use the `glidertest.utilities._check_necessary_variables()` function to check whether or not the required variables are within the dataset passed to the function.
- For plotting functions on a single axis, you should include as optional inputs the `fig` and `ax`, and return the same, to enable their usage within multi-axes plots. For plotting functions with multiple or interrelated axes, perhaps fig and ax shouldn't be included as inputs, but can be provided as outputs for the user to make onward edits.
- For plotting, see the guidance on uniformity (using standard lineswidths, figure sizes and font sizes etc.). These are all described in `glidertest/glidertest.mplstyle`, in case an individual user wants to change these to their preferences.
- Each new function should have a corresponding test, feel free to ask if you're not sure how to write a test!

### Improving The Documentation

Our [documentation](https://callumrollo.github.io/glidertest/) is built from the function docstrings and the [example notebook](https://callumrollo.github.io/glidertest/demo-output.html). If you think the documentation could be better, do not hesitate to suggest an improvement! Either in an Issue or a PR.

## Attribution
This guide is based on the **contributing-gen**. [Make your own](https://github.com/bttger/contributing-gen)!
