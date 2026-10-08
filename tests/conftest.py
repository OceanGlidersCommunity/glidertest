"""Shared pytest setup.

pytest imports this before any test module, so setting the non-interactive Matplotlib backend here
means test files need no ``matplotlib.use("agg")`` statement above their imports (and therefore no
E402 suppression on those imports). ``reports.report`` and the CLI also force Agg at runtime; this
covers tests that touch a plotter directly.
"""

import matplotlib

matplotlib.use("agg")
