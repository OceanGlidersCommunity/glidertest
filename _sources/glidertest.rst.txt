=============
API reference
=============

Entry points
------------

The two functions most users need. Everything else on this page is the toolbox they are
built from.

.. autofunction:: glidertest.reports.report

.. autofunction:: glidertest.reports.navigator

Modules
-------

.. automodule:: glidertest.tools
   :members:
   :undoc-members:

.. automodule:: glidertest.plots
   :members:
   :undoc-members:

.. automodule:: glidertest.qc
   :members:
   :undoc-members:

.. automodule:: glidertest.og1_attrs
   :members:
   :undoc-members:

.. automodule:: glidertest.utilities
   :members:
   :undoc-members:

.. automodule:: glidertest.fetchers
   :members:
   :undoc-members:

.. automodule:: glidertest.interactive
   :members:
   :undoc-members:

.. automodule:: glidertest.cli
   :members: main, build_parser

Legacy
------

``glidertest.summary_sheet`` builds the older RST/PDF "mission summary" (it is what pulls in
the ``pandoc`` and ``rstcloth`` dependencies). The HTML report —
:func:`glidertest.reports.report` — supersedes it; it is kept for existing users for now, and
its removal is planned.

.. automodule:: glidertest.summary_sheet
   :members:
   :undoc-members:
