"""The web application of Dense-Armor: the dashboard of the finance system.

It lives in ``tools/``, as the dashboard of Dense-Evolution does, and not in the ``dense_armor`` package
that goes to PyPI: ``pyproject.toml`` lists only ``dense_armor*`` and ``pipelines*`` as packages, so its
templates and static files are not shipped. It is imported as ``dashboard`` with ``tools`` on the path
(``pythonpath`` of ``pytest.ini``); to run it, from the root of the repository:
``$env:PYTHONPATH = "tools"; python -m dashboard.app --config <file.toml>``.

``app`` is the Flask application and its command line, ``chain`` composes the six slots of the finance
chain, ``sections`` builds the pages of the pieces, and ``templates/`` and ``static/`` are its pages, style
and script. The finance primitives that it shows stay in ``dense_armor.core.finance``.
"""
