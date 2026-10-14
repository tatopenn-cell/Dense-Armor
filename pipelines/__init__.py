"""Ready-to-use pipelines. THE PRODUCT.

A pipeline chains the primitive tools from :mod:`dense_armor.utility`
in a documented order to solve one standard client problem. Pipelines
do not implement new algorithms; they only orchestrate existing ones
and expose them under a single class with a clear input / output.

The user-facing entry point is always::

    from pipelines.<domain> import <PipelineName>

Currently available domains:

    finance/
        sorveglianza.py    watch a market index, flag anomalous days
        rischio.py         daily VaR and volatility alarm
        esecuzione.py      decide whether an order leaves, and how big
        anomalia_titolo.py single-ticker strangeness

Each domain gets its own subfolder. Adding a new domain means creating
a new subfolder here, with the same structure: one file per pipeline,
one ``__init__.py`` re-exporting the classes.

Note:
    Pipelines are expected to be stable. A breaking change in a public
    pipeline is a breaking change for the user, so the classes here
    should be considered frozen once released.
"""

__all__: list[str] = []
