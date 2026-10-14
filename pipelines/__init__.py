"""Ready-to-use pipelines. THE PRODUCT.

A pipeline chains the primitive tools from :mod:`dense_armor.utility`
in a documented order to solve one standard client problem. Pipelines
do not implement new algorithms; they only orchestrate existing ones
and expose them under a single class with a clear input / output.

The user-facing entry point is always::

    from pipelines.<domain> import <PipelineName>

Currently available domains:

    finance/
        surveillance.py        watch a market index, flag anomalous days
        risk.py                daily VaR and volatility alarm
        execution.py           decide whether an order leaves, and how big
        single_ticker.py       single-ticker strangeness
        static_analysis.py     fixed-cutoff baseline
        online_analysis.py     adaptive detectors on a return series
        fraud.py               supervised card-fraud detection
        unsupervised_fraud.py  card-fraud detection without labels
        intrusion.py           network intrusion detection
        credit_scoring.py      score customers for a telemarketing campaign

Each domain gets its own subfolder. Adding a new domain means creating
a new subfolder here, with the same structure: one file per pipeline,
one ``__init__.py`` re-exporting the classes.

Note:
    Pipelines are expected to be stable. A breaking change in a public
    pipeline is a breaking change for the user, so the classes here
    should be considered frozen once released.
"""

__all__: list[str] = []
