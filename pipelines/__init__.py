"""Ready-to-use pipelines. THE PRODUCT.

A pipeline chains the primitive tools from ``dense_armor.utility`` in a
documented order to solve one standard client problem. Pipelines add no
algorithms of their own.

    finance/
        sorveglianza.py    watch a market index, flag anomalous days
        rischio.py         daily VaR and volatility alarm
        esecuzione.py      decide whether an order leaves, and how big
        anomalia_titolo.py single-ticker strangeness

Future domains (robotics, chemistry, physics) get their own subfolder
here. Users import from ``pipelines.<domain>``.
"""

__all__: list[str] = []
