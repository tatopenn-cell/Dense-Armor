"""The optimisers have no batch form: they are declared and nothing else."""

import dense_armor.ml.optim as O


def test_every_optimiser_is_declared_stopped():
    assert set(O.STOPPED) == {"SGD", "AdaGrad", "Adam", "FTRLProximal", "OnlineNewtonStep"}
    assert O.__all__ == []
