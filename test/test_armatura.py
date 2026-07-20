# -*- coding: utf-8 -*-
import numpy as np
from dense_armor import Armatura


def test_detects_spike_and_nan():
    serie = [1.2, 1.3, 9999.0, 1.25, float("nan"), 1.3]
    a = Armatura(livello_ia=1.0)
    pulito, K, anomalie = a.analizza(serie)
    assert anomalie == [2, 4]
    assert K.shape == (len(serie),)


def test_livello_zero_filtra_lo_spike():
    serie = [1.2, 1.3, 9999.0, 1.25, 1.3]
    a = Armatura(livello_ia=0.0)
    pulito, K, anomalie = a.analizza(serie)
    # a clip pieno lo spike deve essere sostanzialmente ridotto, non passante intatto
    assert abs(pulito[2]) < 9999.0 * 0.5


def test_referto_json_e_serializzabile():
    a = Armatura(livello_ia=1.0)
    referto = a.referto_json([1.0, 2.0, float("nan"), 3.0], nome="test")
    import json
    json.dumps(referto)  # non deve sollevare eccezioni
    assert referto["punti"] == 4


def test_gradino_sostenuto_non_viene_appiattito_ma_lo_spike_isolato_si():
    # Proprieta' centrale del motore hybrid (core/hybrid_engine.py) rispetto
    # al vecchio gate ABCollatz: un cambiamento VERO e sostenuto deve passare,
    # uno spike isolato no -- anche se entrambi, al primo istante, sembrano
    # "un salto grande" allo stesso modo. Il gate ABCollatz non garantiva
    # questa distinzione in modo dimostrabile (vedi CHANGELOG 1.0.10).
    rng = np.random.default_rng(7)
    plateau1 = 1.0 + rng.normal(0, 0.05, 15)
    spike = [50.0]
    transizione = 1.0 + rng.normal(0, 0.05, 3)
    plateau2 = 5.0 + rng.normal(0, 0.05, 15)
    serie = np.concatenate([plateau1, spike, transizione, plateau2])

    a = Armatura(livello_ia=0.0)  # clip pieno: filtra attivamente
    pulito, K, anomalie = a.analizza(serie)

    idx_spike = 15
    assert abs(pulito[idx_spike] - 1.0) < 1.0, "lo spike isolato deve essere ricondotto vicino alla baseline"

    # ultimi 5 punti del plateau2: il motore ha avuto tempo di riconoscere
    # il nuovo livello come genuino, deve lasciarlo passare quasi intatto
    coda_plateau2 = slice(len(serie) - 5, len(serie))
    assert np.allclose(pulito[coda_plateau2], serie[coda_plateau2], atol=0.5), \
        "un gradino vero e sostenuto non deve restare appiattito sulla vecchia baseline"
