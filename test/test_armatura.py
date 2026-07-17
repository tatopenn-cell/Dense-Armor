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
