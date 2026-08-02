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


def test_nan_vicino_a_spike_non_viene_sanato_con_la_media_globale():
    # Bug trovato dopo la pubblicazione di 1.1.0 testando l'esempio esatto
    # del README con un utente vero: un NaN a 2 passi da uno spike enorme
    # veniva sostituito con la media dell'INTERA serie (contaminata dallo
    # spike ovunque fosse), non con una stima locale -- su serie brevi il
    # sostituto risultava assurdo (es. 2000.81 invece di ~1.28). Riproduce
    # esattamente l'esempio del README.
    serie = [1.2, 1.3, 9999.0, 1.25, float("nan"), 1.3]
    for livello in (0.0, 1.0):
        a = Armatura(livello_ia=livello)
        pulito, K, anomalie = a.analizza(serie)
        assert 4 in anomalie, f"il NaN deve restare marcato anomalo (livello={livello})"
        assert abs(pulito[4] - 1.28) < 1.0, \
            f"pulito[4] deve essere vicino al vicinato reale (~1.28), non alla media globale (livello={livello}, pulito[4]={pulito[4]})"


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


def test_gradino_genuino_non_si_blocca_su_un_equilibrio_intermedio_sbagliato():
    # Bug reale trovato testando dal vivo: la finestra di baseline presa da
    # `out` (gia' guarito) invece che da `processed` (grezzo) crea un loop di
    # autocontaminazione. I primi 1-2 punti dopo un gradino vengono respinti
    # (inevitabile per un rilevatore causale: al primissimo campione un
    # gradino vero e' indistinguibile da uno spike), ma quei punti respinti
    # finiscono in `out` e la finestra dei punti successivi li rimedia dentro
    # il proprio calcolo -- producendo una baseline "a meta' strada" che
    # respinge ANCHE i punti successivi genuini, che rientrano in `out`
    # alimentando lo stesso problema per la finestra dopo. Risultato: un
    # equilibrio stabile ma SBAGLIATO che non si risolve mai da solo (a
    # differenza di uno spike isolato, che esce dalla finestra dopo `radius`
    # passi). Il test precedente (`test_gradino_sostenuto_...`) non lo
    # intercettava: controllava solo gli ultimi 5 punti con atol=0.5, una
    # tolleranza/finestra troppo larghe per questo specifico fallimento.
    # Qui si controlla OGNI punto di una coda lunga (oltre il raggio adattivo)
    # con una tolleranza stretta.
    rng = np.random.default_rng(0)
    serie = np.concatenate([
        1.0 + rng.normal(0, 0.03, 4),
        5.0 + rng.normal(0, 0.03, 30),
    ])

    a = Armatura(livello_ia=0.0)
    pulito, K, anomalie = a.analizza(serie)

    coda = pulito[-15:]  # ben oltre il raggio adattivo (radius = min(20, max(3, 34//3)) = 11)
    assert np.allclose(coda, 5.0, atol=0.3), (
        f"la coda del plateau genuino deve stabilizzarsi vicino al vero livello (5.0), "
        f"non restare bloccata su un equilibrio intermedio sbagliato: {coda}"
    )
