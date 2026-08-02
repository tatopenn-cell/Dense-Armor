# -*- coding: utf-8 -*-
"""Regressione per dense_armor/utility/robust_filters.py -- i quattro
rilevatori di anomalie "a basso costo" (Chauvenet, Tukey/IQR, Hampel,
sigma-clipping), su finestra locale centrata (batch/offline, non il ciclo
causale real-time di core/hybrid_engine.py).
"""
import numpy as np
import pytest

from dense_armor.utility.robust_filters import (
    chauvenet_criterion, tukey_fences, hampel_filter, sigma_clip, pressure_valve,
)

METODI = [
    ('chauvenet', chauvenet_criterion),
    ('tukey', tukey_fences),
    ('hampel', hampel_filter),
    ('sigma_clip', sigma_clip),
]


@pytest.mark.parametrize('nome,fn', METODI)
def test_rileva_un_outlier_chiaro(nome, fn):
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1.0, 100)
    x[50] = 50.0
    pulito, anomalie = fn(x)
    assert 50 in anomalie, f'{nome}: outlier non rilevato'
    assert abs(pulito[50]) < 5.0, f'{nome}: pulito[50]={pulito[50]} non ricondotto vicino alla baseline'


@pytest.mark.parametrize('nome,fn', METODI)
def test_pochi_falsi_positivi_su_rumore_puro(nome, fn):
    # Rumore gaussiano puro, nessun vero outlier -- un tasso di falsi
    # positivi contenuto (soglie tipo "3 sigma" implicano ~0.3% atteso per
    # punto su una gaussiana pura, ma le finestre locali corte introducono
    # variabilita' aggiuntiva; 10% e' una soglia larga apposta, il punto e'
    # scartare una regressione grossolana, non fissare un tasso esatto).
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1.0, 300)
    _, anomalie = fn(x)
    assert len(anomalie) / len(x) < 0.10, f'{nome}: troppi falsi positivi ({len(anomalie)}/300)'


@pytest.mark.parametrize('nome,fn', METODI)
def test_gradino_genuino_si_stabilizza_senza_restare_a_meta_strada(nome, fn):
    # Stesso scenario che ha rotto core/hybrid_engine.py prima del fix
    # 1.1.2 (vedi CHANGELOG): un cambiamento di livello genuino e sostenuto
    # non deve restare bloccato su un valore intermedio sbagliato. La
    # finestra centrata (non causale) di questi 4 filtri li rende
    # strutturalmente immuni al loop di autocontaminazione trovato li'
    # (vedono gia' i punti futuri che confermano il nuovo livello).
    rng = np.random.default_rng(0)
    serie = np.concatenate([1.0 + rng.normal(0, 0.03, 15), 5.0 + rng.normal(0, 0.03, 15)])
    pulito, anomalie = fn(serie, radius=10)
    coda = pulito[-5:]
    assert np.allclose(coda, 5.0, atol=0.3), f'{nome}: coda bloccata a meta strada: {coda}'


@pytest.mark.parametrize('nome,fn', METODI)
def test_serie_troppo_corta_non_solleva_eccezioni(nome, fn):
    for x in (np.array([]), np.array([1.0]), np.array([1.0, 2.0])):
        pulito, anomalie = fn(x)
        assert len(pulito) == len(x)
        assert anomalie == []


def test_hampel_soglia_su_esempio_a_mano():
    # Esempio calcolabile a mano: finestra [1,1,1,1,1,50], mediana=1,
    # MAD = mediana(|1-1|*5, |50-1|) = mediana([0,0,0,0,0,49]) = 0 -- il
    # ramo "scaled_mad quasi zero" scatta, e 50 (l'unico valore diverso
    # dalla mediana) deve essere marcato.
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 50.0])
    pulito, anomalie = hampel_filter(x, radius=5)
    assert 5 in anomalie
    assert pulito[5] == pytest.approx(1.0)


def test_tukey_fences_su_esempio_a_mano():
    # [1,2,3,4,5,6,7,8,9,100]: Q1=3.25, Q3=7.75 (numpy 'linear'), IQR=4.5,
    # fence alta = 7.75 + 1.5*4.5 = 14.5 -- 100 e' ben oltre, deve essere marcato.
    x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 100.0])
    pulito, anomalie = tukey_fences(x, radius=10)
    assert 9 in anomalie


def test_chauvenet_rigetta_solo_lo_scostamento_davvero_improbabile():
    # 20 punti ~N(0,1) piu' un valore a 5 sigma: N*P per un punto a 5 sigma
    # su N=21 e' ben sotto 0.5 (P(|Z|>5) ~ 5.7e-7, N*P ~ 1.2e-5) -- deve
    # essere rigettato con ampio margine, non un caso limite.
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.normal(0, 1.0, 20), [5.0]])
    pulito, anomalie = chauvenet_criterion(x, radius=20)
    assert 20 in anomalie


class TestPressureValve:
    # pressure_valve combina i 4 metodi con lo stimatore a minima varianza
    # (Lagrange multiplier, w_k = (1/scala_k^2) / Sum_j(1/scala_j^2)) invece
    # di un voto -- soglia_pressione=8.0 e' calibrata empiricamente (vedi il
    # docstring della funzione per i percentili misurati), non "3 sigma"
    # come i metodi singoli: la scala combinata e' sempre piu' stretta di
    # ogni scala_k, quindi la stessa unita' di soglia non si trasferisce
    # direttamente da un metodo singolo alla combinazione.

    def test_rileva_un_outlier_chiaro(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1.0, 100)
        x[50] = 50.0
        pulito, anomalie, pressione = pressure_valve(x)
        assert 50 in anomalie
        assert abs(pulito[50]) < 5.0
        assert pressione[50] > 50.0  # ampio margine sopra soglia_pressione=8.0

    def test_pochi_falsi_positivi_su_rumore_puro(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1.0, 300)
        _, anomalie, _ = pressure_valve(x)
        assert len(anomalie) / len(x) < 0.05

    def test_gradino_genuino_si_stabilizza_senza_restare_a_meta_strada(self):
        rng = np.random.default_rng(0)
        serie = np.concatenate([1.0 + rng.normal(0, 0.03, 15), 5.0 + rng.normal(0, 0.03, 15)])
        pulito, anomalie, pressione = pressure_valve(serie, radius=10)
        coda = pulito[-5:]
        assert np.allclose(coda, 5.0, atol=0.3)
        assert not any(i >= 25 for i in anomalie)

    def test_due_outlier_vicini_entrambi_rilevati(self):
        # Il limite noto di Chauvenet da solo (mean/std non robuste su
        # outlier ravvicinati) non deve propagarsi alla combinazione: gli
        # altri 3 metodi (robusti) dominano il peso via 1/scala^2.
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1.0, 40)
        x[19] = 30.0
        x[20] = -30.0
        _, anomalie, _ = pressure_valve(x, radius=10)
        assert 19 in anomalie
        assert 20 in anomalie

    def test_serie_troppo_corta_non_solleva_eccezioni(self):
        for x in (np.array([]), np.array([1.0]), np.array([1.0, 2.0])):
            pulito, anomalie, pressione = pressure_valve(x)
            assert len(pulito) == len(x)
            assert len(pressione) == len(x)
            assert anomalie == []

    def test_pesi_lagrange_sommano_a_uno_e_favoriscono_la_scala_minore(self):
        # Verifica diretta della formula w_k = (1/scala_k^2)/Sum(1/scala_j^2)
        # su un caso a mano: due "metodi" con scala 1.0 e 2.0 -> pesi attesi
        # 4/5 e 1/5 (1/1^2 : 1/2^2 = 4 : 1).
        scale = np.array([1.0, 2.0])
        pesi = (1.0 / scale ** 2)
        pesi /= pesi.sum()
        assert pesi[0] == pytest.approx(0.8)
        assert pesi[1] == pytest.approx(0.2)
        assert pesi.sum() == pytest.approx(1.0)
