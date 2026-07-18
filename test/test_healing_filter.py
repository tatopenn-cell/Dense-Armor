# -*- coding: utf-8 -*-
"""Regressione per dense_armor/utility/healing.py.

Costruzione degli scenari verificata riga per riga con l'autore dopo un
primo giro di test che divergeva dai numeri attesi (vedi storia in git):
causa reale trovata in due punti, non un bug in healing_filter/
rolling_median:
1) Scenario A usava n=500 invece di 200 -- radius/wide_mult sono in
   CAMPIONI, non in unita' di tempo, quindi una finestra di 2-6 campioni
   copre una frazione diversa del periodo della sinusoide a seconda di n,
   abbastanza da ribaltare completamente vinto/perso a parita' di rumore.
2) Scenari B/C1/C2 usavano uno spike di ampiezza fissa (+/-8 con segno
   casuale uniforme) invece di uno spike gaussiano additivo
   N(0, 8^2) applicato solo dove spike_mask e' vero -- distribuzione
   diversa, margini diversi.

Con la costruzione esatta sotto, i numeri combaciano con quanto riportato
originariamente (B: 40/40, |media|/std=2.56; C1: 30/30, media/std vicini a
quelli gia' nel docstring di healing_filter; C2: 30/30, |media|/std=2.49).

Nota sul criterio di significativita': anche con la costruzione corretta,
il rapporto |media|/std per B (2.56) e C2 (2.49) NON supera 3 -- il
criterio ">3 sigma" proposto inizialmente per l'asserzione non varrebbe
letteralmente nemmeno sui dati giusti. Le asserzioni sotto usano quindi il
conteggio vittorie (deterministico, riprodotto esattamente) come criterio
primario, non un rapporto statistico che fallirebbe anche sul caso corretto.
"""
import numpy as np
from dense_armor.utility.healing import healing_filter

N = 200


def _rolling_median(x, radius=2):
    n = len(x)
    return np.array([np.median(x[max(0, i - radius):min(n, i + radius + 1)]) for i in range(n)])


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _segnale_sinusoide(n=N):
    return np.sin(np.linspace(0, 20, n)) * 5 + 10


def _scenario_salto_spike(rng, n=N, salto=(5.0, 12.0), base_noise=0.3, spike_frac=0.05, spike_std=8.0):
    """Ordine esatto delle chiamate RNG (rilevante per la riproducibilita'
    seed-per-seed): spike_mask -- rumore base -- ampiezza spike."""
    x_clean = np.concatenate([np.full(n // 2, salto[0]), np.full(n - n // 2, salto[1])])
    spike_mask = rng.random(n) < spike_frac
    x_corr = x_clean + rng.normal(0, base_noise, n) + np.where(spike_mask, rng.normal(0, spike_std, n), 0.0)
    return x_clean, x_corr


def test_scenario_a_sinusoide_perde_a_rumore_basso_e_medio_vince_ad_alto():
    """n=200 (verificato con l'autore): healing_filter perde sistematicamente
    contro la mediana mobile a rumore basso (0.05) e medio (0.30) -- 0/30 in
    entrambi i casi. A rumore alto (1.0) il risultato e' misto (21/30 a
    favore di healing, non un vincitore netto). Solo a rumore molto alto
    (3.0) la vittoria e' netta (30/30)."""
    clean = _segnale_sinusoide()

    for livello in [0.05, 0.30]:
        diffs = []
        for seed in range(30):
            rng = np.random.default_rng(seed)
            rumoroso = clean + rng.normal(0, livello, size=clean.shape)
            diffs.append(_rmse(healing_filter(rumoroso), clean) - _rmse(_rolling_median(rumoroso), clean))
        assert all(d > 0 for d in diffs), (
            f"a rumore={livello}: atteso che healing_filter perda sistematicamente "
            f"contro la mediana mobile (limite noto)"
        )

    diffs_alto = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        rumoroso = clean + rng.normal(0, 1.0, size=clean.shape)
        diffs_alto.append(_rmse(healing_filter(rumoroso), clean) - _rmse(_rolling_median(rumoroso), clean))
    vinte_alto = sum(1 for d in diffs_alto if d < 0)
    assert vinte_alto >= 15, f"a rumore=1.0 atteso almeno un pareggio verso l'alto, misurate {vinte_alto}/30"

    diffs_molto_alto = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        rumoroso = clean + rng.normal(0, 3.0, size=clean.shape)
        diffs_molto_alto.append(_rmse(healing_filter(rumoroso), clean) - _rmse(_rolling_median(rumoroso), clean))
    assert all(d < 0 for d in diffs_molto_alto), "a rumore=3.0 atteso 30/30 a favore di healing_filter"


def test_scenario_b_salto_vero_piu_spike_batte_mediana_mobile():
    """Costruzione esatta verificata: 40/40 vittorie, |media|/std=2.56
    (combacia con quanto originariamente riportato)."""
    diffs = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        clean, rumoroso = _scenario_salto_spike(rng)
        diffs.append(_rmse(healing_filter(rumoroso, radius=2, sustain_threshold=0.7, wide_mult=3), clean)
                      - _rmse(_rolling_median(rumoroso, radius=2), clean))
    diffs = np.array(diffs)
    assert np.all(diffs < 0), f"healing_filter dovrebbe battere la mediana mobile su tutti i 40 seed, vinte {int(np.sum(diffs<0))}/40"


def test_scenario_c1_spike_fitti_vince_con_varianza_alta():
    """Variante 15% di spike: 30/30 confermato, varianza alta (media~0.124,
    std~0.177, combacia col docstring di healing_filter) -- il vantaggio e'
    consistente in segno ma non uniforme in ampiezza, come documentato."""
    diffs = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        clean, rumoroso = _scenario_salto_spike(rng, spike_frac=0.15)
        diffs.append(_rmse(healing_filter(rumoroso), clean) - _rmse(_rolling_median(rumoroso, radius=2), clean))
    diffs = np.array(diffs)
    assert np.all(diffs < 0), f"attese 30/30 vittorie, misurate {int(np.sum(diffs<0))}/30"
    assert diffs.std() > abs(diffs.mean()), "varianza attesa alta (std > |media|), come documentato"


def test_scenario_c2_salto_piccolo_vince():
    """Variante salto piccolo 5->7 (invece di 5->12), stessa costruzione
    di spike/rumore di B: 30/30 confermato, |media|/std=2.49."""
    diffs = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        clean, rumoroso = _scenario_salto_spike(rng, salto=(5.0, 7.0))
        diffs.append(_rmse(healing_filter(rumoroso), clean) - _rmse(_rolling_median(rumoroso, radius=2), clean))
    diffs = np.array(diffs)
    assert np.all(diffs < 0), f"attese 30/30 vittorie, misurate {int(np.sum(diffs<0))}/30"
