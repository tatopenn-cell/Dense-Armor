# -*- coding: utf-8 -*-
import numpy as np


def healing_filter(x: np.ndarray, radius: int = 2, sustain_threshold: float = 0.7, wide_mult: int = 3) -> np.ndarray:
    """
    Filtro di recupero segnale ispirato a healing.py (Dense-Evolution) --
    evoluzione portata su segnali IA generici. A differenza di ABCollatz e
    del damping Stadio 1 (entrambi giudicano un punto guardando solo il
    proprio residuo istantaneo), questo classifica ogni punto guardando
    QUANTI VICINI condividono la stessa deviazione dalla baseline locale:
      - deviazione isolata (nessun vicino la condivide) -> rumore/spike ->
        sostituito con la baseline (mediana di una finestra piu' ampia)
      - deviazione sostenuta (la maggioranza dei vicini nella finestra
        stretta condivide segno e ampiezza) -> cambiamento vero del segnale
        -> lasciato passare inalterato

    Parametri tarati per grid search (40 seed, scenario salto+spike,
    vedi test/test_healing_filter.py): radius=2, sustain_threshold=0.7,
    wide_mult=3 -> 40/40 vittorie su mediana mobile r2, rapporto
    media/std ~2.56 (non rumore statistico).

    LIMITI NOTI (misurati, non ipotizzati):
    - su segnale liscio + rumore gaussiano puro (nessun salto vero), perde
      contro una semplice mediana mobile a basso/medio rumore (0/30, 0/30);
      vince solo ad alto/molto alto rumore. Non e' il caso d'uso primario:
      questo filtro serve dove ci sono transizioni vere da preservare, non
      per denoising puro di segnali stazionari.
    - variante spike piu' fitti (15% invece di 5%): vince 30/30 ma con
      varianza alta (std=0.177 > media=0.124) -- il vantaggio e' consistente
      in segno ma non uniforme in ampiezza, da approfondire se si useranno
      densita' di spike elevate in produzione.
    - O(n * wide) per chiamata (loop Python, non vettorizzato/JIT) --
      prima di produzione va portato a JAX con jax.lax.scan o vmap su
      finestre, come il resto della codebase.
    - collasso TEMPORANEO (alcuni punti consecutivi a un valore estremo,
      poi ritorno al livello originale) scambiato per un cambiamento vero
      e lasciato passare: misurato su uno scenario reale con 3 punti
      consecutivi collassati a zero su un fondo costante (poi tornato al
      livello originale), RMSE 20.8 invece di ~0 -- vedi
      test/testKalman.py scenario C e
      test/test_arbiter_orca_integration.py in Dense-Armor per lo stesso
      scenario. Causa: questa funzione decide punto per punto (nessun
      raggruppamento in sequenze), quindi non ha modo di controllare cosa
      succede DOPO una deviazione sostenuta per distinguere un collasso
      temporaneo da un regime che si assesta davvero -- lo stesso
      controllo aggiunto a `utility/arbiter.py` (che invece raggruppa in
      sequenze) risolve questo caso li', ma richiederebbe un cambio di
      design qui, non ancora fatto per non rischiare la calibrazione a
      140 seed gia' verificata sopra.
    """
    n = len(x)
    out = np.zeros(n)
    wide = radius * wide_mult
    for i in range(n):
        lo_wide, hi_wide = max(0, i - wide), min(n, i + wide + 1)
        baseline = np.median(x[lo_wide:hi_wide])
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        window = x[lo:hi]
        dev_i = x[i] - baseline
        devs = window - baseline
        if abs(dev_i) < 1e-9:
            out[i] = x[i]
            continue
        same_sign_share = np.mean(
            (np.sign(devs) == np.sign(dev_i)) & (np.abs(devs) > sustain_threshold * abs(dev_i))
        )
        out[i] = x[i] if same_sign_share > 0.5 else baseline
    return out
