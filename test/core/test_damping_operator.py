import jax.numpy as jnp
import numpy as np

from dense_armor.utility.damping_operator import apply_damping_blend


def test_nessun_nan_in_uscita_anche_con_nan_in_ingresso():
    grezzo = jnp.array([1.0, float("nan"), 0.3, -2.0])
    riferimento = jnp.array([1.0, 0.4, float("nan"), -1.9])
    out = apply_damping_blend(grezzo, riferimento)
    assert not bool(jnp.any(jnp.isnan(out)))


def test_blend_range_limitato():
    """Il coefficiente K non deve mai far collassare l'output esattamente
    sull'uno o sull'altro segnale (mai 0% o 100%)."""
    grezzo = jnp.array([1.0, 50.0, 0.3, -2.0])
    riferimento = jnp.array([1.0, 0.4, 0.35, -1.9])
    out = np.array(apply_damping_blend(grezzo, riferimento))
    for g, r, o in zip(np.array(grezzo), np.array(riferimento), out):
        if g == r:
            continue
        assert min(g, r) - 1e-6 <= o <= max(g, r) + 1e-6


def _peso_del_riferimento(distanza):
    from dense_armor.utility.damping_operator import _ALPHA, _SIGMA

    grezzo = jnp.array([1.0], dtype=jnp.float32)
    riferimento = jnp.array([1.0 + distanza], dtype=jnp.float32)
    out = float(apply_damping_blend(grezzo, riferimento)[0])
    return (out - 1.0) / distanza, _ALPHA - _SIGMA, 1.0 - (_ALPHA - _SIGMA)


def test_il_peso_del_riferimento_parte_vicino_a_un_quarto_a_distanza_quasi_nulla():
    peso, _, _ = _peso_del_riferimento(0.001)
    assert 0.25 < peso < 0.30


def test_il_peso_del_riferimento_cresce_con_la_distanza_fino_al_limite():
    p_piccolo, _, _ = _peso_del_riferimento(0.001)
    p_medio, _, limite = _peso_del_riferimento(0.01)
    assert p_piccolo < p_medio < limite


def test_oltre_una_distanza_di_circa_0_05_il_peso_resta_al_limite_superiore():
    for distanza in (0.05, 0.5, 2.0, 8.0):
        peso, _, limite = _peso_del_riferimento(distanza)
        assert abs(peso - limite) < 1e-5, (distanza, peso, limite)


def test_il_peso_non_scende_mai_sotto_il_limite_inferiore():
    for distanza in (0.0001, 0.001, 0.01, 1.0):
        peso, minimo, _ = _peso_del_riferimento(distanza)
        assert peso >= minimo - 1e-6
