# -*- coding: utf-8 -*-
import numpy as np
import pytest

from dense_armor.core.vector import BitwisePermutationEngine, ParametricScenarioSimulator


def test_bitwise_swap_scambia_gli_elementi_attesi():
    engine = BitwisePermutationEngine(n_elements=2)  # size = 4, stati 00,01,10,11
    data = np.array([0.0, 1.0, 2.0, 3.0])
    # target_bit=1 (stride 1), control_bit=0 (stride 2): scambia indici (2,3)
    out = engine.apply_bitwise_swap(data, target_bit=1, control_bit=0)
    np.testing.assert_array_equal(out, [0.0, 1.0, 3.0, 2.0])


def test_bitwise_swap_non_modifica_l_originale():
    engine = BitwisePermutationEngine(n_elements=2)
    data = np.array([0.0, 1.0, 2.0, 3.0])
    engine.apply_bitwise_swap(data, target_bit=1, control_bit=0)
    np.testing.assert_array_equal(data, [0.0, 1.0, 2.0, 3.0])


def test_run_parallel_scenarios_shape():
    sim = ParametricScenarioSimulator()
    params_batch = np.ones((5, 10)) * 0.3
    history = sim.run_parallel_scenarios(base_state=0.0, parameters_batch=params_batch)
    assert history.shape == (5, 10)


def test_run_parallel_scenarios_converge_verso_il_parametro_costante():
    sim = ParametricScenarioSimulator()
    params_batch = np.ones((1, 200)) * 2.0
    history = sim.run_parallel_scenarios(base_state=0.0, parameters_batch=params_batch)
    assert history[0, -1] == pytest.approx(2.0, abs=1e-3)


def test_collapse_decision_result_binario_e_vettore_normalizzato():
    sim = ParametricScenarioSimulator()
    dist = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result, collapsed = sim.collapse_decision(dist, target_idx=2)
    assert result in (0, 1)
    assert np.sum(np.abs(collapsed)) == pytest.approx(1.0, abs=1e-6)


def test_collapse_decision_non_modifica_l_array_originale():
    sim = ParametricScenarioSimulator()
    dist = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    original = dist.copy()
    sim.collapse_decision(dist, target_idx=2)
    np.testing.assert_array_equal(dist, original)


def test_collapse_decision_vettore_a_energia_zero_solleva_errore():
    sim = ParametricScenarioSimulator()
    with pytest.raises(RuntimeError):
        sim.collapse_decision(np.zeros(5), target_idx=2)
