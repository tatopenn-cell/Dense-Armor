# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dense_armor.core.noise as noise_module
from dense_armor.core.noise import AIHardwareProfiler, StochasticAdversarialNoise


def test_ai_hardware_profiler_espone_i_campi_attesi():
    profiler = AIHardwareProfiler()
    assert profiler.ram_total_gb > 0
    assert profiler.max_tensor_dim > 0
    assert isinstance(profiler.backend_device, str)
    assert isinstance(profiler.get_profile_summary(), str)
    assert "RAM" in profiler.get_profile_summary()


def test_inject_noise_intensita_zero_lascia_il_dato_invariato():
    data = np.array([1.0, 2.0, 3.0])
    out = StochasticAdversarialNoise.inject_noise(data, "bitflip", intensity=0.0)
    np.testing.assert_array_equal(out, data)


def test_inject_noise_clean_lascia_il_dato_invariato():
    data = np.array([1.0, 2.0, 3.0])
    out = StochasticAdversarialNoise.inject_noise(data, "clean", intensity=0.9)
    np.testing.assert_array_equal(out, data)


@pytest.mark.parametrize("noise_type", ["bitflip", "dropout_noise", "gaussian_blur"])
def test_inject_noise_numpy_path_preserva_shape_e_norma(noise_type):
    data = np.ones(50)
    out = StochasticAdversarialNoise.inject_noise(data, noise_type, intensity=0.5, seed=1)
    assert out.shape == data.shape
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("noise_type", ["bitflip", "dropout_noise", "gaussian_blur"])
def test_inject_noise_jax_path_preserva_shape_e_norma(noise_type):
    data = jnp.ones(50)
    out = StochasticAdversarialNoise.inject_noise(data, noise_type, intensity=0.5, seed=1)
    assert out.shape == data.shape
    assert float(jnp.linalg.norm(out)) == pytest.approx(1.0, abs=1e-4)


def test_inject_noise_bitflip_intensita_massima_inverte_il_segno_di_tutto():
    data = np.arange(1, 201, dtype=np.float64)
    out = StochasticAdversarialNoise.inject_noise(data, "bitflip", intensity=1.0, seed=3)
    # intensity=1.0: ogni elemento supera la soglia -> flip garantito su tutti
    np.testing.assert_allclose(out, -data / np.linalg.norm(data), atol=1e-6)


def test_inject_noise_bitflip_intensita_parziale_flippa_solo_alcuni():
    data = np.ones(500)
    out = np.array(
        StochasticAdversarialNoise.inject_noise(data, "bitflip", intensity=0.5, seed=3)
    )
    unique_abs_signs = set(np.sign(out).tolist())
    assert unique_abs_signs == {1.0, -1.0}, "con intensity=0.5 ci si aspettano sia flip che non-flip"


def test_inject_noise_jax_path_tipo_sconosciuto_lascia_il_dato_pre_normalizzazione():
    data = jnp.array([1.0, 2.0, 3.0])
    out = StochasticAdversarialNoise.inject_noise(data, "tipo_inesistente", intensity=0.9, seed=1)
    expected = data / jnp.linalg.norm(data)
    np.testing.assert_allclose(np.array(out), np.array(expected), atol=1e-6)


def test_detect_active_backend_fallback_se_jax_default_backend_fallisce(monkeypatch):
    def _boom():
        raise RuntimeError("nessun backend disponibile")

    monkeypatch.setattr(noise_module.jax, "default_backend", _boom)
    profiler = AIHardwareProfiler()
    assert profiler.backend_device == "CPU (JAX Fallback)"


def test_get_safe_tensor_limit_scala_con_ram_alta():
    profiler = AIHardwareProfiler()
    profiler.ram_total_gb = 40.0
    profiler.backend_device = "CPU (JAX Accelerato)"
    assert profiler._get_safe_tensor_limit() == 8192


def test_get_safe_tensor_limit_scala_con_ram_media():
    profiler = AIHardwareProfiler()
    profiler.ram_total_gb = 16.0
    profiler.backend_device = "CPU (JAX Accelerato)"
    assert profiler._get_safe_tensor_limit() == 4096


def test_get_safe_tensor_limit_raddoppia_con_gpu():
    profiler = AIHardwareProfiler()
    profiler.ram_total_gb = 40.0
    profiler.backend_device = "GPU (JAX Accelerato)"
    assert profiler._get_safe_tensor_limit() == 16384
