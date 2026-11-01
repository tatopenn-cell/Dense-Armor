# -*- coding: utf-8 -*-

import dense_armor.core.noise as noise_module
from dense_armor.core.noise import AIHardwareProfiler


def test_ai_hardware_profiler_espone_i_campi_attesi():
    profiler = AIHardwareProfiler()
    assert profiler.ram_total_gb > 0
    assert profiler.max_tensor_dim > 0
    assert isinstance(profiler.backend_device, str)
    assert isinstance(profiler.get_profile_summary(), str)
    assert "RAM" in profiler.get_profile_summary()


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
