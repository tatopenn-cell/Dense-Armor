# -*- coding: utf-8 -*-
import types

import pytest

import dense_armor.core.memory as memory_module
from dense_armor.core.memory import UniversalMemoryGuard, MemoryPressureError


def _fake_virtual_memory(total, available):
    return types.SimpleNamespace(total=total, available=available)


def test_check_memory_safety_passa_con_ram_abbondante(monkeypatch):
    monkeypatch.setattr(
        "dense_armor.core.memory.psutil.virtual_memory",
        lambda: _fake_virtual_memory(total=100, available=90),
    )
    guard = UniversalMemoryGuard(min_free_ram_percentage=0.15, force_gc=False)
    guard.check_memory_safety()  # non deve sollevare


def test_check_memory_safety_solleva_sotto_soglia(monkeypatch):
    monkeypatch.setattr(
        "dense_armor.core.memory.psutil.virtual_memory",
        lambda: _fake_virtual_memory(total=100, available=1),
    )
    guard = UniversalMemoryGuard(min_free_ram_percentage=0.15, force_gc=False)
    with pytest.raises(MemoryPressureError):
        guard.check_memory_safety()


def test_check_memory_safety_soft_gc_non_cambia_esito_se_ram_resta_bassa(monkeypatch):
    monkeypatch.setattr(
        "dense_armor.core.memory.psutil.virtual_memory",
        lambda: _fake_virtual_memory(total=100, available=1),
    )
    guard = UniversalMemoryGuard(min_free_ram_percentage=0.15, force_gc=True)
    with pytest.raises(MemoryPressureError):
        guard.check_memory_safety()


def test_get_gpu_free_memory_nvidia_ritorna_frazione_valida():
    guard = UniversalMemoryGuard()
    ratio = guard._get_gpu_free_memory_nvidia()
    assert 0.0 <= ratio <= 1.0


def test_calculate_optimal_chunks_un_solo_chunk_se_ci_sta(monkeypatch):
    monkeypatch.setattr(
        "dense_armor.core.memory.psutil.virtual_memory",
        lambda: _fake_virtual_memory(total=1_000_000, available=1_000_000),
    )
    guard = UniversalMemoryGuard(min_free_ram_percentage=0.05, force_gc=False)
    chunks = guard.calculate_optimal_chunks(total_items=10, item_size_bytes=8)
    assert chunks == 1


def test_calculate_optimal_chunks_piu_chunk_se_i_dati_non_ci_stanno(monkeypatch):
    monkeypatch.setattr(
        "dense_armor.core.memory.psutil.virtual_memory",
        lambda: _fake_virtual_memory(total=1_000, available=1_000),
    )
    guard = UniversalMemoryGuard(min_free_ram_percentage=0.05, force_gc=False)
    chunks = guard.calculate_optimal_chunks(total_items=1_000_000, item_size_bytes=8)
    assert chunks > 1


def test_get_gpu_free_memory_nvidia_analizza_output_multi_gpu_reale(monkeypatch):
    fake_csv = "8000, 4000\n16000, 16000\n"
    monkeypatch.setattr(
        memory_module.subprocess, "check_output", lambda *a, **k: fake_csv.encode()
    )
    guard = UniversalMemoryGuard()
    ratio = guard._get_gpu_free_memory_nvidia()
    assert ratio == pytest.approx(0.5)  # min(4000/8000, 16000/16000) = 0.5


def test_check_memory_safety_soft_gc_gestisce_jax_clear_caches_che_fallisce(monkeypatch):
    # free_pct=0.10 < min_free_ram(0.05) + 0.10 -> attiva il soft GC
    monkeypatch.setattr(
        memory_module.psutil, "virtual_memory",
        lambda: _fake_virtual_memory(total=100, available=10),
    )
    monkeypatch.setattr(
        memory_module.jax, "clear_caches",
        lambda: (_ for _ in ()).throw(RuntimeError("guasto simulato")),
    )
    guard = UniversalMemoryGuard(min_free_ram_percentage=0.05, force_gc=True)
    guard.check_memory_safety()  # non deve propagare l'errore di jax.clear_caches


def _fake_gpu_device(platform="gpu", device_kind="Fake GPU"):
    return types.SimpleNamespace(platform=platform, device_kind=device_kind)


def test_check_memory_safety_vram_esaurita_solleva_errore(monkeypatch):
    monkeypatch.setattr(
        memory_module.psutil, "virtual_memory",
        lambda: _fake_virtual_memory(total=100, available=90),
    )
    monkeypatch.setattr(memory_module.jax, "devices", lambda: [_fake_gpu_device()])
    guard = UniversalMemoryGuard(min_free_ram_percentage=0.05, force_gc=False)
    monkeypatch.setattr(guard, "_get_gpu_free_memory_nvidia", lambda: 0.01)

    with pytest.raises(MemoryPressureError, match="VRAM esaurita"):
        guard.check_memory_safety()


def test_check_memory_safety_vram_libera_non_solleva(monkeypatch):
    monkeypatch.setattr(
        memory_module.psutil, "virtual_memory",
        lambda: _fake_virtual_memory(total=100, available=90),
    )
    monkeypatch.setattr(memory_module.jax, "devices", lambda: [_fake_gpu_device()])
    guard = UniversalMemoryGuard(min_free_ram_percentage=0.05, force_gc=False)
    monkeypatch.setattr(guard, "_get_gpu_free_memory_nvidia", lambda: 0.9)

    guard.check_memory_safety()  # non deve sollevare


def test_check_memory_safety_query_dispositivo_jax_fallita_non_blocca(monkeypatch):
    monkeypatch.setattr(
        memory_module.psutil, "virtual_memory",
        lambda: _fake_virtual_memory(total=100, available=90),
    )

    def _boom():
        raise RuntimeError("driver GPU non inizializzato")

    monkeypatch.setattr(memory_module.jax, "devices", _boom)
    guard = UniversalMemoryGuard(min_free_ram_percentage=0.05, force_gc=False)
    guard.check_memory_safety()  # non deve propagare l'errore
