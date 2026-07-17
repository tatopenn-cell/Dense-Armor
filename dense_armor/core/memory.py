# -*- coding: utf-8 -*-
"""
core/memory.py
====================
UniversalMemoryGuard — controlla RAM e VRAM prima di ogni allocazione pesante.
MemoryPressureError  — eccezione lanciata quando la memoria è insufficiente.
"""

import gc
import logging
import math
import subprocess
import psutil

try:
    import jax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

logger = logging.getLogger(__name__)


class MemoryPressureError(Exception):
    """Eccezione lanciata quando la memoria di sistema (RAM/VRAM) è insufficiente."""


class UniversalMemoryGuard:
    """
    Monitora preventivamente RAM e VRAM prima di ogni allocazione pesante.
    Calcola il partizionamento ottimale dei dati per prevenire OOM.
    """

    def __init__(
        self,
        min_free_ram_percentage: float = 0.15,
        force_gc: bool = True,
    ) -> None:
        """min_free_ram_percentage — soglia minima di RAM libera richiesta;
        force_gc — se True tenta un soft garbage-collect prima di bloccare."""
        self.min_free_ram = min_free_ram_percentage
        self.force_gc     = force_gc

    # ── GPU helpers ───────────────────────────────────────────────────────────

    def _get_gpu_free_memory_nvidia(self) -> float:
        """Estrae la percentuale minima di VRAM libera dai driver NVIDIA (Multi-GPU Safe)."""
        try:
            cmd    = "nvidia-smi --query-gpu=memory.total,memory.free --format=csv,nounits,noheader"
            output = subprocess.check_output(cmd, shell=True, timeout=5).decode().strip()
            
            # [FIX MULTI-GPU]: Gestisce l'output in caso di schede video multiple isolate
            lines = [line.strip() for line in output.split("\n") if line.strip()]
            min_free_ratio = 1.0
            
            for line in lines:
                total, free = map(float, line.split(","))
                if total > 0:
                    min_free_ratio = min(min_free_ratio, free / total)
            return min_free_ratio
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError, OSError, ValueError):
            # nvidia-smi assente, fallito, in timeout, o output di formato
            # inatteso: nessuna GPU NVIDIA leggibile, non un bug -- si
            # assume VRAM libera (nessun blocco).
            return 1.0

    # ── RAM check ─────────────────────────────────────────────────────────────

    def check_memory_safety(self) -> None:
        """Verifica lo stato della RAM e della VRAM prima di allocazioni critiche."""
        vm             = psutil.virtual_memory()
        free_pct       = vm.available / vm.total

        # Soft GC se vicini alla soglia
        if self.force_gc and free_pct < (self.min_free_ram + 0.10):
            gc.collect()
            if HAS_JAX:
                try:
                    jax.clear_caches()
                except Exception:
                    # pulizia best-effort e non critica: gli interni di JAX
                    # possono fallire in troppi modi diversi per elencarli,
                    # ma non deve bloccare il check di sicurezza memoria --
                    # loggato (non piu' silenzioso) per restare tracciabile.
                    logger.debug("jax.clear_caches() fallito durante il soft GC", exc_info=True)
            vm       = psutil.virtual_memory()
            free_pct = vm.available / vm.total

        if free_pct < self.min_free_ram:
            raise MemoryPressureError(
                f"RAM insufficiente: {free_pct:.1%} disponibile — "
                f"richiesta minima: {self.min_free_ram:.1%}"
            )

        # VRAM check (solo se JAX con backend GPU)
        if HAS_JAX:
            try:
                for dev in jax.devices():
                    if dev.platform == "gpu":
                        vram_free = self._get_gpu_free_memory_nvidia()
                        if vram_free < 0.05:
                            raise MemoryPressureError(
                                f"VRAM esaurita su {dev.device_kind}: "
                                f"{vram_free:.1%} libera."
                            )
            except MemoryPressureError:
                raise
            except (RuntimeError, AttributeError):
                # query driver/dispositivo JAX fallita (es. driver GPU non
                # inizializzato correttamente): stesso principio del check
                # RAM, non e' un errore dell'utente -- si prosegue senza
                # bloccare su un dato VRAM che non si riesce a leggere.
                pass

    # ── Chunk calculator ──────────────────────────────────────────────────────

    def calculate_optimal_chunks(self, total_items: int, item_size_bytes: int) -> int:
        """Calcola il partizionamento ottimale basato sulla RAM e sul sovraccarico XLA."""
        self.check_memory_safety()
        vm                     = psutil.virtual_memory()
        
        # [FIX XLA-PADDING]: Riduciamo la finestra allocabile al 40% per compensare 
        # i buffer temporanei generati durante il tracciamento dei grafi statici
        safe_allocatable_bytes = int(vm.available * 0.40)
        total_size_bytes       = total_items * item_size_bytes

        if total_size_bytes <= safe_allocatable_bytes:
            return 1

        return max(math.ceil(total_size_bytes / safe_allocatable_bytes), 1)
