"""Power trace and energy: read a sysfs node, integrate the power over time.

The whole-board power of a Jetson is read from the INA3221 sensors
through the sysfs nodes, one integer in mW per read (Shalavi et al.,
2306.13107, page 2). The energy of a run is the integral of the power
over the wall time, by the trapezoid rule, in joules: a milliwatt-second
divided by 1000. The internal sensors underestimate the actual power by
up to 50% in most cases and a linear calibration brings the error within
±3% (page 1); the calibration is available as ``calibrate_mw`` and is
not applied by default.

Run from the repository root:

    python -m edge.power --demo

The demo prints the energy of a synthetic constant power, labelled
synthetic. A real measurement requires a board with the INA3221 sensor
exposed as a sysfs node; the node path is a parameter, without a default.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CALIBRATION_MW: dict[str, tuple[float, float]] = {
    "AGX Orin": (1.02, 3115.39),
    "Xavier NX": (1.10, 3130.41),
    "TX2": (0.90, 1998.80),
    "Nano": (1.11, 232.60),
}


@dataclass
class PowerTrace:
    """A power trace: sample times in seconds and power in mW.

    The two arrays must have the same length and at least two samples,
    because an integral needs a segment. Construction raises
    ``ValueError`` otherwise.
    """

    t_s: np.ndarray
    mw: np.ndarray

    def __post_init__(self) -> None:
        self.t_s = np.asarray(self.t_s, dtype=float)
        self.mw = np.asarray(self.mw, dtype=float)
        if self.t_s.shape != self.mw.shape:
            raise ValueError(
                f"t_s and mw must have the same length, got "
                f"{self.t_s.shape} and {self.mw.shape}"
            )
        if self.t_s.ndim != 1:
            raise ValueError(f"t_s must be 1-D, got {self.t_s.ndim}-D")
        if self.t_s.size < 2:
            raise ValueError(f"a trace needs at least two samples, got {self.t_s.size}")


def energy_j(trace: PowerTrace) -> float:
    """Integral of the power over time, in joules, by the trapezoid rule.

    Milliwatt times second divided by 1000 is a joule. A constant power
    ``P`` held for ``T`` seconds gives ``P*T/1000``. A linear ramp from
    ``P0`` to ``P1`` over ``T`` gives ``(P0+P1)/2 * T/1000``.
    """
    return float(np.trapezoid(trace.mw, trace.t_s) / 1000.0)


def net_energy_j(trace: PowerTrace, idle_mw: float) -> float:
    """Energy above the idle power, in joules, never below zero in total.

    The idle power is subtracted at every sample; a sample below idle
    contributes zero to the integral, so a board that spends less than
    its own idle does not produce a negative energy.
    """
    above = np.clip(trace.mw - float(idle_mw), 0.0, None)
    return float(np.trapezoid(above, trace.t_s) / 1000.0)


def calibrate_mw(x_mw: float, device: str) -> float:
    """Linear calibration of Table I, page 4 of 2306.13107.

    ``y = a*x + b``, with ``x`` the reading of the built-in sensor in mW,
    ``a`` and ``b`` from the table, and ``y`` the estimate of the true
    board power in mW. The four lines are: AGX Orin ``1.02x + 3115.39``,
    Xavier NX ``1.10x + 3130.41``, TX2 ``0.90x + 1998.80``, Nano
    ``1.11x + 232.60``, with errors ±3%, ±2%, ±3%, ±0.8% respectively.

    The four lines hold for the devices and the setup of the paper
    (external oscilloscope, same nvpmodel); the owner decides whether to
    apply them on a different setup. The default of every function that
    could apply them is not to apply them.
    """
    if device not in CALIBRATION_MW:
        raise ValueError(
            f"unknown device {device!r}, expected one of {sorted(CALIBRATION_MW)}"
        )
    a, b = CALIBRATION_MW[device]
    return float(a * float(x_mw) + b)


class SysfsPowerSource:
    """Read an integer in mW from a sysfs node every ``dt_s`` seconds.

    The INA3221 sensors of a Jetson expose the whole-board power as a
    text file under sysfs, one integer in mW per read (2306.13107, page
    2). A background thread reads the file every ``dt_s`` seconds
    between :meth:`start` and :meth:`stop`, and :meth:`stop` returns the
    :class:`PowerTrace` and joins the thread.

    The path is a parameter without a default: the node name depends on
    the board and the kernel, and a guessed default would be wrong on
    most boards. The sampling interval matters: the built-in sensors
    measure average values and miss the current peaks, and the sampling
    rate changes the precision (page 1). My choice: ``dt_s`` has no
    default, so the interval is always declared by the caller and
    written next to the trace.
    """

    def __init__(self, path: str | Path, dt_s: float) -> None:
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt_s}")
        self.path = Path(path)
        self.dt_s = float(dt_s)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._t0 = 0.0
        self._t: list[float] = []
        self._mw: list[float] = []
        self._error: BaseException | None = None

    def _read_mw(self) -> float:
        return float(int(self.path.read_text(encoding="utf-8").strip()))

    def _loop(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            t = time.monotonic() - self._t0
            try:
                mw = self._read_mw()
            except BaseException as exc:
                self._error = exc
                return
            self._t.append(t)
            self._mw.append(mw)
            next_t += self.dt_s
            sleep = next_t - time.monotonic()
            if sleep > 0.0:
                self._stop.wait(sleep)

    def start(self) -> SysfsPowerSource:
        if self._thread is not None:
            raise RuntimeError("already started")
        self._t0 = time.monotonic()
        self._t = []
        self._mw = []
        self._error = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=False)
        self._thread.start()
        return self

    def stop(self) -> PowerTrace:
        if self._thread is None:
            raise RuntimeError("not started")
        self._stop.set()
        self._thread.join()
        thread = self._thread
        self._thread = None
        if self._error is not None:
            err = self._error
            raise RuntimeError(f"read failed: {err}") from err
        t: np.ndarray = np.asarray(self._t, dtype=float)
        mw: np.ndarray = np.asarray(self._mw, dtype=float)
        if t.size < 2:
            raise RuntimeError(f"the trace has {t.size} samples, need at least two")
        del thread
        return PowerTrace(t_s=t, mw=mw)


class SyntheticPowerSource:
    """A constant power, for tests and for the page. The trace is synthetic.

    With ``duration_s`` given, :meth:`stop` returns a trace of exactly
    that duration, on a uniform grid of ``dt_s``. Without it, the
    duration is the wall time elapsed between :meth:`start` and
    :meth:`stop`, which is what a real source does. Either way the trace
    is synthetic, and the caller must say so on the page and in the JSON.
    """

    def __init__(
        self,
        mw: float,
        dt_s: float = 0.1,
        duration_s: float | None = None,
    ) -> None:
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt_s}")
        if duration_s is not None and duration_s <= 0.0:
            raise ValueError(
                f"duration_s must be positive when given, got {duration_s}"
            )
        self.mw = float(mw)
        self.dt_s = float(dt_s)
        self.duration_s = duration_s
        self._t0 = 0.0
        self._running = False

    def start(self) -> SyntheticPowerSource:
        self._t0 = time.monotonic()
        self._running = True
        return self

    def stop(self) -> PowerTrace:
        if not self._running:
            raise RuntimeError("not started")
        self._running = False
        if self.duration_s is None:
            duration = time.monotonic() - self._t0
        else:
            duration = float(self.duration_s)
        duration = max(duration, 1e-6)
        n = max(int(math.ceil(duration / self.dt_s)) + 1, 2)
        t = np.linspace(0.0, duration, n)
        mw = np.full(n, self.mw)
        return PowerTrace(t_s=t, mw=mw)


def _demo() -> int:
    mw = 5000.0
    duration = 2.0
    src = SyntheticPowerSource(mw=mw, dt_s=0.1, duration_s=duration)
    trace = src.start().stop()
    e = energy_j(trace)
    print(f"source: synthetic, {mw:.0f} mW for {duration:.1f} s")
    print(f"energy: {e:.4f} J (expected {mw * duration / 1000.0:.4f})")
    print(f"calibrated AGX Orin: {calibrate_mw(mw, 'AGX Orin'):.2f} mW")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="edge.power")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        return _demo()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
