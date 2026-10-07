"""Root class of the Dense-Armor online-learning stack.

Follows Buitinck et al. (arXiv:1309.0238): hyper-parameters set in
``__init__`` are stored unchanged as public attributes; learned state
lives in attributes with a trailing underscore. Adds robot/LLM extras
(time base, state_dict, describe).
"""

from __future__ import annotations

import copy
import gc
import inspect
import logging
import math
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

_ARGS_KEY = "_args"
_VERSION_KEY = "_dense_armor_version"


def _get_version() -> str:
    """Installed dense-armor version, or "unknown" if unavailable."""
    try:
        from importlib.metadata import version

        return version("dense-armor")
    except Exception:  # noqa: BLE001
        return "unknown"


class InconsistentVersionWarning(UserWarning):
    """Unpickled a Dense-Armor estimator from a different library version."""


def _format_float(v: float) -> str:
    """Float formatting per Task-01 spec.

    Scientific when abs(v) > 1e5 or 0 < abs(v) < 1e-4, else up to six
    decimals with at least one digit after the point; nan / inf as such.
    """
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    a = abs(v)
    if a > 1e5 or (0.0 < a < 1e-4):
        return f"{v:.0e}"
    s = f"{v:.6f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s


def _short_repr(v: Any, depth: int, module_prefix: bool = False) -> str:
    if isinstance(v, Base):
        return v._repr_at_depth(depth + 1, module_prefix)
    if isinstance(v, float):
        return _format_float(v)
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, (set, frozenset)):
        items = sorted(v, key=lambda x: (str(type(x)), str(x)))
        return (
            "{" + ", ".join(_short_repr(i, depth, module_prefix) for i in items) + "}"
        )
    if isinstance(v, tuple):
        inner = ", ".join(_short_repr(i, depth, module_prefix) for i in v)
        return "(" + inner + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_short_repr(i, depth, module_prefix) for i in v) + "]"
    if isinstance(v, dict):
        items = sorted(v.items(), key=lambda kv: str(kv[0]))
        return (
            "{"
            + ", ".join(
                f"{k!r}: {_short_repr(val, depth, module_prefix)}" for k, val in items
            )
            + "}"
        )
    if callable(v):
        return getattr(v, "__name__", repr(v))
    return repr(v)


class Base:
    """Root of the Dense-Armor online-learning stack."""

    _mutable_attributes: frozenset = frozenset()

    # ── parameters ────────────────────────────────────────────────
    def _init_signature(self) -> inspect.Signature:
        return inspect.signature(type(self).__init__)

    def _get_params(self) -> dict[str, Any]:
        """Walk ``__init__`` and read each kwarg from the matching attribute.

        A parameter that is itself a :class:`Base` is stored as
        ``(class, its_params)`` recursively. ``*args`` goes under
        ``_args``; ``**kwargs`` is expanded.
        """
        out: dict[str, Any] = {}
        args_coll: tuple = ()
        for name, p in self._init_signature().parameters.items():
            if name == "self":
                continue
            if p.kind is inspect.Parameter.VAR_POSITIONAL:
                args_coll = tuple(getattr(self, name, ()) or ())
                continue
            if p.kind is inspect.Parameter.VAR_KEYWORD:
                kw = getattr(self, name, None)
                if isinstance(kw, dict):
                    out.update(kw)
                continue
            if not hasattr(self, name):
                continue
            val = getattr(self, name)
            if isinstance(val, Base):
                out[name] = (type(val), val._get_params())
            else:
                out[name] = val
        if args_coll:
            out[_ARGS_KEY] = args_coll
        return out

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the estimator's hyper-parameters.

        Args:
            deep: if True (default), nested ``Base`` parameters are
                expanded to ``(class, params_dict)``. If False, they are
                returned as the estimator instance.

        Returns:
            Mapping from parameter name to value.
        """
        if deep:
            return self._get_params()
        out: dict[str, Any] = {}
        for name, p in self._init_signature().parameters.items():
            if name == "self" or p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if hasattr(self, name):
                out[name] = getattr(self, name)
        return out

    def set_params(self, **params: Any) -> Base:
        """Set hyper-parameters in place and return ``self``."""
        valid = set(self._init_signature().parameters) - {"self"}
        for k, v in params.items():
            if k not in valid:
                raise ValueError(f"{type(self).__name__} has no parameter {k!r}")
            setattr(self, k, v)
        return self

    # ── clone ─────────────────────────────────────────────────────
    def clone(
        self,
        new_params: dict[str, Any] | None = None,
        include_attributes: bool = False,
    ) -> Base:
        """Build a fresh instance from the constructor parameters.

        Args:
            new_params: overrides; a nested ``Base`` parameter may receive
                a dict of its own parameters.
            include_attributes: also deep-copy non-parameter attributes.

        Returns:
            A new estimator with no learned state.
        """
        new_params = dict(new_params or {})
        kwargs: dict[str, Any] = {}
        for name, p in self._init_signature().parameters.items():
            if name == "self" or p.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                continue
            cur = getattr(self, name, None)
            if name in new_params:
                ov = new_params.pop(name)
                if isinstance(cur, Base) and isinstance(ov, dict):
                    kwargs[name] = cur.clone(new_params=ov)
                else:
                    kwargs[name] = copy.deepcopy(ov)
            elif isinstance(cur, Base):
                kwargs[name] = cur.clone()
            else:
                kwargs[name] = copy.deepcopy(cur)
        if new_params:
            raise ValueError(f"Unknown parameter(s) for clone: {sorted(new_params)}")
        inst = type(self)(**kwargs)
        if include_attributes:
            for k, v in self.__dict__.items():
                if k in kwargs or k in inst.__dict__:
                    continue
                setattr(inst, k, copy.deepcopy(v))
        return inst

    # ── mutate ────────────────────────────────────────────────────
    def mutate(self, new_attrs: dict[str, Any]) -> Base:
        """Change declared-mutable attributes in place.

        Recursive into nested estimators. Names not listed in
        ``_mutable_attributes``, or missing, raise ``ValueError``.
        """
        for k, v in new_attrs.items():
            if k not in self._mutable_attributes:
                raise ValueError(f"{type(self).__name__}: {k!r} is not mutable")
            if not hasattr(self, k):
                raise ValueError(f"{type(self).__name__}: {k!r} does not exist")
            cur = getattr(self, k)
            if isinstance(cur, Base) and isinstance(v, dict):
                cur.mutate(v)
            else:
                setattr(self, k, v)
        return self

    # ── stochastic ────────────────────────────────────────────────
    def _is_stochastic(self) -> bool:
        """True if any parameter named ``seed`` at any depth is None."""
        for name in self._init_signature().parameters:
            if name == "seed" and getattr(self, name, None) is None:
                return True
            v = getattr(self, name, None)
            if isinstance(v, Base) and v._is_stochastic():
                return True
        return False

    # ── pickling ──────────────────────────────────────────────────
    def __getstate__(self) -> dict[str, Any]:
        ver = _get_version()
        state = self.__dict__.copy()
        state[_VERSION_KEY] = ver
        return state

    def __setstate__(self, state: Any) -> None:
        from dense_armor import __version__ as ver

        if isinstance(state, tuple):
            dict_state, slots_state = state
        else:
            dict_state, slots_state = state, None
        orig = dict_state.pop(_VERSION_KEY, None)
        if orig is not None and orig != ver:
            warnings.warn(
                f"{type(self).__name__} was pickled with dense-armor "
                f"{orig}, current is {ver}.",
                InconsistentVersionWarning,
                stacklevel=2,
            )
        self.__dict__.update(dict_state)
        if slots_state:
            for k, v in slots_state.items():
                setattr(self, k, v)

    # ── memory ────────────────────────────────────────────────────
    @property
    def _raw_memory_usage(self) -> int:
        """Bytes of the object graph, each object counted once.

        Breadth-first walk with ``gc.get_referents``; classes, modules
        and functions are skipped (they would pull in the interpreter);
        ``ndarray.nbytes`` is added for numpy arrays (not GC-tracked).
        """
        seen: set = set()
        queue = [self]
        total = 0
        while queue:
            obj = queue.pop()
            oid = id(obj)
            if oid in seen:
                continue
            seen.add(oid)
            if isinstance(obj, type) or inspect.ismodule(obj):
                continue
            if inspect.isfunction(obj) or inspect.ismethod(obj):
                continue
            try:
                total += sys.getsizeof(obj)
            except TypeError:
                continue
            if isinstance(obj, np.ndarray):
                total += obj.nbytes
                continue
            try:
                refs: list = list(gc.get_referents(obj))
            except Exception:  # noqa: BLE001
                refs = []
            for r in refs:
                if id(r) not in seen:
                    queue.append(r)
        return total

    def _memory_usage(self) -> str:
        n: float = float(self._raw_memory_usage)
        for unit in ("B", "KiB", "MiB", "GiB"):
            if n < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} TiB"

    # ── repr ──────────────────────────────────────────────────────
    def _repr_at_depth(self, depth: int, module_prefix: bool = False) -> str:
        cls = type(self)
        name = f"{cls.__module__}.{cls.__name__}" if module_prefix else cls.__name__
        lines = [f"{name} ("]
        for pname, p in self._init_signature().parameters.items():
            if pname == "self" or p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if not hasattr(self, pname):
                continue
            v = getattr(self, pname)
            lines.append(
                "    " * (depth + 1) + f"{pname}={_short_repr(v, depth, module_prefix)}"
            )
        lines.append("    " * depth + ")")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self._repr_at_depth(0, module_prefix=False)

    def __str__(self) -> str:
        return type(self).__name__

    # ── call logging ──────────────────────────────────────────────
    @staticmethod
    @contextmanager
    def _call_logger(
        klass: type | None = None, method: str | None = None
    ) -> Iterator[None]:
        """Log every public method call at DEBUG level for the duration.

        Args:
            klass: restrict logging to instances of this class if given.
            method: restrict logging to calls of this method name if given.
        """
        log = logging.getLogger("dense_armor.base")
        old_level = log.level
        log.setLevel(logging.DEBUG)

        def collect(kls, seen=None):
            if seen is None:
                seen = set()
            if kls in seen:
                return seen
            seen.add(kls)
            for sub in kls.__subclasses__():
                collect(sub, seen)
            return seen

        def make_patched():
            def patched(self, name):
                attr = object.__getattribute__(self, name)
                if name.startswith("_") or not callable(attr):
                    return attr
                if klass is not None and klass not in type(self).__mro__:
                    return attr
                if method is not None and name != method:
                    return attr

                def wrapper(*a, **kw):
                    log.debug("%s.%s", type(self).__name__, name)
                    return attr(*a, **kw)

                wrapper.__name__ = name
                return wrapper

            return patched

        targets = list(collect(Base))
        saved: dict[type, Any] = {}
        for k in targets:
            saved[k] = k.__dict__.get("__getattribute__")
            k.__getattribute__ = make_patched()  # type: ignore[method-assign]

        try:
            yield
        finally:
            for k, v in saved.items():
                if v is None:
                    try:
                        delattr(k, "__getattribute__")
                    except AttributeError:
                        pass
                else:
                    k.__getattribute__ = v  # type: ignore[method-assign]
            log.setLevel(old_level)

    # ── unit-test hooks ───────────────────────────────────────────
    @classmethod
    def _unit_test_params(cls) -> Iterator[dict[str, Any]]:
        """Yield constructor kwargs used by the checks. Empty by default."""
        yield {}

    def _unit_test_skips(self) -> set:
        return set()

    # ── learned-state detection ───────────────────────────────────
    def _has_learned(self) -> bool:
        """True if any instance attribute has a trailing underscore."""
        for k in self.__dict__:
            if k.startswith("__") and k.endswith("__"):
                continue
            if k.endswith("_"):
                return True
        return False

    # ── state_dict ────────────────────────────────────────────────
    def state_dict(self) -> dict[str, Any]:
        """Return the learned state (trailing-underscore attributes)."""
        return {
            k: copy.deepcopy(v)
            for k, v in self.__dict__.items()
            if k.endswith("_") and not (k.startswith("__") and k.endswith("__"))
        }

    def load_state_dict(self, state: dict[str, Any]) -> Base:
        """Restore the learned state from ``state_dict()``."""
        for k, v in state.items():
            setattr(self, k, copy.deepcopy(v))
        return self

    # ── robot time base ───────────────────────────────────────────
    _DT_N = 64

    def _time_step(self, t: float | None) -> float | None:
        """Record a timestamp; return the running median dt (or None)."""
        if t is None:
            return None
        buf = self.__dict__.setdefault("_dt_buf", [])
        last = self.__dict__.get("_t_last")
        if last is not None:
            dt = t - last
            if dt <= 0:
                raise ValueError(f"non-increasing t: previous {last}, got {t}")
            buf.append(dt)
            if len(buf) > self._DT_N:
                del buf[: len(buf) - self._DT_N]
        self.__dict__["_t_last"] = t
        return self.dt

    @property
    def dt(self) -> float | None:
        buf = self.__dict__.get("_dt_buf")
        if not buf:
            return None
        return float(np.median(buf))

    @property
    def rate(self) -> float | None:
        d = self.dt
        return None if d is None else 1.0 / d

    @property
    def jitter(self) -> float | None:
        buf = self.__dict__.get("_dt_buf")
        if not buf:
            return None
        arr = np.asarray(buf)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        return 1.4826 * mad

    def window_samples(self, seconds: float) -> int:
        """Convert a window given in seconds to a number of samples."""
        d = self.dt
        if d is None:
            return max(1, round(seconds))
        return max(1, round(seconds / d))

    # ── describe (LLM tool) ───────────────────────────────────────
    def describe(self) -> dict[str, Any]:
        """JSON-serialisable description of the estimator."""
        import json as _json

        params: dict[str, Any] = {}
        for pname, p in self._init_signature().parameters.items():
            if pname == "self" or p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            default = None if p.default is inspect.Parameter.empty else p.default
            try:
                _json.dumps(default)
                dflt = default
            except Exception:  # noqa: BLE001
                dflt = repr(default)
            ann = p.annotation
            if ann is inspect.Parameter.empty:
                tname = "Any"
            else:
                tname = getattr(ann, "__name__", str(ann))
            params[pname] = {"type": tname, "default": dflt}
        methods = [
            m
            for m in (
                "learn_one",
                "predict_one",
                "predict_proba_one",
                "score_one",
                "transform_one",
                "update",
                "learn_many",
                "predict_many",
                "predict_proba_many",
                "score_many",
                "transform_many",
            )
            if callable(getattr(self, m, None))
        ]
        return {
            "name": type(self).__name__,
            "module": type(self).__module__,
            "parameters": params,
            "methods": methods,
            "input_schema": {"type": "object", "description": "feature dict -> number"},
            "output_schema": {"type": "any"},
        }
