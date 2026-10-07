"""Tools for LLM agents.

An LLM agent that operates a robot needs two things from each estimator:

- a **schema**: what the estimator accepts, what it returns, its units,
  its parameters, its latency budget and its memory class. The schema
  is a valid JSON document (draft-07), ready for a tool-calling
  interface.
- a **call**: a single function that takes a JSON payload, dispatches
  it to the estimator's public method, and returns a JSON response.

Both functions are pure with respect to the estimator's parameters: they
never change ``get_params()``. The estimator's learned state does change
when ``learn_one`` is dispatched, and that is the point.

The schema is a superset of :meth:`dense_armor.roles.Base.describe`,
which already provides the name, module, parameters, and method list.
The extra keys are ``units`` (from the estimator's ``units`` attribute,
if any), ``budget_s`` and ``memory_class`` (from the robot contract in
:mod:`dense_armor.roles.realtime`), and the input / output schema
boilerplate.

References
----------
JSON Schema draft-07 (IETF draft-handrews-json-schema-01).
"""
import json
from typing import Any, Mapping, Optional

from dense_armor.roles.physics import unit_spec_of
from dense_armor.roles.signal import Signal


DEFAULT_INPUT_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "description": "feature dict -> number or string",
    "additionalProperties": True,
}

DEFAULT_OUTPUT_SCHEMA: Mapping[str, Any] = {
    "description": "any JSON-serialisable value",
}


def _units_dict(est: Any) -> dict:
    spec = unit_spec_of(est)
    if spec is None:
        return {}
    return {
        "inputs": dict(spec.inputs),
        "outputs": dict(spec.outputs),
    }


def schema(est: Any) -> dict:
    """Return a JSON schema describing ``est`` as an LLM tool.

    The result is a JSON-serialisable dict with:

    - ``name``, ``module``: class name and module path;
    - ``parameters``: the same dict as :meth:`describe`, hyper-parameter
      name -> ``{type, default}``;
    - ``methods``: the public methods the estimator exposes;
    - ``units``: ``{"inputs": {...}, "outputs": {...}}`` if the
      estimator declares units, otherwise ``{}``;
    - ``budget_s``, ``memory_class``: the robot contract, if declared;
    - ``input_schema``, ``output_schema``: JSON schema of one call.

    Args:
        est: the estimator to describe.

    Returns:
        A JSON-serialisable dict.

    Examples:
        >>> from dense_armor.roles.agents import schema
        >>> from dense_armor.utility.stats.moments import RunningMoments
        >>> s = schema(RunningMoments())
        >>> s["name"]
        'RunningMoments'
        >>> "learn_one" in s["methods"]
        True
    """
    base = est.describe() if hasattr(est, "describe") else {}
    return {
        "name": type(est).__name__,
        "module": type(est).__module__,
        "parameters": base.get("parameters", {}),
        "methods": list(base.get("methods", [])),
        "units": _units_dict(est),
        "budget_s": getattr(est, "budget_s", None),
        "memory_class": getattr(est, "memory_class", "unknown"),
        "input_schema": dict(DEFAULT_INPUT_SCHEMA),
        "output_schema": dict(DEFAULT_OUTPUT_SCHEMA),
    }


def _to_signal_or_dict(x: Mapping[str, Any]) -> Signal | dict:
    if x is None:
        return {}
    if "__signal__" in x:
        payload = x["__signal__"]
        return Signal(
            values=payload["values"],
            names=payload["names"],
            units=payload.get("units", [""] * len(payload["names"])),
            t=payload.get("t"),
        )
    return dict(x)


def call_json(est: Any, payload: str | Mapping[str, Any]) -> str:
    """Dispatch a JSON payload to a public method and return JSON.

    The payload is either a JSON string or a dict with the keys:

    - ``method`` (required): the method name to call, for example
      ``"learn_one"``, ``"predict_one"``, ``"score_one"``,
      ``"transform_one"``.
    - ``x`` (required for most methods): the input features, a dict of
      scalars. A special form ``{"__signal__": {"values": [...],
      "names": [...], "units": [...], "t": ...}}`` builds a
      :class:`~dense_armor.roles.signal.Signal`.
    - ``y`` (optional): the target for supervised methods.
    - ``t`` (optional): the timestamp, in seconds.

    The response is a JSON string. On success it has the shape
    ``{"ok": true, "result": ...}``. On error it has the shape
    ``{"ok": false, "error": "<message>"}``.

    Args:
        est: the estimator.
        payload: a JSON string or a dict.

    Returns:
        A JSON string.

    Examples:
        >>> from dense_armor.roles.agents import call_json
        >>> from dense_armor.utility.stats.moments import RunningMoments
        >>> import json
        >>> out = call_json(
        ...     RunningMoments(),
        ...     {"method": "learn_one", "x": {"a": 1.0}, "y": 0.0},
        ... )
        >>> json.loads(out)["ok"]
        True
    """
    if isinstance(payload, str):
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError as e:
            return json.dumps({"ok": False, "error": f"invalid JSON: {e}"})
    else:
        obj = dict(payload)
    method_name = obj.get("method")
    if not method_name:
        return json.dumps({"ok": False, "error": "missing 'method' key"})
    if not hasattr(est, method_name):
        return json.dumps({
            "ok": False,
            "error": f"{type(est).__name__} has no method {method_name!r}",
        })
    method = getattr(est, method_name)
    x_raw = obj.get("x")
    x = _to_signal_or_dict(x_raw) if x_raw is not None else {}
    y = obj.get("y")
    t = obj.get("t")
    try:
        if method_name in ("learn_one",):
            if y is None:
                result = method(x, t=t) if t is not None else method(x)
            else:
                result = method(x, y, t=t) if t is not None else method(x, y)
        elif method_name in ("predict_one", "score_one", "transform_one"):
            result = method(x, t=t) if t is not None else method(x)
        else:
            result = method()
    except Exception as e:  # noqa: BLE001
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})
    return json.dumps({"ok": True, "result": _jsonable(result)})


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(item) for item in v]
    if hasattr(v, "tolist"):
        return v.tolist()
    return repr(v)
