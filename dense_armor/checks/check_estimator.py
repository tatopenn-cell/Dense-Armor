"""Run every applicable check on an estimator."""

from __future__ import annotations

import json
import pickle
from typing import Any

from dense_armor.base import (
    AnomalyDetector,
    Base,
    Classifier,
    DriftDetector,
    Regressor,
    Transformer,
)


def _unit_params(est: Base) -> list[dict]:
    return list(type(est)._unit_test_params())


def check_repr(est: Base) -> None:
    assert isinstance(repr(est), str) and repr(est)
    assert str(est) == type(est).__name__


def check_repr_clone_equal(est: Base) -> None:
    assert repr(est.clone()) == repr(est)


def check_clone(est: Base) -> None:
    c = est.clone()
    assert type(c) is type(est)
    assert c is not est
    assert c.get_params() == est.get_params()


def check_clone_new_params(est: Base) -> None:
    for name, p in est._init_signature().parameters.items():
        if name == "self" or p.kind in (
            p.VAR_POSITIONAL,
            p.VAR_KEYWORD,
        ):
            continue
        cur = getattr(est, name, None)
        new: Any
        if isinstance(cur, bool):
            new = not cur
        elif isinstance(cur, int):
            new = cur + 1
        elif isinstance(cur, float):
            new = cur + 1.0
        else:
            continue
        c = est.clone(new_params={name: new})
        assert getattr(c, name) == new
        assert getattr(est, name) == cur
        return


def check_clone_independent(est: Base) -> None:
    c = est.clone()
    for k in list(est.__dict__):
        if k.endswith("_") and not k.startswith("__"):
            setattr(c, k, "sentinel")
            assert getattr(est, k) != "sentinel"


def check_get_params_signature(est: Base) -> None:
    sig = est._init_signature()
    expected = {
        n
        for n, p in sig.parameters.items()
        if n != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
    got = set(est.get_params())
    assert expected <= got, f"missing params: {expected - got}"


def check_default_params_non_mutable(est: Base) -> None:
    for kwargs in _unit_params(est):
        for k, v in kwargs.items():
            assert not isinstance(v, (list, dict, set)), (
                f"default param {k} is a mutable object"
            )


def check_mutate_idempotent(est: Base) -> None:
    cur = {k: getattr(est, k) for k in est._mutable_attributes if hasattr(est, k)}
    for k in est._mutable_attributes:
        assert hasattr(est, k), f"declared mutable {k} does not exist"
    if cur:
        est.mutate(cur)


def check_pickle_roundtrip(est: Base) -> None:
    blob = pickle.dumps(est)
    est2 = pickle.loads(blob)
    assert type(est2) is type(est)
    assert repr(est2) == repr(est)


def check_docstring(est: Base) -> None:
    assert type(est).__doc__, f"{type(est).__name__} has no docstring"


def check_describe_serializable(est: Base) -> None:
    d = est.describe()
    json.dumps(d)


def check_state_dict_roundtrip(est: Base) -> None:
    s = est.state_dict()
    est.load_state_dict(s)


def _sample_x() -> dict:
    return {"a": 1.0, "b": 2.0, "c": 3.0}


def check_learn_one_does_not_modify_x(est: Base) -> None:
    x = _sample_x()
    snapshot = dict(x)
    try:
        if isinstance(est, Classifier):
            est.learn_one(x, 0)
        elif isinstance(est, (Regressor,)):
            est.learn_one(x, 0.0)
        elif isinstance(est, (Transformer, AnomalyDetector)):
            est.learn_one(x)
        elif isinstance(est, DriftDetector):
            return
        else:
            return
    except NotImplementedError:
        return
    assert x == snapshot, "learn_one modified the input dict"


def check_predict_before_learning(est: Base) -> None:
    fresh = est.clone() if not est._has_learned() else None
    if fresh is None:
        return
    x = _sample_x()
    if isinstance(fresh, Classifier):
        assert fresh.predict_one(x) is None
        assert fresh.predict_proba_one(x) == {}


def check_shuffle_features(est: Base) -> None:
    return


def check_classifier_proba_sum(est: Base) -> None:
    if not isinstance(est, Classifier):
        return
    x = _sample_x()
    for i in range(20):
        try:
            est.learn_one({"a": float(i % 2), "b": float(i)}, i % 2)
        except NotImplementedError:
            return
    try:
        p = est.predict_proba_one(x)
    except NotImplementedError:
        return
    if not p:
        return
    s = sum(p.values())
    assert abs(s - 1.0) < 1e-9, f"proba sums to {s}"


def check_classifier_multiclass_bool(est: Base) -> None:
    if isinstance(est, Classifier):
        assert isinstance(est._multiclass, bool)


def check_dt_accepted(est: Base) -> None:
    est._time_step(0.0)
    est._time_step(0.01)
    est._time_step(0.02)
    assert est.dt is not None
    assert est.rate is not None
    assert est.jitter is not None


def check_non_increasing_t_raises(est: Base) -> None:
    fresh = est.clone()
    fresh._time_step(0.0)
    fresh._time_step(0.1)
    try:
        fresh._time_step(0.05)
    except ValueError:
        return
    raise AssertionError("non-increasing t did not raise")


def check_estimator(est: Base) -> None:
    """Run every applicable check on ``est``.

    Args:
        est: the estimator to check.

    Raises:
        AssertionError: if any check fails.
    """
    skip = est._unit_test_skips()
    checks = [
        check_repr,
        check_repr_clone_equal,
        check_clone,
        check_clone_new_params,
        check_clone_independent,
        check_get_params_signature,
        check_default_params_non_mutable,
        check_mutate_idempotent,
        check_pickle_roundtrip,
        check_docstring,
        check_describe_serializable,
        check_state_dict_roundtrip,
        check_learn_one_does_not_modify_x,
        check_predict_before_learning,
        check_shuffle_features,
        check_classifier_proba_sum,
        check_classifier_multiclass_bool,
        check_dt_accepted,
        check_non_increasing_t_raises,
    ]
    failures = []
    for chk in checks:
        if chk.__name__.removeprefix("check_") in skip:
            continue
        try:
            chk(est)
        except AssertionError as e:
            failures.append((chk.__name__, str(e)))
        except NotImplementedError:
            pass
        except Exception as e:  # noqa: BLE001
            failures.append((chk.__name__, f"{type(e).__name__}: {e}"))
    if failures:
        lines = "\n".join(f"  - {n}: {m}" for n, m in failures)
        raise AssertionError(f"{type(est).__name__} failed check_estimator:\n{lines}")
