import asyncio
import pickle
import subprocess
import sys

import pytest

import dense_armor
from dense_armor.core import visualizer
from dense_armor.roles.safety import SafeEstimator
from dense_armor.utility.drift.cusum import one_sided_arl


class _Model:
    def __init__(self):
        self.w = 1.0

    def state_dict(self):
        return {"w": self.w}

    def load_state_dict(self, d):
        self.w = d["w"]


def test_engine_signature_follows_package_version():
    assert visualizer.AIEngineVisualizer.ENGINE_SIGNATURE == f"dense-armor-v{dense_armor.__version__}"


def test_dynamics_import_does_not_load_osqp():
    code = "import sys, dense_armor.dynamics; print('osqp' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
    assert out == "False"


def test_checkpoint_envelope_refuses_globals(tmp_path):
    p = tmp_path / "bad.pkl"
    p.write_bytes(pickle.dumps(print))
    est = SafeEstimator(_Model())
    with pytest.raises(pickle.UnpicklingError):
        est.restore(p)


def test_checkpoint_with_key_roundtrip_and_wrong_key(tmp_path):
    est = SafeEstimator(_Model())
    est.model.w = 5.0
    p = est.save(tmp_path / "k.pkl", key=b"secret")
    fresh = SafeEstimator(_Model())
    fresh.restore(p, key=b"secret")
    assert fresh.model.w == 5.0
    with pytest.raises(ValueError):
        SafeEstimator(_Model()).restore(p, key=b"other")
    with pytest.raises(ValueError):
        SafeEstimator(_Model()).restore(p)


def test_one_sided_arl_huge_exponent_is_inf_without_warning(recwarn):
    assert one_sided_arl(-50.0, 50.0) == float("inf") or one_sided_arl(-50.0, 50.0) > 1e100
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_mcp_stream_sessions_are_capped_and_expire(monkeypatch):
    pytest.importorskip("mcp")
    import dense_armor.mcp_server.server  # noqa: F401
    from dense_armor.mcp_server import tools

    monkeypatch.setattr(tools, "MAX_STREAM_SESSIONS", 2)
    tools._STREAM_SESSIONS.clear()
    tools._STREAM_LAST_SEEN.clear()
    inp = tools.StreamStartInput(n_channels=1)
    run = lambda: asyncio.run(tools.dense_armor_stream_start(inp))  # noqa: E731
    run()
    run()
    assert run().startswith("Error:")
    monkeypatch.setattr(tools, "STREAM_SESSION_TTL_S", -1.0)
    assert not run().startswith("Error:")
    tools._STREAM_SESSIONS.clear()
    tools._STREAM_LAST_SEEN.clear()
