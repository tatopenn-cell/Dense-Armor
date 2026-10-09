"""Protection before the model, end to end, on a SyntheticArm payload fault.

The model estimates the offset of the torque residual of joint 1 (torque
minus inverse dynamics), which is about zero while the arm is healthy. At
``fault_at_s`` a 5 kg payload adds its gravity torque. A Hampel scorer,
gated so that it never learns flagged samples, decides what the model may
learn: the protected model keeps the healthy offset, the bare model
absorbs the fault.
"""

from pathlib import Path

from dense_armor.roles.anomaly_detector import AnomalyGate
from dense_armor.roles.protection import Protected
from dense_armor.utility.anomaly.filters import HampelScorer
from dense_armor.utility.datasets import SyntheticArm
from dense_armor.utility.learn.online_dynamics import RecursiveLeastSquares

URDF = Path("test/fixtures/urdf/panda.urdf")


def test_protected_model_does_not_learn_the_payload_fault():
    ds = SyntheticArm(
        URDF,
        period_s=2.0,
        rate_hz=50.0,
        n_cycles=2,
        fault_at_s=2.0,
        fault="payload",
        payload_mass=5.0,
        seed=0,
    )
    gate = AnomalyGate(HampelScorer(radius=10, n_sigmas=5.0, feature="r"))
    prot = Protected(model=RecursiveLeastSquares(feature_keys=["one"]), detector=gate)
    bare = RecursiveLeastSquares(feature_keys=["one"])
    false_alarms = caught = 0
    for sig, y in ds.stream():
        q, qd, qdd = ds.trajectory(sig.t)
        r = float(sig["tau_1"] - ds.nominal_torque(q, qd, qdd)[1])
        x = {"r": r, "one": 1.0}
        flagged = prot._flagged(x)
        false_alarms += int(flagged and y == 0)
        caught += int(flagged and y == 1)
        gate.learn_one(x)
        prot.learn_one(x, r)
        bare.learn_one(x, r)
    assert false_alarms <= 2
    assert caught >= 95
    assert abs(prot.model.predict_one({"one": 1.0})) < 0.05
    assert bare.predict_one({"one": 1.0}) < -5.0
