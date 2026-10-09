"""Edge paths of the datasets."""

from pathlib import Path

import numpy as np
import pytest

from dense_armor.utility.datasets import DriftStream, SyntheticArm
from dense_armor.utility.datasets.casper import _first_timestamp

URDF = Path("test/fixtures/urdf/panda.urdf")


def test_synthetic_arm_bad_arguments_and_friction():
    for kw in (
        {"period_s": 0.0},
        {"rate_hz": 0.0},
        {"n_cycles": 0},
        {"payload_link": "nope"},
    ):
        base = {"period_s": 1.0, "rate_hz": 10.0}
        base.update(kw)
        with pytest.raises(ValueError):
            SyntheticArm(URDF, **base)
    ds = SyntheticArm(
        URDF,
        period_s=1.0,
        rate_hz=10.0,
        fault_at_s=0.0,
        fault="friction",
        fault_joint=0,
    )
    q, qd, _ = ds.trajectory(0.3)
    out = ds._fault_term(q, qd, 0.3)
    assert out[0] == pytest.approx(ds.friction_b * qd[0]) and not out[1:].any()


def test_casper_first_timestamp_edges(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("Timestamp,a\n")
    assert _first_timestamp(p, "Timestamp") is None
    p.write_text("Timestamp,a\n1.5,2\n")
    assert _first_timestamp(p, "Timestamp") == 1.5
    assert _first_timestamp(p, "zz") is None
    p.write_text("Timestamp,a\nx,2\n")
    assert _first_timestamp(p, "Timestamp") is None


def test_incremental_drift_moves_the_clusters():
    ds = DriftStream(
        n_samples=2000, n_features=2, n_drifts=1, kind="incremental", seed=0
    )
    X, _ = ds.data()
    assert np.isfinite(X).all()
    assert set(np.unique(ds.concepts_)) == {0, 1}
