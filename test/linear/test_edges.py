"""Edge cases that raise the documented errors.

The constructor validators and the input guards of the optimizers and
the linear models raise ValueError or TypeError on bad inputs. Each
raise is a real path a caller can hit; the tests here hit them once, so
the branch is covered and the message is asserted.
"""

import pytest

from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.utility.linear.bayes import BayesianLinearRegression
from dense_armor.utility.linear.glm import LinearRegression
from dense_armor.utility.linear.payload import PayloadEstimator
from dense_armor.utility.optim import (
    SGD,
    AdaGrad,
    Adam,
    FTRLProximal,
    OnlineNewtonStep,
)


def test_sgd_rejects_non_positive_lr():
    with pytest.raises(ValueError, match="lr"):
        SGD(lr=0.0)


def test_adagrad_rejects_non_positive_lr():
    with pytest.raises(ValueError, match="lr"):
        AdaGrad(lr=-0.1)


def test_adagrad_rejects_non_positive_eps():
    with pytest.raises(ValueError, match="eps"):
        AdaGrad(eps=0.0)


def test_adam_rejects_bad_hyperparameters():
    with pytest.raises(ValueError, match="lr"):
        Adam(lr=0.0)
    with pytest.raises(ValueError, match="beta1"):
        Adam(beta1=1.5)
    with pytest.raises(ValueError, match="beta2"):
        Adam(beta2=-0.1)
    with pytest.raises(ValueError, match="eps"):
        Adam(eps=0.0)


def test_ftrl_rejects_bad_hyperparameters():
    with pytest.raises(ValueError, match="alpha"):
        FTRLProximal(alpha=0.0)
    with pytest.raises(ValueError, match="beta"):
        FTRLProximal(beta=2.0)
    with pytest.raises(ValueError, match="l1"):
        FTRLProximal(l1=-0.1)
    with pytest.raises(ValueError, match="l2"):
        FTRLProximal(l2=-0.1)


def test_newton_rejects_bad_hyperparameters():
    with pytest.raises(ValueError, match="gamma"):
        OnlineNewtonStep(gamma=0.0)
    with pytest.raises(ValueError, match="eps"):
        OnlineNewtonStep(eps=-1.0)


def test_optimizer_eq_and_repr():
    for opt, other_same, other_diff in [
        (SGD(lr=0.1), SGD(lr=0.1), SGD(lr=0.2)),
        (AdaGrad(lr=0.5), AdaGrad(lr=0.5), AdaGrad(lr=0.6)),
        (Adam(lr=0.01), Adam(lr=0.01), Adam(lr=0.02)),
        (FTRLProximal(alpha=1.0), FTRLProximal(alpha=1.0), FTRLProximal(alpha=2.0)),
        (
            OnlineNewtonStep(gamma=1.0),
            OnlineNewtonStep(gamma=1.0),
            OnlineNewtonStep(gamma=2.0),
        ),
    ]:
        assert opt == other_same
        assert opt != other_diff
        assert opt != "not-an-optimizer"
        assert type(opt).__name__ in repr(opt)


def test_glm_rejects_non_dict_input():
    reg = LinearRegression(optimizer=SGD(lr=0.1))
    reg.learn_one({"x": 1.0}, 0.0)
    with pytest.raises(TypeError, match="dict"):
        reg.predict_one(42)


def test_bayes_rejects_bad_alpha_beta():
    with pytest.raises(ValueError, match="alpha"):
        BayesianLinearRegression(alpha=0.0)
    with pytest.raises(ValueError, match="beta"):
        BayesianLinearRegression(beta=-1.0)


def test_bayes_rejects_non_dict_input():
    model = BayesianLinearRegression()
    with pytest.raises(TypeError, match="dict"):
        model.predict_one([1.0, 2.0])


def test_payload_rejects_bad_lam_delta():
    model = RigidBodyModel("test/fixtures/urdf/panda.urdf")
    with pytest.raises(ValueError, match="lam"):
        PayloadEstimator(model, link="panda_link7", lam=0.0)
    with pytest.raises(ValueError, match="delta"):
        PayloadEstimator(model, link="panda_link7", delta=-1.0)


def test_payload_rejects_unknown_link():
    model = RigidBodyModel("test/fixtures/urdf/panda.urdf")
    with pytest.raises(ValueError, match="unknown link"):
        PayloadEstimator(model, link="not_a_link")


def test_payload_extract_returns_none_on_missing_channels():
    model = RigidBodyModel("test/fixtures/urdf/panda.urdf")
    est = PayloadEstimator(model, link="panda_link7")
    assert est.learn_one({"x": 1.0}) is est
    assert est.predict_one({}) == 0.0
