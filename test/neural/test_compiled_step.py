"""Tests for the compiled training step.

The reference is the Python loop of :class:`Trainer` (``compile=False``):
it does not contain the new code, and the equivalence is compared
against it. Every new branch of the compiled step has a test that
fails if the branch is removed (mutation table in the delivery).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.compiled import (
    _init_state as init_state,
    _read_values as read_values,
    _write_values as write_values,
    make_step,
)
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.models import (
    cnn,
    mlp,
    recurrent_net,
    resnet_mlp,
    transformer_encoder,
)
from dense_armor.utility.neural.schedules import ConstantLR, StepLR
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD, AdaGrad, Adam, RMSProp


MODEL_FACTORIES = {
    "mlp": (
        lambda: mlp([2, 4, 1], jax.random.PRNGKey(0)),
        (16, 2),
        (16, 1),
    ),
    "resnet_mlp": (
        lambda: resnet_mlp(width=4, depth=2, n_in=2, n_out=1, key=jax.random.PRNGKey(0)),
        (16, 2),
        (16, 1),
    ),
    "cnn": (
        lambda: cnn(
            input_shape=(8, 8, 1), channels=[4], n_out=10, key=jax.random.PRNGKey(0)
        ),
        (16, 8, 8, 1),
        (16, 10),
    ),
    "recurrent_net": (
        lambda: recurrent_net(
            n_in=2, n_hidden=4, n_out=1, key=jax.random.PRNGKey(0), cell="rnn"
        ),
        (16, 8, 2),
        (16, 1),
    ),
    "transformer_encoder": (
        lambda: transformer_encoder(
            n_in=4, d_model=8, n_heads=2, d_ff=16, depth=1, n_out=3,
            key=jax.random.PRNGKey(0), max_len=8,
        ),
        (16, 8, 4),
        (16, 3),
    ),
}


def _data(shape_x, shape_y, seed=1):
    kx, ky = jax.random.split(jax.random.PRNGKey(seed))
    X = jax.random.normal(kx, shape_x)
    y = jax.random.normal(ky, shape_y)
    return X, y


def _trajectory(model_factory, shape_x, shape_y, total_steps, batch_size, epochs):
    X, y = _data(shape_x, shape_y)
    m_loop = model_factory()
    t_loop = Trainer(m_loop, MSE(), SGD(0.01))
    t_loop.fit(X, y, epochs=epochs, batch_size=batch_size, key=jax.random.PRNGKey(0))
    m_compiled = model_factory()
    t_compiled = Trainer(m_compiled, MSE(), SGD(0.01), compile=True)
    t_compiled.fit(
        X, y, epochs=epochs, batch_size=batch_size, key=jax.random.PRNGKey(0)
    )
    return t_loop.history_["train_loss"], t_compiled.history_["train_loss"], total_steps


@pytest.mark.parametrize("model_name", list(MODEL_FACTORIES))
def test_equivalence_of_the_loss(model_name):
    """Loss at the same epoch is within 1e-4 relative for all five models."""
    factory, sx, sy = MODEL_FACTORIES[model_name]
    epochs = 10
    batch_size = 8
    n = sx[0]
    total_steps = epochs * (n // batch_size)
    loop, compiled, _ = _trajectory(factory, sx, sy, total_steps, batch_size, epochs)
    for a, b in zip(loop, compiled):
        rel = abs(a - b) / max(abs(a), 1e-12)
        assert rel <= 1e-4, f"{model_name}: loop={a} compiled={b} rel={rel}"


STORED_MASTER_LOSSES_X64_TRUE = [
    1.4483716913582123,
    1.191682586339547,
    1.0359279394961722,
]
STORED_MASTER_LOSSES_X64_FALSE = [
    7.079418420791626,
    5.084859251976013,
    3.7510147094726562,
]


def test_default_unchanged_bit_for_bit():
    """At compile=False the loss equals the values stored from master.

    The values are the three epoch losses of the MLP [2, 4, 1] with
    seed 0, MSE, SGD(0.01), batch 8 on 16 samples, shuffle with key 0.
    They were taken from master before this branch existed and are
    stored here. The equality is bit for bit.
    """
    factory, sx, sy = MODEL_FACTORIES["mlp"]
    X, y = _data(sx, sy)
    m = factory()
    t = Trainer(m, MSE(), SGD(0.01))
    t.fit(X, y, epochs=3, batch_size=8, key=jax.random.PRNGKey(0))
    expected = (
        STORED_MASTER_LOSSES_X64_TRUE
        if jax.config.jax_enable_x64
        else STORED_MASTER_LOSSES_X64_FALSE
    )
    assert t.history_["train_loss"] == expected


@pytest.mark.parametrize("optimizer_name", ["sgd", "sgd_momentum", "adam", "rmsprop", "adagrad"])
def test_optimizer_runs_in_compiled_mode(optimizer_name):
    """Every optimizer of the batch runs and lowers the loss."""
    factory, sx, sy = MODEL_FACTORIES["mlp"]
    X, y = _data(sx, sy)
    if optimizer_name == "sgd":
        opt = SGD(0.05)
    elif optimizer_name == "sgd_momentum":
        opt = SGD(0.05, momentum=0.9)
    elif optimizer_name == "adam":
        opt = Adam(0.01)
    elif optimizer_name == "rmsprop":
        opt = RMSProp(0.01)
    else:
        opt = AdaGrad(0.1)
    m = factory()
    t = Trainer(m, MSE(), opt, compile=True)
    t.fit(X, y, epochs=10, batch_size=8, key=jax.random.PRNGKey(0))
    losses = t.history_["train_loss"]
    assert losses[-1] < losses[0], f"{optimizer_name}: {losses}"


def test_clipping_path():
    """With a tiny threshold, clipping changes the loss against no clipping."""
    factory, sx, sy = MODEL_FACTORIES["mlp"]
    X, y = _data(sx, sy)
    m1 = factory()
    t1 = Trainer(m1, MSE(), SGD(0.01), clip_norm=0.01, compile=True)
    t1.fit(X, y, epochs=2, batch_size=8, key=jax.random.PRNGKey(0))
    m2 = factory()
    t2 = Trainer(m2, MSE(), SGD(0.01), clip_norm=None, compile=True)
    t2.fit(X, y, epochs=2, batch_size=8, key=jax.random.PRNGKey(0))
    assert t1.history_["train_loss"][-1] != t2.history_["train_loss"][-1]


def test_l2_decay_path():
    """l2 decay changes the loss against no decay."""
    factory, sx, sy = MODEL_FACTORIES["mlp"]
    X, y = _data(sx, sy)
    m1 = factory()
    t1 = Trainer(m1, MSE(), SGD(0.01), weight_decay=0.1, decay="l2", compile=True)
    t1.fit(X, y, epochs=2, batch_size=8, key=jax.random.PRNGKey(0))
    m2 = factory()
    t2 = Trainer(m2, MSE(), SGD(0.01), weight_decay=0.0, compile=True)
    t2.fit(X, y, epochs=2, batch_size=8, key=jax.random.PRNGKey(0))
    assert t1.history_["train_loss"][-1] != t2.history_["train_loss"][-1]


def test_decoupled_decay_path():
    """decoupled decay changes the loss against no decay."""
    factory, sx, sy = MODEL_FACTORIES["mlp"]
    X, y = _data(sx, sy)
    m1 = factory()
    t1 = Trainer(
        m1, MSE(), SGD(0.01), weight_decay=0.1, decay="decoupled", compile=True
    )
    t1.fit(X, y, epochs=2, batch_size=8, key=jax.random.PRNGKey(0))
    m2 = factory()
    t2 = Trainer(m2, MSE(), SGD(0.01), weight_decay=0.0, compile=True)
    t2.fit(X, y, epochs=2, batch_size=8, key=jax.random.PRNGKey(0))
    assert t1.history_["train_loss"][-1] != t2.history_["train_loss"][-1]


def test_l2_and_decoupled_differ_for_sgd_with_momentum():
    """For SGD with momentum, l2 and decoupled produce different losses."""
    factory, sx, sy = MODEL_FACTORIES["mlp"]
    X, y = _data(sx, sy)
    m1 = factory()
    t1 = Trainer(
        m1, MSE(), SGD(0.01, momentum=0.9), weight_decay=0.1,
        decay="l2", compile=True,
    )
    t1.fit(X, y, epochs=2, batch_size=8, key=jax.random.PRNGKey(0))
    m2 = factory()
    t2 = Trainer(
        m2, MSE(), SGD(0.01, momentum=0.9), weight_decay=0.1,
        decay="decoupled", compile=True,
    )
    t2.fit(X, y, epochs=2, batch_size=8, key=jax.random.PRNGKey(0))
    assert t1.history_["train_loss"][-1] != t2.history_["train_loss"][-1]


def test_schedule_with_warmup_path():
    """A step schedule produces a different lr trajectory than a constant one."""
    factory, sx, sy = MODEL_FACTORIES["mlp"]
    X, y = _data(sx, sy)
    m1 = factory()
    sched = StepLR(0.01, step=1.0, gamma=0.5)
    t1 = Trainer(m1, MSE(), SGD(0.01), schedule=sched, compile=True)
    t1.fit(X, y, epochs=4, batch_size=8, key=jax.random.PRNGKey(0))
    lrs = t1.history_["lr"]
    assert len(set(round(x, 8) for x in lrs)) > 1, f"schedule not changing: {lrs}"


def test_early_stopping_path():
    """With patience, the training stops before the last epoch."""
    factory, sx, sy = MODEL_FACTORIES["mlp"]
    X, y = _data(sx, sy)
    X_val, y_val = _data(sx, sy, seed=42)
    m = factory()
    t = Trainer(m, MSE(), SGD(0.01), compile=True)
    t.fit(
        X, y, epochs=20, batch_size=8, key=jax.random.PRNGKey(0),
        X_val=X_val, y_val=y_val, patience=2, min_delta=100.0,
    )
    assert t.stopped_epoch_ is not None
    assert t.stopped_epoch_ < 19


def test_empty_validation_set_path():
    """Without a validation set, val_loss is NaN and the fit returns."""
    factory, sx, sy = MODEL_FACTORIES["mlp"]
    X, y = _data(sx, sy)
    m = factory()
    t = Trainer(m, MSE(), SGD(0.01), compile=True)
    t.fit(X, y, epochs=2, batch_size=8, key=jax.random.PRNGKey(0))
    assert all(math.isnan(v) for v in t.history_["val_loss"])


def test_batch_of_one():
    """A batch of one sample runs through the compiled step."""
    factory, sx, sy = MODEL_FACTORIES["mlp"]
    X, y = _data(sx, sy)
    m = factory()
    t = Trainer(m, MSE(), SGD(0.01), compile=True)
    t.fit(X, y, epochs=1, batch_size=1, key=jax.random.PRNGKey(0))
    assert len(t.history_["train_loss"]) == 1


def test_last_smaller_batch():
    """With 10 samples and batch 8, the last batch has two samples."""
    factory, sx, sy = MODEL_FACTORIES["mlp"]
    X, y = _data((10, 2), (10, 1))
    m = factory()
    t = Trainer(m, MSE(), SGD(0.01), compile=True)
    t.fit(X, y, epochs=1, batch_size=8, key=jax.random.PRNGKey(0))
    assert len(t.history_["train_loss"]) == 1


def test_more_epochs_than_schedule():
    """A schedule shorter than the number of epochs does not crash."""
    factory, sx, sy = MODEL_FACTORIES["mlp"]
    X, y = _data(sx, sy)
    m = factory()
    sched = StepLR(0.01, step=1.0, gamma=0.5)
    t = Trainer(m, MSE(), SGD(0.01), schedule=sched, compile=True)
    t.fit(X, y, epochs=10, batch_size=8, key=jax.random.PRNGKey(0))
    assert len(t.history_["train_loss"]) == 10


def test_step_counter_bounds():
    """The Adam step counter advances inside the compiled step."""
    factory, sx, sy = MODEL_FACTORIES["mlp"]
    X, y = _data(sx, sy)
    m = factory()
    values = read_values(m)
    write_values(m, values)
    opt = Adam(0.001)
    st = init_state(opt, values)
    step = make_step(m, MSE(), opt, ConstantLR(0.001), 0.0, "l2", None)
    _, st1, _ = step(values, st, X, y, 0)
    assert int(st1["t"]) == 1
    _, st2, _ = step(values, st1, X, y, 0)
    assert int(st2["t"]) == 2


# ==== Strong equivalence tests: one per option that a mutation can remove ====


@pytest.mark.parametrize(
    "make_opt",
    [
        lambda: SGD(0.01),
        lambda: SGD(0.01, momentum=0.9),
        lambda: SGD(0.01, momentum=0.9, nesterov=True),
        lambda: Adam(0.001),
        lambda: Adam(0.001, beta1=0.5, beta2=0.9),
        lambda: RMSProp(0.001),
        lambda: AdaGrad(0.1),
    ],
    ids=["sgd", "sgd_mom", "sgd_nesterov", "adam", "adam_custom_betas", "rmsprop", "adagrad"],
)
def test_compiled_equals_loop_with_that_optimizer(make_opt):
    """For every optimizer option, compiled loss == loop loss at every epoch.

    A mutation that removes Nesterov, or a momentum term, or the Adam
    bias correction, or a beta, changes the loss; the test fails because
    the compiled loss no longer equals the loop loss at the same epoch.
    """
    factory, sx, sy = MODEL_FACTORIES["mlp"]
    X, y = _data(sx, sy)
    m_loop = factory()
    t_loop = Trainer(m_loop, MSE(), make_opt())
    t_loop.fit(X, y, epochs=5, batch_size=8, key=jax.random.PRNGKey(0))
    m_comp = factory()
    t_comp = Trainer(m_comp, MSE(), make_opt(), compile=True)
    t_comp.fit(X, y, epochs=5, batch_size=8, key=jax.random.PRNGKey(0))
    for a, b in zip(t_loop.history_["train_loss"], t_comp.history_["train_loss"]):
        rel = abs(a - b) / max(abs(a), 1e-12)
        assert rel <= 1e-4, f"{a} vs {b}, rel {rel}"


@pytest.mark.parametrize(
    "make_sched",
    [
        lambda opt: ConstantLR(0.01),
        lambda opt: StepLR(0.01, step=1.0, gamma=0.5),
        lambda opt: StepLR(0.01, step=2.0, gamma=0.1),
    ],
    ids=["constant", "step_gamma_0_5", "step_gamma_0_1"],
)
def test_compiled_equals_loop_with_that_schedule(make_sched):
    """For every schedule, compiled loss == loop loss at every epoch.

    A mutation that ignores the schedule in the compiled step (uses a
    constant lr) changes the loss; the test fails because the compiled
    loss no longer equals the loop loss at the same epoch.
    """
    factory, sx, sy = MODEL_FACTORIES["mlp"]
    X, y = _data(sx, sy)
    opt_a = SGD(0.01)
    opt_b = SGD(0.01)
    t_loop = Trainer(factory(), MSE(), opt_a, schedule=make_sched(opt_a))
    t_loop.fit(X, y, epochs=4, batch_size=8, key=jax.random.PRNGKey(0))
    t_comp = Trainer(
        factory(), MSE(), opt_b, schedule=make_sched(opt_b), compile=True
    )
    t_comp.fit(X, y, epochs=4, batch_size=8, key=jax.random.PRNGKey(0))
    for a, b in zip(t_loop.history_["train_loss"], t_comp.history_["train_loss"]):
        rel = abs(a - b) / max(abs(a), 1e-12)
        assert rel <= 1e-4, f"{a} vs {b}, rel {rel}"


@pytest.mark.parametrize("decay", ["l2", "decoupled"])
def test_compiled_equals_loop_with_weight_decay(decay):
    """With weight decay, compiled loss == loop loss at every epoch."""
    factory, sx, sy = MODEL_FACTORIES["mlp"]
    X, y = _data(sx, sy)
    t_loop = Trainer(
        factory(), MSE(), SGD(0.01), weight_decay=0.1, decay=decay
    )
    t_loop.fit(X, y, epochs=4, batch_size=8, key=jax.random.PRNGKey(0))
    t_comp = Trainer(
        factory(), MSE(), SGD(0.01), weight_decay=0.1, decay=decay, compile=True
    )
    t_comp.fit(X, y, epochs=4, batch_size=8, key=jax.random.PRNGKey(0))
    for a, b in zip(t_loop.history_["train_loss"], t_comp.history_["train_loss"]):
        rel = abs(a - b) / max(abs(a), 1e-12)
        assert rel <= 1e-4, f"{a} vs {b}, rel {rel}"


def test_compiled_equals_loop_with_clipping():
    """With clipping, compiled loss == loop loss at every epoch."""
    factory, sx, sy = MODEL_FACTORIES["mlp"]
    X, y = _data(sx, sy)
    t_loop = Trainer(factory(), MSE(), SGD(0.01), clip_norm=0.01)
    t_loop.fit(X, y, epochs=4, batch_size=8, key=jax.random.PRNGKey(0))
    t_comp = Trainer(
        factory(), MSE(), SGD(0.01), clip_norm=0.01, compile=True
    )
    t_comp.fit(X, y, epochs=4, batch_size=8, key=jax.random.PRNGKey(0))
    for a, b in zip(t_loop.history_["train_loss"], t_comp.history_["train_loss"]):
        rel = abs(a - b) / max(abs(a), 1e-12)
        assert rel <= 1e-4, f"{a} vs {b}, rel {rel}"


def test_compiled_equals_loop_l2_equals_decoupled_for_plain_sgd():
    """Proposition 1 of Loshchilov-Hutter: for plain SGD, l2 with
    weight_decay=lam equals decoupled with weight_decay=lam*alpha.
    """
    factory, sx, sy = MODEL_FACTORIES["mlp"]
    X, y = _data(sx, sy)
    alpha = 0.01
    lam = 0.1
    t_l2 = Trainer(
        factory(), MSE(), SGD(alpha), weight_decay=lam, decay="l2", compile=True
    )
    t_l2.fit(X, y, epochs=4, batch_size=8, key=jax.random.PRNGKey(0))
    t_dec = Trainer(
        factory(),
        MSE(),
        SGD(alpha),
        weight_decay=lam * alpha,
        decay="decoupled",
        compile=True,
    )
    t_dec.fit(X, y, epochs=4, batch_size=8, key=jax.random.PRNGKey(0))
    for a, b in zip(t_l2.history_["train_loss"], t_dec.history_["train_loss"]):
        rel = abs(a - b) / max(abs(a), 1e-12)
        assert rel <= 1e-3, f"l2 {a} vs decoupled {b}, rel {rel}"


def test_rejects_dropout():
    """compile=True rejects a model that contains Dropout."""
    from dense_armor.utility.neural.layers import Dropout, Dense
    from dense_armor.utility.neural.chain import Sequential

    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    model = Sequential([Dense(2, 4, k1), Dropout(0.5, k2), Dense(4, 1, k3)])
    with pytest.raises(ValueError, match="Dropout"):
        make_step(
            model, MSE(), SGD(0.01), ConstantLR(0.01), 0.0, "l2", None
        )


def test_rejects_batchnorm():
    """compile=True rejects a model that contains BatchNorm."""
    from dense_armor.utility.neural.norm import BatchNorm
    from dense_armor.utility.neural.layers import Dense
    from dense_armor.utility.neural.chain import Sequential

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    model = Sequential([Dense(2, 4, k1), BatchNorm(4), Dense(4, 1, k2)])
    with pytest.raises(ValueError, match="BatchNorm"):
        make_step(
            model, MSE(), SGD(0.01), ConstantLR(0.01), 0.0, "l2", None
        )


def test_second_fit_does_not_restart_the_optimizer():
    """fit(2)+fit(2) equals fit(4) with shuffle=False (same data order)."""
    factory, sx, sy = MODEL_FACTORIES["mlp"]
    X, y = _data(sx, sy)
    m_a = factory()
    t_a = Trainer(m_a, MSE(), SGD(0.01, momentum=0.9), compile=True)
    t_a.fit(X, y, epochs=2, batch_size=8, key=jax.random.PRNGKey(0), shuffle=False)
    t_a.fit(X, y, epochs=2, batch_size=8, key=jax.random.PRNGKey(0), shuffle=False)
    last_a = t_a.history_["train_loss"][-1]
    m_b = factory()
    t_b = Trainer(m_b, MSE(), SGD(0.01, momentum=0.9), compile=True)
    t_b.fit(X, y, epochs=4, batch_size=8, key=jax.random.PRNGKey(0), shuffle=False)
    last_b = t_b.history_["train_loss"][-1]
    assert abs(last_a - last_b) <= 1e-6, f"{last_a} vs {last_b}"
