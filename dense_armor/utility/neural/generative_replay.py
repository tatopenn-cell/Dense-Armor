"""Deep generative replay (Shin et al. 2017, sec. 2.3 page 3).

The paper uses a single conditional generator trained with self-replay.
This code uses one generator per task, declared as a deviation: the
generator of task ``i`` is trained only on ``(x_real, y_real)`` of task
``i``, and the solver of task ``j`` is trained on the real batch of task
``j`` plus one fake batch from every generator ``0..j-1``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.utility.neural.activations import Relu
from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.losses import MSE, SoftmaxCrossEntropy
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import Adam


class DeepGenerativeReplay:
    """Deep generative replay (Shin et al. 2017, sec. 2.3 p. 3).

    Args:
        n_x: number of input features.
        n_y: number of classes.
        dim_z: dimension of the noise vector of the generator.
        hidden: hidden units of the solver and the generators.
        n_fake: fake samples per past task at each training step.
        lr: learning rate of Adam, both for the solver and the generators.
        seed: seed of the generator initialisation and of the noise.

    Deviations from the paper:
        - one generator per task, not a single conditional one;
        - the generator is a small MLP with an MSE loss, not a GAN;
        - the solver is a small MLP with a softmax cross-entropy.

    References:
        Shin et al. 2017, Continual Learning with Deep Generative Replay,
        NeurIPS, sec. 2.3 page 3.
    """

    def __init__(
        self,
        n_x: int,
        n_y: int,
        dim_z: int = 8,
        hidden: int = 16,
        n_fake: int = 32,
        lr: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.n_x = int(n_x)
        self.n_y = int(n_y)
        self.dim_z = int(dim_z)
        self.hidden = int(hidden)
        self.n_fake = int(n_fake)
        self.lr = float(lr)
        self.seed = int(seed)
        key = jax.random.PRNGKey(seed)
        self._key = key
        self.solver = Sequential(
            [Dense(n_x, hidden, key), Relu(), Dense(hidden, n_y, key)]
        )
        self.generators: list[Sequential] = []
        self.n_tasks_ = 0
        self.loss = SoftmaxCrossEntropy()

    def _new_generator(self) -> Sequential:
        self._key, sub = jax.random.split(self._key)
        return Sequential(
            [
                Dense(self.dim_z, self.hidden, sub),
                Relu(),
                Dense(self.hidden, self.n_x + self.n_y, sub),
            ]
        )

    def sample(
        self, generator: Sequential, n: int
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample ``n`` fake ``(x, y)`` pairs from ``generator``."""
        self._key, sub = jax.random.split(self._key)
        z = jax.random.normal(sub, (n, self.dim_z))
        out = generator.forward(z, training=False)
        x = out[:, : self.n_x]
        y_logits = out[:, self.n_x :]
        return x, jnp.argmax(y_logits, axis=-1)

    def train_task(
        self,
        x_real: Any,
        y_real: Any,
        n_epochs: int = 5,
        batch_size: int = 32,
    ) -> "DeepGenerativeReplay":
        """Train the solver on the real batch and on the past fake batches.

        Then train a new generator on the real batch of the current task.
        """
        x_real = jnp.asarray(x_real)
        y_real = jnp.asarray(y_real)
        n_real = len(x_real)
        fakes_x: list[jnp.ndarray] = []
        fakes_y: list[jnp.ndarray] = []
        for gen in self.generators:
            xf, yf = self.sample(gen, self.n_fake)
            fakes_x.append(xf)
            fakes_y.append(yf)
        if fakes_x:
            X = jnp.concatenate([x_real] + fakes_x, axis=0)
            Y = jnp.concatenate([y_real] + fakes_y, axis=0)
        else:
            X, Y = x_real, y_real
        trainer = Trainer(self.solver, self.loss, Adam(lr=self.lr))
        trainer.fit(
            X, Y, epochs=n_epochs, batch_size=batch_size, key=self._key
        )
        gen = self._new_generator()
        y_onehot = jax.nn.one_hot(y_real, self.n_y)
        target = jnp.concatenate([x_real, y_onehot], axis=-1)
        trainer_g = Trainer(gen, MSE(), Adam(lr=self.lr))
        for _ in range(n_epochs):
            self._key, sub = jax.random.split(self._key)
            z = jax.random.normal(sub, (n_real, self.dim_z))
            trainer_g.fit(z, target, epochs=1, batch_size=batch_size, key=sub)
        self.generators.append(gen)
        self.n_tasks_ += 1
        return self

    def predict(self, x: Any) -> jnp.ndarray:
        """Return the solver logits on ``x``, shape ``(n, n_y)``."""
        return self.solver.forward(jnp.asarray(x), training=False)
