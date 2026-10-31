"""Speed of ``Trainer.fit`` with ``compile=False`` and ``compile=True``.

The path a user takes: epochs of ``Trainer.fit`` timed after one warm-up
epoch, 3 sessions, the clock stopped when the parameters are ready.
The mlp has 4 layers of 512 and 10 batches of 256 samples per epoch.
On a GPU the transformer encoder of 4 layers, ``d_model`` 256, sequence
128, has 10 batches of 32 samples per epoch. Synthetic data, seed 0.

The script prints the median and the range of steps per second for each
mode and the ratio of the medians. It writes no file.

Usage:
    python benchmarks/compiled_fit_speed.py
"""

import statistics
import time

import jax

from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.models import mlp, transformer_encoder
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD

SESSIONS = 3
KEY = jax.random.PRNGKey


def _cases():
    cases = {
        "mlp 4x512 batch 256 (10 batches)": (
            lambda: mlp([512] * 5, KEY(0)),
            (256 * 10, 512),
            (256 * 10, 512),
            256,
        )
    }
    if jax.default_backend() != "cpu":
        cases["transformer 4x256 seq128 batch 32 (10 batches)"] = (
            lambda: transformer_encoder(
                n_in=128,
                d_model=256,
                n_heads=8,
                d_ff=1024,
                depth=4,
                n_out=128,
                key=KEY(0),
                max_len=128,
            ),
            (320, 128, 128),
            (320, 128),
            32,
        )
    return cases


def main() -> None:
    print("backend", jax.default_backend())
    for name, (make, shape_x, shape_y, batch) in _cases().items():
        kx, ky = jax.random.split(KEY(0))
        X = jax.random.normal(kx, shape_x)
        y = jax.random.normal(ky, shape_y)
        n_batches = shape_x[0] // batch
        medians = {}
        for compile in (False, True):
            model = make()
            trainer = Trainer(model, MSE(), SGD(0.01), compile=compile)
            trainer.fit(X, y, epochs=1, batch_size=batch, key=KEY(0))
            sps = []
            for s in range(SESSIONS):
                start = time.perf_counter()
                trainer.fit(X, y, epochs=1, batch_size=batch, key=KEY(s))
                jax.block_until_ready([p.value for p in model.params()])
                sps.append(n_batches / (time.perf_counter() - start))
            medians[compile] = statistics.median(sps)
            print(
                name,
                f"compile={compile}",
                f"steps/s median {medians[compile]:.2f}",
                f"range [{min(sps):.2f}, {max(sps):.2f}]",
            )
        print(f"   ratio {medians[True] / medians[False]:.2f}x")


if __name__ == "__main__":
    main()
