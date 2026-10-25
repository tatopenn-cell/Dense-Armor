"""Figure and timings for the composable attention page.

Synthetic data only. Writes docs/assets/neural/composable_attention.png and
prints, for each brick, median, p90 and max of one call over 200 calls
after 20 warm-ups, and the number of parameters.
"""

from __future__ import annotations

# ruff: noqa: E402
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from dense_armor.utility.neural.bottleneck import BottleneckFusion
from dense_armor.utility.neural.cross_attention import MultiHeadCrossAttention
from dense_armor.utility.neural.gated_attention import GatedCrossAttention
from dense_armor.utility.neural.latent_attention import LatentCrossAttention

OUT = Path(__file__).resolve().parents[2] / "docs" / "assets" / "neural"
OUT.mkdir(parents=True, exist_ok=True)
KEY = jax.random.PRNGKey(0)


def timing(fn, n_warm: int = 20, n_meas: int = 200) -> tuple[float, float, float]:
    for _ in range(n_warm):
        fn()
    times = []
    for _ in range(n_meas):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    med = float(np.median(times))
    p90 = float(times[int(0.9 * n_meas) - 1])
    mx = float(times[-1])
    return med, p90, mx


def count_params(brick) -> int:
    return sum(int(np.prod(p.value.shape)) for p in brick.params())


def main() -> None:
    key = KEY
    d_model, d_context = 4, 4
    Tq, Tc = 5, 7

    xq = jax.random.normal(key, (Tq, d_model))
    xc = jax.random.normal(key, (Tc, d_context))
    signal = jnp.array([0.2, 0.5, 0.8, 0.4, 0.9])

    cross = MultiHeadCrossAttention(d_model, 2, key, d_context=d_context)
    gated = GatedCrossAttention(cross, 0.5)
    latent = LatentCrossAttention(3, d_model, 2, key, d_context=d_context)
    fusion = BottleneckFusion(d_model, 2, 2, key, d_ff=8)

    a = jax.random.normal(key, (6, d_model))
    b = jax.random.normal(key, (8, d_model))

    cross.forward(xq, xc)
    assert cross.weights_ is not None
    w_cross = np.asarray(cross.weights_[0, 0])

    gated.forward(xq, xc, signal)
    gate = np.asarray(gated.gate_)

    latent.forward(xc)
    assert latent.attention.weights_ is not None
    w_lat = np.asarray(latent.attention.weights_[0, 0])

    a1, _, fsn1 = fusion.forward(a, b)
    a2, _, _ = fusion.forward(a1, b, fsn=fsn1)
    b2 = jax.random.normal(jax.random.PRNGKey(99), (8, d_model))
    a1b, _, fsn1b = fusion.forward(a, b2)
    a2b, _, _ = fusion.forward(a1b, b2, fsn=fsn1b)
    diff_one = float(np.max(np.abs(np.asarray(a1) - np.asarray(a1b))))
    diff_two = float(np.max(np.asarray(a2) - np.asarray(a2b)))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].imshow(w_cross, cmap="viridis", aspect="auto")
    axes[0, 0].set_title("A. cross-attention weights (synthetic)")
    axes[0, 0].set_xlabel("context step")
    axes[0, 0].set_ylabel("query step")

    axes[0, 1].plot(np.arange(Tq), np.asarray(signal), "o-", label="signal")
    axes[0, 1].axhline(0.5, color="k", ls="--", label="threshold")
    axes[0, 1].bar(np.arange(Tq), gate, alpha=0.3, label="gate")
    axes[0, 1].set_title("B. gate open where signal > threshold")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].imshow(w_lat, cmap="magma", aspect="auto")
    axes[1, 0].set_title("C. latent cross-attention weights (synthetic)")
    axes[1, 0].set_xlabel("input step")
    axes[1, 0].set_ylabel("latent index")

    axes[1, 1].bar(["1 layer", "2 layers"], [diff_one, diff_two])
    axes[1, 1].set_title("D. |a - a'| when b changes")
    axes[1, 1].set_ylabel("max absolute difference")

    fig.suptitle("composable attention, synthetic data", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "composable_attention.png", dpi=120)
    plt.close(fig)
    print(f"figure: {OUT / 'composable_attention.png'}")

    print("\ntimings (ms, 200 calls after 20 warm-ups):")
    for name, brick, call in [
        ("MultiHeadCrossAttention", cross, lambda: cross.forward(xq, xc)),
        ("GatedCrossAttention", gated, lambda: gated.forward(xq, xc, signal)),
        ("LatentCrossAttention", latent, lambda: latent.forward(xc)),
        ("BottleneckFusion", fusion, lambda: fusion.forward(a, b)),
    ]:
        med, p90, mx = timing(call)
        print(
            f"  {name}: median={med:.3f} p90={p90:.3f} max={mx:.3f} params={count_params(brick)}"
        )

    print(f"\nD flow: 1 layer diff={diff_one:.6f}, 2 layers diff={diff_two:.6f}")


if __name__ == "__main__":
    main()
