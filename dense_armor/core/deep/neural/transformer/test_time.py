"""Test-time compute: generate candidates and pick the best."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

_VERIFIERS = ("confidence", "margin", "entropy", "ensemble")


class TestTimeCompute:
    """Test-time compute wrapper.

    Runs ``model`` on ``n_candidates`` perturbed copies of the input, scores
    each copy with a ``verifier``, and returns the best one. Four verifiers:

    - ``"confidence"`` (default): the maximum of the softmax output.
    - ``"margin"``: the gap between the top-1 and the top-2 softmax value.
    - ``"entropy"``: negative entropy of the softmax output.
    - ``"ensemble"``: the average of the softmax over the candidates, in
      log-space. Not a verifier: the return value is the averaged prediction
      of the ensemble, not a selected candidate.

    Args:
        model: the wrapped model. It must have ``forward(x, training=...)``
            and, if a gradient is needed, ``backward(g)`` and ``params()``.
        n_candidates: number of perturbations. Default 4.
        noise: standard deviation of the perturbations. Default 0.01.
        seed: seed of the perturbations. Default 0.
        verifier: one of ``"confidence"``, ``"margin"``, ``"entropy"``,
            ``"ensemble"``. Default ``"confidence"``.

    Raises:
        ValueError: if ``n_candidates < 1``, ``noise < 0``, or ``verifier``
            is not in ``_VERIFIERS``.
    """

    def __init__(
        self,
        model: Any,
        n_candidates: int = 4,
        noise: float = 0.01,
        seed: int = 0,
        verifier: str = "confidence",
    ) -> None:
        if n_candidates < 1:
            raise ValueError(f"n_candidates must be >= 1, got {n_candidates!r}")
        if noise < 0.0:
            raise ValueError(f"noise must be >= 0, got {noise!r}")
        if verifier not in _VERIFIERS:
            raise ValueError(f"verifier must be one of {_VERIFIERS}, got {verifier!r}")
        self.model = model
        self.n_candidates = int(n_candidates)
        self.noise = float(noise)
        self.seed = int(seed)
        self.verifier = verifier
        self._key = jax.random.PRNGKey(seed)
        self._call_count = 0
        self._candidates: jax.Array | None = None
        self._outputs: list[jax.Array] = []
        self._selected: jax.Array | None = None

    def __repr__(self) -> str:
        return (
            f"TestTimeCompute(n_candidates={self.n_candidates}, "
            f"noise={self.noise}, seed={self.seed}, "
            f"verifier={self.verifier!r})"
        )

    def _score(self, probs: jax.Array) -> jax.Array:
        if self.verifier == "margin":
            s = jnp.sort(probs, axis=-1)
            return s[..., -1] - s[..., -2]
        if self.verifier == "entropy":
            return jnp.sum(probs * jnp.log(probs + 1e-9), axis=-1)
        return probs.max(axis=-1)

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Run the model on the candidates and return the best."""
        x = jnp.asarray(x)
        key = jax.random.fold_in(self._key, self._call_count)
        self._call_count += 1
        noise = self.noise * jax.random.normal(key, (self.n_candidates,) + x.shape)
        candidates = x[None, ...] + noise
        ys = [self.model.forward(c, training=training) for c in candidates]
        probs = [jax.nn.softmax(y, axis=-1) for y in ys]

        if self.verifier == "ensemble":
            mean_probs = jnp.mean(jnp.stack(probs), axis=0)
            out = jnp.log(mean_probs + 1e-9)
            self._candidates = candidates
            self._outputs = ys
            self._selected = None
            return out

        scores = jnp.stack([self._score(p) for p in probs])  # (n_cand, B)
        best = jnp.argmax(scores, axis=0)  # (B,)
        ys_stacked = jnp.stack(ys)  # (n_cand, B, n_y)
        out = ys_stacked[best, jnp.arange(best.shape[0])]
        self._candidates = candidates
        self._outputs = ys
        self._selected = best
        return out

    def backward(self, g: jax.Array) -> jax.Array:
        """Backward through the selected candidate (not for ensemble)."""
        if self.verifier == "ensemble":
            raise RuntimeError("backward is not implemented for verifier='ensemble'")
        if self._selected is None or self._candidates is None:
            raise RuntimeError("call forward() before backward()")
        candidates = self._candidates
        selected = self._selected
        g = jnp.asarray(g)
        dx = jnp.zeros_like(candidates[0])
        for i in range(self.n_candidates):
            mask = (selected == i).astype(g.dtype)
            if not bool(jnp.any(mask)):
                continue
            self.model.forward(candidates[i], training=True)
            self.model.backward(g * mask[:, None])
        return dx

    def params(self) -> list:
        return self.model.params()
