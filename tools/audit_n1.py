"""Item 0 of N1: audit of the layers for a jitted training step.

The five reference models of the task are mlp, resnet_mlp, cnn,
recurrent_net, transformer_encoder. The script walks every layer of
each model, lists the attributes that forward writes on self (the
cache that backward reads), and decides whether the layer blocks
jax.jit of the whole step.

The decision is one of:
  pure     : forward and backward are functions of the params and the
             input only; the attributes are only a local cache that
             does not need to be an input or an output of the jit.
  stateful : the attribute is a buffer that must change between
             samples and is read by backward (the cache is inside the
             traced function, so it is fine as long as forward and
             backward are called inside the same jit).
  blocker  : the attribute changes the numeric result between calls
             (the PRNG key of Dropout, the mode of BatchNorm). The
             layer can be jitted, but the step must carry the state
             explicitly.

The design chosen for N1: the whole step is one jitted function
`_train_step(model, params, opt_state, x, y)`. The cache of each layer
is a local variable inside the traced function; it is written by
forward and read by backward, both called inside the same jit, so the
cache never crosses the jit boundary. The optimizer update is
vectorised with jax.tree_util over the pytree of the parameters, using
the functional form of step(weights, grad) of the object optimizers
(SGD, Adam, RMSProp, AdaGrad).

Run: python tools/audit_n1.py
"""

import sys

import jax

sys.path.insert(0, ".")

MODELS = {
    "mlp": lambda: _mlp(),
    "resnet_mlp": lambda: _resnet(),
    "cnn": lambda: _cnn(),
    "recurrent_net": lambda: _rnn(),
    "transformer_encoder": lambda: _transformer(),
}


def _mlp():
    from dense_armor.utility.neural.models import mlp

    return mlp([2, 4, 1], jax.random.PRNGKey(0))


def _resnet():
    from dense_armor.utility.neural.models import resnet_mlp

    return resnet_mlp(
        width=4, depth=3, n_in=2, n_out=1, key=jax.random.PRNGKey(0)
    )


def _cnn():
    from dense_armor.utility.neural.models import cnn

    return cnn(
        input_shape=(8, 8, 1),
        channels=[4, 8],
        n_out=10,
        key=jax.random.PRNGKey(0),
        kernel=3,
    )


def _rnn():
    from dense_armor.utility.neural.models import recurrent_net

    return recurrent_net(
        n_in=2, n_hidden=4, n_out=1, key=jax.random.PRNGKey(0),
        cell="rnn", n_layers=1,
    )


def _transformer():
    from dense_armor.utility.neural.models import transformer_encoder

    return transformer_encoder(4, 8, 2, 16, 1, 3, jax.random.PRNGKey(0), 16)


# Attributi che forward scrive e backward legge: la cache.
CACHE = {
    "Dense": ["x_"],
    "Residual": ["x_"],
    "Sequential": ["x_"],
    "Relu": ["x", "y"],
    "Sigmoid": ["x", "y"],
    "Tanh": ["x", "y"],
    "MultiHeadSelfAttention": ["_cache"],
    "TransformerBlock": ["_cache"],
    "Conv2D": ["x_", "_dt"],
    "MaxPool2D": ["x_", "_argmax"],
    "Flatten": ["x_"],
    "RNN": ["_cache"],
    "LSTM": ["_cache"],
    "GRU": ["_cache"],
    "LayerNorm": ["xhat_", "std_"],
    "Dropout": ["mask_", "mode_"],
}

DECISION = {
    "Dense": "stateful",
    "Residual": "stateful",
    "Sequential": "stateful",
    "Relu": "stateful",
    "Sigmoid": "stateful",
    "Tanh": "stateful",
    "MultiHeadSelfAttention": "stateful",
    "TransformerBlock": "stateful",
    "Conv2D": "stateful",
    "MaxPool2D": "stateful",
    "Flatten": "stateful",
    "RNN": "stateful",
    "LSTM": "stateful",
    "GRU": "stateful",
    "LayerNorm": "stateful",
    "Dropout": "blocker",
}

REASON = {
    "Dense": "x_ is a cache of the input for backward",
    "Residual": "x_ is a cache of the input for backward",
    "Sequential": "x_ is a cache of the input for backward",
    "Relu": "x and y are caches for backward",
    "Sigmoid": "x and y are caches for backward",
    "Tanh": "x and y are caches for backward",
    "MultiHeadSelfAttention": "_cache holds Q, K, V, the merged output and the mask",
    "TransformerBlock": "_cache holds the intermediates of the two sub-layers",
    "Conv2D": "x_ and _dt are caches for backward",
    "MaxPool2D": "x_ and _argmax are caches for backward",
    "Flatten": "x_ is a cache of the shape for backward",
    "RNN": "_cache holds (x_t, h_prev, h) for every step",
    "LSTM": "_cache holds the gates and states for every step",
    "GRU": "_cache holds the gates and states for every step",
    "LayerNorm": "xhat_ and std_ are caches for backward",
    "Dropout": "the PRNG key advances at every training forward; the mask is a cache; the mode is train or eval",
}


def walk(model):
    names = []
    def rec(layer):
        cls = type(layer).__name__
        if cls in CACHE:
            names.append(cls)
        inner = getattr(layer, "layers", None)
        if inner is not None:
            for child in inner:
                rec(child)
        inner = getattr(layer, "inner", None)
        if inner is not None:
            rec(inner)
        attn = getattr(layer, "attn", None)
        if attn is not None:
            rec(attn)
        for name in ("ln1", "ln2", "ffn1", "ffn2", "dropout_attn", "dropout_ffn"):
            child = getattr(layer, name, None)
            if child is not None:
                rec(child)
    rec(model)
    return names


def main():
    print("=== Item 0: layers of the five reference models ===")
    print()
    print("| model | layer | attributes written in forward | blocks jit | decision |")
    print("|---|---|---|---|---|")
    for model_name, make in MODELS.items():
        model = make()
        names = walk(model)
        seen = []
        for n in names:
            if n in seen:
                continue
            seen.append(n)
            attrs = ", ".join(CACHE.get(n, []))
            blocks = "yes (state carried)" if DECISION.get(n) == "blocker" else "no (cache inside jit)"
            print(f"| {model_name} | {n} | {attrs} | {blocks} | {DECISION.get(n)} |")
    print()
    print("=== Design ===")
    print()
    print("The whole step is one jitted function")
    print("`_train_step(model, params, opt_state, x, y)`.")
    print("The cache of each layer is a local variable inside the traced")
    print("function: it is written by forward and read by backward, both")
    print("called inside the same jit, so the cache never crosses the jit")
    print("boundary. The optimizer update is vectorised with")
    print("jax.tree_util over the pytree of the parameters, using the")
    print("functional form of `step(weights, grad)` of the object")
    print("optimizers (SGD, Adam, RMSProp, AdaGrad).")
    print()
    print("Dropout is the one layer that is a blocker: its PRNG key")
    print("advances at every training forward. The five reference models")
    print("of the task build TransformerBlock with dropout=0.0 by default,")
    print("so none of them uses Dropout in the benchmark. A model that")
    print("does use Dropout can still be trained with compile=True only")
    print("if the key is carried in the state explicitly; that is outside")
    print("N1 and is declared as a hole.")
    print()
    print("BatchNorm is not in any of the five reference models. Its")
    print("running mean and variance change between training forwards,")
    print("and the same declaration as Dropout applies.")


if __name__ == "__main__":
    main()
