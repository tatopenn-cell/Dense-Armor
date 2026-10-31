# Neural networks: API

Every public class and function of `dense_armor.utility.neural` and of the optimizers of `dense_armor.utility.optim`, with its signature and what it does. The title of each section is the guide page that explains the pieces with examples.

## [Foundations](../neural/index.md)

| Name | Signature | What it does |
|---|---|---|
| `Param` | `Param(value: Any) -> None` | A jax array with a gradient of the same shape and dtype |
| `fan_in` | `fan_in(shape: tuple[int, ...]) -> int` | The fan-in of a weight array: ``shape[0]`` |
| `he_normal` | `he_normal(key: Any, shape: tuple[int, ...], a: float = 0.0) -> jax.Array` | Zero-mean Gaussian with standard deviation ``sqrt(2/((1+a**2) fan_in))`` |
| `xavier_normal` | `xavier_normal(key: Any, shape: tuple[int, ...]) -> jax.Array` | Zero-mean Gaussian with standard deviation ``sqrt(1 / fan_in)`` |
| `zeros_bias` | `zeros_bias(n: int) -> jax.Array` | Zero biases of length ``n`` |
| `glorot_uniform` | `glorot_uniform(key: Any, shape: tuple[int, ...]) -> jax.Array` | Uniform initialisation of Glorot and Bengio (2010), Eqn.(16) p.5 |
| `relu_forward` | `relu_forward(x: jax.Array) -> jax.Array` | ``max(x, 0)``. The derivative at 0 is 0 (choice of this code) |
| `relu_backward` | `relu_backward(x: jax.Array, y: jax.Array, g: jax.Array) -> jax.Array` | Derivative of ReLU: ``g * (x > 0)``. At ``x = 0`` it returns 0 |
| `prelu_forward` | `prelu_forward(x: jax.Array, a: float = 0.25) -> jax.Array` | ``x`` where ``x > 0``, ``a*x`` otherwise (Eqn.(1), page 2) |
| `prelu_backward` | `prelu_backward(x: jax.Array, y: jax.Array, g: jax.Array, a: float = 0.25) -> jax.Array` | Derivative of PReLU: ``g`` where ``x > 0``, ``a*g`` otherwise |
| `sigmoid_forward` | `sigmoid_forward(x: jax.Array) -> jax.Array` | ``1 / (1 + exp(-x))``, stable for large negative ``x`` |
| `sigmoid_backward` | `sigmoid_backward(x: jax.Array, y: jax.Array, g: jax.Array) -> jax.Array` | ``g * y * (1 - y)`` |
| `tanh_forward` | `tanh_forward(x: jax.Array) -> jax.Array` | ``tanh(x)`` |
| `tanh_backward` | `tanh_backward(x: jax.Array, y: jax.Array, g: jax.Array) -> jax.Array` | ``g * (1 - y**2)`` |
| `softmax_forward` | `softmax_forward(x: jax.Array) -> jax.Array` | Softmax over the last axis, stable with the subtraction of the max |
| `softmax_backward` | `softmax_backward(x: jax.Array, y: jax.Array, g: jax.Array) -> jax.Array` | Jacobian-vector product of softmax: ``y * (g - sum(g*y))`` |
| `Relu` | `Relu() -> None` | ReLU: ``forward`` caches ``x`` and ``y``; ``backward`` returns ``dx`` |
| `PRelu` | `PRelu(a: float = 0.25) -> None` | PReLU with ``a`` as a ``Param``; ``backward`` fills ``a.grad`` |
| `Sigmoid` | `Sigmoid() -> None` | Sigmoid: ``forward`` caches ``x`` and ``y``; ``backward`` returns ``dx`` |
| `Tanh` | `Tanh() -> None` | Tanh: ``forward`` caches ``x`` and ``y``; ``backward`` returns ``dx`` |
| `Softmax` | `Softmax() -> None` | Softmax over the last axis: ``forward`` caches ``x`` and ``y``; ``backward`` returns ``dx`` |
| `mse_forward` | `mse_forward(pred: jax.Array, target: jax.Array) -> jax.Array` | Mean over all elements of ``(pred - target)**2`` |
| `mse_backward` | `mse_backward(pred: jax.Array, target: jax.Array) -> jax.Array` | ``2 * (pred - target) / size`` |
| `softmax_cross_entropy_forward` | `softmax_cross_entropy_forward(logits: jax.Array, labels: jax.Array) -> jax.Array` | Mean over the batch of ``-log softmax(logits)[label]``, in log-sum-exp form |
| `softmax_cross_entropy_backward` | `softmax_cross_entropy_backward(logits: jax.Array, labels: jax.Array) -> jax.Array` | ``(softmax(logits) - onehot(labels)) / n`` |
| `bce_with_logits_forward` | `bce_with_logits_forward(logits: jax.Array, targets: jax.Array) -> jax.Array` | Mean over the batch of ``max(z,0) - z*t + log(1 + exp(-/z/))`` |
| `bce_with_logits_backward` | `bce_with_logits_backward(logits: jax.Array, targets: jax.Array) -> jax.Array` | ``(sigmoid(z) - t) / n``, with ``n`` the number of elements |
| `MSE` | `MSE() -> None` | Mean squared error with a cached last prediction and target |
| `SoftmaxCrossEntropy` | `SoftmaxCrossEntropy() -> None` | Softmax cross-entropy with a cached last logits and labels |
| `BCEWithLogits` | `BCEWithLogits() -> None` | Binary cross-entropy with logits, with a cached last logits and targets |
| `huber_forward` | `huber_forward(pred: jax.Array, target: jax.Array, delta: float = 1.0) -> jax.Array` | Mean of the Huber loss with parameter ``delta`` |
| `huber_backward` | `huber_backward(pred: jax.Array, target: jax.Array, delta: float = 1.0) -> jax.Array` | Gradient of the Huber loss with respect to ``pred`` |
| `Huber` | `Huber(delta: float = 1.0) -> None` | Huber loss with a cached last prediction and target |
| `gradcheck` | `gradcheck(forward: Callable[[Any], Any], backward: Callable[[Any, Any, Any], Any], x: Any, eps: float = 1e-06) -> float` | Largest relative error between ``backward`` and finite differences |

## [Layers](../neural/layers.md)

| Name | Signature | What it does |
|---|---|---|
| `dense_forward` | `dense_forward(w: jax.Array, b: jax.Array, x: jax.Array) -> jax.Array` | ``x @ w + b`` |
| `dense_backward` | `dense_backward(w: jax.Array, x: jax.Array, g: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]` | Gradients of ``dense_forward``: ``(dx, dw, db)`` |
| `Dense` | `Dense(n_in: int, n_out: int, key: Any, init: str = he, a: float = 0.0) -> None` | A dense layer ``y = x @ W + b`` |
| `Embedding` | `Embedding(n_tokens: int, dim: int, key: Any) -> None` | A lookup table ``E[ids]`` |
| `Dropout` | `Dropout(p: float, key: Any) -> None` | Random dropout with the scaling of Hinton et al. 2012 |
| `Residual` | `Residual(inner: Any) -> None` | ``y = x + inner.forward(x)`` (He et al. 2015, Eqn.(1) page 3) |
| `BatchNorm` | `BatchNorm(n_features: int, eps: float = 1e-05, momentum: float = 0.1) -> None` | Batch normalisation of a 2-D batch ``(m, n_features)`` |
| `LayerNorm` | `LayerNorm(n_features: int, eps: float = 1e-05) -> None` | Layer normalisation over the features of each sample |
| `RMSNorm` | `RMSNorm(n_features: int, eps: float = 1e-05) -> None` | RMS layer normalisation over the features of each sample |
| `GLUFeedForward` | `GLUFeedForward(d_model: int, d_ff: int, activation: str, key: Any, bias: bool = False) -> None` | The feed-forward layer ``(act(x W) * (x V)) W2`` of Shazeer 2020 |
| `AsTransformer` | `AsTransformer(layer: Any, input_keys: list[str], prefix: str = z) -> None` | A layer exposed as a ``Transformer`` |

## [Chains](../neural/chains.md)

| Name | Signature | What it does |
|---|---|---|
| `Sequential` | `Sequential(layers: list[Any]) -> None` | A chain of layers, applied in order |
| `Parallel` | `Parallel(branches: list[Any], merge: str = add) -> None` | Several branches on the same input, merged on the output |
| `mlp` | `mlp(sizes: list[int], key: Any, activation: str = relu) -> Sequential` | A chain of dense layers with an activation between them |
| `resnet_mlp` | `resnet_mlp(width: int, depth: int, n_in: int, n_out: int, key: Any) -> Sequential` | A chain of residual blocks, the dense form of He et al. 2015 |
| `LastStep` | `LastStep() -> None` | Take the last step of a sequence ``(B, T, n)``, giving ``(B, n)`` |
| `MeanOverTime` | `MeanOverTime() -> None` | Mean over the time axis of ``(B, T, n)`` or ``(T, n)`` |
| `AddPositions` | `AddPositions(max_len: int) -> None` | Add the sinusoidal positions of page 6 of Vaswani et al. 2017 |
| `cnn` | `cnn(input_shape: tuple[int, int, int], channels: list[int], n_out: int, key: Any, kernel: int = 3, activation: str = relu) -> Sequential` | A chain of ``Conv2D`` + activation + ``MaxPool2D(2)``, then a head |
| `recurrent_net` | `recurrent_net(n_in: int, n_hidden: int, n_out: int, key: Any, cell: str = lstm, n_layers: int = 1) -> Sequential` | A chain of recurrent layers, then ``LastStep`` and a dense head |
| `transformer_encoder` | `transformer_encoder(n_in: int, d_model: int, n_heads: int, d_ff: int, depth: int, n_out: int, key: Any, max_len: int, causal: bool = False) -> Sequential` | The encoder of Vaswani et al. 2017, as a chain |

## [Convolution and pooling](../neural/vision_layers.md)

| Name | Signature | What it does |
|---|---|---|
| `Conv1D` | `Conv1D(c_in: int, c_out: int, kernel: int, key: Any, stride: int = 1, padding: int = 0, causal: bool = False) -> None` | 1-D cross-correlation with zero padding, optional causal shift |
| `Conv2D` | `Conv2D(c_in: int, c_out: int, kernel: int, key: Any, stride: int = 1, padding: int = 0) -> None` | 2-D cross-correlation with zero padding, channels last |
| `MaxPool1D` | `MaxPool1D(kernel: int, stride: int \| None = None) -> None` | 1-D max pooling with the first maximum going to the gradient |
| `MaxPool2D` | `MaxPool2D(kernel: int, stride: int \| None = None) -> None` | 2-D max pooling with the first maximum going to the gradient |
| `AvgPool1D` | `AvgPool1D(kernel: int, stride: int \| None = None) -> None` | 1-D average pooling. Input ``(L, c)`` or ``(B, L, c)`` |
| `AvgPool2D` | `AvgPool2D(kernel: int, stride: int \| None = None) -> None` | 2-D average pooling. Input ``(H, W, c)`` or ``(B, H, W, c)`` |
| `Flatten` | `Flatten() -> None` | Flatten every axis after the batch axis |

## [Recurrent layers](../neural/sequence_layers.md)

| Name | Signature | What it does |
|---|---|---|
| `RNN` | `RNN(n_in: int, n_hidden: int, key: Any, activation: str = tanh) -> None` | Elman recurrent layer ``h = tanh(W x + U h_prev + b)`` |
| `GRU` | `GRU(n_in: int, n_hidden: int, key: Any) -> None` | GRU layer, Eqs.(5) to (8), page 3 of Cho et al. 2014 |
| `LSTM` | `LSTM(n_in: int, n_hidden: int, key: Any, peepholes: bool = True) -> None` | Vanilla LSTM with peepholes, Greff et al. 2017 pages 1-2 |

## [Attention](../neural/attention.md)

| Name | Signature | What it does |
|---|---|---|
| `scaled_dot_product_attention` | `scaled_dot_product_attention(q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array \| None = None) -> jax.Array` | ``softmax(q k^T / sqrt(d_k)) v`` (Eq.(1), page 4) |
| `scaled_dot_product_attention_backward` | `scaled_dot_product_attention_backward(q: jax.Array, k: jax.Array, v: jax.Array, g: jax.Array, mask: jax.Array \| None = None) -> tuple[jax.Array, jax.Array, jax.Array]` | Gradients ``(dq, dk, dv)`` of the scaled dot-product attention |
| `sinusoidal_positions` | `sinusoidal_positions(length: int, d_model: int) -> jax.Array` | The sinusoidal positional encoding of page 6 |
| `rotary_angles` | `rotary_angles(length: int, d_k: int, offset: int = 0, base: float = 10000.0) -> tuple[jax.Array, jax.Array]` | The cosines and sines of the rotary rotation at each position |
| `apply_rotary` | `apply_rotary(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array` | Rotate each adjacent pair of the last axis of ``x`` |
| `apply_rotary_transpose` | `apply_rotary_transpose(g: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array` | The gradient of ``apply_rotary``: the rotation by the opposite angle |
| `MultiHeadSelfAttention` | `MultiHeadSelfAttention(d_model: int, n_heads: int, key: Any, causal: bool = False, rotary: bool = False) -> None` | Multi-head self-attention of Vaswani et al. 2017, page 5 |
| `TransformerBlock` | `TransformerBlock(d_model: int, n_heads: int, d_ff: int, key: Any, causal: bool = False, dropout: float = 0.0) -> None` | The encoder block of Vaswani et al. 2017 |

## [Training](../neural/training.md)

| Name | Signature | What it does |
|---|---|---|
| `Trainer` | `Trainer(model: Any, loss: Any, optimizer: Any, schedule: Any = None, weight_decay: float = 0.0, decay: str = l2, clip_norm: float \| None = None) -> None` | Train a model on a dataset, with mini-batches and schedules |
| `ConstantLR` | `ConstantLR(lr: float) -> None` | A constant learning rate |
| `StepLR` | `StepLR(lr: float, step: float, gamma: float) -> None` | Step schedule: ``lr * gamma ** floor(epoch / step)`` |
| `CosineWarmRestarts` | `CosineWarmRestarts(lr_max: float, lr_min: float, t0: float, t_mult: float = 1.0) -> None` | Cosine annealing with warm restarts |
| `minibatches` | `minibatches(n: int, batch_size: int, key: Any = None, drop_last: bool = False) -> Iterator[jax.Array]` | Yield index arrays that cover ``range(n)`` once |

## [Continuous training](../neural/continuous.md)

| Name | Signature | What it does |
|---|---|---|
| `OnlineNetRegressor` | `OnlineNetRegressor(model: Any, loss: Any, optimizer: Any, feature_keys: Any = None, schedule: Any = None, weight_decay: float = 0.0, decay: str = l2, clip_norm: float \| None = None) -> None` | A network trained one sample at a time |
| `OnlineNetClassifier` | `OnlineNetClassifier(model: Any, loss: Any, optimizer: Any, classes: Any, feature_keys: Any = None, schedule: Any = None, weight_decay: float = 0.0, decay: str = l2, clip_norm: float \| None = None) -> None` | A network classifier trained one sample at a time |
| `BackgroundLearner` | `BackgroundLearner(estimator: Any, maxsize: int = 64, sync_every: int = 1) -> None` | Learn on a worker thread; predict on the last served copy |
| `HedgeNetClassifier` | `HedgeNetClassifier(sizes: Any, key: Any, classes: Any, beta: float = 0.99, lr: float = 0.1, smoothing: float = 0.2, feature_keys: Any = None) -> None` | Online deep learning with Hedge Backpropagation |

## [Encoders and decoders](../neural/encoder_decoder.md)

| Name | Signature | What it does |
|---|---|---|
| `Autoencoder` | `Autoencoder(encoder: Any, decoder: Any) -> None` | An encoder and a decoder, joined end to end |
| `Seq2Seq` | `Seq2Seq(encoder: Any, decoder: Any, head: Any, horizon: int, reverse_source: bool = True) -> None` | A recurrent encoder, a recurrent decoder, a dense head |
| `VectorQuantizer` | `VectorQuantizer(n_codes: int, dim: int, key: Any, beta: float = 0.25) -> None` | Nearest-code quantizer with the straight-through gradient |
| `MatryoshkaLoss` | `MatryoshkaLoss(dims: Any, d: int, n_classes: int, key: Any, weights: Any = None) -> None` | The multi-size loss of Matryoshka Representation Learning |
| `OnlineAutoencoder` | `OnlineAutoencoder(model: Autoencoder, optimizer: Any, feature_keys: Any = None, schedule: Any = None, clip_norm: float \| None = None) -> None` | An autoencoder trained one sample at a time, as an anomaly score |

## [Deep](../neural/deep.md)

| Name | Signature | What it does |
|---|---|---|
| `gelu_forward` | `gelu_forward(x: jax.Array) -> jax.Array` | ``x * Phi(x)`` (Eq. on page 1 of Hendrycks and Gimpel 2016) |
| `gelu_backward` | `gelu_backward(x: jax.Array, y: jax.Array, g: jax.Array) -> jax.Array` | Derivative of the exact GELU: ``Phi(x) + x phi(x)`` |
| `gelu_approx_forward` | `gelu_approx_forward(x: jax.Array) -> jax.Array` | ``0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x**3)))`` |
| `gelu_approx_backward` | `gelu_approx_backward(x: jax.Array, y: jax.Array, g: jax.Array) -> jax.Array` | Derivative of the tanh approximation of the GELU |
| `silu_forward` | `silu_forward(x: jax.Array) -> jax.Array` | ``x * sigmoid(x)`` (SiLU, pages 1 and 2 of Elfwing et al. 2017) |
| `silu_backward` | `silu_backward(x: jax.Array, y: jax.Array, g: jax.Array) -> jax.Array` | Derivative of SiLU: ``s + x s (1 - s)`` with ``s = sigmoid(x)`` |
| `elu_forward` | `elu_forward(x: jax.Array, alpha: float = 1.0) -> jax.Array` | ``x`` if ``x > 0``, ``alpha (exp(x) - 1)`` otherwise (Eq.(15)) |
| `elu_backward` | `elu_backward(x: jax.Array, y: jax.Array, g: jax.Array, alpha: float = 1.0) -> jax.Array` | Derivative of ELU: ``1`` if ``x > 0``, ``alpha exp(x)`` otherwise |
| `leaky_relu_forward` | `leaky_relu_forward(x: jax.Array, slope: float = 0.01) -> jax.Array` | ``x`` if ``x > 0``, ``slope * x`` otherwise |
| `leaky_relu_backward` | `leaky_relu_backward(x: jax.Array, y: jax.Array, g: jax.Array, slope: float = 0.01) -> jax.Array` | Derivative of LeakyReLU: ``1`` if ``x > 0``, ``slope`` otherwise |
| `softplus_forward` | `softplus_forward(x: jax.Array, beta: float = 1.0) -> jax.Array` | ``log(1 + exp(beta x)) / beta``, stable with ``jax.nn.softplus`` |
| `softplus_backward` | `softplus_backward(x: jax.Array, y: jax.Array, g: jax.Array, beta: float = 1.0) -> jax.Array` | Derivative of the softplus: ``sigmoid(beta x)`` |
| `GELU` | `GELU(approximate: bool = False) -> None` | GELU (exact or approximate) with a forward and a backward |
| `SiLU` | `SiLU() -> None` | SiLU: ``x * sigmoid(x)`` |
| `ELU` | `ELU(alpha: float = 1.0) -> None` | ELU with a scale ``alpha`` (Eq.(15), page 5 of Clevert et al.) |
| `LeakyReLU` | `LeakyReLU(slope: float = 0.01) -> None` | Leaky ReLU with a fixed slope (no Param) |
| `Softplus` | `Softplus(beta: float = 1.0) -> None` | Softplus with a scale ``beta`` |
| `clip_grad_norm` | `clip_grad_norm(params: list[Param], max_norm: float) -> float` | Clip the global L2 norm of the gradients of ``params`` |
| `LinearWarmup` | `LinearWarmup(lr_start: float, lr_end: float, warmup: float) -> None` | Linear warmup: ``lr_start`` then a linear ramp to ``lr_end`` |
| `NoamSchedule` | `NoamSchedule(d_model: float, warmup: float, scale: float = 1.0) -> None` | Noam schedule: linear warmup then inverse-square-root decay |
| `ConvTranspose1D` | `ConvTranspose1D(c_in: int, c_out: int, kernel: int, key: Any, stride: int = 1, padding: int = 0, output_padding: int = 0) -> None` | 1-D transposed convolution, channels last |
| `ConvTranspose2D` | `ConvTranspose2D(c_in: int, c_out: int, kernel: int, key: Any, stride: int = 1, padding: int = 0, output_padding: int = 0) -> None` | 2-D transposed convolution, channels last |
| `Upsample1D` | `Upsample1D(factor: int) -> None` | Nearest-neighbour upsampling along the length, channels last |
| `Upsample2D` | `Upsample2D(factor: int) -> None` | Nearest-neighbour upsampling along both spatial axes, channels last |
| `Bidirectional` | `Bidirectional(forward_layer: Any, backward_layer: Any) -> None` | A forward and a backward recurrent layer, concatenated |
| `GroupNorm` | `GroupNorm(num_groups: int, num_channels: int, eps: float = 1e-05, spatial_ndim: int = 2) -> None` | Group normalization of the channels of a sample |
| `InstanceNorm` | `InstanceNorm(num_channels: int, eps: float = 1e-05, spatial_ndim: int = 2) -> None` | Instance normalization: one group per channel |
| `Conv3D` | `Conv3D(c_in: int, c_out: int, kernel: int, key: Any, stride: int = 1, padding: int = 0) -> None` | 3-D cross-correlation with zero padding, channels last |
| `DepthwiseConv2D` | `DepthwiseConv2D(channels: int, kernel: int, key: Any, stride: int = 1, padding: int = 0) -> None` | Depthwise 2-D convolution: one square kernel per channel |
| `SeparableConv2D` | `SeparableConv2D(c_in: int, c_out: int, kernel: int, key: Any, stride: int = 1, padding: int = 0) -> None` | Separable 2-D convolution: depthwise then pointwise |
| `LearnedPositions` | `LearnedPositions(max_len: int, d_model: int, key: Any, std: float = 0.02) -> None` | Add a learned vector to each position of a sequence |
| `l1_forward` | `l1_forward(pred: jax.Array, target: jax.Array) -> jax.Array` | Mean of ``abs(pred - target)`` over all elements |
| `l1_backward` | `l1_backward(pred: jax.Array, target: jax.Array) -> jax.Array` | ``sign(pred - target) / size`` (0 where the difference is 0) |
| `label_smoothing_ce_forward` | `label_smoothing_ce_forward(logits: jax.Array, labels: jax.Array, eps: float = 0.1) -> jax.Array` | Mean over the batch of the softmax cross-entropy with smoothed labels |
| `label_smoothing_ce_backward` | `label_smoothing_ce_backward(logits: jax.Array, labels: jax.Array, eps: float = 0.1) -> jax.Array` | ``(softmax(logits) - q'(k)) / n`` with ``n`` the batch size |
| `kldiv_forward` | `kldiv_forward(logits: jax.Array, target_probs: jax.Array) -> jax.Array` | Mean over the batch of ``sum_k y_k (log y_k - log softmax(logits)_k)`` |
| `kldiv_backward` | `kldiv_backward(logits: jax.Array, target_probs: jax.Array) -> jax.Array` | ``(softmax(logits) - y) / n`` with ``n`` the batch size |
| `hinge_forward` | `hinge_forward(pred: jax.Array, y: jax.Array) -> jax.Array` | Mean of ``max(0, 1 - y * pred)`` over all elements |
| `hinge_backward` | `hinge_backward(pred: jax.Array, y: jax.Array) -> jax.Array` | ``-y / size`` where ``1 - y * pred > 0``, 0 elsewhere |
| `focal_forward` | `focal_forward(logits: jax.Array, y: jax.Array, gamma: float = 2.0, alpha: float \| None = None) -> jax.Array` | Mean of the ``alpha``-balanced focal loss over all elements |
| `focal_backward` | `focal_backward(logits: jax.Array, y: jax.Array, gamma: float = 2.0, alpha: float \| None = None) -> jax.Array` | ``d loss / d logits`` of the focal loss, mean over all elements |
| `L1Loss` | `L1Loss() -> None` | Mean absolute error with cached last call |
| `LabelSmoothingCE` | `LabelSmoothingCE(eps: float = 0.1) -> None` | Softmax cross-entropy with smoothed labels, cached last call |
| `KLDivLoss` | `KLDivLoss() -> None` | Kullback-Leibler divergence ``sum_k y_k (log y_k - log p_k)`` |
| `HingeLoss` | `HingeLoss() -> None` | Hinge loss ``max(0, 1 - y * pred)`` for ``y`` in {-1, +1} |
| `FocalLoss` | `FocalLoss(gamma: float = 2.0, alpha: float \| None = None) -> None` | Focal loss ``-alpha_t (1 - p_t)**gamma * log(p_t)`` |
| `OnlineNetVector` | `OnlineNetVector(model: Any, loss: Any, optimizer: Any, feature_keys: Any = None, schedule: Any = None, weight_decay: float = 0.0, decay: str = l2, clip_norm: float \| None = None) -> None` | A network trained one sample at a time, on vector targets |

## Optimizers

| Name | Signature | What it does |
|---|---|---|
| `SGD` | `SGD(lr: float = 0.1, momentum: float = 0.0, nesterov: bool = False) -> None` | Gradient descent with optional momentum and Nesterov momentum |
| `RMSProp` | `RMSProp(lr: float = 0.001, rho: float = 0.9, eps: float = 1e-08) -> None` | RMSProp with an exponentially decaying average of squared gradients |
| `Adam` | `Adam(lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-08) -> None` | Adam with bias-corrected moment estimates |
| `AdamW` | `AdamW(lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-08, weight_decay: float = 0.0) -> None` | Adam with weight decay decoupled from the gradient |
| `AdaGrad` | `AdaGrad(lr: float = 0.1, eps: float = 1e-08) -> None` | AdaGrad with per-coordinate adaptive step sizes |
| `FTRLProximal` | `FTRLProximal(alpha: float = 1.0, beta: float = 0.5, l1: float = 0.0, l2: float = 0.0) -> None` | FTRL with a quadratic + L1 + L2 regularizer |
| `OnlineNewtonStep` | `OnlineNewtonStep(gamma: float = 1.0, eps: float = 1e-06, D: float \| None = None) -> None` | Second-order online update with a rank-one running Hessian |
