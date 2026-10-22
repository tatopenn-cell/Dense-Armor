"""Neural bricks: parameters, initialisations, activations, losses."""

from dense_armor.utility.neural.adapter import AsTransformer
from dense_armor.utility.neural.layers import (
    Dense,
    Dropout,
    Embedding,
    Residual,
    dense_backward,
    dense_forward,
)
from dense_armor.utility.neural.norm import BatchNorm, LayerNorm
from dense_armor.utility.neural.params import Param

__all__ = [
    "Param",
    "AsTransformer",
    "BatchNorm",
    "Dense",
    "Dropout",
    "Embedding",
    "LayerNorm",
    "Residual",
    "dense_backward",
    "dense_forward",
]

from dense_armor.utility.neural.attention import (
    MultiHeadSelfAttention,
    TransformerBlock,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    sinusoidal_positions,
)
from dense_armor.utility.neural.chain import Parallel, Sequential
from dense_armor.utility.neural.conv import (
    AvgPool1D,
    AvgPool2D,
    Conv1D,
    Conv2D,
    Flatten,
    MaxPool1D,
    MaxPool2D,
)
from dense_armor.utility.neural.models import (
    AddPositions,
    LastStep,
    MeanOverTime,
    cnn,
    mlp,
    recurrent_net,
    resnet_mlp,
    transformer_encoder,
)
from dense_armor.utility.neural.recurrent import GRU, LSTM, RNN

__all__ = list(__all__) + [
    "Parallel",
    "Sequential",
    "mlp",
    "resnet_mlp",
    "AddPositions",
    "LastStep",
    "MeanOverTime",
    "cnn",
    "recurrent_net",
    "transformer_encoder",
    "Conv1D",
    "Conv2D",
    "MaxPool1D",
    "MaxPool2D",
    "AvgPool1D",
    "AvgPool2D",
    "Flatten",
    "GRU",
    "LSTM",
    "RNN",
    "MultiHeadSelfAttention",
    "TransformerBlock",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "sinusoidal_positions",
]

from dense_armor.utility.neural.batching import minibatches
from dense_armor.utility.neural.schedules import (
    ConstantLR,
    CosineWarmRestarts,
    StepLR,
)
from dense_armor.utility.neural.trainer import Trainer

__all__ = list(__all__) + [
    "minibatches",
    "ConstantLR",
    "StepLR",
    "CosineWarmRestarts",
    "Trainer",
]

from dense_armor.utility.neural.background import BackgroundLearner
from dense_armor.utility.neural.hedge import HedgeNetClassifier
from dense_armor.utility.neural.online import (
    OnlineNetClassifier,
    OnlineNetRegressor,
)

__all__ = list(__all__) + [
    "BackgroundLearner",
    "HedgeNetClassifier",
    "OnlineNetClassifier",
    "OnlineNetRegressor",
]

from dense_armor.utility.neural.autoencoder import Autoencoder
from dense_armor.utility.neural.matryoshka import MatryoshkaLoss
from dense_armor.utility.neural.online_autoencoder import OnlineAutoencoder
from dense_armor.utility.neural.quantize import VectorQuantizer
from dense_armor.utility.neural.seq2seq import Seq2Seq

__all__ = list(__all__) + [
    "Autoencoder",
    "MatryoshkaLoss",
    "OnlineAutoencoder",
    "Seq2Seq",
    "VectorQuantizer",
]


from dense_armor.utility.neural.activations_more import (
    ELU,
    GELU,
    LeakyReLU,
    SiLU,
    Softplus,
)

__all__ = list(__all__) + [
    "ELU",
    "GELU",
    "LeakyReLU",
    "SiLU",
    "Softplus",
]


from dense_armor.utility.neural.schedules_more import LinearWarmup, NoamSchedule

__all__ = list(__all__) + ["LinearWarmup", "NoamSchedule"]


from dense_armor.utility.neural.convtranspose import (
    ConvTranspose1D,
    ConvTranspose2D,
    Upsample1D,
    Upsample2D,
)

__all__ = list(__all__) + [
    "ConvTranspose1D",
    "ConvTranspose2D",
    "Upsample1D",
    "Upsample2D",
]


from dense_armor.utility.neural.bidirectional import Bidirectional

__all__ = list(__all__) + ["Bidirectional"]


from dense_armor.utility.neural.norm_more import GroupNorm, InstanceNorm

__all__ = list(__all__) + ["GroupNorm", "InstanceNorm"]


from dense_armor.utility.neural.conv_more import (
    Conv3D,
    DepthwiseConv2D,
    SeparableConv2D,
)

__all__ = list(__all__) + [
    "Conv3D",
    "DepthwiseConv2D",
    "SeparableConv2D",
]
