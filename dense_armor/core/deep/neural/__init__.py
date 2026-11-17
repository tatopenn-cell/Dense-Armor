"""Neural bricks: parameters, initialisations, activations, losses."""

from dense_armor.core.deep.neural.layers.adapter import AsTransformer
from dense_armor.core.deep.neural.layers.glu import GLUFeedForward
from dense_armor.core.deep.neural.layers import (
    Dense,
    Dropout,
    Embedding,
    Residual,
    dense_backward,
    dense_forward,
)
from dense_armor.core.deep.neural.layers.norm import BatchNorm, LayerNorm
from dense_armor.core.deep.neural.layers.norm_rms import RMSNorm
from dense_armor.core.deep.neural.layers.params import Param

__all__ = [
    "Param",
    "AsTransformer",
    "BatchNorm",
    "Dense",
    "Dropout",
    "Embedding",
    "GLUFeedForward",
    "LayerNorm",
    "RMSNorm",
    "Residual",
    "dense_backward",
    "dense_forward",
]

from dense_armor.core.deep.neural.attention import (
    MultiHeadSelfAttention,
    TransformerBlock,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    sinusoidal_positions,
)
from dense_armor.core.deep.neural.layers.chain import Parallel, Sequential
from dense_armor.core.deep.neural.layers.conv import (
    AvgPool1D,
    AvgPool2D,
    Conv1D,
    Conv2D,
    Flatten,
    MaxPool1D,
    MaxPool2D,
)
from dense_armor.core.deep.neural.transformer.decoder import TransformerDecoderBlock
from dense_armor.core.deep.neural.transformer.generate import generate
from dense_armor.core.deep.neural.losses.losses_lm import NextTokenLoss, perplexity
from dense_armor.core.deep.neural.models import (
    AddPositions,
    LastStep,
    MeanOverTime,
    cnn,
    mlp,
    recurrent_net,
    resnet_mlp,
    transformer_decoder,
    transformer_encoder,
)
from dense_armor.core.deep.neural.layers.recurrent import GRU, LSTM, RNN

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
    "transformer_decoder",
    "TransformerDecoderBlock",
    "rotary_decoder",
    "RotaryDecoder",
    "RotaryDecoderBlock",
    "NextTokenLoss",
    "perplexity",
    "generate",
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

from dense_armor.core.deep.neural.training.batching import minibatches
from dense_armor.core.deep.neural.training.data_parallel import linear_scaling
from dense_armor.core.deep.neural.training.schedules import (
    ConstantLR,
    CosineWarmRestarts,
    StepLR,
)
from dense_armor.core.deep.neural.training.stream import DataStream
from dense_armor.core.deep.neural.training.trainer import Trainer

__all__ = list(__all__) + [
    "minibatches",
    "DataStream",
    "ConstantLR",
    "StepLR",
    "CosineWarmRestarts",
    "Trainer",
    "linear_scaling",
]

from dense_armor.core.deep.neural.training.background import BackgroundLearner
from dense_armor.core.deep.neural.online.hedge import HedgeNetClassifier
from dense_armor.core.deep.neural.online import (
    OnlineNetClassifier,
    OnlineNetRegressor,
)

__all__ = list(__all__) + [
    "BackgroundLearner",
    "HedgeNetClassifier",
    "OnlineNetClassifier",
    "OnlineNetRegressor",
]

from dense_armor.core.deep.neural.models.autoencoder import Autoencoder
from dense_armor.core.deep.neural.losses.matryoshka import MatryoshkaLoss
from dense_armor.core.deep.neural.online.online_autoencoder import OnlineAutoencoder
from dense_armor.core.deep.neural.models.quantize import VectorQuantizer
from dense_armor.core.deep.neural.transformer.seq2seq import Seq2Seq

__all__ = list(__all__) + [
    "Autoencoder",
    "MatryoshkaLoss",
    "OnlineAutoencoder",
    "Seq2Seq",
    "VectorQuantizer",
]


from dense_armor.core.deep.neural.layers.activations_more import (
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


from dense_armor.core.deep.neural.training.schedules_more import LinearWarmup, NoamSchedule

__all__ = list(__all__) + ["LinearWarmup", "NoamSchedule"]


from dense_armor.core.deep.neural.layers.convtranspose import (
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


from dense_armor.core.deep.neural.layers.bidirectional import Bidirectional

__all__ = list(__all__) + ["Bidirectional"]


from dense_armor.core.deep.neural.layers.norm_more import GroupNorm, InstanceNorm

__all__ = list(__all__) + ["GroupNorm", "InstanceNorm"]


from dense_armor.core.deep.neural.layers.conv_more import (
    Conv3D,
    DepthwiseConv2D,
    SeparableConv2D,
)

__all__ = list(__all__) + [
    "Conv3D",
    "DepthwiseConv2D",
    "SeparableConv2D",
]


from dense_armor.core.deep.neural.attention.positions import LearnedPositions

__all__ = list(__all__) + ["LearnedPositions"]


from dense_armor.core.deep.neural.losses.losses_more import (
    FocalLoss,
    HingeLoss,
    KLDivLoss,
    L1Loss,
    LabelSmoothingCE,
)

__all__ = list(__all__) + [
    "FocalLoss",
    "HingeLoss",
    "KLDivLoss",
    "L1Loss",
    "LabelSmoothingCE",
]


from dense_armor.core.deep.neural.online.online_vector import OnlineNetVector

__all__ = list(__all__) + ["OnlineNetVector"]


from dense_armor.core.deep.neural.attention.cross_attention import MultiHeadCrossAttention

__all__ = list(__all__) + ["MultiHeadCrossAttention"]


from dense_armor.core.deep.neural.consolidation import (
    ArchivalMemory,
    DualMemory,
    LifecycleMemory,
    PQMemory,
    ResidualTree,
    TheoreticalBound,
    VQMemory,
)
from dense_armor.core.deep.neural.online.continual import AGEM, EWC, GEM
from dense_armor.core.deep.neural.online.generative_replay import DeepGenerativeReplay

__all__ = list(globals().get("__all__", [])) + [
    "AGEM",
    "ArchivalMemory",
    "DeepGenerativeReplay",
    "DualMemory",
    "EWC",
    "GEM",
    "LifecycleMemory",
    "PQMemory",
    "ResidualTree",
    "TheoreticalBound",
    "VQMemory",
]

from dense_armor.core.deep.neural.attention.moe import SparseMoE
from dense_armor.core.deep.neural.transformer.test_time import TestTimeCompute

__all__ = list(globals().get("__all__", [])) + [
    "SparseMoE",
    "TestTimeCompute",
]

from dense_armor.core.deep.neural import _moved  # noqa: E402,F401
