"""Building blocks for trading systems: schedules, pre-trade limits,
exchange message decoding and a preallocated ring buffer."""

from dense_armor.utility.finance.config import (
    FinanceConfig,
    PathsConfig,
    PreTradeConfig,
    RingConfig,
    ScheduleConfig,
    load_config,
)
from dense_armor.utility.finance.decision_log import GENESIS_HASH, DecisionLog
from dense_armor.utility.finance.feed import frame, iter_itch
from dense_armor.utility.finance.itch import decode_message
from dense_armor.utility.finance.pretrade import Decision, PreTradeChecker
from dense_armor.utility.finance.ring import RingBuffer
from dense_armor.utility.finance.schedule import twap_schedule, vwap_schedule

__all__ = [
    "RingBuffer",
    "decode_message",
    "Decision",
    "DecisionLog",
    "FinanceConfig",
    "GENESIS_HASH",
    "PathsConfig",
    "PreTradeChecker",
    "PreTradeConfig",
    "RingConfig",
    "ScheduleConfig",
    "frame",
    "iter_itch",
    "load_config",
    "twap_schedule",
    "vwap_schedule",
]
