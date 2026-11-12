"""Building blocks for trading systems: schedules, pre-trade limits,
exchange message decoding and a preallocated ring buffer."""

from dense_armor.core.finance.audit_db import AuditDB
from dense_armor.core.finance.config import (
    FinanceConfig,
    PathsConfig,
    PreTradeConfig,
    RingConfig,
    ScheduleConfig,
    load_config,
)
from dense_armor.core.finance.decision_log import (
    GENESIS_HASH,
    DecisionLog,
    record_hash,
)
from dense_armor.core.finance.feed import frame, iter_itch
from dense_armor.core.finance.itch import decode_message
from dense_armor.core.finance.pretrade import Decision, PreTradeChecker
from dense_armor.core.finance.ring import RingBuffer
from dense_armor.core.finance.schedule import twap_schedule, vwap_schedule
from dense_armor.core.finance.synthetic import synthetic_itch

__all__ = [
    "RingBuffer",
    "decode_message",
    "Decision",
    "AuditDB",
    "DecisionLog",
    "record_hash",
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
    "synthetic_itch",
    "twap_schedule",
    "vwap_schedule",
]
