# Finance primitives — the map the dashboard must expose

Modules: 13.  Public names: 25.

| module | kind | signature |
|---|---|---|
| `audit_db` | class | `AuditDB(append, last_hash, verify)` |
| `audit_db` | func | `metadata()` |
| `config` | class | `FinanceConfig(make_checker, make_ring)` |
| `config` | class | `PathsConfig()` |
| `config` | class | `PreTradeConfig()` |
| `config` | class | `RingConfig()` |
| `config` | class | `ScheduleConfig()` |
| `config` | func | `load_config(path: 'str | Path') -> 'FinanceConfig'` |
| `dashboard` | func | `create_app(config_path: 'str') -> 'Any'` |
| `dashboard` | func | `main(argv: 'list[str] | None' = None) -> 'int'` |
| `decision_log` | class | `DecisionLog(GENESIS_HASH, append, verify)` |
| `decision_log` | func | `record_hash(seq: 'int', t: 'float', payload: 'Any', prev: 'str') -> 'str'` |
| `demo` | func | `main(argv: 'list[str] | None' = None) -> 'int'` |
| `feed` | func | `frame(message: 'bytes | bytearray | memoryview') -> 'bytes'` |
| `feed` | func | `iter_itch(path: 'str | Path') -> 'Iterator[dict]'` |
| `fix_gateway` | func | `send_orders(settings_path: 'str | Path', orders: 'list[dict[str, Any]]', timeout_s: 'float' = 10.0) -> 'list[dict[str, Any]]'` |
| `itch` | func | `decode_message(buf: bytes) -> dict` |
| `pretrade` | class | `Decision()` |
| `pretrade` | class | `PreTradeChecker(check, open_notional, release)` |
| `ring` | class | `RingBuffer(pop_into, push)` |
| `schedule` | func | `twap_schedule(total: float, n: int) -> numpy.ndarray` |
| `schedule` | func | `vwap_schedule(total: float, history: numpy.ndarray) -> numpy.ndarray` |
| `status` | func | `collect_status(config_path: 'str') -> 'dict'` |
| `status` | func | `main(argv: 'list[str] | None' = None) -> 'int'` |
| `synthetic` | func | `synthetic_itch(n: 'int', seed: 'int') -> 'bytes'` |
