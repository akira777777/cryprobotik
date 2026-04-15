# AGENTS.md

Guidance for agentic coding agents working in the **cryprobotik** repository — an autonomous multi-exchange (OKX + Bybit) crypto perpetual-futures trading bot. Read `CLAUDE.md` first for runtime architecture (signal pipeline, exchange layer, risk modules). This file covers commands and code style.

## Build / lint / test commands

```bash
# Unit tests (no exchange access, no DB) — DEFAULT before any commit
pytest -v -m "not integration"

# Single test file
pytest tests/test_risk_manager.py -v

# Single test by node-id (function or class::function)
pytest tests/test_ensemble.py::test_confidence_capped_at_one -v
pytest tests/test_kill_switch.py::TestKillSwitch::test_halt_survives_day_boundary -v

# Filter by keyword across all tests
pytest -v -m "not integration" -k "regime and hysteresis"

# Integration tests (require live testnet API keys in .env)
pytest -v -m integration

# Lint + auto-fix
ruff check src/ tests/ --fix
ruff format src/ tests/

# Type-check (strict mode, pydantic plugin enabled)
mypy src/

# Apply DB schema (idempotent)
python -m src.data.storage --apply-schema

# Run bot — paper mode is the SAFE default
python -m src.main --mode paper
# Live mode REQUIRES both flags — never omit --confirm-live
python -m src.main --mode live --confirm-live

# Docker (rebuild after .py changes; restart only suffices for .env / config.yaml)
docker compose build bot && docker compose up -d bot
docker compose logs -f bot
```

`pytest.ini_options.asyncio_mode = "auto"` is set in `pyproject.toml`, so async tests run without `@pytest.mark.asyncio`. Use `@pytest.mark.integration` to mark exchange-touching tests.

## Code style

### Imports
- `from __future__ import annotations` at the top of every module (already standard).
- Order: stdlib → third-party → `src.*` (ruff `I` rule enforces this — let ruff format).
- Use `from typing import TYPE_CHECKING` + a guarded block for type-only imports to avoid circular deps and runtime cost. Example: `if TYPE_CHECKING: from src.settings import RiskConfig`.
- Never `import *`. Avoid relative imports — always `from src.foo import bar`.

### Formatting & line length
- `ruff format` is the source of truth. Line length 110, target Python 3.12.
- Ruff lint rules enabled: `E F W I B UP ASYNC RUF`. `E501` and `B008` are intentionally ignored.
- Run `ruff check src/ tests/` before committing.

### Types
- `mypy --strict` is enabled with `pydantic.mypy` plugin. **Annotate every function signature and return type.**
- Prefer modern syntax: `list[int]`, `dict[str, float]`, `X | None` (PEP 604) — never `List`, `Dict`, `Optional`.
- Use `TYPE_CHECKING` to import heavy or circular types for annotations only.
- Avoid `Any` and `# type: ignore` — fix the underlying type instead. If unavoidable, narrow the ignore: `# type: ignore[attr-defined]`.
- Connector parameters must be typed as `ExchangeConnector`, never `object`.

### Naming
- `snake_case` for functions, methods, variables, modules.
- `PascalCase` for classes, dataclasses, `StrEnum` types.
- `_leading_underscore` for private/internal attributes and helpers.
- `SCREAMING_SNAKE_CASE` for module-level constants (e.g. `MIN_SAMPLES`, `BOT_STATE_KEY`).
- Test functions begin with `test_`; test classes with `Test`.

### Dataclasses & data types
- Prefer `@dataclass(slots=True)` for value objects (`Signal`, `SizedTrade`, `KlineEvent`, etc.) — saves memory and prevents typo-bugs.
- Use `StrEnum` (Python 3.12+) for string enums (`Regime`, `OrderSide`, `SignalAction`). Compare with `==` against the enum value, not raw strings.
- Configuration is pydantic v2 models in `src/settings.py`. Add new tunables there, never as magic numbers in business code.

### Async
- Bot runs on a single asyncio event loop via `asyncio.TaskGroup` in `Orchestrator`. **Never** call blocking I/O (sync `requests`, `time.sleep`, sync DB drivers) from coroutines.
- Wrap CPU-bound work (sklearn, heavy pandas) with `await loop.run_in_executor(None, fn, ...)`.
- Always wrap network calls with `asyncio.wait_for(..., timeout=N)`. Hung exchange REST stalls the entire pipeline.
- Track every `asyncio.create_task` in a `set` and `task.add_done_callback(set.discard)` — never fire-and-forget. Unsupervised tasks swallow exceptions and break shutdown.
- Don't hold locks across `await` to a callback or external service (Telegram, exchange). Compute under the lock, release, then await.
- Take a snapshot (`list(d.values())`) before iterating dicts/sets that other coroutines mutate.

### Error handling
- **Never** `except Exception: pass`. Every caught exception must be logged with structured context (at least `error=str(e)` and the operation name) or re-raised.
- Catch ccxt-specific exception types in the executor: `ccxt.InsufficientFunds`/`InvalidOrder` → `PermanentOrderError` (no retry); `ccxt.NetworkError`/`RequestTimeout`/`RateLimitExceeded` → `TransientOrderError` (retry once). Default unknown exceptions to **permanent** to avoid double-fills.
- NaN-safe numeric checks: use `math.isnan(x)` / `math.isfinite(x)`, not the `x != x` idiom.
- Validate inputs at boundaries (pydantic at config-load, explicit checks in `RiskManager.size_trade`). Once past validation, trust types.

### Logging
- Always use `structlog` via `from src.utils.logging import get_logger; log = get_logger(__name__)`. Never `print` and never the stdlib `logging` module directly.
- Pass structured fields as kwargs: `log.info("orchestrator.signal_emitted", symbol=sym, side=side.value, conf=round(c, 3))`. Use a dotted `module.event` name.
- **Never** spread WS dicts as kwargs (`**msg`) — OKX/Bybit messages contain an `event` key that collides with structlog's reserved `event` kwarg.
- **Never** log full private-channel WS payloads. Extract only safe fields (`instId`, `ordId`, `state`). API keys and order details must not appear in logs.
- Use `log.debug` for hot-path noise, `log.info` for state transitions, `log.warning` for recoverable anomalies, `log.error` for actionable failures.

### Tests
- Place tests in `tests/` mirroring `src/` layout. Use `tests/conftest.py` fixtures (`sample_ohlcv_df`, etc.) instead of inline synthetic data.
- Async tests need no decorator — `asyncio_mode = "auto"`.
- Mock external dependencies with `unittest.mock.AsyncMock` (DB, exchanges, Telegram). Never hit a real network in `not integration` tests.
- One assertion per logical concern. Prefer many small tests over one mega-test. Use `pytest.mark.parametrize` for table-driven cases.
- Risk-critical changes (sizing, kill-switch, regime, ML labeling) MUST land with new tests in the same change.

## Critical safety rules

- **Never** weaken the hard ceilings in `src/settings.py` (`MAX_LEVERAGE_CEILING`, `MAX_RISK_PER_TRADE_CEILING`, etc.) — they are the last defense against config typos.
- **Never** remove the `--confirm-live` requirement in `src/main.py`.
- **Never** commit `.env` or any file containing API keys / secrets / DSNs. Verify `.gitignore` covers `.env*` before staging.
- The kill switch (`src/risk/kill_switch.py`) state is persisted to the DB and **does not auto-clear** on day roll. Use `python -m scripts.reset_halt` to reset deliberately.
- Only the alphabetically-first enabled exchange triggers `_evaluate_and_execute` (signal-source guard in `kline_pump`). Don't accidentally remove it.
- Closed bars only — never trade on a forming bar. Higher-timeframe indicator reads in strategies must use `iloc[-2]` (last closed bar), not `iloc[-1]` (forming).
- After editing `.py` files in Docker, **rebuild** — `docker compose restart bot` reuses the cached image and only picks up `.env` / `config.yaml` changes.
