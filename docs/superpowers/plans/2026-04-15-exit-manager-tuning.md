# Exit Manager Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the `bars_open` time-exit counter bug (was counting 30s ticks, not 15m bars) and expose all exit parameters in `config.yaml` via a validated pydantic `ExitConfig` model.

**Architecture:** Move `ExitConfig` from a dataclass in `exit_manager.py` into `src/settings.py` as a pydantic model with a `model_validator` enforcing R-multiple ordering. Wire `on_bar_close()` into `kline_pump` so `bars_open` increments only on real 15m bar closes.

**Tech Stack:** Python 3.12+, pydantic v2 (`model_validator`), pytest + `AsyncMock`, structlog, prometheus-client.

---

## File Map

| File | What changes |
|---|---|
| `src/settings.py` | Add `ExitConfig` pydantic model; add `exit` field to `AppConfig` |
| `config/config.yaml` | Add `exit:` section |
| `src/execution/exit_manager.py` | Remove `ExitConfig` dataclass; accept pydantic `ExitConfig`; add `on_bar_close()`; remove stray `bars_open += 1` |
| `src/orchestrator.py` | Update import; pass `settings.config.exit` to `ExitManager`; call `exit_manager.on_bar_close()` in `kline_pump` |
| `tests/test_exit_manager.py` | Add two new tests |

---

### Task 1: Add `ExitConfig` pydantic model to `src/settings.py`

**Files:**
- Modify: `src/settings.py` (after `ExecutionConfig`, before `AppConfig`)

- [ ] **Step 1: Add `ExitConfig` to `src/settings.py`**

Open `src/settings.py`. After the `ExecutionConfig` class and before `AppConfig`, insert:

```python
class ExitConfig(BaseModel):
    """Dynamic exit-management parameters (breakeven, trailing SL, partial TP, time exit)."""

    atr_period: int = Field(14, ge=2)
    atr_trailing_mult: float = Field(1.5, gt=0.0)
    breakeven_trigger_r: float = Field(1.0, gt=0.0)
    partial_tp_trigger_r: float = Field(1.5, gt=0.0)
    partial_tp_fraction: float = Field(0.5, gt=0.0, lt=1.0)
    trailing_trigger_r: float = Field(2.0, gt=0.0)
    max_bars_open: int = Field(48, ge=1)
    time_exit_min_r: float = Field(0.5, ge=0.0)
    check_interval_sec: float = Field(30.0, gt=0.0)

    @model_validator(mode="after")
    def _r_ordering(self) -> "ExitConfig":
        if not (self.breakeven_trigger_r < self.partial_tp_trigger_r):
            raise ValueError(
                f"breakeven_trigger_r ({self.breakeven_trigger_r}) must be"
                f" < partial_tp_trigger_r ({self.partial_tp_trigger_r})"
            )
        if not (self.partial_tp_trigger_r <= self.trailing_trigger_r):
            raise ValueError(
                f"partial_tp_trigger_r ({self.partial_tp_trigger_r}) must be"
                f" <= trailing_trigger_r ({self.trailing_trigger_r})"
            )
        return self
```

`model_validator` is already imported at the top of `settings.py` (it's used by `RiskConfig`). No new imports needed.

- [ ] **Step 2: Add `exit` field to `AppConfig`**

In `src/settings.py`, find `class AppConfig(BaseModel)` and add the `exit` field after `execution`:

```python
class AppConfig(BaseModel):
    """Merged config.yaml — the `config` attribute of Settings."""

    mode: RuntimeMode = RuntimeMode.TESTNET
    exchanges: ExchangesConfig = Field(default_factory=ExchangesConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    exit: ExitConfig = Field(default_factory=ExitConfig)          # ← add this line
    paper: PaperConfig = Field(default_factory=PaperConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
```

- [ ] **Step 3: Verify settings load cleanly**

```bash
cd c:/Users/fear7/Desktop/cryprobotik
C:/Python314/python.exe -c "
from src.settings import ExitConfig, AppConfig
cfg = ExitConfig()
print(f'defaults OK: breakeven={cfg.breakeven_trigger_r} partial_tp={cfg.partial_tp_trigger_r} trailing={cfg.trailing_trigger_r}')
app = AppConfig()
print(f'AppConfig.exit OK: {app.exit}')
"
```

Expected output (no errors):
```
defaults OK: breakeven=1.0 partial_tp=1.5 trailing=2.0
AppConfig.exit OK: atr_period=14 ...
```

- [ ] **Step 4: Verify validator rejects bad ordering**

```bash
C:/Python314/python.exe -c "
from src.settings import ExitConfig
try:
    ExitConfig(breakeven_trigger_r=2.0, partial_tp_trigger_r=1.5)
    print('ERROR: should have raised')
except Exception as e:
    print(f'Correctly rejected: {e}')
"
```

Expected: `Correctly rejected: breakeven_trigger_r (2.0) must be < partial_tp_trigger_r (1.5)`

- [ ] **Step 5: Commit**

```bash
git add src/settings.py
git commit -m "feat: add ExitConfig pydantic model to settings with R-ordering validator"
```

---

### Task 2: Add `exit:` section to `config/config.yaml`

**Files:**
- Modify: `config/config.yaml` (after `execution:` block)

- [ ] **Step 1: Add `exit:` section**

Open `config/config.yaml`. After the `execution:` block, insert:

```yaml
# --- Exit management ----------------------------------------------------------
exit:
  atr_period: 14
  atr_trailing_mult: 1.5        # trailing SL = price ± (mult × ATR)
  breakeven_trigger_r: 1.0      # move SL to entry after 1R gain
  partial_tp_trigger_r: 1.5     # close partial_tp_fraction at 1.5R gain
  partial_tp_fraction: 0.5      # fraction of position closed at partial TP
  trailing_trigger_r: 2.0       # start trailing SL after 2R gain
  max_bars_open: 48              # time-exit after 48 × 15m bars (~12h)
  time_exit_min_r: 0.5          # skip time-exit if position already at ≥ this R
  check_interval_sec: 30.0      # how often the polling loop wakes up (seconds)
```

- [ ] **Step 2: Verify full config loads without errors**

```bash
C:/Python314/python.exe -c "
import yaml
from src.settings import AppConfig
cfg = AppConfig.model_validate(yaml.safe_load(open('config/config.yaml')))
print(f'exit config loaded: max_bars_open={cfg.exit.max_bars_open} check_interval={cfg.exit.check_interval_sec}')
"
```

Expected: `exit config loaded: max_bars_open=48 check_interval=30.0`

- [ ] **Step 3: Commit**

```bash
git add config/config.yaml
git commit -m "feat: add exit: section to config.yaml"
```

---

### Task 3: Update `exit_manager.py` — replace dataclass, accept pydantic config, add `on_bar_close`, fix counter

**Files:**
- Modify: `src/execution/exit_manager.py`

- [ ] **Step 1: Remove the `ExitConfig` dataclass and update imports**

In `src/execution/exit_manager.py`:

1. Remove the entire `ExitConfig` dataclass block (lines ~58–84, the `@dataclass(slots=True)` block named `ExitConfig`).

2. In the `TYPE_CHECKING` block at the top, add the import for the pydantic `ExitConfig`:

```python
if TYPE_CHECKING:
    from src.data.storage import Storage
    from src.exchanges.base import ExchangeConnector
    from src.portfolio.tracker import PortfolioTracker, TrackedPosition
    from src.settings import ExecutionConfig, ExitConfig  # ← add ExitConfig here
```

3. In `__init__`, update the type annotation for `config`:

```python
def __init__(
    self,
    tracker: "PortfolioTracker",
    feature_store: FeatureStore,
    connectors: dict[str, "ExchangeConnector"],
    config: "ExitConfig | None" = None,
) -> None:
    self._tracker = tracker
    self._store = feature_store
    self._connectors = connectors
    if config is None:
        from src.settings import ExitConfig as _ExitConfig  # noqa: PLC0415
        config = _ExitConfig()
    self._config = config
    self._states: dict[tuple[str, str], _ExitState] = {}
    self._lock = asyncio.Lock()
```

- [ ] **Step 2: Add `on_bar_close()` method**

In `src/execution/exit_manager.py`, after the `register_position` method and before `run()`, add:

```python
def on_bar_close(self, exchange: str, symbol: str, timeframe: str) -> None:
    """
    Increment the bar counter for a tracked position.
    Called by kline_pump on every closed bar — only 15m bars count.
    """
    if timeframe != "15m":
        return
    key = (exchange, symbol)
    if key in self._states:
        self._states[key].bars_open += 1
        log.debug(
            "exit_manager.bar_close",
            exchange=exchange,
            symbol=symbol,
            bars_open=self._states[key].bars_open,
        )
```

- [ ] **Step 3: Remove stray `bars_open += 1` from `_check_position`**

In `_check_position`, find and delete this line (currently around line 230):

```python
        # Increment bar count (approximate: loop fires every check_interval_sec;
        # 15m bars = 900 s, so each fire ≈ check_interval_sec / 900 bars).
        state.bars_open += 1
```

Delete both the comment and the `state.bars_open += 1` line. Bar counting is now exclusively handled by `on_bar_close()`.

- [ ] **Step 4: Verify the module imports cleanly**

```bash
C:/Python314/python.exe -c "from src.execution.exit_manager import ExitManager; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/execution/exit_manager.py
git commit -m "fix: replace ExitConfig dataclass with pydantic model; add on_bar_close(); remove stray bars_open increment"
```

---

### Task 4: Update `src/orchestrator.py` — pass config, wire `on_bar_close`

**Files:**
- Modify: `src/orchestrator.py`

- [ ] **Step 1: Update the `ExitConfig` import**

In `src/orchestrator.py`, find the current import:

```python
from src.execution.exit_manager import ExitConfig, ExitManager
```

Replace with (since `ExitConfig` now lives in `settings.py`):

```python
from src.execution.exit_manager import ExitManager
```

`ExitConfig` is accessed via `self._settings.config.exit` — no direct import needed.

- [ ] **Step 2: Pass config when constructing `ExitManager`**

Find the `ExitManager` construction block (~line 232):

```python
        # Exit manager — dynamic SL/TP management (breakeven, trailing, partial TP)
        self._exit_manager = ExitManager(
            tracker=self._tracker,
            feature_store=self._feature_store,
            connectors=self._connectors,
        )
```

Replace with:

```python
        # Exit manager — dynamic SL/TP management (breakeven, trailing, partial TP)
        self._exit_manager = ExitManager(
            tracker=self._tracker,
            feature_store=self._feature_store,
            connectors=self._connectors,
            config=self._settings.config.exit,
        )
```

- [ ] **Step 3: Wire `on_bar_close` in `kline_pump`**

In `kline_pump`, find the CVD bar-close call (~line 647):

```python
            # Finalise CVD bar-delta on every closed bar (any timeframe, any exchange).
            if evt.closed and self._cvd_store is not None:
                self._cvd_store.on_bar_close(exchange, evt.symbol)
```

Add the exit manager call immediately after:

```python
            # Finalise CVD bar-delta on every closed bar (any timeframe, any exchange).
            if evt.closed and self._cvd_store is not None:
                self._cvd_store.on_bar_close(exchange, evt.symbol)

            # Advance exit-manager bar counter (only counts 15m bars internally).
            if evt.closed and self._exit_manager is not None:
                self._exit_manager.on_bar_close(exchange, evt.symbol, evt.timeframe)
```

- [ ] **Step 4: Verify the orchestrator imports cleanly**

```bash
C:/Python314/python.exe -c "
import ast, sys
try:
    ast.parse(open('src/orchestrator.py').read())
    print('syntax OK')
except SyntaxError as e:
    print(f'syntax error: {e}')
    sys.exit(1)
"
```

Expected: `syntax OK`

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator.py
git commit -m "feat: wire ExitManager.on_bar_close into kline_pump; pass pydantic ExitConfig from settings"
```

---

### Task 5: Write and run the two new tests

**Files:**
- Modify: `tests/test_exit_manager.py` (create if it doesn't exist)

- [ ] **Step 1: Check whether `tests/test_exit_manager.py` exists**

```bash
ls tests/test_exit_manager.py 2>&1
```

If it doesn't exist, create a file with just the module docstring and imports before adding tests:

```python
"""Tests for src/execution/exit_manager.py."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, UTC

from src.execution.exit_manager import ExitManager, _ExitState
from src.settings import ExitConfig
```

If the file already exists, add the imports it's missing (check which ones are already there).

- [ ] **Step 2: Write `test_bars_open_counts_15m_only`**

Add this test to `tests/test_exit_manager.py`:

```python
def test_bars_open_counts_15m_only() -> None:
    """on_bar_close increments bars_open only for 15m bars."""
    tracker = MagicMock()
    tracker.open_positions.return_value = []
    store = MagicMock()
    cfg = ExitConfig()
    mgr = ExitManager(tracker=tracker, feature_store=store, connectors={}, config=cfg)

    # Register a position so the key exists in _states
    mgr.register_position(
        exchange="bybit",
        symbol="BTC/USDT:USDT",
        entry_ts=datetime.now(UTC),
        original_sl=29000.0,
        risk_usd=100.0,
    )

    # 1h bar — should NOT increment
    mgr.on_bar_close("bybit", "BTC/USDT:USDT", "1h")
    assert mgr._states[("bybit", "BTC/USDT:USDT")].bars_open == 0

    # 15m bar — should increment
    mgr.on_bar_close("bybit", "BTC/USDT:USDT", "15m")
    assert mgr._states[("bybit", "BTC/USDT:USDT")].bars_open == 1

    # 4h bar — should NOT increment
    mgr.on_bar_close("bybit", "BTC/USDT:USDT", "4h")
    assert mgr._states[("bybit", "BTC/USDT:USDT")].bars_open == 1
```

- [ ] **Step 3: Run it to verify it passes**

```bash
C:/Python314/python.exe -m pytest tests/test_exit_manager.py::test_bars_open_counts_15m_only -v
```

Expected: `PASSED`

- [ ] **Step 4: Write `test_time_exit_fires_at_correct_bar_count`**

Add this test to `tests/test_exit_manager.py`:

```python
@pytest.mark.asyncio
async def test_time_exit_fires_at_correct_bar_count() -> None:
    """Time exit fires after exactly max_bars_open 15m bars when R < time_exit_min_r."""
    from src.exchanges.base import PositionSide
    from src.portfolio.tracker import TrackedPosition

    tracker = MagicMock()
    store = MagicMock()
    store.as_df.return_value = None  # ATR unavailable — OK, time exit doesn't need it

    cfg = ExitConfig(max_bars_open=3, time_exit_min_r=0.5)
    conn = AsyncMock()
    # place_order returns a result with a non-rejected status
    from src.exchanges.base import OrderResult, OrderStatus
    conn.place_order.return_value = OrderResult(
        exchange_order_id="x1",
        client_order_id=None,
        symbol="BTC/USDT:USDT",
        status=OrderStatus.OPEN,
        filled_qty=0.0,
        avg_price=0.0,
    )

    mgr = ExitManager(
        tracker=tracker,
        feature_store=store,
        connectors={"bybit": conn},
        config=cfg,
    )

    entry_price = 30000.0
    original_sl = 29000.0  # SL distance = 1000; at 30100 price, R = 0.1 < 0.5

    mgr.register_position(
        exchange="bybit",
        symbol="BTC/USDT:USDT",
        entry_ts=datetime.now(UTC),
        original_sl=original_sl,
        risk_usd=100.0,
    )

    # Advance to exactly max_bars_open bars
    for _ in range(3):
        mgr.on_bar_close("bybit", "BTC/USDT:USDT", "15m")

    assert mgr._states[("bybit", "BTC/USDT:USDT")].bars_open == 3

    # Build a fake position where R < 0.5 (price barely moved)
    pos = TrackedPosition(
        exchange="bybit",
        symbol="BTC/USDT:USDT",
        side=PositionSide.LONG,
        quantity=0.01,
        entry_price=entry_price,
        mark_price=30100.0,   # R = (30100-30000)/1000 = 0.1 < 0.5
        unrealized_pnl=1.0,
        updated_at=datetime.now(UTC),
    )

    state = mgr._states[("bybit", "BTC/USDT:USDT")]
    await mgr._check_position(pos, state, conn)

    # Assert a market close order was placed
    conn.place_order.assert_called_once()
    call_kwargs = conn.place_order.call_args[0][0]  # first positional arg = OrderRequest
    assert call_kwargs.reduce_only is True
    assert call_kwargs.meta["exit"] == "time_stop"
```

- [ ] **Step 5: Run the new test**

```bash
C:/Python314/python.exe -m pytest tests/test_exit_manager.py::test_time_exit_fires_at_correct_bar_count -v
```

Expected: `PASSED`

- [ ] **Step 6: Run the full test suite to check for regressions**

```bash
C:/Python314/python.exe -m pytest -v -m "not integration" -q 2>&1 | tail -15
```

Expected: same 7 pre-existing failures, no new failures, `test_bars_open_counts_15m_only` and `test_time_exit_fires_at_correct_bar_count` both `PASSED`.

- [ ] **Step 7: Commit**

```bash
git add tests/test_exit_manager.py
git commit -m "test: add bar-counter and time-exit tests for ExitManager"
```

---

## Post-Implementation Checklist

- [ ] `pytest -v -m "not integration" -q` — no new failures
- [ ] `C:/Python314/python.exe -c "import yaml; from src.settings import AppConfig; cfg = AppConfig.model_validate(yaml.safe_load(open('config/config.yaml'))); print(cfg.exit)"` — loads cleanly
- [ ] Bot starts in testnet mode without error: `PYTHONIOENCODING=utf-8 C:/Python314/python.exe -m src.main --mode testnet` (check first 30s of logs for `orchestrator.setup_complete`)
