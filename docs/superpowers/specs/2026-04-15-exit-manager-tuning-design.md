# Exit Manager Tuning — Design Spec

**Date:** 2026-04-15
**Scope:** Fix `bars_open` time-exit bug; expose all exit parameters in `config.yaml`
**Status:** Approved

---

## Problem

Two issues in the current `ExitManager` (`src/execution/exit_manager.py`):

1. **`bars_open` bug** — the counter increments on every 30-second polling tick, not on actual 15m bar closes. `max_bars_open: 48` was intended to mean 48 × 15m = 12 hours, but actually fires after 48 × 30s = 24 minutes.

2. **Hardcoded parameters** — all `ExitConfig` values (ATR multiplier, R-multiple triggers, partial TP fraction, time-exit limits) live only as Python dataclass defaults. Tuning requires code changes and a redeploy rather than a config edit.

---

## Approach

**Option A (chosen):** Hook `on_bar_close` into the existing `kline_pump` bar-close event flow. Expose all `ExitConfig` parameters under a new `exit:` section in `config.yaml` backed by a pydantic model with startup validation.

Rejected alternatives:
- **Option B** (event-driven queue refactor): larger blast radius, premature for a bug fix.
- **Option C** (wall-clock time): ignores market structure; `max_hold_hours` is less meaningful than `max_bars_open` for a bar-driven strategy.

---

## Design

### 1. Config exposure

New `exit:` section in `config/config.yaml`:

```yaml
exit:
  atr_period: 14
  atr_trailing_mult: 1.5        # trailing SL = price ± (mult × ATR)
  breakeven_trigger_r: 1.0      # move SL to entry after 1R gain
  partial_tp_trigger_r: 1.5     # close partial_tp_fraction at 1.5R gain
  partial_tp_fraction: 0.5      # fraction of position closed at partial TP
  trailing_trigger_r: 2.0       # start trailing SL after 2R gain
  max_bars_open: 48              # time-exit after 48 × 15m bars = 12h
  time_exit_min_r: 0.5          # skip time-exit if position already at ≥ this R
  check_interval_sec: 30.0      # how often the polling loop wakes up
```

New pydantic model `ExitConfig` in `src/settings.py`:

```python
class ExitConfig(BaseModel):
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
            raise ValueError("breakeven_trigger_r must be < partial_tp_trigger_r")
        if not (self.partial_tp_trigger_r <= self.trailing_trigger_r):
            raise ValueError("partial_tp_trigger_r must be <= trailing_trigger_r")
        return self
```

`AppConfig` gains `exit: ExitConfig = Field(default_factory=ExitConfig)`.

`ExitManager.__init__` accepts `config: ExitConfig` (replacing the current `ExitConfig | None` with dataclass defaults). The orchestrator passes `settings.config.exit`.

### 2. Bar-close counter fix

**`ExitManager`** gains a new public method:

```python
def on_bar_close(self, exchange: str, symbol: str, timeframe: str) -> None:
    """Increment bar counter for 15m bars only. Called by kline_pump."""
    if timeframe != "15m":
        return
    key = (exchange, symbol)
    if key in self._states:
        self._states[key].bars_open += 1
```

**`_check_position`** — remove the existing `state.bars_open += 1` line (currently inside the polling loop body). Bar counting now comes exclusively from `on_bar_close`.

**`orchestrator.py` `kline_pump`** — after the existing `cvd_store.on_bar_close(...)` call:

```python
if self._exit_manager is not None:
    self._exit_manager.on_bar_close(exchange, symbol, evt.timeframe)
```

The polling loop (`run()`) continues to run every `check_interval_sec` for breakeven/trailing/partial-TP checks — these correctly fire at tick resolution. Only the time-exit counter moves to bar-close events.

### 3. Testing

Two new tests in `tests/test_exit_manager.py`:

**`test_bars_open_counts_15m_only`**
- Register a position on `ExitManager`
- Call `on_bar_close(exchange, symbol, "1h")` → assert `bars_open == 0`
- Call `on_bar_close(exchange, symbol, "15m")` → assert `bars_open == 1`
- Call `on_bar_close(exchange, symbol, "4h")` → assert `bars_open == 1` (still)

**`test_time_exit_fires_at_correct_bar_count`**
- Create `ExitConfig(max_bars_open=3, time_exit_min_r=0.5)`
- Register a position with `original_sl` set so R-multiple stays below 0.5
- Call `on_bar_close` 3 times on `"15m"`
- Call `_check_position` with a mocked connector (`AsyncMock`)
- Assert `conn.place_order` was called once with `reduce_only=True` and `meta["exit"] == "time_stop"`
- Assert `exits_time_stop_total` Prometheus counter incremented

Both tests use `AsyncMock` for the connector — no real exchange calls.

### 4. Validation at startup

If `config.yaml` has `breakeven_trigger_r >= partial_tp_trigger_r`, the pydantic `model_validator` raises `ValueError` during `AppConfig` load in `src/settings.py`. The bot exits before connecting to any exchange, with a clear error message.

---

## Files Changed

| File | Change |
|---|---|
| `config/config.yaml` | Add `exit:` section with all parameters |
| `src/settings.py` | Add `ExitConfig` pydantic model; add `exit` field to `AppConfig` |
| `src/execution/exit_manager.py` | Add `on_bar_close()`; remove `bars_open += 1` from polling loop; accept `ExitConfig` from settings |
| `src/orchestrator.py` | Pass `settings.config.exit` to `ExitManager`; call `exit_manager.on_bar_close()` in `kline_pump` |
| `tests/test_exit_manager.py` | Add two new tests |

---

## Non-goals

- Regime-aware exit parameters (different trailing multiplier in trending vs ranging) — separate future improvement
- Second partial TP level — separate future improvement
- ATR timeframe selection per entry timeframe — separate future improvement
