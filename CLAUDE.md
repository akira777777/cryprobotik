# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Unit tests (no exchange access, no DB required)
pytest -v -m "not integration"

# Single test file / single test
pytest tests/test_risk_manager.py -v
pytest tests/test_ensemble.py::test_name -v

# Integration tests (requires testnet API keys in .env)
pytest -v -m integration

# Lint / type-check
ruff check src/ tests/
mypy src/

# Apply DB schema manually (idempotent)
python -m src.data.storage --apply-schema

# Start bot (paper mode — safe default, no real orders)
python -m src.main --mode paper

# Start bot with local Python 3.13 venv (when Docker unavailable)
.venv313/Scripts/python.exe -m src.main --mode paper

# Start bot + localtunnel together (sets MINIAPP_URL automatically)
bash start_bot.sh paper

# Live mode requires BOTH --mode live AND --confirm-live
python -m src.main --mode live --confirm-live

# Docker workflow
docker compose up -d --build          # full stack
docker compose restart bot            # picks up config.yaml / .env changes only
docker compose build bot && docker compose up -d bot   # after code changes
docker compose logs -f bot
```

## Architecture

### Runtime flow

```
src/main.py → Orchestrator.setup() → Orchestrator.run() [asyncio.TaskGroup]
```

`Orchestrator` (`src/orchestrator.py`) is the single wiring point. It constructs every module, connects exchanges, bootstraps the instrument universe, and launches all long-running async tasks. Nothing else instantiates top-level objects.

### Signal pipeline (per closed bar)

```
Exchange WS → KlineEvent queue → kline_pump task
                                        ↓
                                 FeatureStore (in-memory rolling bars per FeatureKey)
                                   CVDStore (taker buy/sell flow per bar)
                                   OIStore  (open-interest snapshots)
                                        ↓
                                 Ensemble.evaluate_symbol()
                                   ├─ RegimeClassifier → Regime (ADX + realized vol → 5 regimes)
                                   ├─ MomentumStrategy, MeanReversionStrategy,
                                   │  VolatilityBreakoutStrategy, FundingContrarianStrategy
                                   └─ weighted net vote → Signal | None
                                        ↓
                                 MLSignalFilter.evaluate() → MLDecision (pass-through if cold start)
                                        ↓
                                 RiskManager.size_trade() → SizedTrade | RejectedTrade
                                        ↓
                                 PortfolioLimits.check() (correlation, max positions, margin)
                                        ↓
                                 OrderExecutor → OrderRouter → exchange REST
```

Only closed bars trigger evaluation — forming bars update the store but are not scored. Only the alphabetically-first **enabled** exchange (`signal_source` guard in `kline_pump`) triggers `_evaluate_and_execute` to prevent duplicate signals.

`FundingArbStrategy` is entirely separate — it emits `PairSignal` (two legs), bypasses the Ensemble and ML filter, and is handled directly by the orchestrator via `scan_arb_opportunities()`.

### Exchange layer (`src/exchanges/`)

- `ExchangeConnector` ABC (`base.py`) is the only interface used by strategies and the executor. Strategies never import from `okx.py` or `bybit.py`.
- REST via `ccxt.async_support`; WebSocket via native `websockets` through `WSManager` (`ws_manager.py`).
- Event queues exposed by every connector: `kline_events`, `funding_events`, `order_events`, `fill_events`, `position_events`, **`trade_events`** (individual aggressor trades for CVD), **`oi_events`** (open-interest snapshots).
- Abstract methods each connector must implement: `subscribe_klines`, `subscribe_funding`, **`subscribe_trades`**, **`unsubscribe_trades`**, **`fetch_open_interest`**, plus REST account/trading methods.
- `PaperConnector` wraps a real connector's market data, intercepts orders, and simulates fills locally. It also forwards `trade_events` from the underlying connector.
- **OKX candle WS subscriptions must go to `/ws/v5/business`**, not `/ws/v5/public` (OKX error 60018 otherwise). The OKX connector uses a separate `_ws_business` WSManager for kline channels. Trade subscriptions use the public WS.
- OKX trades WS channel: `{"channel": "trades", "instId": "BTC-USDT-SWAP"}` on the public WS.
- Bybit trades WS topic: `publicTrade.{SYMBOL}` on the public WS.
- Open interest is fetched via REST (`fetch_open_interest`) — polled every 5 minutes (configurable) by `_oi_poll_loop` in the orchestrator.

### Strategy system (`src/strategies/`)

| Strategy | File | Trigger |
|---|---|---|
| `MomentumStrategy` | `momentum.py` | EMA stack + RSI + MACD across 15m/1h/4h; volume confirmation (last bar ≥ `volume_multiplier × 20-bar mean`) |
| `MeanReversionStrategy` | `mean_reversion.py` | BB(20) + RSI(2) in ranging market (ADX < `adx_max`) |
| `VolatilityBreakoutStrategy` | `volatility_breakout.py` | Donchian squeeze + volume surge on 1h |
| `FundingArbStrategy` | `funding_arb.py` | Cross-exchange funding rate delta > `min_rate_delta`; emits `PairSignal` |
| `FundingContrarianStrategy` | `funding_contrarian.py` | Funding rate in top/bottom `extreme_threshold` percentile of rolling history → contrarian trade |

`FundingContrarianStrategy` requires a `FundingHistory` instance (injected at construction from `Orchestrator._funding_history`). It needs ≥20 historical samples before emitting any signal (cold-start guard). When funding is in the top 15% of its historical distribution, longs are paying excessive premium → emits SELL. Bottom 15% → emits BUY. **Hardening**: effective threshold is tightened to ~10th/90th percentile internally (`extreme_threshold * 0.67`). **Trend veto**: if 4h EMA21 is well below EMA55 (strong downtrend), SELL signals are vetoed; if EMA21 is well above EMA55 (strong uptrend), BUY signals are vetoed.

`MeanReversionStrategy` (`mean_reversion.py`): **Falling knife veto** — before emitting LONG, checks that 4h EMA21 is not declining (`EMA21[-1] >= EMA21[-5]`); suppresses if trending down. **ADX slope filter** — only enters when ADX is flat or falling (`ADX[-1] <= ADX[-3] + 1.0`), confirming trend momentum is fading.

`RegimeClassifier` (`regime.py`): ADX + realized vol + **Hurst exponent** → 5 regimes (`trend_high_vol`, `trend_low_vol`, `range_high_vol`, `range_low_vol`, `chop`). Hysteresis: last N bars must agree before regime switches (`regime_hysteresis_bars` in config, default 2). The `_hurst(prices, max_lag=20)` function uses the R/S (rescaled range) method; H > 0.55 = trending, H < 0.45 = mean-reverting, otherwise chop. It acts as a tiebreaker: e.g. ADX says `chop` but H > 0.55 → upgraded to `trend_*`; ADX says `trend_*` but H < 0.42 → downgraded to `range_*`.

`Ensemble` (`ensemble.py`): multiplies each signal's confidence by the regime weight matrix, nets long vs short votes, emits a consolidated `Signal` if `abs(net_vote) ≥ min_net_vote`. Both `min_net_vote` and the weight matrix live in `config/config.yaml` under `regime:`.

### Perp-native signal stores (`src/data/feature_store.py`)

Three stores are instantiated in `Orchestrator.setup()` alongside `FeatureStore`:

- **`CVDStore`** — per-bar taker buy/sell flow. `on_trade(exchange, symbol, side, qty)` accumulates each trade. `on_bar_close(exchange, symbol)` finalises the bar's net delta. `cvd_ratio(exchange, symbol, lookback=20)` returns taker-buy fraction `[0,1]` (>0.55 = buy pressure). `trend_aligned()` checks if CVD aligns with an intended trade side.
- **`OIStore`** — rolling open-interest history. `update(exchange, symbol, oi_contracts)` appends snapshots; `oi_roc(exchange, symbol, periods=5)` returns the rate of change as a fraction.
- **`FundingHistory`** — rolling funding rate distribution. `update(exchange, symbol, rate)` feeds history; `percentile(exchange, symbol, rate)` returns `[0,1]` rank. Used by `FundingContrarianStrategy`.

### Risk layer (`src/risk/`)

Three components with distinct responsibilities:
- `RiskManager` — **stateless**. ATR-based sizing: `qty = (equity × risk_per_trade_pct) / (SL_distance × contract_size)`. Computes SL/TP, enforces leverage and min-notional.
- `PortfolioLimits` — **stateful**. Enforces max open positions, per-symbol cap, correlation cap (1h close returns), and margin utilization. Correlation matrix is recomputed from `FeatureStore` data at check time.
- `KillSwitch` — persisted to `bot_state` DB table. Trips when daily drawdown exceeds `max_daily_drawdown_pct`. Requires a manual DB row delete to reset.

Hard ceilings (leverage ≤ 10, risk_per_trade ≤ 5%, etc.) are enforced at startup in `src/settings.py` — exceeding them causes a validation error before the bot touches any exchange.

### ML signal filter (`src/ml/`)

Online learning filter between Ensemble and RiskManager:
- Cold start (`n_samples < MIN_SAMPLES = 50`): all signals pass through at score 0.5.
- Warm: `GradientBoostingClassifier` scores signals; rejected below `ACCEPT_THRESHOLD = 0.60`.
- Retrains every `RETRAIN_EVERY = 20` new labeled outcomes in a thread pool (`run_in_executor`).
- Labels come from `realized_pnl` on fill: `1` if profitable, `0` otherwise. Labels within `FEE_FLOOR_PCT = 0.1%` of notional are skipped as noise.
- Model + last 500 training examples persisted via `joblib` to `bot_state` DB table (key `ml.model_state`). HMAC-SHA256 signed with `MODEL_HMAC_SECRET` env var — if unset, persistence is disabled.
- **`FEATURE_VERSION = 3`** — bumping this discards all persisted models on restart (intentional cold-start). Current vector is **25 features**: RSI/MACD/EMA-slope/BB-width/ADX/Donchian/ATR across 15m+1h+4h, regime one-hot (5 flags), hour/weekday sin-cos, **`cvd_ratio`** (taker buy fraction), **`oi_roc`** (OI rate of change). `MLSignalFilter` accepts optional `cvd_store` and `oi_store` constructor params and passes them to `extract_features`.
- **Feature cache** (`src/ml/features.py`): module-level `_feature_cache: dict[tuple, list[float]]` (max 50 entries, LRU-evict). Key is `(symbol, exchange, bar_ts_ms)`. The 9 indicator functions run only once per symbol+bar; subsequent calls within the same bar are O(1) lookups — ~80% CPU reduction when the ensemble evaluates multiple strategies for the same bar.
- **River online learning** (`src/ml/model.py`, optional): if `river` is installed (`pip install river`), `_online_model` (River `AdaptiveRandomForestClassifier`) learns after every single trade outcome via `learn_one()`. Score is blended `0.6 × sklearn + 0.4 × river` when the online model has ≥10 samples. Gracefully disabled via `try/except ImportError` when river is absent — zero behavior change without it. `stats()` includes `"online_model_samples"` count.

### Config and secrets

- All tunables: `config/config.yaml` (loaded as typed `AppConfig` via pydantic).
- All secrets: `.env` via `pydantic-settings` (`DATABASE_URL` has no default — bot fails to start without it).
- Mode precedence: `--mode` CLI > `CRYPROBOTIK_MODE` env var > `mode:` in config.yaml.
- Config hard ceilings: `src/settings.py` constants (`MAX_LEVERAGE_CEILING`, `MAX_RISK_PER_TRADE_CEILING`, etc.).
- New config sections under `strategies:`: `funding_contrarian` (`extreme_threshold`, `low_threshold`, `base_confidence`), `cvd` (`enabled`, `max_bars`), `oi` (`enabled`, `poll_interval_sec`, `max_samples`).

### Database (`src/data/storage.py`)

Thin asyncpg wrapper. Schema at `src/data/schema.sql` applied idempotently on boot. Key tables: `ohlcv`, `signals`, `orders`, `fills`, `positions`, `bot_state`, `ml_decisions`, `events`. The `equity_daily` materialized view requires manual refresh.

### Monitoring endpoints (port 8080)

| Endpoint | Description |
|---|---|
| `/app` | **Telegram Mini App** — full-screen live dashboard (4 tabs: signals, positions, ML, regime) |
| `/ml/dashboard` | Full live HTML dashboard (P&L, equity curve, ML stats, SSE feed) |
| `/ml/stats` | JSON model status |
| `/ml/stream` | Server-Sent Events — one event per ML decision (zero-delay) |
| `/ml/decisions` | JSON: last 50 ML decisions from DB |
| `/trading/stats` | Paper account summary |
| `/trading/fills` | Last 50 fills |
| `/api/positions` | JSON: current open positions (for Mini App initial load) |
| `/ws/live` | WebSocket — pushed by orchestrator on signal/regime events (zero-delay) |
| `/metrics` | Prometheus metrics |

**Mini App real-time stack**: `/app` uses SSE (`/ml/stream`) for zero-delay ML decisions and WebSocket (`/ws/live`) for regime + signal push from orchestrator. Falls back to polling `/trading/stats` and `/ml/stats` every 10s. Telegram `/webapp` command sends an inline button opening the Mini App — set `MINIAPP_URL=https://your-host:8080/app` env var to enable the Telegram WebApp button (requires HTTPS); defaults to a plain link to `http://localhost:8080/app`.

**Public deployment**: `start_bot.sh` uses `localtunnel` (`npm install -g localtunnel`) to expose port 8080 as `https://cryprobotik.loca.lt`, sets `MINIAPP_URL` automatically, then starts the bot. Active deployment: `https://cryprobotik.loca.lt/app`.

`LiveBroadcaster` (`src/monitoring/health.py`) is instantiated in `Orchestrator.__init__` as `self._broadcaster`, passed to `build_app()`, and called via `.push(event_dict)` at each of the 4 signal rejection/execution points in `_evaluate_and_execute_inner` plus on every bar close (regime snapshot). The `.push()` call is non-blocking and drops events if a subscriber queue is full — intentional to prevent slow WebSocket clients from backpressuring the trading loop.

### Logging

All logs are structlog JSON. Use `get_logger(__name__)`. Never pass WS message dicts with `**msg` as kwargs — OKX/Bybit messages have an `event` field that conflicts with structlog's reserved `event` kwarg. Extract specific fields instead.

### Key gotchas

- **Telegram commands**: `/status`, `/positions`, `/pnl`, `/halt`, `/resume`, `/minimap` (auto-refreshing pinned text dashboard, 60s interval), `/webapp` (sends inline button for the Mini App). All commands are auth-gated by `TELEGRAM_CHAT_IDS`.
- **Adding a new abstract method to `ExchangeConnector` in `base.py`** requires implementing it in **all three** connectors: `okx.py`, `bybit.py`, and `paper.py`.
- **Stale signal guard**: OKX sends the last confirmed candle as a WS subscription snapshot. `kline_pump` skips any bar with `evt.ts <= _startup_ts` (set after backfill completes) to avoid trading on stale data.
- **CVD bar finalisation**: `cvd_store.on_bar_close(exchange, symbol)` is called in `kline_pump` on **every** closed bar regardless of timeframe. CVD features are thus always based on the most recently closed bar of any timeframe.
- **`bybit.enabled: false`** in the default config — enable only if `BYBIT_API_KEY` is set in `.env`.
- **`-m "not integration"`** is required for unit tests — integration tests hit real exchange REST and will fail without valid testnet credentials.
- `pyproject.toml` sets `asyncio_mode = "auto"` — all async tests run automatically without `@pytest.mark.asyncio`.
- `docker compose restart bot` reuses the existing image — it picks up `.env` and `config.yaml` changes but **not** code changes. Always rebuild after editing `.py` files.
- **`positions_snapshot` not `positions`**: the schema table for position tracking is `positions_snapshot` (no `is_open` column, keyed by `(ts, exchange, symbol)`). Any raw SQL querying positions must use `DISTINCT ON (exchange, symbol) ... ORDER BY exchange, symbol, ts DESC` and filter `side != 'flat'` to get current open positions.
