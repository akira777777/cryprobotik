# Cryprobotik

Autonomous multi-exchange cryptocurrency perpetual-futures trading bot for **OKX** and **Bybit**.

Python 3.12, asyncio, multi-strategy ensemble with market-regime detection, hard risk ceilings, TimescaleDB persistence, Telegram operator interface, Prometheus/Grafana observability.

---

## ⚠️ Risk warning

Trading leveraged crypto derivatives is **extremely high risk**. Autonomous bots can lose 100% of allocated capital faster than a human can react. This codebase ships with conservative defaults, hard-coded safety ceilings, and a mandatory multi-phase validation procedure. **Do not skip the testnet phase.** See `DEPLOYMENT.md` for the full rollout protocol.

The author(s) and contributors assume no liability for losses incurred by running this software. You are solely responsible for understanding what the bot is doing on your behalf before funding it with real money.

---

## Features

- **Unified connector layer** for OKX v5 + Bybit v5 (REST via `ccxt.async_support`, WebSockets native).
- **Five strategy modules**: momentum (multi-TF EMA/RSI/MACD), Bollinger mean reversion, volatility breakout, cross-exchange funding arbitrage, regime-aware ensemble.
- **ADX + realized-vol regime classifier** weights strategies dynamically per market condition.
- **Strict risk manager**: ATR-based position sizing, min 1.5:1 RR, correlation caps, per-symbol exposure limits.
- **Hard-coded safety ceilings** in `src/settings.py` — leverage, drawdown, risk-per-trade cannot be configured above fixed limits.
- **Daily drawdown kill switch** — persisted to DB, requires manual reset.
- **Three runtime modes** toggled via config/CLI: `testnet`, `paper` (mainnet data, simulated fills), `live`.
- **Live mode guard** — requires `--confirm-live` CLI flag in addition to config change.
- **Telegram notifications** with `/status`, `/positions`, `/pnl`, `/halt`, `/resume` commands.
- **FastAPI health endpoint** (`/health`, `/ready`, `/metrics`) for external monitoring and Prometheus scraping.
- **TimescaleDB** hypertables for OHLCV, orders, fills, equity curve, events.
- **Structlog JSON logging** — stdout → Docker json-file driver → any log shipper.
- **Graceful shutdown** and auto-reconnect on WS drops / DB restarts.

## Quickstart (local dev)

```bash
# 1. Clone, create env file
cp .env.example .env
#    fill in OKX_*, BYBIT_*, TELEGRAM_*, POSTGRES_PASSWORD

# 2. Start the stack (bot + TimescaleDB)
docker compose up -d --build

# 3. Tail the logs
docker compose logs -f bot

# 4. Hit the health endpoint
curl http://127.0.0.1:8080/health
```

The bot starts in **testnet mode** by default. Do not change the mode in `config/config.yaml` until you've completed the validation protocol in `DEPLOYMENT.md`.

## Running tests

```bash
# Unit tests
pytest -v -m "not integration"

# Integration tests against exchange testnets (requires valid testnet API keys in .env)
pytest -v -m integration
```

## Project layout

```
src/
  main.py                 # CLI entrypoint
  orchestrator.py         # top-level async task supervisor
  settings.py             # pydantic config + HARD CEILINGS
  exchanges/              # OKX, Bybit, paper connectors
  strategies/             # momentum, mean_reversion, funding_arb, vol_breakout, regime, ensemble
  risk/                   # position sizing, limits, kill switch
  execution/              # order executor, rate limiter, exchange router
  portfolio/              # position tracker, analytics
  data/                   # TimescaleDB storage, instrument universe, feature store
  notifications/          # Telegram client
  monitoring/             # FastAPI health + Prometheus metrics
  utils/                  # logging, indicators, time
tests/
config/config.yaml        # single source of truth for tunables
docker/                   # Dockerfile, entrypoint, Prometheus config
docker-compose.yml        # bot + timescaledb + optional prometheus/grafana profile
```

## Configuration

All tunables live in `config/config.yaml`. Secrets (API keys, DB password, Telegram token) live in `.env` and are loaded via `pydantic-settings`. Never commit `.env`.

### Runtime modes

| Mode | Data source | Order fills | Use for |
|---|---|---|---|
| `testnet` | OKX demo + Bybit testnet | real (on testnet) | initial validation |
| `paper` | live mainnet | simulated locally | dry-run on real prices |
| `live` | live mainnet | real money | only after full validation protocol |

Switch via `mode:` in `config.yaml` **or** `CRYPROBOTIK_MODE` env var **or** `--mode` CLI flag. Live mode additionally requires `--confirm-live`.

### Hard ceilings

`src/settings.py` refuses to load any config that exceeds:

| Setting | Ceiling |
|---|---|
| `risk.max_daily_drawdown_pct` | 0.30 |
| `risk.risk_per_trade_pct` | 0.05 |
| `risk.leverage` | 10 |
| `risk.max_open_positions` | 20 |
| `risk.max_margin_utilization` | 0.90 |

Raising a ceiling requires editing `settings.py` directly and is treated as a deliberate safety review event.

## Deployment

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for the full zero-to-live rollout procedure including VPS provisioning, testnet validation checklist, paper staging phase, and the `--confirm-live` cutover.

## License

Proprietary / unlicensed. Do not redistribute.
