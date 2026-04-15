# Cryprobotik — Deployment Guide

> **Read this entire document before putting real money on the line.**
>
> Trading crypto perpetual futures with leverage is extremely high risk. A
> fully autonomous bot can lose 100% of allocated capital faster than a human
> can react. Every shortcut you take on the steps below increases the
> probability of losing money you cannot afford to lose. **Capital
> preservation > returns** is a non-negotiable principle, enforced in the
> risk module and reinforced by this checklist.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Phase 0 — Local smoke test](#2-phase-0--local-smoke-test)
3. [Phase 1 — VPS provisioning](#3-phase-1--vps-provisioning)
4. [Phase 2 — First boot on testnet](#4-phase-2--first-boot-on-testnet)
5. [Phase 3 — Testnet validation (2 weeks minimum)](#5-phase-3--testnet-validation-2-weeks-minimum)
6. [Phase 4 — Paper mode on mainnet (1 week minimum)](#6-phase-4--paper-mode-on-mainnet-1-week-minimum)
7. [Phase 5 — Live cutover](#7-phase-5--live-cutover)
8. [Phase 6 — Ongoing operations](#8-phase-6--ongoing-operations)
9. [Incident playbooks](#9-incident-playbooks)
10. [Appendices](#10-appendices)

---

## 1. Prerequisites

### 1.1 Accounts and credentials

- **OKX account** with perpetual futures (USDT-M) enabled.
  - Create **two** API key sets:
    - `OKX_TESTNET` — from `https://www.okx.com/account/my-api` with the "Demo Trading" / testnet toggle on.
    - `OKX_LIVE` — create *after* testnet validation passes, not before.
  - Permissions: **Read** + **Trade** only. **NEVER enable Withdraw.**
  - IP whitelist the API key to your VPS public IP (create the key *after* provisioning the VPS).
- **Bybit account** with USDT perpetuals enabled.
  - Create **two** API key sets:
    - `BYBIT_TESTNET` — from `https://testnet.bybit.com/app/user/api-management`.
    - `BYBIT_LIVE` — create after testnet validation passes.
  - Permissions: **Read** + **Derivatives Trade** only. No Withdraw, no Transfer.
  - IP whitelist to the VPS.
- **Telegram bot**:
  - Talk to [@BotFather](https://t.me/BotFather), run `/newbot`, save the token.
  - Message your new bot once, then visit `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your `chat_id`. Save it.
- **(Optional but recommended) Grafana Cloud free tier** for a hosted dashboard if you prefer not to run Grafana yourself.

### 1.2 Local tooling

On your workstation:

- Docker 24+ and Docker Compose v2
- Python 3.12 (for running unit tests locally)
- SSH client + a hardware-protected SSH key (ed25519 recommended)
- A password manager for all secrets

### 1.3 Hard rules

- **No secret ever lives in `config.yaml` or git.** Secrets belong in `.env`, which is `.gitignore`d. `config.yaml` is safe to commit.
- **No withdraw permission** on any API key, ever.
- **Live mode requires a dual switch**: `mode: live` in `config.yaml` *and* `--confirm-live` on the CLI. Anything less will refuse to start.
- **First live capital: $100–200 MAX**, regardless of how confident you feel.

---

## 2. Phase 0 — Local smoke test

Before touching a VPS, verify the code runs on your workstation.

```bash
# 1. Clone (or unpack) the repo and enter it
cd cryprobotik

# 2. Install deps in a venv
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the full unit-test suite — must be 100% green
pytest tests/ -v

# 4. Lint + type-check (optional but recommended)
ruff check src tests
mypy src
```

If anything fails here, **stop and fix it**. Do not proceed to a VPS with a
red test suite.

---

## 3. Phase 1 — VPS provisioning

### 3.1 Choose a region close to the matching engines

Lower latency directly improves fill quality on market orders and reduces slippage.

| Provider | Recommended region | Proximity to |
|---|---|---|
| **Hetzner** | `fsn1` (Falkenstein, DE) or `hel1` (Helsinki) | OKX London PoP, Bybit Singapore via peering |
| **AWS** | `ap-northeast-1` (Tokyo) | Bybit HK, OKX HK |
| **Vultr** | `nrt` (Tokyo), `fra` (Frankfurt) | Either |

Minimum spec: **2 vCPU, 4 GB RAM, 40 GB SSD**. Hetzner CX22 (~€4/month) is plenty for v1.

### 3.2 Base hardening

After first SSH:

```bash
# Create a non-root user
adduser cryprobotik
usermod -aG sudo cryprobotik
rsync --archive --chown=cryprobotik:cryprobotik ~/.ssh /home/cryprobotik

# Disable root SSH and password auth
sudo sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sudo sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl reload ssh

# UFW firewall — allow SSH + Telegram outbound, nothing inbound on Prometheus
sudo apt-get update && sudo apt-get install -y ufw
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp
sudo ufw enable

# Time sync is critical — exchange signatures have ±5s tolerance
sudo apt-get install -y chrony
sudo systemctl enable --now chrony

# Unattended security upgrades
sudo apt-get install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

### 3.3 Install Docker + Compose

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker cryprobotik
newgrp docker
docker --version
docker compose version
```

Log out and back in so the group change takes effect.

### 3.4 Deploy the code

```bash
# Transfer the code via rsync from your workstation
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '.git' \
      cryprobotik/ cryprobotik@<vps-ip>:/home/cryprobotik/cryprobotik/

# On the VPS
cd ~/cryprobotik
```

### 3.5 Create `.env` (secrets)

```bash
cp .env.example .env
chmod 600 .env
nano .env
```

Fill in (use **testnet** keys for now):

```ini
# --- Database (Compose wires this to the timescaledb service) ---
DATABASE_URL=postgresql://cryprobotik:<strong-db-password>@timescaledb:5432/cryprobotik
POSTGRES_PASSWORD=<same-strong-db-password>

# --- OKX (testnet first) ---
OKX_API_KEY=<okx-testnet-key>
OKX_API_SECRET=<okx-testnet-secret>
OKX_API_PASSPHRASE=<okx-testnet-passphrase>
OKX_TESTNET=true

# --- Bybit (testnet first) ---
BYBIT_API_KEY=<bybit-testnet-key>
BYBIT_API_SECRET=<bybit-testnet-secret>
BYBIT_TESTNET=true

# --- Telegram ---
TELEGRAM_BOT_TOKEN=<botfather-token>
TELEGRAM_CHAT_ID=<your-chat-id>

# --- Monitoring / logging ---
HEALTH_PORT=8080
LOG_LEVEL=INFO
```

Double-check `config/config.yaml` has `mode: testnet` at the top. **It must.**

---

## 4. Phase 2 — First boot on testnet

### 4.1 Bring up the database first

```bash
docker compose up -d timescaledb
docker compose logs -f timescaledb  # wait for "database system is ready to accept connections"
```

### 4.2 Apply the schema

The bot entrypoint does this automatically on first start, but you can run
it manually for a cleaner first boot:

```bash
docker compose run --rm bot python -m src.data.storage --apply-schema
```

### 4.3 Start the bot

```bash
docker compose up -d bot
docker compose logs -f bot
```

You should see, in order:

1. `settings.loaded mode=testnet`
2. `storage.pool_created`
3. `universe.refreshed count=N symbols=[...]`
4. `exchange.okx.connected ws=public`
5. `exchange.okx.authenticated ws=private`
6. `exchange.bybit.connected ws=public`
7. `exchange.bybit.authenticated ws=private`
8. `health.server_started port=8080`
9. `orchestrator.ready`
10. Periodic `kline.received` events

If any of those are missing, see [§9 Incident playbooks](#9-incident-playbooks).

### 4.4 Confirm the operator loop

- **Health**: `curl http://localhost:8080/health` → `200 OK`
- **Readiness**: `curl http://localhost:8080/ready` → `200 OK` (only if all WS feeds are up and DB is reachable)
- **Metrics**: `curl http://localhost:8080/metrics | head -40` — you should see counters and gauges
- **Telegram**: you should have received a startup message from the bot. Send `/status` in your chat — it should reply.

---

## 5. Phase 3 — Testnet validation (2 weeks minimum)

**Do not skip or shorten this phase.** Two weeks is the floor, not a target.
The goal is to uncover bugs *before* they cost money, under real (testnet)
WebSocket conditions, reconnects, funding cycles, and universe rotations.

### 5.1 Daily checklist (each morning)

- [ ] Telegram daily report arrived at the scheduled UTC hour.
- [ ] `docker compose ps` — both services healthy.
- [ ] `curl localhost:8080/ready` returns 200.
- [ ] Check logs for the last 24h: `docker compose logs --since 24h bot | grep -iE 'error|halt|reconnect'` — investigate anything you don't recognize.
- [ ] Equity curve in the DB: `docker compose exec timescaledb psql -U cryprobotik -c "SELECT ts, equity FROM equity WHERE mode='testnet' ORDER BY ts DESC LIMIT 10;"`
- [ ] Kill-switch state: `SELECT value FROM bot_state WHERE key='kill_switch';` — should be `{"halted": false, ...}`.

### 5.2 Required validation drills

You must complete **every** drill below and document the result before
proceeding. No exceptions.

#### Drill A — Kill switch forced halt

Goal: prove the halt mechanism works end-to-end, including Telegram alert,
position flatten, and persistence across restart.

1. Temporarily edit `config/config.yaml`: set `risk.max_daily_drawdown_pct: 0.02` (2%).
2. `docker compose up -d bot` and wait for it to open positions.
3. Watch the equity curve; as soon as a single losing trade pushes drawdown > 2%, the bot must:
   - Send a `CRITICAL` Telegram alert (`KILL SWITCH TRIGGERED`).
   - Cancel all open orders.
   - Close all positions at market.
   - Set `halted=true` in `bot_state`.
   - Refuse to open any new positions.
4. `docker compose restart bot` — verify it comes back **still halted** and refuses to trade.
5. Manual reset: `docker compose exec bot python scripts/reset_halt.py`.
6. Restore `max_daily_drawdown_pct: 0.30` in `config.yaml` and restart.

**Pass criterion**: every step behaved as above, including persistence across restart. If not, **do not continue**.

#### Drill B — WebSocket reconnect storm

1. On the VPS: `sudo iptables -I OUTPUT -d <okx-ws-ip> -j DROP` (resolve the IP with `dig`).
2. Watch logs — you should see `ws.disconnected` followed by exponential-backoff reconnect attempts and eventually a successful `ws.reconnected` *after you remove the rule*.
3. Remove the rule: `sudo iptables -D OUTPUT -d <okx-ws-ip> -j DROP`.
4. Verify the bot re-subscribed to all previous kline/funding channels without restart, and that positions/balances were re-fetched and match DB snapshots.
5. Repeat for Bybit.

**Pass criterion**: no unmanaged positions, no duplicate signals after reconnect, `ws_reconnects_total` metric increments correctly.

#### Drill C — Database restart

1. `docker compose restart timescaledb`.
2. The bot should log DB connection errors briefly, enter a reduced-function state (no new writes), and auto-recover once the DB is back (~30s).
3. During the outage the bot **must not** place new orders (signal flow is blocked on failed `record_signal`).

**Pass criterion**: no crash, no lost fills (fills that arrived during the outage are persisted on recovery via the in-memory queue).

#### Drill D — Funding-rate arbitrage end-to-end

1. Wait for a funding cycle where |OKX funding - Bybit funding| > 5bps for the same symbol.
2. Verify the ensemble emits a `PairSignal`, the router opens a long leg on one exchange and a short leg on the other, and both legs close before the next funding tick.
3. Check the `fills` table for both legs; net PnL should be approximately the captured funding differential minus fees and slippage.

**Pass criterion**: delta-neutral pair executed atomically, no dangling leg.

#### Drill E — Universe rotation

1. Wait for a scheduled universe refresh (default every 4h).
2. Confirm in logs that removed symbols unsubscribe cleanly (`feature_store.dropped`) and new symbols subscribe with a REST backfill.
3. No leaked tasks, no stale indicator state.

### 5.3 Graduation criteria

Only proceed past testnet when **all** of these are true:

- [ ] 14+ consecutive days of clean operation (no manual restarts).
- [ ] All 5 drills (A–E) passed and documented.
- [ ] Daily Telegram reports delivered without gaps.
- [ ] No unexplained errors in logs.
- [ ] Equity curve trends up or is flat — consistent loss means a strategy bug or misconfig, *not* "more time will fix it". Stop and investigate.
- [ ] You personally understand what every strategy is doing and why.

---

## 6. Phase 4 — Paper mode on mainnet (1 week minimum)

Paper mode uses **real mainnet prices** but simulated fills in an in-memory
matching engine. This catches bugs that only show up under real liquidity
and volatility.

### 6.1 Switch to paper mode

1. Create **live API keys** (still no withdraw permission) and IP-whitelist the VPS.
2. Update `.env`:
   ```ini
   OKX_API_KEY=<okx-live-key>
   OKX_API_SECRET=<okx-live-secret>
   OKX_API_PASSPHRASE=<okx-live-passphrase>
   OKX_TESTNET=false
   BYBIT_API_KEY=<bybit-live-key>
   BYBIT_API_SECRET=<bybit-live-secret>
   BYBIT_TESTNET=false
   ```
3. Update `config/config.yaml`: `mode: paper`.
4. `docker compose restart bot`.
5. Verify logs: `exchange.okx.connected ws=public testnet=false` — the bot is reading **live** market data.
6. Place a tiny paper order manually via `/status` and verify it shows up in the `orders` table with `mode='paper'`.

### 6.2 What to watch for

- **Sizing matches expectations**: on a $10k paper account with 2% risk and an ATR-based stop of $500, you should see ~$0.4 position size * price notional. Verify.
- **Routing picks the right venue**: inspect the `orders` table — are momentum signals going to the venue with tighter spreads?
- **Slippage is realistic**: compare simulated fill prices to actual mid at the fill timestamp.
- **PnL math reconciles**: sum of fills should equal change in equity ± fees.

Run this for **at least 7 days**. If anything looks off, do not proceed.

---

## 7. Phase 5 — Live cutover

**Do this when calm, well-rested, and with 2+ hours of uninterrupted time to monitor the first trades.** Not late at night. Not during a volatile news event.

### 7.1 Pre-flight checklist

- [ ] Testnet validation passed (§5.3)
- [ ] Paper mainnet validation passed (§6.2)
- [ ] Capital for first live run: **$100–200 MAX**
- [ ] Funded only a sub-account or a dedicated account with **no other balance** on it
- [ ] Withdrawal whitelist enabled on both exchange accounts (extra defense in depth)
- [ ] 2FA enabled on both exchange accounts
- [ ] VPS snapshots enabled (Hetzner/Vultr/AWS — whichever you use)
- [ ] Telegram `/halt` command tested and known to work
- [ ] You can SSH into the VPS within 60 seconds

### 7.2 The cutover

1. Transfer **$100–200** to the perpetuals wallet on each exchange.
2. Edit `config/config.yaml`:
   ```yaml
   mode: live
   ```
3. Verify hard-ceiling risk parameters are at or below defaults:
   ```yaml
   risk:
     max_daily_drawdown_pct: 0.30   # or lower — consider 0.10 for first run
     risk_per_trade_pct: 0.02       # or lower — consider 0.01 for first run
     max_open_positions: 3          # or lower
     leverage: 3                    # or lower — consider 2 for first run
   ```
4. Restart the bot with the explicit live confirmation flag:
   ```bash
   docker compose down
   CRYPROBOTIK_CONFIRM_LIVE=1 docker compose run --rm --service-ports bot \
       python -m src.main --config config/config.yaml --mode live --confirm-live
   ```
   Or, if you prefer detached mode, modify the compose command to include `--confirm-live`:
   ```yaml
   # docker-compose.yml (bot service)
   command: ["python", "-m", "src.main", "--mode", "live", "--confirm-live"]
   ```
   Then `docker compose up -d bot`.
5. **Without** `--confirm-live`, the bot will refuse to start in live mode — this is by design.

### 7.3 First 72 hours

- **Watch live** for the first 2 hours. Keep the logs open on one screen, the exchange UI on another.
- Verify the first trade end-to-end: signal → order placed → fill received → position tracked → SL/TP armed.
- **Do not tune parameters** during the first 72 hours. Resist the urge.
- If anything looks even slightly wrong, `/halt` in Telegram and investigate.

### 7.4 Scaling capital

Only after:

- 30+ days of clean live operation
- Positive rolling Sharpe (>1.0) verified in the daily analytics report
- Zero manual interventions required
- You have re-read this document

…should you consider scaling capital. Scale in **2× increments**, never more. A bot that works at $200 may not work at $20,000 because of liquidity and slippage asymmetries.

---

## 8. Phase 6 — Ongoing operations

### 8.1 Routine maintenance

| Frequency | Action |
|---|---|
| Daily | Read Telegram report. Spot-check logs for errors. |
| Weekly | Review equity curve, win rate, Sharpe. Grafana or ad-hoc SQL. |
| Monthly | Update pinned dependencies (`pip-audit` + re-run full test suite + 1 week of paper mode before re-deploying to live). |
| Quarterly | Re-read this document. Review and re-sign off on risk parameters. |
| After any incident | Add a postmortem entry to a local `POSTMORTEMS.md`. Do not skip. |

### 8.2 Backups

- TimescaleDB volume: nightly `pg_dump` to an off-VPS destination (S3, Backblaze, or rsync to your workstation).
- `.env`: stored in your password manager, not on the VPS.
- `config/config.yaml`: committed to a private git repo (it contains no secrets).

### 8.3 Upgrades

```bash
# On your workstation
git pull   # or rsync your updated code
pytest tests/ -v                    # must be green
ruff check src tests && mypy src

# Deploy
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '.git' \
      cryprobotik/ cryprobotik@<vps-ip>:/home/cryprobotik/cryprobotik/

# On the VPS — always run a paper cycle after updating
ssh cryprobotik@<vps-ip>
cd ~/cryprobotik
# 1. Switch to paper temporarily
sed -i 's/^mode:.*/mode: paper/' config/config.yaml
docker compose up -d --build bot
# 2. Watch for 24h. If clean, flip back to live.
sed -i 's/^mode:.*/mode: live/' config/config.yaml
docker compose restart bot   # must include --confirm-live in command
```

---

## 9. Incident playbooks

### 9.1 Bot is halted

```bash
# Inspect the halt reason
docker compose exec timescaledb psql -U cryprobotik -c \
    "SELECT value FROM bot_state WHERE key='kill_switch';"

# Inspect the events leading up to the halt
docker compose logs --since 6h bot | grep -iE 'halt|drawdown|error'
```

**Do not blindly reset the halt.** Find root cause first. Only once you've
confirmed it was a data glitch, a one-off bad trade, or a real drawdown you
can live with, run:

```bash
docker compose exec bot python scripts/reset_halt.py
```

### 9.2 Exchange WS is stuck disconnected

```bash
# Check the reconnect counter
curl -s localhost:8080/metrics | grep ws_reconnects_total

# Check time sync — bad clock causes signature failures
chronyc tracking

# Verify the API key still has the VPS IP whitelisted (log in to the exchange UI)

# Last resort
docker compose restart bot
```

### 9.3 DB is full / slow

TimescaleDB with default retention grows ~100MB/day. If the volume is getting full:

```sql
-- Add a retention policy (drops chunks older than 90d)
SELECT add_retention_policy('ohlcv', INTERVAL '90 days');
SELECT add_retention_policy('equity', INTERVAL '365 days');
SELECT add_retention_policy('events', INTERVAL '30 days');
```

### 9.4 Unexpected loss spike

1. `/halt` in Telegram immediately.
2. SSH in, check open positions: `SELECT * FROM positions_snapshot ORDER BY ts DESC LIMIT 20;`
3. If a position is stuck without a stop, close it manually in the exchange UI.
4. Pull logs for the last hour, search for the symbol and strategy.
5. Write a postmortem before resetting.

### 9.5 I can't reach Telegram

Telegram is **not** safety-critical — the kill switch runs on the bot side,
not on your ability to see alerts. If Telegram is down, SSH in and use
`/metrics` plus the SQL queries above.

---

## 10. Appendices

### 10.1 Useful SQL queries

```sql
-- Last 20 trades with PnL
SELECT ts, symbol, strategy, side, qty, price, realized_pnl
FROM fills ORDER BY ts DESC LIMIT 20;

-- Equity curve for last 7 days
SELECT time_bucket('1 hour', ts) AS bucket, last(equity, ts) AS equity
FROM equity WHERE mode='live' AND ts > now() - interval '7 days'
GROUP BY bucket ORDER BY bucket;

-- Win rate by strategy, last 30 days
SELECT strategy,
       COUNT(*) FILTER (WHERE realized_pnl > 0) * 1.0 / COUNT(*) AS win_rate,
       SUM(realized_pnl) AS total_pnl, COUNT(*) AS n_trades
FROM fills WHERE ts > now() - interval '30 days' AND realized_pnl IS NOT NULL
GROUP BY strategy;

-- Open position snapshot
SELECT ts, symbol, side, qty, entry_price, unrealized_pnl
FROM positions_snapshot
WHERE ts = (SELECT MAX(ts) FROM positions_snapshot);

-- Kill switch state
SELECT key, value FROM bot_state WHERE key LIKE 'kill_switch%';
```

### 10.2 Telegram commands

| Command | Effect |
|---|---|
| `/status` | Show mode, equity, drawdown, open positions count, halt state |
| `/positions` | List currently open positions with unrealized PnL |
| `/pnl` | Show daily, weekly, 30-day PnL |
| `/halt` | Manually halt the bot (identical effect to the kill switch) |
| `/resume` | Clear a manual halt (does **not** clear a drawdown halt — use `scripts/reset_halt.py`) |

Commands are authorized by `chat_id` — only the IDs in `TELEGRAM_CHAT_ID` can issue them.

### 10.3 Risk parameter reference

All values are **hard-ceilinged** in `src/settings.py` — configs that exceed
them will refuse to load. These ceilings exist to prevent catastrophic
misconfig, not as recommended operating points.

| Parameter | Default | Hard ceiling |
|---|---|---|
| `max_daily_drawdown_pct` | 0.30 | 0.30 |
| `risk_per_trade_pct` | 0.02 | 0.05 |
| `leverage` | 3 | 10 |
| `max_open_positions` | 4 | 20 |
| `max_margin_utilization` | 0.70 | 0.90 |

**For first live runs, use values well below the defaults.** A reasonable
starting point: `risk_per_trade_pct: 0.01`, `leverage: 2`, `max_open_positions: 2`,
`max_daily_drawdown_pct: 0.10`.

### 10.4 Directory layout on the VPS

```
/home/cryprobotik/cryprobotik/
├── config/config.yaml
├── .env                  # chmod 600
├── docker-compose.yml
├── src/
├── tests/
└── docker/
    └── entrypoint.sh

Docker volumes (persisted):
- timescale_data     # DB
- bot_logs           # structured logs
```

### 10.5 What this bot is NOT

To save you pain:

- **It is not a printing press.** Positive-expectancy is not a law of physics. Any strategy can break when market structure shifts.
- **It is not ML-based.** v1 uses classical technical indicators. The architecture supports ML plug-ins, but you'd have to train and validate them yourself.
- **It is not a backtest engine.** Validation is done via paper mode on live data, not historical backtesting.
- **It is not a custodial service.** It does not and cannot move your funds off the exchange. If the exchange goes down, your funds are at the mercy of the exchange.
- **It is not a replacement for your judgment.** Review the logs. Kill it when your gut says something's off. Never "set and forget" leveraged crypto.

---

## Final reminder

The code in this repo has been written carefully, with hard ceilings,
persistent kill switches, and multiple layers of validation. None of that
matters if you skip the testnet phase, over-leverage the first live run, or
deploy on a day when you're tired.

**Capital preservation > returns.** Read that again. Then go follow the checklist.
