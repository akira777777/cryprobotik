# Cryprobotik: Profitability Analysis & Improvement Roadmap

## Executive Summary

After a full audit of the codebase, config, strategies, ML pipeline, and risk layer — here's the honest assessment: **the architecture is solid but the bot is currently configured to be conservative to the point of inaction, and it's missing several high-edge strategy classes that dominate perp futures markets.** The ML filter with only 25 features and GradientBoosting is a decent start but leaves significant alpha on the table. Below is everything broken into what's wrong, what to add, and how to tune it.

---

## PART 1: CURRENT WEAKNESSES (What's Holding You Back)

### 1.1 The Ensemble Is Too Restrictive

**Problem:** `min_net_vote: 0.30` combined with the weight matrix means you need near-unanimous agreement from multiple strategies before any trade fires. In practice, during "chop" regime (which is the most common market state), ALL strategies except funding_arb have weight 0.0. The bot literally does nothing most of the time.

**The math:** In chop regime, only funding_arb (weight 1.0) and funding_contrarian (weight 0.2) contribute. Funding contrarian's max confidence is ~0.55 × 0.2 = 0.11 weighted contribution. That's never hitting 0.30. So in chop, the ensemble is dead silent except for the arb path that bypasses ensemble entirely.

**Fix:** Either lower `min_net_vote` to 0.15-0.20 for certain regimes, or add strategies that specifically perform in chop/sideways markets (see Part 2).

### 1.2 No Exit Management (Biggest Single Gap)

**Problem:** The bot computes SL/TP at entry and then... that's it. Static stops. No trailing stops, no partial profit-taking, no time-based exits, no volatility-adjusted stop tightening. This is the #1 profitability killer. You're leaving money on the table on winners (TP too early or too late) and taking full losses on losers that could have been cut earlier.

What's missing:
- **Trailing stop logic** — move SL to breakeven after 1R profit, trail at 1.5 ATR behind price after 2R
- **Partial take-profit** — close 50% at 1.5R, let the rest run with trailing stop
- **Time-based exit** — if a trade hasn't moved 0.5R in your direction after N bars, cut it (the thesis is dead)
- **Volatility expansion exit** — if ATR doubles after entry, tighten stop (regime changed)

### 1.3 ML Filter Is Underpowered

**Problems:**
- **GradientBoosting with 100 trees and max_depth=3** is a very weak learner for this task. It's essentially a slightly-better-than-random filter after 50 samples.
- **25 features but no market microstructure features** — no orderbook imbalance, no liquidation cascade detection, no whale flow.
- **Binary labels (win/lose) ignore magnitude** — a trade that makes 5R and one that makes 0.01R above fees both get label=1. You're not learning which signals produce big winners.
- **No feature importance tracking** — you have no idea which features actually matter. Some of those 25 features are likely noise.
- **500-sample cap is tiny** — for GBM to learn meaningful patterns in 25-dimensional space, you need thousands of samples minimum.
- **Retrain every 20 samples** is too frequent for GBM — causes unstable decision boundaries.

### 1.4 Risk Sizing Doesn't Scale With Confidence

The risk manager applies flat 2% risk regardless of signal confidence. A 0.95-confidence signal from perfect multi-timeframe alignment gets the same size as a borderline 0.30 signal. This is leaving edge on the table.

### 1.5 No Correlation With Macro/BTC Dominance

Every altcoin perp is correlated with BTC. The bot trades 10 symbols but has no concept of "the entire market is dumping because of a BTC crash." The correlation check in PortfolioLimits uses 1h close returns, but that's backward-looking. If BTC drops 5% in 15 minutes, your bot might enter 4 long altcoin positions before the correlation catches up.

### 1.6 Funding Arb Requires Both Exchanges

Funding arb needs Bybit enabled, but `bybit.enabled: false` in config. So your highest-confidence strategy (0.8 base) is effectively dead.

---

## PART 2: NEW STRATEGIES TO ADD (Ranked by Expected Edge)

### 2.1 Liquidation Cascade Strategy (HIGH PRIORITY)

**Edge:** Perp-specific. When large leveraged positions get liquidated, they create forced selling/buying that moves price through levels. Detect when OI drops sharply while price moves in one direction → that's liquidations. Trade the bounce after the cascade exhausts.

**Implementation:**
- Monitor OI drops > 3% in a single bar combined with price move > 1.5 ATR
- Enter counter-trend after the cascade bar closes (liquidation exhaustion)
- Tight stop (0.5 ATR), target 1-2 ATR bounce
- Works in ALL regimes — add to ensemble with chop weight > 0

**Data needed:** You already have OIStore. Just need faster polling (every 60s instead of 300s) and bar-level OI change detection.

### 2.2 Order Flow Imbalance / VWAP Strategy (HIGH PRIORITY)

**Edge:** VWAP is the institutional benchmark. Price consistently reverts to VWAP during ranging markets and uses it as support/resistance during trends.

**Implementation:**
- Calculate session VWAP (UTC 00:00 reset) from your existing OHLCV data
- Long when price pulls back to VWAP from above in uptrend (VWAP as support)
- Short when price rejects VWAP from below in downtrend (VWAP as resistance)
- Confluence: CVD ratio confirms direction

### 2.3 Multi-Timeframe RSI Divergence Strategy (MEDIUM PRIORITY)

**Edge:** Classic but effective on perps. When price makes a new high but RSI makes a lower high (bearish divergence), it signals momentum exhaustion. Combined with funding rate data, this catches tops/bottoms.

**Implementation:**
- Detect swing highs/lows in price and RSI on 1h timeframe
- Require divergence confirmation on 4h
- Enter on the 15m after divergence confirms with a candle close
- Higher confidence when funding rate is extreme (convergence with funding_contrarian)

### 2.4 Ichimoku Cloud Strategy (MEDIUM PRIORITY)

**Edge:** Provides both trend direction and support/resistance levels in a single indicator. Extremely effective on 4h charts for crypto.

**Implementation:**
- Tenkan/Kijun cross for entry signal
- Cloud as dynamic support/resistance for SL placement
- Chikou Span for confirmation
- Weight heavily in trend regimes, zero in chop

### 2.5 Market Making / Grid Strategy (LOWER PRIORITY — Different Risk Profile)

**Edge:** In ranging markets (where your current bot does almost nothing), a simple grid strategy that places limit orders above and below current price can capture the spread + funding.

**Caveat:** Requires limit order execution (your current default is market orders). Would need changes to the executor.

---

## PART 3: CONFIGURATION TUNING FOR PROFITABILITY

### 3.1 Config Changes (config.yaml)

```yaml
# --- These changes should be applied ---

# REGIME WEIGHTS — give strategies a chance in chop
regime:
  min_net_vote: 0.20  # was 0.30 — too restrictive
  weights:
    chop:
      momentum: 0.0
      volatility_breakout: 0.0
      mean_reversion: 0.3    # was 0.0 — mean reversion CAN work in chop
      funding_arb: 1.0
      funding_contrarian: 0.4  # was 0.2 — funding contrarian IS a chop strategy

    range_high_vol:
      mean_reversion: 1.0     # was 0.8 — this is its best regime
      funding_contrarian: 0.6  # was 0.4

    trend_high_vol:
      momentum: 1.0
      volatility_breakout: 1.0  # was 0.8 — this IS the vol breakout regime
      funding_contrarian: 0.7   # was 0.6

# RISK — scale with confidence
risk:
  risk_per_trade_pct: 0.015   # was 0.02 — slightly lower base, but scale up with confidence
  max_open_positions: 6       # was 4 — with correlation check, 6 is fine for top-10 universe
  leverage: 5                 # was 3 — with 1.5% risk per trade, 5x is still conservative
  sl_atr_multiplier: 1.5     # was 2.0 — tighter stops with trailing stop logic
  min_reward_to_risk: 2.5    # was 2.0 — be pickier about reward

# MOMENTUM — loosen the volume filter
strategies:
  momentum:
    volume_multiplier: 1.2    # was 1.5 — too many missed entries
    rsi_long_threshold: 50    # was 55 — catch trends earlier
    rsi_short_threshold: 50   # was 45

  mean_reversion:
    adx_max: 25               # was 20 — too restrictive, misses early range formations
    rsi_long_threshold: 15    # was 10 — still deep oversold but catches more setups
    rsi_short_threshold: 85   # was 90

  volatility_breakout:
    squeeze_bars: 4           # was 6 — 6 hours of squeeze is too long, miss breakouts
    squeeze_atr_ratio_max: 0.85  # was 0.8 — slightly more permissive

# ENABLE BYBIT for funding arb
exchanges:
  bybit:
    enabled: true             # was false — kills your best strategy
```

### 3.2 Confidence-Scaled Position Sizing

Add to `RiskManager.size_trade()`:

```python
# Scale risk with confidence: base_risk * (0.5 + 0.5 * confidence)
# At confidence 0.3: risk = base * 0.65 (smaller)
# At confidence 1.0: risk = base * 1.0 (full size)
confidence_scalar = 0.5 + 0.5 * min(1.0, confidence)
risk_dollars = equity * self._config.risk_per_trade_pct * confidence_scalar
```

### 3.3 Paper Trading Starting Balance

`starting_balance_usd: 10000` is fine for testing, but make sure your live deployment matches the actual capital you'll trade with. Position sizing is proportional to equity, so a 10k paper test tells you nothing about how it'll behave at 1k or 100k.

---

## PART 4: ML FILTER IMPROVEMENTS

### 4.1 Upgrade the Model

Replace GradientBoosting with **LightGBM** or **XGBoost**:
- Much faster training (matters for online learning)
- Native handling of categorical features (regime)
- Better with small datasets
- GPU support if needed later

```python
# In requirements.txt, add:
lightgbm>=4.0.0

# In src/ml/model.py, replace sklearn GBM with:
import lightgbm as lgb
model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=5,          # was 3 — too shallow
    learning_rate=0.05,   # slower learning, more stable
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=10,
    reg_alpha=0.1,
    reg_lambda=0.1,
)
```

### 4.2 Add Features (FEATURE_VERSION → 4)

New features to add to the vector:

```python
# Liquidation proxy: OI drop rate on the signal bar
"oi_bar_change",        # OI change on the current bar (not ROC over 5 periods)

# Funding rate raw + percentile
"funding_rate_raw",     # raw funding rate (not just contrarian signal)
"funding_percentile",   # where current rate sits historically

# Volume profile
"volume_ratio_15m",     # current bar vol / 20-bar avg on 15m
"volume_ratio_1h",      # same on 1h

# Cross-asset correlation
"btc_correlation_24h",  # rolling 24h correlation with BTC
"btc_rsi_15m",          # BTC's RSI — market leader signal

# Spread / microstructure (if available)
"bid_ask_spread_bps",   # from orderbook snapshots

# Multi-bar momentum
"returns_3bar_15m",     # 3-bar return on 15m (short momentum)
"returns_12bar_1h",     # 12-bar return on 1h (medium momentum)
```

This would bring the vector to ~35 features. Bump FEATURE_VERSION to 4.

### 4.3 Regression Instead of Classification

Instead of binary win/lose, predict the **magnitude of return** (regression). Then filter signals by predicted return > threshold.

```python
# Replace GradientBoostingClassifier with GradientBoostingRegressor
# Label = realized_pnl / notional (percentage return)
# Accept signal if predicted_return > 0.005 (0.5% expected return)
```

This teaches the model not just "will this win" but "how much will this win." Much more informative.

### 4.4 Increase Training Buffer

```python
ML_BUFFER_MAX = 2000      # was 500 — need more data for deeper trees
RETRAIN_EVERY = 50        # was 20 — more stable retraining
MIN_SAMPLES = 100         # was 50 — need more examples before trusting the model
```

---

## PART 5: KNOWLEDGE GAPS TO FILL

### 5.1 On-Chain Data Integration

The bot operates purely on exchange data. Adding on-chain signals would give it an information edge:

- **Exchange inflows/outflows** — large BTC/ETH deposits to exchanges precede selling. Services: Glassnode API, CryptoQuant API.
- **Whale wallet tracking** — when known whale wallets move funds, it often precedes large trades.
- **Stablecoin mint/burn** — USDT/USDC minting events correlate with upcoming buying pressure.

**Implementation:** Create an `OnChainStore` similar to `OIStore` that polls an API every 5 minutes and feeds features to the ML vector.

### 5.2 Sentiment / News Data

- **Fear & Greed Index** — available via API, simple but effective regime filter
- **Social sentiment** (Twitter/X, Reddit) — services like LunarCrush or Santiment
- **News events** — CPI releases, FOMC, major hacks/exploits cause regime shifts

Add as ML features + potential regime override (e.g., during FOMC announcement, force regime to "chop" and reduce position sizes).

### 5.3 Cross-Exchange Orderbook Depth

Your bot uses last price and OHLCV but never looks at the orderbook. Adding bid/ask imbalance at top-of-book would significantly improve:
- Entry timing (enter when orderbook supports your direction)
- Slippage estimation (avoid thin orderbooks)
- Liquidation level detection (where are the large resting orders)

**Implementation:** Subscribe to OKX orderbook WS channel (`books5` for top 5 levels) and compute imbalance ratio = bid_volume / (bid_volume + ask_volume).

### 5.4 Backtest Infrastructure

**Critical missing piece.** You have no backtesting framework. You're flying blind on strategy parameter changes. You need:

1. Historical data pipeline (download OHLCV from exchange into PostgreSQL)
2. Replay engine that feeds stored bars through the same `Ensemble.evaluate_symbol()` pipeline
3. Performance metrics: Sharpe ratio, max drawdown, win rate, profit factor, avg R-multiple
4. Walk-forward optimization (train on 6 months, test on next 2 months, roll forward)

Without backtesting, every config change is a guess. This should be priority #1 before any other optimization.

---

## PART 6: IMPLEMENTATION PRIORITY (What To Do First)

### Phase 1 — Immediate (1-2 weeks)
1. **Build backtesting framework** — you need this before changing anything else
2. **Add trailing stop + partial TP logic** — biggest single improvement to profitability
3. **Apply config tuning from Part 3** — quick wins, test in paper mode
4. **Enable Bybit** — unlock funding arb

### Phase 2 — Short Term (2-4 weeks)
5. **Add Liquidation Cascade strategy** — you already have the data (OIStore)
6. **Add VWAP strategy** — easy to implement from existing OHLCV
7. **Upgrade ML to LightGBM** — drop-in replacement, better performance
8. **Add confidence-scaled sizing** — 10-line change to risk manager

### Phase 3 — Medium Term (1-2 months)
9. **Add RSI Divergence strategy**
10. **Expand ML features to v4** (volume ratios, BTC correlation, funding features)
11. **Switch ML from classification to regression**
12. **Add orderbook imbalance data**

### Phase 4 — Long Term (2-3 months)
13. **On-chain data integration** (exchange flows, whale tracking)
14. **Sentiment data feeds**
15. **Ichimoku strategy**
16. **Walk-forward optimization pipeline**

---

## APPENDIX: Quick Wins Checklist

- [ ] Lower `min_net_vote` from 0.30 to 0.20
- [ ] Increase `chop` regime weights for mean_reversion (0.0 → 0.3) and funding_contrarian (0.2 → 0.4)
- [ ] Reduce `squeeze_bars` from 6 to 4
- [ ] Reduce `volume_multiplier` from 1.5 to 1.2
- [ ] Enable Bybit connector
- [ ] Add confidence scalar to position sizing
- [ ] Increase leverage from 3 to 5 (with lower base risk_per_trade)
- [ ] Poll OI every 60s instead of 300s
- [ ] Increase ML buffer to 2000, retrain interval to 50
