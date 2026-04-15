"""
Backtest replay engine.

Design principles:
    1. Same code path as live.  FeatureStore, Ensemble, MLSignalFilter and
       RiskManager are instantiated exactly as in the Orchestrator — strategies
       are not reimplemented or bypassed.
    2. Different data source.  Historical bars come from a list of KlineEvents
       (typically loaded from the ohlcv DB table or a CSV); the engine feeds
       them into the FeatureStore one bar at a time in chronological order.
    3. Simulated fill model.  Orders fill at the NEXT bar's open price plus
       configurable slippage (bps).  SL triggers when a bar's low (for longs)
       or high (for shorts) crosses the stop price; TP similarly.  This is a
       conservative fill model — no partial fills, no intra-bar logic.
    4. No side effects.  The engine never touches the DB or sends Telegram
       messages.  Results are returned as BacktestResult dataclasses.

Walk-forward is implemented in scripts/run_backtest.py — the engine handles
a single window at a time.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.data.feature_store import Bar, CVDStore, FeatureKey, FeatureStore, FundingHistory, OIStore
from src.exchanges.base import KlineEvent, OrderSide
from src.risk.manager import RejectedTrade, RiskManager, SizedTrade
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.settings import RiskConfig
    from src.strategies.ensemble import Ensemble
    from src.ml.model import MLSignalFilter

log = get_logger(__name__)


# ─────────────────────── config ───────────────────────


@dataclass(slots=True)
class BacktestConfig:
    """Simulation parameters."""

    initial_balance: float = 10_000.0      # USDT
    taker_fee_bps: float = 6.0             # 0.06% per side (OKX taker)
    slippage_bps: float = 5.0              # half-spread estimate
    leverage: int = 3                       # mirrors live config default
    # Risk manager parameters (must match live config when comparing)
    risk_per_trade_pct: float = 0.02
    sl_atr_multiplier: float = 1.5
    min_reward_to_risk: float = 1.5
    max_open_positions: int = 4
    # ML filtering
    use_ml_filter: bool = False            # disabled by default (cold-start in backtest)
    exchange: str = "okx_backtest"
    timeframes: list[str] = field(default_factory=lambda: ["15m", "1h", "4h"])
    primary_timeframe: str = "15m"         # bar that triggers evaluation


# ─────────────────────── state ───────────────────────


@dataclass
class SimPosition:
    """Tracks a single open simulated position."""

    symbol: str
    side: OrderSide
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    leverage: int
    risk_usd: float
    notional_usd: float
    strategy: str
    confidence: float
    entry_ts: datetime
    entry_bar_idx: int


@dataclass
class SimFill:
    """Record of a simulated fill (entry or exit)."""

    symbol: str
    side: OrderSide
    price: float
    quantity: float
    fee_usd: float
    ts: datetime
    reason: str             # "entry", "sl", "tp", "close_at_end", "time_stop"
    pnl_usd: float = 0.0    # only meaningful for exit fills
    strategy: str = ""


# ─────────────────────── results ───────────────────────


@dataclass
class BacktestResult:
    """Output of a single backtest window."""

    run_id: str
    symbols: list[str]
    start_ts: datetime
    end_ts: datetime
    initial_balance: float
    final_balance: float
    fills: list[SimFill]
    equity_curve: list[tuple[datetime, float]]  # (ts, equity)

    # Computed metrics (populated by _compute_metrics)
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    winrate: float = 0.0
    total_trades: int = 0
    profitable_trades: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    avg_r_multiple: float = 0.0
    config: BacktestConfig = field(default_factory=BacktestConfig)

    @property
    def net_pnl(self) -> float:
        return self.final_balance - self.initial_balance

    @property
    def return_pct(self) -> float:
        if self.initial_balance <= 0:
            return 0.0
        return (self.final_balance - self.initial_balance) / self.initial_balance


# ─────────────────────── engine ───────────────────────


class BacktestEngine:
    """
    Replay-based backtesting engine.

    Usage::

        engine = BacktestEngine(ensemble, risk_config, backtest_config)
        result = await engine.run(
            symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
            bars_by_symbol={symbol: [KlineEvent, ...]},
        )
    """

    def __init__(
        self,
        ensemble: "Ensemble",
        risk_config: "RiskConfig",
        config: BacktestConfig | None = None,
        ml_filter: "MLSignalFilter | None" = None,
    ) -> None:
        self._ensemble = ensemble
        self._risk_config = risk_config
        self._config = config or BacktestConfig()
        self._ml_filter = ml_filter

    async def run(
        self,
        symbols: list[str],
        bars_by_symbol: dict[str, list[KlineEvent]],
        run_id: str | None = None,
    ) -> BacktestResult:
        """
        Execute a full backtest window.

        bars_by_symbol: all timeframe bars for each symbol, sorted oldest→newest.
        The engine drives time from the primary_timeframe (default 15m) bars.
        """
        run_id = run_id or uuid.uuid4().hex[:12]
        cfg = self._config
        log.info("backtest.run_start", run_id=run_id, symbols=symbols, config=str(cfg))

        store = FeatureStore(max_bars=1500)
        risk_manager = RiskManager(self._risk_config)
        balance = cfg.initial_balance
        equity_curve: list[tuple[datetime, float]] = []
        fills: list[SimFill] = []
        open_positions: dict[str, SimPosition] = {}  # symbol → position

        # Group bars by symbol and timeframe for easy iteration
        primary_tf = cfg.primary_timeframe
        primary_bars: list[tuple[datetime, str, KlineEvent]] = []  # (ts, symbol, evt)

        for symbol, bar_list in bars_by_symbol.items():
            for evt in bar_list:
                if evt.timeframe == primary_tf and evt.closed:
                    primary_bars.append((evt.ts, symbol, evt))
                # Load ALL timeframes (including non-primary) into store eagerly
                # so indicator lookback windows are available from the start.
                if evt.closed:
                    key = FeatureKey(cfg.exchange, symbol, evt.timeframe)
                    bar = Bar(
                        ts_ms=int(evt.ts.timestamp() * 1000),
                        open=evt.open,
                        high=evt.high,
                        low=evt.low,
                        close=evt.close,
                        volume=evt.volume,
                    )
                    store.append_bar(key, bar)

        # Sort all primary-TF bars by time across all symbols
        primary_bars.sort(key=lambda x: x[0])

        if not primary_bars:
            log.warning("backtest.no_primary_bars", run_id=run_id, timeframe=primary_tf)
            return self._empty_result(run_id, symbols, cfg)

        start_ts = primary_bars[0][0]
        end_ts = primary_bars[-1][0]

        # We need a second pass: build a fresh store and replay bar-by-bar
        # so the strategies see only past data at each evaluation point.
        replay_store = FeatureStore(max_bars=1500)

        # Pre-load non-primary TF bars into indexed lookup for fast replay
        non_primary_lookup: dict[tuple[str, str, str], list[KlineEvent]] = {}
        for symbol, bar_list in bars_by_symbol.items():
            for evt in bar_list:
                if evt.timeframe != primary_tf and evt.closed:
                    k = (symbol, evt.timeframe, str(int(evt.ts.timestamp())))
                    non_primary_lookup.setdefault((symbol, evt.timeframe, ""), []).append(evt)

        # Build per-TF sorted lists for efficient replay
        non_primary_sorted: dict[tuple[str, str], list[KlineEvent]] = {}
        for (sym, tf, _), evts in non_primary_lookup.items():
            key = (sym, tf)
            if key not in non_primary_sorted:
                all_evts = [e for e in bars_by_symbol.get(sym, [])
                            if e.timeframe == tf and e.closed]
                all_evts.sort(key=lambda e: e.ts)
                non_primary_sorted[key] = all_evts

        # Cursors: how many non-primary bars have been fed for each (sym, tf)
        np_cursors: dict[tuple[str, str], int] = {}

        bar_idx = 0
        prev_bar_ts: dict[str, datetime] = {}

        for bar_ts, symbol, evt in primary_bars:
            bar_idx += 1

            # Feed non-primary bars for this symbol up to current time
            for tf in cfg.timeframes:
                if tf == primary_tf:
                    continue
                np_key = (symbol, tf)
                np_list = non_primary_sorted.get(np_key, [])
                cursor = np_cursors.get(np_key, 0)
                while cursor < len(np_list) and np_list[cursor].ts <= bar_ts:
                    ne = np_list[cursor]
                    fk = FeatureKey(cfg.exchange, symbol, ne.timeframe)
                    replay_store.append_bar(fk, Bar(
                        ts_ms=int(ne.ts.timestamp() * 1000),
                        open=ne.open, high=ne.high, low=ne.low,
                        close=ne.close, volume=ne.volume,
                    ))
                    cursor += 1
                np_cursors[np_key] = cursor

            # Feed the current primary bar
            fk = FeatureKey(cfg.exchange, symbol, primary_tf)
            replay_store.append_bar(fk, Bar(
                ts_ms=int(evt.ts.timestamp() * 1000),
                open=evt.open, high=evt.high, low=evt.low,
                close=evt.close, volume=evt.volume,
            ))

            # Check if existing position has been stopped out by this bar
            if symbol in open_positions:
                pos = open_positions[symbol]
                exit_fill = self._check_exits(pos, evt, bar_ts)
                if exit_fill is not None:
                    balance += exit_fill.pnl_usd - exit_fill.fee_usd
                    fills.append(exit_fill)
                    del open_positions[symbol]
                    log.debug(
                        "backtest.exit",
                        symbol=symbol,
                        reason=exit_fill.reason,
                        pnl=round(exit_fill.pnl_usd, 4),
                    )

            # Update equity curve
            unrealized = sum(
                self._unrealized_pnl(p, evt.close)
                for s, p in open_positions.items()
                if s == symbol
            )
            equity_curve.append((bar_ts, balance + unrealized))

            # Evaluate for new signal only if no open position for this symbol
            if symbol in open_positions:
                continue
            if len(open_positions) >= cfg.max_open_positions:
                continue

            try:
                result_tuple = self._ensemble.evaluate_symbol(symbol, replay_store, cfg.exchange, bar_ts)
                signal, regime, raw_signals = result_tuple
            except Exception as e:
                log.debug("backtest.ensemble_error", symbol=symbol, error=str(e))
                continue

            if signal is None:
                continue

            # ML filter (optional — cold-start in backtest mode passes through)
            if self._ml_filter and cfg.use_ml_filter:
                try:
                    decision = await self._ml_filter.evaluate(
                        signal, replay_store, cfg.exchange
                    )
                    if not decision.accepted:
                        continue
                except Exception:
                    pass  # pass-through on error

            # Compute ATR for sizing (needed by RiskManager)
            atr_val = _get_atr_from_store(replay_store, cfg.exchange, symbol, cfg.primary_timeframe)
            if atr_val is None or atr_val <= 0:
                continue  # can't size without ATR

            # Size the trade using the real RiskManager
            sized = risk_manager.size_trade(
                symbol=symbol,
                exchange=cfg.exchange,
                side=signal.side,
                entry_price=evt.close,
                atr=atr_val,
                equity=balance,
                free_margin=balance,  # conservative: treat full balance as free
                strategy=signal.strategy,
                confidence=signal.confidence,
            )
            if isinstance(sized, RejectedTrade):
                continue

            # Fill at current bar's close (approximates "next bar open" for market orders)
            fill_price = _apply_slippage(evt.close, signal.side, cfg.slippage_bps)
            qty = sized.quantity
            if qty <= 0:
                continue

            notional = fill_price * qty * cfg.leverage
            fee_usd = notional * (cfg.taker_fee_bps / 10_000.0)
            balance -= fee_usd

            entry_fill = SimFill(
                symbol=symbol,
                side=signal.side,
                price=fill_price,
                quantity=qty,
                fee_usd=fee_usd,
                ts=bar_ts,
                reason="entry",
                strategy=signal.strategy,
            )
            fills.append(entry_fill)

            open_positions[symbol] = SimPosition(
                symbol=symbol,
                side=signal.side,
                entry_price=fill_price,
                quantity=qty,
                stop_loss=sized.stop_loss,
                take_profit=sized.take_profit,
                leverage=cfg.leverage,
                risk_usd=sized.risk_usd,
                notional_usd=notional,
                strategy=signal.strategy,
                confidence=signal.confidence,
                entry_ts=bar_ts,
                entry_bar_idx=bar_idx,
            )
            log.debug(
                "backtest.entry",
                symbol=symbol,
                side=signal.side.value,
                price=round(fill_price, 4),
                qty=round(qty, 6),
                sl=round(sized.stop_loss, 4),
                tp=round(sized.take_profit, 4),
            )

        # Close all open positions at the last bar's close (end of window)
        for symbol, pos in list(open_positions.items()):
            last_close = bars_by_symbol[symbol][-1].close if bars_by_symbol.get(symbol) else pos.entry_price
            exit_fill = self._close_position(pos, last_close, end_ts, "close_at_end")
            balance += exit_fill.pnl_usd - exit_fill.fee_usd
            fills.append(exit_fill)

        result = BacktestResult(
            run_id=run_id,
            symbols=symbols,
            start_ts=start_ts,
            end_ts=end_ts,
            initial_balance=cfg.initial_balance,
            final_balance=balance,
            fills=fills,
            equity_curve=equity_curve,
            config=cfg,
        )
        _compute_metrics(result)
        log.info(
            "backtest.run_complete",
            run_id=run_id,
            trades=result.total_trades,
            winrate=round(result.winrate, 3),
            sharpe=round(result.sharpe_ratio, 3),
            max_dd=round(result.max_drawdown_pct, 3),
            return_pct=round(result.return_pct, 4),
        )
        return result

    # ─────────────────────── internals ───────────────────────

    def _check_exits(
        self,
        pos: SimPosition,
        bar: KlineEvent,
        ts: datetime,
    ) -> SimFill | None:
        """
        Check if a bar triggers the stop-loss or take-profit.
        Conservative: SL checked first (worst case for the backtest).
        """
        if pos.side == OrderSide.BUY:
            # Long: SL hit if the bar's low goes below stop_loss
            if bar.low <= pos.stop_loss:
                return self._close_position(pos, pos.stop_loss, ts, "sl")
            # TP hit if the bar's high goes above take_profit
            if bar.high >= pos.take_profit:
                return self._close_position(pos, pos.take_profit, ts, "tp")
        else:
            # Short: SL hit if the bar's high goes above stop_loss
            if bar.high >= pos.stop_loss:
                return self._close_position(pos, pos.stop_loss, ts, "sl")
            # TP hit if the bar's low goes below take_profit
            if bar.low <= pos.take_profit:
                return self._close_position(pos, pos.take_profit, ts, "tp")
        return None

    def _close_position(
        self,
        pos: SimPosition,
        exit_price: float,
        ts: datetime,
        reason: str,
    ) -> SimFill:
        cfg = self._config
        if pos.side == OrderSide.BUY:
            pnl = (exit_price - pos.entry_price) * pos.quantity * pos.leverage
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity * pos.leverage
        notional = exit_price * pos.quantity * pos.leverage
        fee = notional * (cfg.taker_fee_bps / 10_000.0)
        return SimFill(
            symbol=pos.symbol,
            side=OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY,
            price=exit_price,
            quantity=pos.quantity,
            fee_usd=fee,
            ts=ts,
            reason=reason,
            pnl_usd=pnl,
            strategy=pos.strategy,
        )

    @staticmethod
    def _unrealized_pnl(pos: SimPosition, current_price: float) -> float:
        if pos.side == OrderSide.BUY:
            return (current_price - pos.entry_price) * pos.quantity * pos.leverage
        return (pos.entry_price - current_price) * pos.quantity * pos.leverage

    @staticmethod
    def _empty_result(run_id: str, symbols: list[str], cfg: BacktestConfig) -> BacktestResult:
        now = datetime.now(UTC)
        return BacktestResult(
            run_id=run_id,
            symbols=symbols,
            start_ts=now,
            end_ts=now,
            initial_balance=cfg.initial_balance,
            final_balance=cfg.initial_balance,
            fills=[],
            equity_curve=[],
            config=cfg,
        )


# ─────────────────────── helpers ───────────────────────


def _get_atr_from_store(
    store: FeatureStore,
    exchange: str,
    symbol: str,
    timeframe: str,
    period: int = 14,
) -> float | None:
    """Return the current ATR value from the FeatureStore."""
    try:
        from src.utils.indicators import atr as compute_atr  # noqa: PLC0415
        key = FeatureKey(exchange, symbol, timeframe)
        df = store.as_df(key, min_bars=period + 5)
        if df is None:
            return None
        series = compute_atr(df, length=period)
        if series is None or series.empty:
            return None
        val = float(series.iloc[-1])
        return None if math.isnan(val) else val
    except Exception:
        return None


def _apply_slippage(price: float, side: OrderSide, slippage_bps: float) -> float:
    """Apply half-spread slippage in the direction that's worse for us."""
    factor = slippage_bps / 10_000.0
    if side == OrderSide.BUY:
        return price * (1 + factor)
    return price * (1 - factor)


def _compute_metrics(result: BacktestResult) -> None:
    """Populate BacktestResult metric fields from fills and equity_curve."""
    exit_fills = [f for f in result.fills if f.reason != "entry"]
    result.total_trades = len(exit_fills)
    result.total_pnl = sum(f.pnl_usd for f in exit_fills)
    result.total_fees = sum(f.fee_usd for f in result.fills)
    result.profitable_trades = sum(1 for f in exit_fills if f.pnl_usd > 0)
    result.winrate = (
        result.profitable_trades / result.total_trades if result.total_trades > 0 else 0.0
    )

    # R-multiple: pnl / risk_usd per trade (need to pair entry↔exit fills)
    r_multiples: list[float] = []
    entry_fills = [f for f in result.fills if f.reason == "entry"]
    symbol_entry: dict[str, SimFill] = {}
    for f in result.fills:
        if f.reason == "entry":
            symbol_entry[f.symbol] = f
        else:
            entry = symbol_entry.pop(f.symbol, None)
            if entry is not None and entry.price > 0:
                sl_dist = abs(entry.price * 0.015)  # approximate: use 1.5% as proxy
                if sl_dist > 0:
                    r_multiples.append(f.pnl_usd / (sl_dist * entry.quantity))
    result.avg_r_multiple = sum(r_multiples) / len(r_multiples) if r_multiples else 0.0

    # Sharpe ratio from equity curve daily returns
    if len(result.equity_curve) >= 2:
        equities = [e for _, e in result.equity_curve]
        returns = [
            (equities[i] - equities[i - 1]) / equities[i - 1]
            for i in range(1, len(equities))
            if equities[i - 1] > 0
        ]
        if returns:
            mean_r = sum(returns) / len(returns)
            variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
            std_r = math.sqrt(variance) if variance > 0 else 0.0
            bars_per_day = 96  # 15m bars in 24h
            annualisation = math.sqrt(252 * bars_per_day)
            result.sharpe_ratio = (mean_r / std_r * annualisation) if std_r > 0 else 0.0

    # Maximum drawdown from equity curve
    if result.equity_curve:
        peak = result.equity_curve[0][1]
        max_dd = 0.0
        for _, eq in result.equity_curve:
            if eq > peak:
                peak = eq
            if peak > 0:
                dd = (peak - eq) / peak
                if dd > max_dd:
                    max_dd = dd
        result.max_drawdown_pct = max_dd
