"""
Backtest runner with walk-forward analysis.

Usage examples::

    # Simple single-window backtest (last 90 days)
    python scripts/run_backtest.py --window 90d

    # Walk-forward: 180d train → 60d test, stepping 30d
    python scripts/run_backtest.py --train-window 180d --test-window 60d --step 30d

    # Limit to specific symbols
    python scripts/run_backtest.py --symbols BTC/USDT:USDT ETH/USDT:USDT

    # Save report to custom directory
    python scripts/run_backtest.py --report-dir reports/

Output:
    - Console summary table (Sharpe, max DD, winrate per window)
    - JSON report at reports/backtest_<run_id>.json
    - Saved to backtest_runs + backtest_fills DB tables
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Add project root to sys.path so `src` is importable without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from src.data.storage import Storage
from src.exchanges.base import KlineEvent
from src.settings import RegimeConfig, RiskConfig, Settings, StrategiesConfig
from src.strategies.ensemble import Ensemble
from src.strategies.funding_contrarian import FundingContrarianStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.regime import RegimeClassifier
from src.strategies.volatility_breakout import VolatilityBreakoutStrategy
from src.strategies.vwap import VWAPStrategy
from src.data.feature_store import FundingHistory
from src.utils.logging import get_logger

log = get_logger(__name__)


# ─────────────────────── CLI ───────────────────────


def _parse_duration(s: str) -> timedelta:
    """Parse strings like '90d', '4h', '30m' → timedelta."""
    s = s.strip()
    if s.endswith("d"):
        return timedelta(days=int(s[:-1]))
    if s.endswith("h"):
        return timedelta(hours=int(s[:-1]))
    if s.endswith("m"):
        return timedelta(minutes=int(s[:-1]))
    raise ValueError(f"Cannot parse duration '{s}'. Use e.g. '90d', '4h', '30m'.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a backtest (or walk-forward) using historical OHLCV data from the DB."
    )
    # Window mode (simple)
    parser.add_argument(
        "--window", default=None,
        help="Single window duration, e.g. '90d'. Mutually exclusive with --train-window.",
    )
    # Walk-forward mode
    parser.add_argument("--train-window", default="180d", help="Training window (default: 180d)")
    parser.add_argument("--test-window", default="60d", help="Test window (default: 60d)")
    parser.add_argument("--step", default="30d", help="Walk-forward step size (default: 30d)")
    # Scope
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Symbols to include (default: top symbols from config.universe.force_include, "
             "or all symbols in DB if none specified).",
    )
    parser.add_argument("--exchange", default="okx", help="Exchange to pull data from (default: okx)")
    parser.add_argument("--timeframes", nargs="+", default=["15m", "1h", "4h"],
                        help="Timeframes to load (default: 15m 1h 4h)")
    # Simulation parameters
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--taker-fee-bps", type=float, default=6.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--leverage", type=int, default=3)
    # Output
    parser.add_argument("--report-dir", default="reports", help="Directory for JSON reports.")
    parser.add_argument("--no-db-save", action="store_true", help="Skip saving results to DB.")
    return parser.parse_args()


# ─────────────────────── DB loading ───────────────────────


async def load_bars(
    storage: Storage,
    exchange: str,
    symbols: list[str],
    timeframes: list[str],
    start_ts: datetime,
    end_ts: datetime,
) -> dict[str, list[KlineEvent]]:
    """Load OHLCV bars from the DB for the given window, returning KlineEvent lists."""
    log.info(
        "backtest.loading_bars",
        exchange=exchange,
        symbols=symbols,
        timeframes=timeframes,
        start=start_ts.isoformat(),
        end=end_ts.isoformat(),
    )
    bars_by_symbol: dict[str, list[KlineEvent]] = {s: [] for s in symbols}

    async with storage._pool.acquire() as conn:
        for symbol in symbols:
            for tf in timeframes:
                rows = await conn.fetch(
                    """
                    SELECT ts, open, high, low, close, volume
                    FROM ohlcv
                    WHERE exchange = $1 AND symbol = $2 AND timeframe = $3
                      AND ts >= $4 AND ts < $5
                    ORDER BY ts ASC
                    """,
                    exchange, symbol, tf, start_ts, end_ts,
                )
                for row in rows:
                    bars_by_symbol[symbol].append(KlineEvent(
                        exchange=exchange,
                        symbol=symbol,
                        timeframe=tf,
                        ts=row["ts"].replace(tzinfo=UTC),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                        closed=True,
                    ))

    log.info(
        "backtest.bars_loaded",
        total={s: len(v) for s, v in bars_by_symbol.items()},
    )
    return bars_by_symbol


async def save_result(storage: Storage, result: BacktestResult) -> None:
    """Persist a BacktestResult to backtest_runs + backtest_fills tables."""
    async with storage._pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO backtest_runs
                (run_id, symbols, start_ts, end_ts, initial_balance, final_balance,
                 total_trades, profitable_trades, winrate, sharpe_ratio,
                 max_drawdown_pct, total_pnl, total_fees, avg_r_multiple, config)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
            ON CONFLICT (run_id) DO NOTHING
            """,
            result.run_id,
            result.symbols,
            result.start_ts,
            result.end_ts,
            result.initial_balance,
            result.final_balance,
            result.total_trades,
            result.profitable_trades,
            result.winrate,
            result.sharpe_ratio,
            result.max_drawdown_pct,
            result.total_pnl,
            result.total_fees,
            result.avg_r_multiple,
            json.dumps(dataclasses.asdict(result.config)),
        )
        if result.fills:
            await conn.executemany(
                """
                INSERT INTO backtest_fills
                    (run_id, ts, symbol, side, price, quantity, fee_usd, pnl_usd, reason, strategy)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                """,
                [
                    (
                        result.run_id,
                        f.ts,
                        f.symbol,
                        f.side.value,
                        f.price,
                        f.quantity,
                        f.fee_usd,
                        f.pnl_usd,
                        f.reason,
                        f.strategy,
                    )
                    for f in result.fills
                ],
            )
    log.info("backtest.saved_to_db", run_id=result.run_id)


def save_report(result: BacktestResult, report_dir: str) -> str:
    """Save JSON report and return the file path."""
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    path = Path(report_dir) / f"backtest_{result.run_id}.json"

    report: dict = {
        "run_id": result.run_id,
        "start_ts": result.start_ts.isoformat(),
        "end_ts": result.end_ts.isoformat(),
        "symbols": result.symbols,
        "initial_balance": result.initial_balance,
        "final_balance": result.final_balance,
        "net_pnl": result.net_pnl,
        "return_pct": result.return_pct,
        "total_trades": result.total_trades,
        "profitable_trades": result.profitable_trades,
        "winrate": result.winrate,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown_pct": result.max_drawdown_pct,
        "total_pnl": result.total_pnl,
        "total_fees": result.total_fees,
        "avg_r_multiple": result.avg_r_multiple,
        "equity_curve": [
            {"ts": ts.isoformat(), "equity": eq}
            for ts, eq in result.equity_curve[-500:]  # cap at 500 points for file size
        ],
        "fills": [
            {
                "ts": f.ts.isoformat(),
                "symbol": f.symbol,
                "side": f.side.value,
                "price": f.price,
                "quantity": f.quantity,
                "fee_usd": f.fee_usd,
                "pnl_usd": f.pnl_usd,
                "reason": f.reason,
                "strategy": f.strategy,
            }
            for f in result.fills
        ],
    }
    path.write_text(json.dumps(report, indent=2))
    log.info("backtest.report_saved", path=str(path))
    return str(path)


# ─────────────────────── ensemble factory ───────────────────────


def _build_ensemble(settings: Settings) -> Ensemble:
    """Build the same Ensemble as the live Orchestrator."""
    sc = settings.config.strategies
    rc = settings.config.regime

    strategies = []

    if sc.momentum.enabled:
        strategies.append(MomentumStrategy(
            timeframes=sc.momentum.timeframes,
            ema_fast=sc.momentum.ema_fast,
            ema_mid=sc.momentum.ema_mid,
            ema_slow=sc.momentum.ema_slow,
            rsi_period=sc.momentum.rsi_period,
            rsi_long_threshold=sc.momentum.rsi_long_threshold,
            rsi_short_threshold=sc.momentum.rsi_short_threshold,
            macd_fast=sc.momentum.macd_fast,
            macd_slow=sc.momentum.macd_slow,
            macd_signal=sc.momentum.macd_signal,
            base_confidence=sc.momentum.base_confidence,
            volume_multiplier=sc.momentum.volume_multiplier,
        ))

    if sc.mean_reversion.enabled:
        strategies.append(MeanReversionStrategy(
            timeframe=sc.mean_reversion.timeframe,
            bb_period=sc.mean_reversion.bb_period,
            bb_std=sc.mean_reversion.bb_std,
            rsi_period=sc.mean_reversion.rsi_period,
            rsi_long_threshold=sc.mean_reversion.rsi_long_threshold,
            rsi_short_threshold=sc.mean_reversion.rsi_short_threshold,
            adx_max=sc.mean_reversion.adx_max,
            base_confidence=sc.mean_reversion.base_confidence,
        ))

    if sc.volatility_breakout.enabled:
        strategies.append(VolatilityBreakoutStrategy(
            timeframe=sc.volatility_breakout.timeframe,
            donchian_period=sc.volatility_breakout.donchian_period,
            squeeze_atr_ratio_max=sc.volatility_breakout.squeeze_atr_ratio_max,
            squeeze_bars=sc.volatility_breakout.squeeze_bars,
            volume_multiple=sc.volatility_breakout.volume_multiple,
            base_confidence=sc.volatility_breakout.base_confidence,
        ))

    if sc.vwap.enabled:
        strategies.append(VWAPStrategy(
            timeframe=sc.vwap.timeframe,
            ema_period=sc.vwap.ema_period,
            vwap_band_pct=sc.vwap.vwap_band_pct,
            base_confidence=sc.vwap.base_confidence,
        ))

    if sc.funding_contrarian.enabled:
        fh = FundingHistory()
        strategies.append(FundingContrarianStrategy(
            funding_history=fh,
            extreme_threshold=sc.funding_contrarian.extreme_threshold,
            low_threshold=sc.funding_contrarian.low_threshold,
            base_confidence=sc.funding_contrarian.base_confidence,
        ))

    regime_classifier = RegimeClassifier(rc)

    return Ensemble(
        strategies=strategies,
        regime_classifier=regime_classifier,
        min_net_vote=rc.min_net_vote,
        regime_weights=rc.weights,
    )


# ─────────────────────── summary printing ───────────────────────


def _print_summary(results: list[BacktestResult]) -> None:
    header = (
        f"{'Run ID':<14} {'Start':>10} {'End':>10} {'Trades':>7} "
        f"{'Winrate':>8} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8}"
    )
    separator = "-" * len(header)
    print("\n" + separator)
    print(header)
    print(separator)
    for r in results:
        print(
            f"{r.run_id:<14} "
            f"{r.start_ts.strftime('%Y-%m-%d'):>10} "
            f"{r.end_ts.strftime('%Y-%m-%d'):>10} "
            f"{r.total_trades:>7} "
            f"{r.winrate:>8.1%} "
            f"{r.sharpe_ratio:>7.2f} "
            f"{r.max_drawdown_pct:>7.1%} "
            f"{r.return_pct:>8.2%}"
        )
    print(separator)
    if len(results) > 1:
        avg_sharpe = sum(r.sharpe_ratio for r in results) / len(results)
        avg_dd = sum(r.max_drawdown_pct for r in results) / len(results)
        avg_wr = sum(r.winrate for r in results) / len(results)
        print(
            f"{'AVERAGE':<14} {'':>10} {'':>10} "
            f"{sum(r.total_trades for r in results):>7} "
            f"{avg_wr:>8.1%} "
            f"{avg_sharpe:>7.2f} "
            f"{avg_dd:>7.1%} "
            f"{'':>8}"
        )
        print(separator)
    print()


# ─────────────────────── main ───────────────────────


async def main() -> None:
    args = _parse_args()

    # Load settings from config.yaml + .env
    settings = Settings()

    # Build the Ensemble with the real strategies
    ensemble = _build_ensemble(settings)

    # Build risk config for sizing
    rc = settings.config.risk

    bt_config = BacktestConfig(
        initial_balance=args.initial_balance,
        taker_fee_bps=args.taker_fee_bps,
        slippage_bps=args.slippage_bps,
        leverage=args.leverage,
        risk_per_trade_pct=rc.risk_per_trade_pct,
        sl_atr_multiplier=rc.sl_atr_multiplier,
        min_reward_to_risk=rc.min_reward_to_risk,
        max_open_positions=rc.max_open_positions,
        exchange=args.exchange,
        timeframes=args.timeframes,
    )

    engine = BacktestEngine(ensemble=ensemble, risk_config=rc, config=bt_config)

    # Connect to DB
    storage = Storage(settings.DATABASE_URL)
    await storage.connect()

    try:
        # Determine symbols
        symbols = args.symbols or settings.config.universe.force_include or []
        if not symbols:
            print("No symbols specified. Use --symbols or add force_include to config.yaml.")
            return

        # Determine time windows
        now = datetime.now(UTC)
        windows: list[tuple[datetime, datetime]] = []

        if args.window is not None:
            w = _parse_duration(args.window)
            windows = [(now - w, now)]
        else:
            train_w = _parse_duration(args.train_window)
            test_w = _parse_duration(args.test_window)
            step_w = _parse_duration(args.step)
            # Walk-forward: train+test windows sliding forward by step_w
            # We start train_w before the earliest (now - test_w) and step forward.
            total_lookback = train_w + test_w
            start = now - total_lookback
            while start + train_w + test_w <= now:
                train_end = start + train_w
                test_start = train_end
                test_end = test_start + test_w
                windows.append((test_start, test_end))  # only test on out-of-sample portion
                start += step_w

        results: list[BacktestResult] = []
        for w_start, w_end in windows:
            bars = await load_bars(
                storage, args.exchange, symbols, args.timeframes, w_start, w_end
            )
            result = await engine.run(symbols=symbols, bars_by_symbol=bars)
            results.append(result)

            # Save report
            report_path = save_report(result, args.report_dir)

            if not args.no_db_save:
                try:
                    await save_result(storage, result)
                except Exception as e:
                    log.warning("backtest.db_save_failed", error=str(e))

        _print_summary(results)

    finally:
        await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
