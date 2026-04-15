"""
Top-level async orchestrator — owns every long-running task and wires all the
modules together.

Task topology (all children of one asyncio.TaskGroup):

    exchange_connectors ─── kline_pump (per connector)
                         ─── funding_pump (per connector)
                         ─── order_pump (per connector)
                         ─── fill_pump (per connector)
                         ─── position_pump (per connector)
    portfolio_tracker ─── reconcile_loop
    universe_selector ─── refresh_loop
    funding_arb_scanner ─── periodic_loop
    health_server ─── uvicorn task
    telegram ─── command polling + daily report

Graceful shutdown flow (SIGTERM/SIGINT):
    1. cancel_all in TaskGroup
    2. if risk.flatten_on_shutdown: close all positions at market
    3. close connectors + DB pool
    4. stop telegram
"""

from __future__ import annotations

import asyncio
import signal
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.data.feature_store import Bar, CVDStore, FeatureKey, FeatureStore, FundingHistory, OIStore
from src.data.storage import Storage
from src.data.universe import UniverseSelector, UniverseSnapshot
from src.exchanges.base import (
    ExchangeConnector,
    FillEvent,
    FundingRateEvent,
    KlineEvent,
    OIEvent,
    OrderSide,
    OrderUpdateEvent,
    PositionUpdateEvent,
    TradeEvent,
)
from src.exchanges.bybit import BybitConnector
from src.exchanges.okx import OKXConnector
from src.exchanges.paper import PaperConnector
from src.execution.executor import OrderExecutor
from src.execution.exit_manager import ExitConfig, ExitManager
from src.execution.order_router import OrderRouter
from src.monitoring import prom_metrics as m
from src.monitoring.health import LiveBroadcaster, build_app, serve_health
from src.notifications.telegram import TelegramNotifier
from src.portfolio.analytics import Analytics
from src.portfolio.tracker import PortfolioTracker
from src.risk.kill_switch import KillSwitch
from src.risk.limits import PortfolioLimits
from src.risk.manager import RejectedTrade, RiskManager
from src.settings import RuntimeMode, Settings
from src.strategies.ensemble import Ensemble
from src.strategies.funding_arb import FundingArbStrategy
from src.strategies.funding_contrarian import FundingContrarianStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.regime import RegimeClassifier
from src.strategies.volatility_breakout import VolatilityBreakoutStrategy
from src.strategies.vwap import VWAPStrategy
from src.strategies.liquidation_cascade import LiquidationCascadeStrategy
from src.ml.model import MLSignalFilter
from src.utils.indicators import atr
from src.utils.logging import get_logger
from src.utils.time import now_utc

if TYPE_CHECKING:
    from src.strategies.base import PairSignal, Signal

log = get_logger(__name__)

# Which timeframes we subscribe to and keep in the feature store. The
# momentum strategy's three TFs + 15m for regime + 1h for breakout.
REQUIRED_TIMEFRAMES: list[str] = ["15m", "1h", "4h"]


class Orchestrator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._mode = settings.config.mode
        self._storage: Storage | None = None
        self._connectors: dict[str, ExchangeConnector] = {}
        self._feature_store = FeatureStore(max_bars=1500)
        self._universe: UniverseSelector | None = None
        self._ensemble: Ensemble | None = None
        self._funding_arb: FundingArbStrategy | None = None
        self._risk_manager: RiskManager | None = None
        self._limits: PortfolioLimits | None = None
        self._kill_switch: KillSwitch | None = None
        self._tracker: PortfolioTracker | None = None
        self._executor: OrderExecutor | None = None
        self._router: OrderRouter | None = None
        self._telegram: TelegramNotifier | None = None
        self._analytics: Analytics | None = None
        self._ml_filter: MLSignalFilter | None = None
        self._regime_classifier: RegimeClassifier | None = None
        self._broadcaster = LiveBroadcaster()
        self._shutdown_event = asyncio.Event()
        self._startup_ts: datetime | None = None
        # In-flight dedup: prevent two concurrent evaluations for the same symbol
        # from both passing the open_positions check before either places an order.
        self._in_flight_symbols: set[str] = set()
        # Perp-native signal stores — populated by trade/OI/funding pumps
        self._cvd_store: CVDStore | None = None
        self._oi_store: OIStore | None = None
        self._funding_history: FundingHistory | None = None
        self._exit_manager: "ExitManager | None" = None

    def _regime_snapshot(self) -> dict[str, str]:
        """Return {symbol: regime_name} for all symbols in the current universe.
        Called by the Telegram minimap to display current market regime per symbol."""
        if self._regime_classifier is None or self._universe is None:
            return {}
        snap = self._universe.current()
        if snap is None:
            return {}
        result: dict[str, str] = {}
        # Use the first available exchange as the regime source.
        exchange = next(iter(self._connectors), "okx")
        for symbol in snap.symbols:
            try:
                regime = self._regime_classifier.classify(symbol, self._feature_store, exchange)
                result[symbol] = regime.value
            except Exception:
                pass
        return result

    # ─────────────────────── bootstrap ───────────────────────

    async def setup(self) -> None:
        log.info("orchestrator.setup_start", mode=self._mode.value)

        # DB
        self._storage = Storage(
            dsn=self._settings.database_url.get_secret_value(),
            pool_min=self._settings.config.database.pool_min,
            pool_max=self._settings.config.database.pool_max,
            statement_cache_size=self._settings.config.database.statement_cache_size,
        )
        await self._storage.connect()
        await self._storage.apply_schema()

        # Perp-native signal stores (initialised before ML so the filter can use them)
        strat_cfg = self._settings.config.strategies
        if strat_cfg.cvd.enabled:
            self._cvd_store = CVDStore(max_bars=strat_cfg.cvd.max_bars)
        if strat_cfg.oi.enabled:
            self._oi_store = OIStore(max_samples=strat_cfg.oi.max_samples)
        self._funding_history = FundingHistory()

        # ML signal filter — loaded before strategies so it's ready at first bar
        self._ml_filter = MLSignalFilter(
            storage=self._storage,
            cvd_store=self._cvd_store,
            oi_store=self._oi_store,
        )
        await self._ml_filter.load()

        # Kill switch needs storage; wire the telegram halt callback AFTER
        # telegram is created, via force_halt below.
        self._kill_switch = KillSwitch(
            config=self._settings.config.risk,
            storage=self._storage,
            on_halt=self._on_halt_callback,
            on_warning=self._on_warning_callback,
        )
        await self._kill_switch.load()

        # Exchange connectors — built per configured mode
        self._connectors = self._build_connectors()
        for name, conn in self._connectors.items():
            await conn.connect()
            log.info("orchestrator.connector_connected", exchange=name)

        # Universe selector (needs connectors + storage)
        # Note: if paper mode, the universe is queried against the underlying
        # venue (live mainnet).
        okx_for_universe = self._connectors.get("okx") or self._connectors.get("paper-okx")
        bybit_for_universe = (
            self._connectors.get("bybit")
            or self._connectors.get("paper-bybit")
            or okx_for_universe  # fall back to OKX-only universe if Bybit is disabled
        )
        if okx_for_universe is None:
            raise RuntimeError("OKX connector is required")
        self._universe = UniverseSelector(
            config=self._settings.config.universe,
            okx=okx_for_universe,
            bybit=bybit_for_universe,
            storage=self._storage,
        )
        snap = await self._universe.bootstrap()
        await self._subscribe_universe(snap, previous=None)
        await self._backfill_feature_store(snap)
        self._startup_ts = datetime.now(timezone.utc)

        # Risk
        self._risk_manager = RiskManager(config=self._settings.config.risk)
        self._tracker = PortfolioTracker(
            connectors=self._connectors,
            storage=self._storage,
            mode=self._mode,
            kill_switch=self._kill_switch,
            reconcile_interval_sec=self._settings.config.monitoring.reconcile_interval_sec,
        )
        self._limits = PortfolioLimits(config=self._settings.config.risk, tracker=self._tracker)

        # Execution
        self._router = OrderRouter(connectors=self._connectors, tracker=self._tracker)
        self._executor = OrderExecutor(
            router=self._router,
            storage=self._storage,
            config=self._settings.config.execution,
            mode=self._mode,
        )

        # Exit manager — dynamic SL/TP management (breakeven, trailing, partial TP)
        self._exit_manager = ExitManager(
            tracker=self._tracker,
            feature_store=self._feature_store,
            connectors=self._connectors,
        )

        # Strategies + ensemble
        self._build_strategies()

        # Analytics + telegram (telegram is last — it wires into everything)
        self._analytics = Analytics(storage=self._storage, mode=self._mode.value)
        import os as _os
        _miniapp_url = _os.getenv("MINIAPP_URL", f"http://localhost:{self._settings.health_port}/app")
        self._telegram = TelegramNotifier(
            settings=self._settings,
            notif_config=self._settings.config.notifications,
            tracker=self._tracker,
            analytics=self._analytics,
            kill_switch=self._kill_switch,
            ml_filter=self._ml_filter,
            get_regimes=self._regime_snapshot,
            miniapp_url=_miniapp_url,
        )
        await self._telegram.start()

        log.info("orchestrator.setup_complete")

    def _build_connectors(self) -> dict[str, ExchangeConnector]:
        """Construct connectors for enabled exchanges appropriate for the current mode."""
        exc_cfg = self._settings.config.exchanges
        result: dict[str, ExchangeConnector] = {}

        if exc_cfg.okx.enabled:
            if self._mode == RuntimeMode.PAPER:
                okx_live = OKXConnector(
                    api_key=self._settings.okx_api_key.get_secret_value(),
                    api_secret=self._settings.okx_api_secret.get_secret_value(),
                    api_passphrase=self._settings.okx_api_passphrase.get_secret_value(),
                    mode=RuntimeMode.LIVE,
                    rest_rate_limit_per_sec=exc_cfg.okx.rest_rate_limit_per_sec,
                    ws_ping_interval_sec=exc_cfg.okx.ws_ping_interval_sec,
                    ws_reconnect_max_backoff_sec=exc_cfg.okx.ws_reconnect_max_backoff_sec,
                )
                result["okx"] = PaperConnector(underlying=okx_live, paper_config=self._settings.config.paper)
            else:
                result["okx"] = OKXConnector(
                    api_key=self._settings.okx_api_key.get_secret_value(),
                    api_secret=self._settings.okx_api_secret.get_secret_value(),
                    api_passphrase=self._settings.okx_api_passphrase.get_secret_value(),
                    mode=self._mode,
                    rest_rate_limit_per_sec=exc_cfg.okx.rest_rate_limit_per_sec,
                    ws_ping_interval_sec=exc_cfg.okx.ws_ping_interval_sec,
                    ws_reconnect_max_backoff_sec=exc_cfg.okx.ws_reconnect_max_backoff_sec,
                )

        if exc_cfg.bybit.enabled:
            if self._mode == RuntimeMode.PAPER:
                bybit_live = BybitConnector(
                    api_key=self._settings.bybit_api_key.get_secret_value(),
                    api_secret=self._settings.bybit_api_secret.get_secret_value(),
                    mode=RuntimeMode.LIVE,
                    rest_rate_limit_per_sec=exc_cfg.bybit.rest_rate_limit_per_sec,
                    ws_ping_interval_sec=exc_cfg.bybit.ws_ping_interval_sec,
                    ws_reconnect_max_backoff_sec=exc_cfg.bybit.ws_reconnect_max_backoff_sec,
                )
                result["bybit"] = PaperConnector(
                    underlying=bybit_live, paper_config=self._settings.config.paper
                )
            else:
                result["bybit"] = BybitConnector(
                    api_key=self._settings.bybit_api_key.get_secret_value(),
                    api_secret=self._settings.bybit_api_secret.get_secret_value(),
                    mode=self._mode,
                    rest_rate_limit_per_sec=exc_cfg.bybit.rest_rate_limit_per_sec,
                    ws_ping_interval_sec=exc_cfg.bybit.ws_ping_interval_sec,
                    ws_reconnect_max_backoff_sec=exc_cfg.bybit.ws_reconnect_max_backoff_sec,
                )

        if not result:
            raise RuntimeError("no exchange connectors enabled in config")
        return result

    async def _backfill_feature_store(self, snap: UniverseSnapshot) -> None:
        """Seed each (symbol, timeframe) buffer with recent history from REST."""
        for symbol in snap.symbols:
            for exchange_name, conn in self._connectors.items():
                for tf in REQUIRED_TIMEFRAMES:
                    try:
                        bars = await conn.fetch_ohlcv_backfill(symbol, tf, limit=500)
                    except Exception as e:
                        log.warning(
                            "orchestrator.backfill_failed",
                            exchange=exchange_name,
                            symbol=symbol,
                            tf=tf,
                            error=str(e),
                        )
                        continue
                    key = FeatureKey(exchange=exchange_name, symbol=symbol, timeframe=tf)
                    store_bars = [
                        Bar(
                            ts_ms=int(b.ts.timestamp() * 1000),
                            open=b.open,
                            high=b.high,
                            low=b.low,
                            close=b.close,
                            volume=b.volume,
                        )
                        for b in bars
                    ]
                    self._feature_store.bulk_load(key, store_bars)

    async def _subscribe_universe(
        self,
        current: UniverseSnapshot,
        previous: UniverseSnapshot | None,
    ) -> None:
        """Diff-subscribe klines + funding for each (symbol, tf, exchange)."""
        prev_set = set(previous.symbols) if previous else set()
        curr_set = set(current.symbols)
        added = curr_set - prev_set
        removed = prev_set - curr_set

        for sym in added:
            for exchange_name, conn in self._connectors.items():
                for tf in REQUIRED_TIMEFRAMES:
                    try:
                        await conn.subscribe_klines(sym, tf)
                    except Exception as e:
                        log.warning(
                            "orchestrator.subscribe_kline_failed",
                            exchange=exchange_name,
                            symbol=sym,
                            tf=tf,
                            error=str(e),
                        )
                try:
                    await conn.subscribe_funding(sym)
                except Exception as e:
                    log.warning(
                        "orchestrator.subscribe_funding_failed",
                        exchange=exchange_name,
                        symbol=sym,
                        error=str(e),
                    )
                if self._cvd_store is not None:
                    try:
                        await conn.subscribe_trades(sym)
                    except Exception as e:
                        log.warning(
                            "orchestrator.subscribe_trades_failed",
                            exchange=exchange_name,
                            symbol=sym,
                            error=str(e),
                        )
        for sym in removed:
            for exchange_name, conn in self._connectors.items():
                for tf in REQUIRED_TIMEFRAMES:
                    try:
                        await conn.unsubscribe_klines(sym, tf)
                    except Exception:
                        pass
                try:
                    await conn.unsubscribe_funding(sym)
                except Exception:
                    pass
                if self._cvd_store is not None:
                    try:
                        await conn.unsubscribe_trades(sym)
                    except Exception:
                        pass
                # Drop feature store buffers for removed symbols
                for tf in REQUIRED_TIMEFRAMES:
                    self._feature_store.drop(FeatureKey(exchange_name, sym, tf))

    def _build_strategies(self) -> None:
        cfg = self._settings.config.strategies
        strategies = []
        if cfg.momentum.enabled:
            strategies.append(
                MomentumStrategy(
                    timeframes=cfg.momentum.timeframes,
                    ema_fast=cfg.momentum.ema_fast,
                    ema_mid=cfg.momentum.ema_mid,
                    ema_slow=cfg.momentum.ema_slow,
                    rsi_period=cfg.momentum.rsi_period,
                    rsi_long_threshold=cfg.momentum.rsi_long_threshold,
                    rsi_short_threshold=cfg.momentum.rsi_short_threshold,
                    macd_fast=cfg.momentum.macd_fast,
                    macd_slow=cfg.momentum.macd_slow,
                    macd_signal=cfg.momentum.macd_signal,
                    base_confidence=cfg.momentum.base_confidence,
                    volume_multiplier=cfg.momentum.volume_multiplier,
                )
            )
        if cfg.mean_reversion.enabled:
            strategies.append(
                MeanReversionStrategy(
                    timeframe=cfg.mean_reversion.timeframe,
                    bb_period=cfg.mean_reversion.bb_period,
                    bb_std=cfg.mean_reversion.bb_std,
                    rsi_period=cfg.mean_reversion.rsi_period,
                    rsi_long_threshold=cfg.mean_reversion.rsi_long_threshold,
                    rsi_short_threshold=cfg.mean_reversion.rsi_short_threshold,
                    adx_max=cfg.mean_reversion.adx_max,
                    base_confidence=cfg.mean_reversion.base_confidence,
                )
            )
        if cfg.volatility_breakout.enabled:
            strategies.append(
                VolatilityBreakoutStrategy(
                    timeframe=cfg.volatility_breakout.timeframe,
                    donchian_period=cfg.volatility_breakout.donchian_period,
                    squeeze_atr_ratio_max=cfg.volatility_breakout.squeeze_atr_ratio_max,
                    squeeze_bars=cfg.volatility_breakout.squeeze_bars,
                    volume_multiple=cfg.volatility_breakout.volume_multiple,
                    base_confidence=cfg.volatility_breakout.base_confidence,
                )
            )

        self._funding_arb = None
        if cfg.funding_arb.enabled:
            self._funding_arb = FundingArbStrategy(
                min_rate_delta=cfg.funding_arb.min_rate_delta,
                min_notional_usd=cfg.funding_arb.min_notional_usd,
                close_before_funding_sec=cfg.funding_arb.close_before_funding_sec,
                base_confidence=cfg.funding_arb.base_confidence,
            )
            strategies.append(self._funding_arb)

        if cfg.funding_contrarian.enabled and self._funding_history is not None:
            strategies.append(
                FundingContrarianStrategy(
                    funding_history=self._funding_history,
                    extreme_threshold=cfg.funding_contrarian.extreme_threshold,
                    low_threshold=cfg.funding_contrarian.low_threshold,
                    base_confidence=cfg.funding_contrarian.base_confidence,
                )
            )

        if cfg.vwap.enabled:
            strategies.append(
                VWAPStrategy(
                    timeframe=cfg.vwap.timeframe,
                    ema_period=cfg.vwap.ema_period,
                    vwap_band_pct=cfg.vwap.vwap_band_pct,
                    base_confidence=cfg.vwap.base_confidence,
                )
            )

        if cfg.liquidation_cascade.enabled and self._oi_store is not None:
            strategies.append(
                LiquidationCascadeStrategy(
                    oi_store=self._oi_store,
                    timeframe=cfg.liquidation_cascade.timeframe,
                    oi_roc_threshold=cfg.liquidation_cascade.oi_roc_threshold,
                    atr_period=cfg.liquidation_cascade.atr_period,
                    atr_multiplier=cfg.liquidation_cascade.atr_multiplier,
                    base_confidence=cfg.liquidation_cascade.base_confidence,
                )
            )

        self._regime_classifier = RegimeClassifier(config=self._settings.config.regime, base_timeframe="15m")
        self._ensemble = Ensemble(
            strategies=strategies,
            regime_classifier=self._regime_classifier,
            min_net_vote=self._settings.config.regime.min_net_vote,
        )
        log.info("orchestrator.strategies_loaded", names=[s.name for s in strategies])

    # ─────────────────────── run loop ───────────────────────

    async def run(self) -> None:
        """Main async entrypoint — blocks until shutdown."""
        assert self._storage is not None
        assert self._tracker is not None
        assert self._kill_switch is not None

        health_app = build_app(
            self._connectors, self._storage, self._kill_switch,
            self._ml_filter, self._broadcaster,
        )

        self._install_signal_handlers()

        try:
            async with asyncio.TaskGroup() as tg:
                # Per-connector pump tasks
                for name, conn in self._connectors.items():
                    tg.create_task(self._kline_pump(name, conn), name=f"kline.{name}")
                    tg.create_task(self._funding_pump(name, conn), name=f"funding.{name}")
                    tg.create_task(self._order_pump(name, conn), name=f"order.{name}")
                    tg.create_task(self._fill_pump(name, conn), name=f"fill.{name}")
                    tg.create_task(self._position_pump(name, conn), name=f"position.{name}")
                    if self._cvd_store is not None:
                        tg.create_task(self._trade_pump(name, conn), name=f"trades.{name}")

                # OI polling loop (one per connector, fires every poll_interval_sec)
                if self._oi_store is not None:
                    for name, conn in self._connectors.items():
                        tg.create_task(self._oi_poll_loop(name, conn), name=f"oi_poll.{name}")

                # Exit manager — trailing stops, breakeven, partial TP, time-exit
                if self._exit_manager is not None:
                    tg.create_task(self._exit_manager.run(), name="exit_manager")

                # Portfolio reconcile
                tg.create_task(self._tracker.run_reconcile_loop(), name="reconcile")

                # Universe refresh
                assert self._universe is not None
                tg.create_task(self._universe_loop(), name="universe")

                # Funding arb scanner
                if self._funding_arb is not None:
                    tg.create_task(self._funding_arb_loop(), name="funding-arb")

                # Health/metrics server
                tg.create_task(
                    serve_health(
                        health_app,
                        host="0.0.0.0",
                        port=self._settings.health_port,
                    ),
                    name="health",
                )

                # Shutdown waiter
                tg.create_task(self._shutdown_waiter(tg), name="shutdown-waiter")

        except* Exception as eg:
            for exc in eg.exceptions:
                log.error("orchestrator.task_error", error=str(exc), exc_info=exc)

        await self._teardown()

    async def _shutdown_waiter(self, tg: asyncio.TaskGroup) -> None:
        await self._shutdown_event.wait()
        log.info("orchestrator.shutdown_requested")
        # Cancel every other task in the group by raising inside it.
        raise asyncio.CancelledError("shutdown requested")

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._shutdown_event.set)
            except NotImplementedError:
                # Windows: signal handlers don't work in asyncio.
                pass

    async def _teardown(self) -> None:
        log.info("orchestrator.teardown_start")

        # Optional flatten on shutdown (off by default)
        if self._settings.config.risk.flatten_on_shutdown and self._executor is not None:
            assert self._tracker is not None
            for pos in self._tracker.open_positions():
                conn = self._connectors.get(pos.exchange)
                if conn is not None:
                    await self._executor.flatten_position(conn, pos.symbol)

        for name, conn in self._connectors.items():
            try:
                await conn.close()
            except Exception as e:
                log.warning("orchestrator.connector_close_failed", exchange=name, error=str(e))

        if self._telegram is not None:
            await self._telegram.stop()

        if self._storage is not None:
            await self._storage.close()

        log.info("orchestrator.teardown_complete")

    # ─────────────────────── event pumps ───────────────────────

    async def _kline_pump(self, exchange: str, conn: ExchangeConnector) -> None:
        assert self._ensemble is not None
        assert self._storage is not None
        while True:
            evt: KlineEvent = await conn.kline_events.get()
            bar = Bar(
                ts_ms=int(evt.ts.timestamp() * 1000),
                open=evt.open,
                high=evt.high,
                low=evt.low,
                close=evt.close,
                volume=evt.volume,
            )
            key = FeatureKey(exchange=exchange, symbol=evt.symbol, timeframe=evt.timeframe)
            self._feature_store.append_bar(key, bar)

            # Persist to DB only on closed bars (forming bars would cause
            # excessive writes — up to 20 symbols × 3 TFs per second).
            if evt.closed:
                try:
                    await self._storage.upsert_ohlcv(
                        exchange=exchange,
                        symbol=evt.symbol,
                        timeframe=evt.timeframe,
                        bars=[(evt.ts, evt.open, evt.high, evt.low, evt.close, evt.volume)],
                    )
                except Exception as e:
                    log.warning("orchestrator.ohlcv_persist_failed", error=str(e))
                # Update staleness gauge — Prometheus can derive age via time() - metric.
                from src.utils.time import now_utc as _now_utc

                m.last_bar_age_gauge.labels(
                    exchange=exchange, symbol=evt.symbol, timeframe=evt.timeframe
                ).set(_now_utc().timestamp())

            # Finalise CVD bar-delta on every closed bar (any timeframe, any exchange).
            if evt.closed and self._cvd_store is not None:
                self._cvd_store.on_bar_close(exchange, evt.symbol)

            # Only evaluate strategies on CLOSED bars of the base timeframe.
            if not evt.closed or evt.timeframe != "15m":
                continue

            # Skip stale snapshot bars delivered by the exchange at subscription
            # time — they carry a timestamp from before startup and would cause
            # trades based on an old signal at the current (different) price.
            if self._startup_ts is not None and evt.ts <= self._startup_ts:
                log.debug(
                    "orchestrator.skipping_stale_bar",
                    symbol=evt.symbol,
                    bar_ts=evt.ts,
                    startup=self._startup_ts,
                )
                continue

            # Only one exchange is the "signal source" per symbol to avoid
            # duplicate signals — we pick the first connector in alphabetical order.
            signal_source = sorted(self._connectors.keys())[0]
            if exchange != signal_source:
                continue

            if self._kill_switch and self._kill_switch.is_halted:
                continue

            await self._evaluate_and_execute(evt.symbol, evt.ts, exchange)

    async def _funding_pump(self, exchange: str, conn: ExchangeConnector) -> None:
        assert self._storage is not None
        while True:
            evt: FundingRateEvent = await conn.funding_events.get()
            # Strip the paper- prefix so funding_arb sees the underlying name.
            normalized = exchange.replace("paper-", "")
            if self._funding_arb is not None:
                self._funding_arb.update_rate(
                    exchange=normalized,
                    symbol=evt.symbol,
                    rate=evt.rate,
                    next_funding_ts=evt.next_funding_ts,
                    ts=evt.ts,
                )
            if self._funding_history is not None:
                self._funding_history.update(normalized, evt.symbol, evt.rate)
            if self._router is not None:
                self._router.update_funding(normalized, evt.symbol, evt.rate)
            try:
                await self._storage.record_funding_rate(
                    ts=evt.ts,
                    exchange=normalized,
                    symbol=evt.symbol,
                    rate=evt.rate,
                    next_funding_ts=evt.next_funding_ts,
                )
            except Exception:
                pass

    async def _order_pump(self, exchange: str, conn: ExchangeConnector) -> None:
        assert self._storage is not None
        while True:
            evt: OrderUpdateEvent = await conn.order_events.get()
            if evt.client_order_id:
                try:
                    await self._storage.update_order_status(
                        client_order_id=evt.client_order_id,
                        status=evt.status.value,
                        exchange_order_id=evt.exchange_order_id,
                    )
                except Exception as e:
                    log.warning("orchestrator.order_update_persist_failed", error=str(e))
            m.orders_placed_total.labels(
                exchange=exchange,
                symbol=evt.symbol,
                side=evt.side.value,
                status=evt.status.value,
            ).inc()

    async def _fill_pump(self, exchange: str, conn: ExchangeConnector) -> None:
        assert self._tracker is not None
        while True:
            evt: FillEvent = await conn.fill_events.get()
            await self._tracker.on_fill(evt)
            m.orders_filled_total.labels(
                exchange=exchange,
                symbol=evt.symbol,
                side=evt.side.value,
            ).inc()
            # Label the ML training example if this fill closes a position.
            # The closing fill side is OPPOSITE to the opening signal side
            # (e.g. a BUY entry is closed with a SELL fill).  We must invert
            # so the lookup in _pending matches the key stored at entry time.
            # Also skip fills with zero pnl (opening fills) via strict epsilon.
            if self._ml_filter is not None and evt.realized_pnl is not None and abs(evt.realized_pnl) > 1e-9:
                opening_side = OrderSide.BUY if evt.side == OrderSide.SELL else OrderSide.SELL
                await self._ml_filter.record_outcome(
                    evt.symbol,
                    opening_side,
                    evt.realized_pnl,
                    entry_notional=evt.quantity * evt.price,
                )
            if self._telegram is not None:
                await self._telegram.notify_trade_close(
                    evt.symbol,
                    evt.realized_pnl or 0.0,
                    reason="fill",
                )

    async def _position_pump(self, exchange: str, conn: ExchangeConnector) -> None:
        assert self._tracker is not None
        while True:
            evt: PositionUpdateEvent = await conn.position_events.get()
            await self._tracker.on_position_update(evt)

    async def _trade_pump(self, exchange: str, conn: ExchangeConnector) -> None:
        """Feed individual aggressor trades into CVDStore for buy/sell flow tracking."""
        assert self._cvd_store is not None
        # Strip paper prefix — CVD is keyed by underlying exchange name
        normalized = exchange.replace("paper-", "")
        while True:
            evt: TradeEvent = await conn.trade_events.get()
            self._cvd_store.on_trade(normalized, evt.symbol, evt.side, evt.qty)

    async def _oi_poll_loop(self, exchange: str, conn: ExchangeConnector) -> None:
        """Poll open interest via REST and update OIStore for ROC signals."""
        assert self._oi_store is not None
        assert self._universe is not None
        normalized = exchange.replace("paper-", "")
        interval = self._settings.config.strategies.oi.poll_interval_sec
        while True:
            await asyncio.sleep(interval)
            snap = self._universe.current
            if snap is None:
                continue
            for symbol in snap.symbols:
                try:
                    oi_evt: OIEvent = await conn.fetch_open_interest(symbol)
                    self._oi_store.update(normalized, symbol, oi_evt.oi_contracts)
                except Exception as e:
                    log.debug(
                        "orchestrator.oi_poll_failed",
                        exchange=exchange,
                        symbol=symbol,
                        error=str(e),
                    )

    # ─────────────────────── periodic loops ───────────────────────

    async def _universe_loop(self) -> None:
        assert self._universe is not None
        interval = self._settings.config.universe.refresh_interval_hours * 3600
        while True:
            await asyncio.sleep(interval)
            previous = self._universe.current
            try:
                new_snap = await self._universe.refresh()
                await self._subscribe_universe(new_snap, previous)
                await self._backfill_feature_store(new_snap)
            except Exception as e:
                log.error("orchestrator.universe_refresh_failed", error=str(e), exc_info=True)

    async def _funding_arb_loop(self) -> None:
        """Scan for cross-exchange funding opportunities every 60s."""
        assert self._funding_arb is not None
        assert self._ensemble is not None
        while True:
            await asyncio.sleep(60)
            if self._kill_switch and self._kill_switch.is_halted:
                continue
            try:
                opportunities = self._funding_arb.scan_arb_opportunities()
                for pair in opportunities:
                    await self._execute_pair_signal(pair)
            except Exception as e:
                log.error("orchestrator.funding_arb_scan_failed", error=str(e), exc_info=True)

    # ─────────────────────── signal → order ───────────────────────

    def _build_return_frames(self, symbol: str, exchange: str) -> dict | None:
        """Build close-price series map for portfolio correlation limit checking."""
        assert self._tracker is not None
        open_symbols = [p.symbol for p in self._tracker.open_positions()]
        frames: dict = {}
        for sym in [*open_symbols, symbol]:
            sym_df = self._feature_store.as_df(
                FeatureKey(exchange=exchange, symbol=sym, timeframe="1h"), min_bars=10
            )
            if sym_df is not None and "close" in sym_df.columns:
                frames[sym] = sym_df["close"]
        return frames if frames else None

    async def _evaluate_and_execute(self, symbol: str, ts: datetime, exchange: str) -> None:
        """Main decision pipeline on every closed base-TF bar."""
        assert self._tracker is not None

        # Dedup: skip if already sizing/placing for this symbol OR if an open
        # position exists.  The in-flight set prevents a race where two
        # concurrent 15m closes both pass the position check before either
        # places an order.
        if symbol in self._in_flight_symbols:
            return
        if any(p.symbol == symbol for p in self._tracker.open_positions()):
            return
        self._in_flight_symbols.add(symbol)

        try:
            await self._evaluate_and_execute_inner(symbol, ts, exchange)
        finally:
            self._in_flight_symbols.discard(symbol)

    async def _evaluate_and_execute_inner(self, symbol: str, ts: datetime, exchange: str) -> None:
        """Inner implementation — called only when not in-flight for this symbol."""
        assert self._ensemble is not None
        assert self._risk_manager is not None
        assert self._limits is not None
        assert self._tracker is not None
        assert self._executor is not None
        assert self._router is not None
        assert self._storage is not None

        signal, regime, raw = self._ensemble.evaluate_symbol(
            symbol=symbol,
            store=self._feature_store,
            exchange=exchange,
            ts=ts,
        )
        for s in raw:
            m.signals_emitted_total.labels(
                strategy=s.strategy,
                side=s.side.value,
                regime=regime.value,
            ).inc()
        # Push regime update to WebSocket clients
        try:
            self._broadcaster.push({"type": "regime", "regimes": self._regime_snapshot()})
        except Exception:
            pass
        if signal is None:
            return

        # ML signal filter — score the signal and store features for outcome labeling
        if self._ml_filter is not None:
            ml_dec = await self._ml_filter.evaluate(signal, self._feature_store, exchange)
            if not ml_dec.accepted:
                log.info(
                    "orchestrator.ml_rejected",
                    symbol=symbol,
                    score=round(ml_dec.ml_score, 3),
                    model_v=ml_dec.model_version,
                )
                if self._telegram is not None:
                    self._telegram.push_signal_event(
                        symbol=symbol, side=signal.side.value,
                        strategy=signal.strategy, confidence=signal.confidence,
                        regime=regime.value, outcome="ml_rej",
                    )
                self._broadcaster.push({
                    "type": "signal", "symbol": symbol, "side": signal.side.value,
                    "strategy": signal.strategy, "confidence": signal.confidence,
                    "regime": regime.value, "ml_score": ml_dec.ml_score,
                    "accepted": False, "cold_start": ml_dec.cold_start, "outcome": "ml_rej",
                    "ts": ts.isoformat(),
                })
                return
            if ml_dec.features:
                self._ml_filter.store_pending(symbol, signal.side, ml_dec.features)

        # Route
        try:
            leg = self._router.route(signal)
        except Exception as e:
            log.error("orchestrator.routing_failed", error=str(e))
            return
        target_exchange = leg.exchange_name

        # Get ATR for position sizing (use the signal-source exchange's feature store)
        df = self._feature_store.as_df(
            FeatureKey(exchange=exchange, symbol=symbol, timeframe="1h"),
            min_bars=30,
        )
        if df is None:
            log.warning("orchestrator.no_atr_data", symbol=symbol)
            return
        atr_val = atr(df, length=14).iloc[-1]
        import math as _math

        if atr_val is None or not _math.isfinite(float(atr_val)):
            log.warning("orchestrator.atr_nan", symbol=symbol)
            return

        # Get entry price (last close)
        last_bar = self._feature_store.latest(FeatureKey(exchange=exchange, symbol=symbol, timeframe="15m"))
        if last_bar is None:
            return
        entry_price = last_bar.close

        equity = self._tracker.total_equity()
        free_margin = self._tracker.free_margin(target_exchange)
        sized = self._risk_manager.size_trade(
            symbol=symbol,
            exchange=target_exchange,
            side=signal.side,
            entry_price=entry_price,
            atr=float(atr_val),
            equity=equity,
            free_margin=free_margin,
            strategy=signal.strategy,
            confidence=signal.confidence,
        )
        if isinstance(sized, RejectedTrade):
            m.orders_rejected_total.labels(reason=sized.reason).inc()
            log.info(
                "orchestrator.trade_rejected_by_risk",
                symbol=symbol,
                reason=sized.reason,
                details=sized.details,
            )
            if self._telegram is not None:
                self._telegram.push_signal_event(
                    symbol=symbol, side=signal.side.value,
                    strategy=signal.strategy, confidence=signal.confidence,
                    regime=regime.value, outcome="risk_rej",
                )
            self._broadcaster.push({
                "type": "signal", "symbol": symbol, "side": signal.side.value,
                "strategy": signal.strategy, "confidence": signal.confidence,
                "regime": regime.value, "accepted": False, "outcome": "risk_rej",
                "ts": ts.isoformat(),
            })
            return

        # Portfolio-level limits (correlation, max positions).
        decision = self._limits.check(sized, return_frames=self._build_return_frames(symbol, exchange))
        if not decision.allowed:
            m.orders_rejected_total.labels(reason=decision.reason or "limit").inc()
            log.info(
                "orchestrator.trade_rejected_by_limits",
                symbol=symbol,
                reason=decision.reason,
                details=decision.details,
            )
            if self._telegram is not None:
                self._telegram.push_signal_event(
                    symbol=symbol, side=signal.side.value,
                    strategy=signal.strategy, confidence=signal.confidence,
                    regime=regime.value, outcome="limit_rej",
                )
            self._broadcaster.push({
                "type": "signal", "symbol": symbol, "side": signal.side.value,
                "strategy": signal.strategy, "confidence": signal.confidence,
                "regime": regime.value, "accepted": False, "outcome": "limit_rej",
                "ts": ts.isoformat(),
            })
            return

        # Record the consolidated signal only after it passes all filters and limits —
        # prevents phantom records for rejected signals in the signals table.
        signal_id = await self._storage.record_signal(
            ts=ts,
            strategy=signal.strategy,
            exchange=exchange,
            symbol=symbol,
            timeframe=None,
            side=signal.side.value,
            confidence=signal.confidence,
            regime=regime.value,
            suggested_sl=signal.suggested_sl,
            suggested_tp=signal.suggested_tp,
            meta=signal.meta,
        )

        # Stash risk_usd so MLSignalFilter can compute R-multiple when the fill comes back.
        if self._ml_filter is not None:
            self._ml_filter.update_pending_risk(symbol, signal.side, sized.risk_usd)

        # Execute
        result = await self._executor.execute(sized, signal_id=signal_id)
        if result is not None and self._exit_manager is not None:
            # Register with the exit manager so it can apply trailing/breakeven/TP rules.
            risk_usd = sized.risk_usd
            self._exit_manager.register_position(
                exchange=exchange,
                symbol=symbol,
                entry_ts=ts,
                original_sl=sized.stop_loss,
                risk_usd=risk_usd,
                sl_order_id=result.exchange_order_id,  # may be updated by arm_exits
            )
        if result is not None and self._telegram is not None:
            self._telegram.push_signal_event(
                symbol=symbol, side=signal.side.value,
                strategy=signal.strategy, confidence=signal.confidence,
                regime=regime.value, outcome="exec",
            )
            self._broadcaster.push({
                "type": "signal", "symbol": symbol, "side": signal.side.value,
                "strategy": signal.strategy, "confidence": signal.confidence,
                "regime": regime.value, "accepted": True, "outcome": "exec",
                "ts": ts.isoformat(),
            })
            await self._telegram.notify_trade_open(
                strategy=signal.strategy,
                symbol=symbol,
                side=sized.side.value,
                qty=sized.quantity,
                entry=sized.entry_price,
                sl=sized.stop_loss,
                tp=sized.take_profit,
            )

    async def _execute_pair_signal(self, pair: "PairSignal") -> None:
        """Handle a funding-arb PairSignal: two simultaneous legs."""
        assert self._risk_manager is not None
        assert self._tracker is not None
        assert self._executor is not None
        assert self._storage is not None

        long_conn = self._connectors.get(pair.long_exchange)
        short_conn = self._connectors.get(pair.short_exchange)
        if long_conn is None or short_conn is None:
            return

        # Size each leg at half the normal per-trade risk so the combined pair
        # uses one "slot" of risk budget. ATR is looked up from 1h bars on the
        # long venue.
        df = self._feature_store.as_df(FeatureKey(pair.long_exchange, pair.symbol, "1h"), min_bars=30)
        if df is None:
            return
        atr_val = atr(df, length=14).iloc[-1]
        if atr_val is None or atr_val != atr_val:
            return
        latest = self._feature_store.latest(FeatureKey(pair.long_exchange, pair.symbol, "15m"))
        if latest is None:
            return
        entry_price = latest.close

        equity = self._tracker.total_equity()

        long_sized = self._risk_manager.size_trade(
            symbol=pair.symbol,
            exchange=pair.long_exchange,
            side=OrderSide.BUY,
            entry_price=entry_price,
            atr=float(atr_val),
            equity=equity * 0.5,
            free_margin=self._tracker.free_margin(pair.long_exchange),
            strategy=pair.strategy,
            confidence=pair.confidence,
        )
        short_sized = self._risk_manager.size_trade(
            symbol=pair.symbol,
            exchange=pair.short_exchange,
            side=OrderSide.SELL,
            entry_price=entry_price,
            atr=float(atr_val),
            equity=equity * 0.5,
            free_margin=self._tracker.free_margin(pair.short_exchange),
            strategy=pair.strategy,
            confidence=pair.confidence,
        )
        if isinstance(long_sized, RejectedTrade) or isinstance(short_sized, RejectedTrade):
            log.info("orchestrator.funding_arb_sized_rejected")
            return

        # Enforce configured min_notional for funding-arb legs
        arb_min = self._settings.config.strategies.funding_arb.min_notional_usd
        if long_sized.notional_usd < arb_min or short_sized.notional_usd < arb_min:
            log.info(
                "orchestrator.funding_arb_below_min_notional",
                symbol=pair.symbol,
                long_notional=round(long_sized.notional_usd, 2),
                short_notional=round(short_sized.notional_usd, 2),
                min=arb_min,
            )
            return

        # Portfolio-level limits for both legs — include correlation frames so the
        # arb path does not bypass the correlation cap.
        arb_frames = self._build_return_frames(pair.symbol, pair.long_exchange)
        long_limit = self._limits.check(long_sized, return_frames=arb_frames)
        if not long_limit.allowed:
            log.info("orchestrator.funding_arb_rejected_by_limits", leg="long", reason=long_limit.reason)
            return
        short_limit = self._limits.check(short_sized, return_frames=arb_frames)
        if not short_limit.allowed:
            log.info("orchestrator.funding_arb_rejected_by_limits", leg="short", reason=short_limit.reason)
            return

        log.info(
            "orchestrator.funding_arb_executing",
            symbol=pair.symbol,
            long=pair.long_exchange,
            short=pair.short_exchange,
            delta=pair.meta.get("delta"),
        )
        # Execute both legs; if the second fails, unwind the first.
        long_result = await self._executor.execute(long_sized)
        if long_result is None:
            return
        short_result = await self._executor.execute(short_sized)
        if short_result is None:
            log.warning("orchestrator.funding_arb_second_leg_failed_unwinding")
            await self._executor.flatten_position(long_conn, pair.symbol)

    # ─────────────────────── halt & warning callbacks ───────────────────────

    async def _on_halt_callback(self, reason: str, drawdown: float) -> None:
        log.error("orchestrator.HALT_RECEIVED", reason=reason, drawdown=drawdown)
        m.halt_state_gauge.set(2)
        if self._telegram is not None:
            await self._telegram.notify_halt(reason, drawdown)
        # Flatten all open positions
        if self._tracker is not None and self._executor is not None:
            for pos in self._tracker.open_positions():
                conn = self._connectors.get(pos.exchange)
                if conn is None:
                    continue
                await self._executor.cancel_all_for_symbol(conn, pos.symbol)
                await self._executor.flatten_position(conn, pos.symbol)

    async def _on_warning_callback(self, drawdown: float) -> None:
        m.halt_state_gauge.set(1)
        if self._telegram is not None:
            await self._telegram.notify_warning(drawdown)
