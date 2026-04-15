"""
Bybit v5 exchange connector.

REST: ccxt.async_support
WS :  native websockets client with:
    - public feed : wss://stream.bybit.com/v5/public/linear
    - private feed: wss://stream.bybit.com/v5/private   (HMAC-SHA256 auth frame)

Testnet endpoints:
    - wss://stream-testnet.bybit.com/v5/public/linear
    - wss://stream-testnet.bybit.com/v5/private

Bybit v5 symbol format: "BTCUSDT" (no slash). ccxt unified symbol: "BTC/USDT:USDT".
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import uuid
from datetime import datetime
from typing import Any

import ccxt.async_support as ccxt_async

from src.exchanges.base import (
    Balance,
    ExchangeConnector,
    FillEvent,
    FundingRateEvent,
    KlineEvent,
    OIEvent,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderUpdateEvent,
    PositionSide,
    PositionUpdateEvent,
    TickerInfo,
    TradeEvent,
)
from src.exchanges.ws_manager import Subscription, WSManager, WSManagerConfig
from src.settings import RuntimeMode
from src.utils.logging import get_logger
from src.utils.time import ms_to_datetime, now_ms, now_utc

log = get_logger(__name__)

BYBIT_WS_PUBLIC_LIVE = "wss://stream.bybit.com/v5/public/linear"
BYBIT_WS_PRIVATE_LIVE = "wss://stream.bybit.com/v5/private"
BYBIT_WS_PUBLIC_TESTNET = "wss://stream-testnet.bybit.com/v5/public/linear"
BYBIT_WS_PRIVATE_TESTNET = "wss://stream-testnet.bybit.com/v5/private"

# Bybit uses lowercase for sub-hour timeframes and uppercase for >= 1 day.
BYBIT_TF_MAP: dict[str, str] = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
}


def _bybit_order_status(s: str) -> OrderStatus:
    # Bybit order statuses (v5): New, PartiallyFilled, Untriggered, Rejected,
    # PartiallyFilledCanceled, Filled, Cancelled, Triggered, Deactivated
    s = (s or "").lower()
    return {
        "new": OrderStatus.OPEN,
        "partiallyfilled": OrderStatus.PARTIALLY_FILLED,
        "filled": OrderStatus.FILLED,
        "cancelled": OrderStatus.CANCELLED,
        "canceled": OrderStatus.CANCELLED,
        "rejected": OrderStatus.REJECTED,
        "partiallyfilledcanceled": OrderStatus.CANCELLED,
        "deactivated": OrderStatus.CANCELLED,
    }.get(s, OrderStatus.NEW)


class BybitConnector(ExchangeConnector):
    name = "bybit"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        mode: RuntimeMode,
        rest_rate_limit_per_sec: float = 10.0,
        ws_ping_interval_sec: float = 20.0,
        ws_reconnect_max_backoff_sec: float = 30.0,
    ) -> None:
        super().__init__()
        self._mode = mode
        self._api_key = api_key
        self._api_secret = api_secret

        # ccxt REST client
        ccxt_opts: dict[str, Any] = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "rateLimit": max(1, int(1000 / rest_rate_limit_per_sec)),
            "options": {
                "defaultType": "swap",
                "defaultSubType": "linear",
            },
        }
        self._rest: ccxt_async.bybit = ccxt_async.bybit(ccxt_opts)
        if mode == RuntimeMode.TESTNET:
            self._rest.set_sandbox_mode(True)

        is_testnet = mode == RuntimeMode.TESTNET
        self._ws_public = WSManager(
            WSManagerConfig(
                url=BYBIT_WS_PUBLIC_TESTNET if is_testnet else BYBIT_WS_PUBLIC_LIVE,
                name="bybit.public",
                ping_interval_sec=ws_ping_interval_sec,
                max_backoff_sec=ws_reconnect_max_backoff_sec,
            ),
            handle_message=self._handle_public_message,
        )
        self._ws_private = WSManager(
            WSManagerConfig(
                url=BYBIT_WS_PRIVATE_TESTNET if is_testnet else BYBIT_WS_PRIVATE_LIVE,
                name="bybit.private",
                ping_interval_sec=ws_ping_interval_sec,
                max_backoff_sec=ws_reconnect_max_backoff_sec,
                on_connected=self._login_private,
            ),
            handle_message=self._handle_private_message,
        )
        self._markets_loaded = False

    # ─────────────────────── lifecycle ───────────────────────

    async def connect(self) -> None:
        if not self._markets_loaded:
            await self._rest.load_markets()
            self._markets_loaded = True
            log.info("bybit.markets_loaded", count=len(self._rest.markets))

        await self._ws_public.start()
        await self._ws_private.start()
        await self._ws_public.wait_connected(timeout=15)
        await self._ws_private.wait_connected(timeout=15)

        await self._subscribe_private_channels()

    async def close(self) -> None:
        await self._ws_public.stop()
        await self._ws_private.stop()
        try:
            await self._rest.close()
        except Exception:
            # ccxt close() can raise if the aiohttp session is already gone during
            # interpreter shutdown. Safe to ignore during connector teardown.
            pass

    # ─────────────────────── private WS login ───────────────────────

    async def _login_private(self, mgr: WSManager) -> None:
        """Bybit v5 private WS auth: sign expires+string with API secret."""
        expires = now_ms() + 10_000  # valid for 10s
        to_sign = f"GET/realtime{expires}"
        sign = hmac.new(self._api_secret.encode(), to_sign.encode(), hashlib.sha256).hexdigest()
        auth_msg = {
            "op": "auth",
            "args": [self._api_key, expires, sign],
        }
        await mgr.send(auth_msg)

    async def _subscribe_private_channels(self) -> None:
        topics = ["order", "position", "execution"]
        sub = Subscription(
            key="private:order+position+execution",
            payload={"op": "subscribe", "args": topics},
        )
        await self._ws_private.subscribe(sub)

    # ─────────────────────── public WS subscriptions ───────────────────────

    async def subscribe_klines(self, symbol: str, timeframe: str) -> None:
        bybit_sym = self._symbol_to_bybit(symbol)
        bybit_tf = BYBIT_TF_MAP.get(timeframe)
        if bybit_tf is None:
            raise ValueError(f"unsupported timeframe for bybit: {timeframe}")
        topic = f"kline.{bybit_tf}.{bybit_sym}"
        sub = Subscription(
            key=f"kline:{symbol}:{timeframe}",
            payload={"op": "subscribe", "args": [topic]},
        )
        await self._ws_public.subscribe(sub)

    async def unsubscribe_klines(self, symbol: str, timeframe: str) -> None:
        bybit_sym = self._symbol_to_bybit(symbol)
        bybit_tf = BYBIT_TF_MAP.get(timeframe, timeframe)
        topic = f"kline.{bybit_tf}.{bybit_sym}"
        await self._ws_public.unsubscribe(
            key=f"kline:{symbol}:{timeframe}",
            unsub_payload={"op": "unsubscribe", "args": [topic]},
        )

    async def subscribe_funding(self, symbol: str) -> None:
        bybit_sym = self._symbol_to_bybit(symbol)
        # Bybit publishes funding as part of the tickers stream.
        topic = f"tickers.{bybit_sym}"
        sub = Subscription(
            key=f"tickers:{symbol}",
            payload={"op": "subscribe", "args": [topic]},
        )
        await self._ws_public.subscribe(sub)

    async def unsubscribe_funding(self, symbol: str) -> None:
        bybit_sym = self._symbol_to_bybit(symbol)
        topic = f"tickers.{bybit_sym}"
        await self._ws_public.unsubscribe(
            key=f"tickers:{symbol}",
            unsub_payload={"op": "unsubscribe", "args": [topic]},
        )

    async def subscribe_trades(self, symbol: str) -> None:
        bybit_sym = self._symbol_to_bybit(symbol)
        topic = f"publicTrade.{bybit_sym}"
        sub = Subscription(
            key=f"trades:{symbol}",
            payload={"op": "subscribe", "args": [topic]},
        )
        await self._ws_public.subscribe(sub)

    async def unsubscribe_trades(self, symbol: str) -> None:
        bybit_sym = self._symbol_to_bybit(symbol)
        topic = f"publicTrade.{bybit_sym}"
        await self._ws_public.unsubscribe(
            key=f"trades:{symbol}",
            unsub_payload={"op": "unsubscribe", "args": [topic]},
        )

    # ─────────────────────── WS message handlers ───────────────────────

    async def _handle_public_message(self, msg: dict[str, Any]) -> None:
        if msg.get("op") == "pong" or msg.get("ret_msg") == "pong":
            self._ws_public.mark_pong()
            return
        if msg.get("op") == "subscribe":
            if not msg.get("success", True):
                log.warning("bybit.public_subscribe_failed", ret_msg=msg.get("ret_msg"), op=msg.get("op"))
            return

        topic: str = msg.get("topic", "")
        if topic.startswith("kline."):
            await self._emit_klines(topic, msg.get("data") or [])
        elif topic.startswith("tickers."):
            await self._emit_funding_from_ticker(topic, msg.get("data") or {})
        elif topic.startswith("publicTrade."):
            await self._emit_trades(topic, msg.get("data") or [])

    async def _emit_klines(self, topic: str, data: list[dict[str, Any]]) -> None:
        # topic format: "kline.<interval>.<symbol>"
        try:
            _, interval, bybit_sym = topic.split(".", 2)
        except ValueError:
            return
        unified_tf = next((k for k, v in BYBIT_TF_MAP.items() if v == interval), interval)
        symbol = self._bybit_to_symbol(bybit_sym)

        for row in data:
            try:
                evt = KlineEvent(
                    exchange=self.name,
                    symbol=symbol,
                    timeframe=unified_tf,
                    ts=ms_to_datetime(int(row["start"])),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    closed=bool(row.get("confirm", False)),
                )
                self._kline_q.put_nowait(evt)
            except (KeyError, ValueError, asyncio.QueueFull):
                log.warning("bybit.kline_parse_error", row=row)

    async def _emit_trades(self, topic: str, data: list[dict[str, Any]]) -> None:
        # topic format: "publicTrade.BTCUSDT"
        bybit_sym = topic.split(".", 1)[1] if "." in topic else ""
        symbol = self._bybit_to_symbol(bybit_sym)
        for row in data:
            try:
                side = OrderSide.BUY if row.get("S") == "Buy" else OrderSide.SELL
                evt = TradeEvent(
                    exchange=self.name,
                    symbol=symbol,
                    ts=ms_to_datetime(int(row["T"])),
                    side=side,
                    qty=float(row["v"]),
                    price=float(row["p"]),
                )
                try:
                    self._trade_q.put_nowait(evt)
                except asyncio.QueueFull:
                    pass  # trades are high-frequency; drop on overflow, CVD will be approximate
            except (KeyError, ValueError) as e:
                from src.monitoring import prom_metrics as m
                m.ws_parse_errors_total.labels(
                    exchange=self.name, channel="trades", reason="parse_error"
                ).inc()
                log.debug("bybit.trade_parse_error", row=row, error=str(e))

    async def _emit_funding_from_ticker(
        self, topic: str, data: dict[str, Any] | list[dict[str, Any]]
    ) -> None:
        # Bybit tickers stream: the first message is a full snapshot, subsequent
        # messages are deltas. Both may carry fundingRate + nextFundingTime.
        if isinstance(data, list):
            rows = data
        else:
            rows = [data]
        bybit_sym = topic.split(".", 1)[1] if "." in topic else ""
        symbol = self._bybit_to_symbol(bybit_sym)
        for row in rows:
            rate_str = row.get("fundingRate")
            if rate_str is None or rate_str == "":
                continue
            try:
                rate = float(rate_str)
                next_ts_ms = row.get("nextFundingTime")
                next_ts = ms_to_datetime(int(next_ts_ms)) if next_ts_ms else None
                evt = FundingRateEvent(
                    exchange=self.name,
                    symbol=symbol,
                    ts=now_utc(),
                    rate=rate,
                    next_funding_ts=next_ts,
                )
                self._funding_q.put_nowait(evt)
            except asyncio.QueueFull:
                from src.monitoring import prom_metrics as m
                m.ws_parse_errors_total.labels(
                    exchange=self.name, channel="tickers", reason="queue_full"
                ).inc()
                # Funding WS is supplementary to the REST poll; safe to drop one event.
            except ValueError as e:
                from src.monitoring import prom_metrics as m
                m.ws_parse_errors_total.labels(
                    exchange=self.name, channel="tickers", reason="parse_error"
                ).inc()
                log.debug("bybit.funding_parse_error", row=row, error=str(e))

    async def _handle_private_message(self, msg: dict[str, Any]) -> None:
        op = msg.get("op")
        if op == "auth":
            if msg.get("success"):
                log.info("bybit.private_auth_ok")
            else:
                log.error(
                    "bybit.private_auth_failed", ret_msg=msg.get("ret_msg"), ret_code=msg.get("ret_code")
                )
            return
        if op == "pong" or msg.get("ret_msg") == "pong":
            self._ws_private.mark_pong()
            return
        if op == "subscribe":
            if not msg.get("success", True):
                log.warning("bybit.private_subscribe_failed", ret_msg=msg.get("ret_msg"), op=msg.get("op"))
            return

        topic: str = msg.get("topic", "")
        data = msg.get("data") or []
        if topic == "order" and data:
            await self._emit_orders(data)
        elif topic == "execution" and data:
            await self._emit_executions(data)
        elif topic == "position" and data:
            await self._emit_positions(data)

    async def _emit_orders(self, data: list[dict[str, Any]]) -> None:
        for row in data:
            try:
                bybit_sym = row.get("symbol", "")
                symbol = self._bybit_to_symbol(bybit_sym)
                side = OrderSide.BUY if row.get("side") == "Buy" else OrderSide.SELL
                update = OrderUpdateEvent(
                    exchange=self.name,
                    symbol=symbol,
                    client_order_id=row.get("orderLinkId") or None,
                    exchange_order_id=row.get("orderId", ""),
                    status=_bybit_order_status(row.get("orderStatus", "")),
                    side=side,
                    quantity=float(row.get("qty", 0) or 0),
                    filled_quantity=float(row.get("cumExecQty", 0) or 0),
                    average_price=float(row.get("avgPrice", 0) or 0) or None,
                    ts=ms_to_datetime(int(row.get("updatedTime", now_ms()))),
                    raw=row,
                )
                self._order_q.put_nowait(update)
            except asyncio.QueueFull:
                log.error(
                    "bybit.order_queue_full_dropping_event",
                    symbol=row.get("symbol"),
                    ord_id=row.get("orderId"),
                    status=row.get("orderStatus"),
                )
                raise
            except ValueError:
                log.warning(
                    "bybit.order_parse_error",
                    symbol=row.get("symbol"),
                    ord_id=row.get("orderId"),
                    status=row.get("orderStatus"),
                )

    async def _emit_executions(self, data: list[dict[str, Any]]) -> None:
        for row in data:
            try:
                bybit_sym = row.get("symbol", "")
                symbol = self._bybit_to_symbol(bybit_sym)
                side = OrderSide.BUY if row.get("side") == "Buy" else OrderSide.SELL
                fill = FillEvent(
                    exchange=self.name,
                    symbol=symbol,
                    client_order_id=row.get("orderLinkId") or None,
                    exchange_order_id=row.get("orderId", ""),
                    side=side,
                    quantity=float(row.get("execQty", 0) or 0),
                    price=float(row.get("execPrice", 0) or 0),
                    fee=abs(float(row.get("execFee", 0) or 0)),
                    fee_currency=row.get("feeCurrency"),
                    realized_pnl=float(row.get("closedPnl", 0) or 0) or None,
                    ts=ms_to_datetime(int(row.get("execTime", now_ms()))),
                    raw=row,
                )
                self._fill_q.put_nowait(fill)
            except asyncio.QueueFull:
                log.error(
                    "bybit.fill_queue_full_dropping_fill",
                    symbol=row.get("symbol"),
                    ord_id=row.get("orderId"),
                )
                raise
            except ValueError:
                log.warning(
                    "bybit.exec_parse_error",
                    symbol=row.get("symbol"),
                    ord_id=row.get("orderId"),
                )

    async def _emit_positions(self, data: list[dict[str, Any]]) -> None:
        for row in data:
            try:
                bybit_sym = row.get("symbol", "")
                symbol = self._bybit_to_symbol(bybit_sym)
                size = float(row.get("size", 0) or 0)
                side_str = (row.get("side") or "").lower()
                if size == 0 or side_str == "none" or side_str == "":
                    side = PositionSide.FLAT
                elif side_str == "buy":
                    side = PositionSide.LONG
                else:
                    side = PositionSide.SHORT
                evt = PositionUpdateEvent(
                    exchange=self.name,
                    symbol=symbol,
                    side=side,
                    quantity=abs(size),
                    entry_price=float(row.get("entryPrice", 0) or 0) or None,
                    mark_price=float(row.get("markPrice", 0) or 0) or None,
                    liquidation_price=float(row.get("liqPrice", 0) or 0) or None,
                    unrealized_pnl=float(row.get("unrealisedPnl", 0) or 0) or None,
                    leverage=float(row.get("leverage", 0) or 0) or None,
                    ts=now_utc(),
                )
                self._position_q.put_nowait(evt)
            except asyncio.QueueFull:
                log.error(
                    "bybit.position_queue_full_dropping_event",
                    symbol=row.get("symbol"),
                )
                from src.monitoring import prom_metrics as m
                m.ws_parse_errors_total.labels(
                    exchange=self.name, channel="position", reason="queue_full"
                ).inc()
            except ValueError as e:
                log.warning(
                    "bybit.position_parse_error",
                    symbol=row.get("symbol"),
                    error=str(e),
                )

    # ─────────────────────── REST: market data ───────────────────────

    async def fetch_ohlcv_backfill(self, symbol: str, timeframe: str, limit: int = 500) -> list[KlineEvent]:
        bars = await self._rest.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return [
            KlineEvent(
                exchange=self.name,
                symbol=symbol,
                timeframe=timeframe,
                ts=ms_to_datetime(b[0]),
                open=float(b[1]),
                high=float(b[2]),
                low=float(b[3]),
                close=float(b[4]),
                volume=float(b[5]),
                closed=True,
            )
            for b in bars
        ]

    async def fetch_24h_tickers(self, quote: str = "USDT", instrument_type: str = "swap") -> list[TickerInfo]:
        tickers = await self._rest.fetch_tickers(params={"category": "linear"})
        out: list[TickerInfo] = []
        for sym, t in tickers.items():
            if not sym.endswith(f"/{quote}:{quote}"):
                continue
            last = float(t.get("last") or 0)
            vol_quote = float(t.get("quoteVolume") or 0)
            out.append(
                TickerInfo(
                    symbol=sym,
                    volume_usd_24h=vol_quote,
                    last_price=last,
                    raw=t,
                )
            )
        return out

    async def fetch_funding_rate(self, symbol: str) -> FundingRateEvent:
        data = await self._rest.fetch_funding_rate(symbol)
        rate = float(data.get("fundingRate") or 0.0)
        next_ts_ms = data.get("fundingTimestamp") or data.get("nextFundingTimestamp")
        next_ts = ms_to_datetime(int(next_ts_ms)) if next_ts_ms else None
        return FundingRateEvent(
            exchange=self.name,
            symbol=symbol,
            ts=now_utc(),
            rate=rate,
            next_funding_ts=next_ts,
        )

    async def fetch_open_interest(self, symbol: str) -> OIEvent:
        data = await self._rest.fetch_open_interest(symbol, params={"category": "linear"})
        oi = float(data.get("openInterestAmount") or data.get("info", {}).get("openInterest") or 0.0)
        return OIEvent(
            exchange=self.name,
            symbol=symbol,
            ts=now_utc(),
            oi_contracts=oi,
        )

    # ─────────────────────── REST: account ───────────────────────

    async def fetch_balance(self) -> Balance:
        bal = await self._rest.fetch_balance(params={"accountType": "UNIFIED"})
        usdt = bal.get("USDT") or {}
        return Balance(
            total=float(usdt.get("total", 0) or 0),
            free=float(usdt.get("free", 0) or 0),
            used=float(usdt.get("used", 0) or 0),
            currency="USDT",
        )

    async def fetch_positions(self) -> list[PositionUpdateEvent]:
        positions = await self._rest.fetch_positions(params={"category": "linear"})
        out: list[PositionUpdateEvent] = []
        for p in positions:
            contracts = float(p.get("contracts") or 0)
            if contracts == 0:
                continue
            side_str = (p.get("side") or "long").lower()
            side = PositionSide.LONG if side_str == "long" else PositionSide.SHORT
            out.append(
                PositionUpdateEvent(
                    exchange=self.name,
                    symbol=p.get("symbol", ""),
                    side=side,
                    quantity=contracts,
                    entry_price=float(p.get("entryPrice") or 0) or None,
                    mark_price=float(p.get("markPrice") or 0) or None,
                    liquidation_price=float(p.get("liquidationPrice") or 0) or None,
                    unrealized_pnl=float(p.get("unrealizedPnl") or 0) or None,
                    leverage=float(p.get("leverage") or 0) or None,
                    ts=now_utc(),
                )
            )
        return out

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            await self._rest.set_leverage(leverage, symbol, params={"category": "linear"})
        except Exception as e:
            log.debug("bybit.set_leverage_noop", symbol=symbol, lev=leverage, error=str(e))

    # ─────────────────────── trading ───────────────────────

    async def place_order(self, req: OrderRequest) -> OrderResult:
        client_order_id = req.client_order_id or self._make_client_id()
        params: dict[str, Any] = {
            "category": "linear",
            "orderLinkId": client_order_id.replace("-", "")[:36],
        }
        if req.reduce_only:
            params["reduceOnly"] = True
        if req.leverage is not None:
            await self.set_leverage(req.symbol, req.leverage)

        order_type_str = "market" if req.order_type == OrderType.MARKET else "limit"
        side_str = "buy" if req.side == OrderSide.BUY else "sell"

        response = await self._rest.create_order(
            symbol=req.symbol,
            type=order_type_str,
            side=side_str,
            amount=req.quantity,
            price=req.price,
            params=params,
        )

        return OrderResult(
            client_order_id=client_order_id,
            exchange_order_id=str(response.get("id", "")),
            status=_bybit_order_status(str(response.get("status", "new"))),
            filled_quantity=float(response.get("filled") or 0),
            average_price=float(response.get("average") or 0) or None,
            raw=response,
        )

    async def cancel_order(
        self, symbol: str, *, client_order_id: str | None = None, exchange_order_id: str | None = None
    ) -> None:
        params: dict[str, Any] = {"category": "linear"}
        if client_order_id:
            params["orderLinkId"] = client_order_id.replace("-", "")[:36]
        try:
            await self._rest.cancel_order(id=exchange_order_id, symbol=symbol, params=params)
        except Exception as e:
            log.warning("bybit.cancel_order_failed", symbol=symbol, error=str(e))

    async def cancel_all(self, symbol: str | None = None) -> None:
        try:
            await self._rest.cancel_all_orders(symbol=symbol, params={"category": "linear"})
        except Exception as e:
            log.warning("bybit.cancel_all_failed", symbol=symbol, error=str(e))

    async def close_position(self, symbol: str) -> OrderResult:
        positions = await self.fetch_positions()
        pos = next((p for p in positions if p.symbol == symbol), None)
        if pos is None or pos.quantity == 0:
            return OrderResult(
                client_order_id=self._make_client_id(),
                exchange_order_id=None,
                status=OrderStatus.REJECTED,
                filled_quantity=0.0,
                average_price=None,
                raw={"reason": "no open position"},
            )
        side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
        req = OrderRequest(
            exchange=self.name,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=pos.quantity,
            reduce_only=True,
        )
        return await self.place_order(req)

    # ─────────────────────── helpers ───────────────────────

    def _symbol_to_bybit(self, symbol: str) -> str:
        """'BTC/USDT:USDT' → 'BTCUSDT'."""
        mkt = self._rest.markets.get(symbol)
        if mkt and "id" in mkt:
            return str(mkt["id"])
        base, rest = symbol.split("/", 1)
        quote = rest.split(":", 1)[0]
        return f"{base}{quote}"

    def _bybit_to_symbol(self, bybit_sym: str) -> str:
        """'BTCUSDT' → 'BTC/USDT:USDT'."""
        for sym, mkt in self._rest.markets.items():
            if mkt.get("id") == bybit_sym:
                return sym
        # Fallback — assume USDT quote
        if bybit_sym.endswith("USDT"):
            base = bybit_sym[:-4]
            return f"{base}/USDT:USDT"
        return bybit_sym

    @staticmethod
    def _make_client_id() -> str:
        return f"cpb{uuid.uuid4().hex[:30]}"
