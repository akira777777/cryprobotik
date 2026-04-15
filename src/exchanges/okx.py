"""
OKX v5 exchange connector.

REST: ccxt.async_support — battle-tested, handles signing + rate limits + retries.
WS : native websockets client with:
    - public feed (wss://ws.okx.com:8443/ws/v5/public)
    - private feed (wss://ws.okx.com:8443/ws/v5/private)  with HMAC-SHA256 login

Demo (testnet) endpoints are selected when config.mode == RuntimeMode.TESTNET.

OKX unified-symbol mapping (ccxt):
    ccxt         : "BTC/USDT:USDT"
    OKX instId   : "BTC-USDT-SWAP"

We rely on ccxt's markets() for the translation.
"""

from __future__ import annotations

import asyncio
import base64
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

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
OKX_WS_PUBLIC_LIVE = "wss://ws.okx.com:8443/ws/v5/public"
OKX_WS_PRIVATE_LIVE = "wss://ws.okx.com:8443/ws/v5/private"
OKX_WS_BUSINESS_LIVE = "wss://ws.okx.com:8443/ws/v5/business"
OKX_WS_PUBLIC_DEMO = "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999999"
OKX_WS_PRIVATE_DEMO = "wss://wspap.okx.com:8443/ws/v5/private?brokerId=9999999"
OKX_WS_BUSINESS_DEMO = "wss://wspap.okx.com:8443/ws/v5/business?brokerId=9999999"

# OKX timeframe mapping — v5 uses bars in lowercase for <= 1h and uppercase for >= 1D.
OKX_TF_MAP: dict[str, str] = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "12h": "12H",
    "1d": "1D",
}


def _side_to_okx(side: OrderSide) -> str:
    return "buy" if side == OrderSide.BUY else "sell"


def _okx_order_status(status: str) -> OrderStatus:
    return {
        "live": OrderStatus.OPEN,
        "partially_filled": OrderStatus.PARTIALLY_FILLED,
        "filled": OrderStatus.FILLED,
        "canceled": OrderStatus.CANCELLED,
        "cancelled": OrderStatus.CANCELLED,
        "rejected": OrderStatus.REJECTED,
    }.get(status, OrderStatus.NEW)


class OKXConnector(ExchangeConnector):
    name = "okx"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        mode: RuntimeMode,
        rest_rate_limit_per_sec: float = 8.0,
        ws_ping_interval_sec: float = 20.0,
        ws_reconnect_max_backoff_sec: float = 30.0,
    ) -> None:
        super().__init__()
        self._mode = mode
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_passphrase = api_passphrase
        self._ws_ping_interval = ws_ping_interval_sec
        self._ws_max_backoff = ws_reconnect_max_backoff_sec

        # ccxt REST client
        ccxt_opts: dict[str, Any] = {
            "apiKey": api_key,
            "secret": api_secret,
            "password": api_passphrase,
            "enableRateLimit": True,
            "rateLimit": max(1, int(1000 / rest_rate_limit_per_sec)),
            "options": {
                "defaultType": "swap",
            },
        }
        self._rest: ccxt_async.okx = ccxt_async.okx(ccxt_opts)
        if mode == RuntimeMode.TESTNET:
            self._rest.set_sandbox_mode(True)

        # WS managers
        is_demo = mode == RuntimeMode.TESTNET
        self._ws_public = WSManager(
            WSManagerConfig(
                url=OKX_WS_PUBLIC_DEMO if is_demo else OKX_WS_PUBLIC_LIVE,
                name="okx.public",
                ping_interval_sec=ws_ping_interval_sec,
                max_backoff_sec=ws_reconnect_max_backoff_sec,
            ),
            handle_message=self._handle_public_message,
        )
        # OKX v5: candle channels must go to /ws/v5/business, not /ws/v5/public
        self._ws_business = WSManager(
            WSManagerConfig(
                url=OKX_WS_BUSINESS_DEMO if is_demo else OKX_WS_BUSINESS_LIVE,
                name="okx.business",
                ping_interval_sec=ws_ping_interval_sec,
                max_backoff_sec=ws_reconnect_max_backoff_sec,
            ),
            handle_message=self._handle_public_message,
        )
        self._ws_private = WSManager(
            WSManagerConfig(
                url=OKX_WS_PRIVATE_DEMO if is_demo else OKX_WS_PRIVATE_LIVE,
                name="okx.private",
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
            await self._rest.load_markets({"type": "swap"})
            self._markets_loaded = True
            log.info("okx.markets_loaded", count=len(self._rest.markets))

        await self._ws_public.start()
        await self._ws_business.start()
        await self._ws_private.start()

        # Wait briefly for both sockets so the caller knows state before subscribing.
        await self._ws_public.wait_connected(timeout=15)
        await self._ws_business.wait_connected(timeout=15)
        await self._ws_private.wait_connected(timeout=15)

        # Subscribe to private channels (orders + positions + account)
        await self._subscribe_private_channels()

    async def close(self) -> None:
        await self._ws_public.stop()
        await self._ws_business.stop()
        await self._ws_private.stop()
        try:
            await self._rest.close()
        except Exception:
            pass

    # ─────────────────────── private-channel login ───────────────────────

    async def _login_private(self, mgr: WSManager) -> None:
        """Sign and send the login message on the private WS connection."""
        ts = str(now_ms() / 1000)
        message = f"{ts}GET/users/self/verify"
        sign = base64.b64encode(
            hmac.new(self._api_secret.encode(), message.encode(), hashlib.sha256).digest()
        ).decode()
        login_msg = {
            "op": "login",
            "args": [
                {
                    "apiKey": self._api_key,
                    "passphrase": self._api_passphrase,
                    "timestamp": ts,
                    "sign": sign,
                }
            ],
        }
        await mgr.send(login_msg)
        # Wait for the login response — handled by _handle_private_message
        # which will log success/failure. No synchronous ack needed here.

    async def _subscribe_private_channels(self) -> None:
        """Subscribe to orders, positions, account on the private WS."""
        for channel in ("orders", "positions", "account"):
            args = (
                [{"channel": channel, "instType": "SWAP"}] if channel != "account" else [{"channel": channel}]
            )
            sub = Subscription(
                key=f"private:{channel}",
                payload={"op": "subscribe", "args": args},
            )
            await self._ws_private.subscribe(sub)

    # ─────────────────────── public WS subscriptions ───────────────────────

    async def subscribe_klines(self, symbol: str, timeframe: str) -> None:
        inst_id = self._symbol_to_inst_id(symbol)
        okx_tf = OKX_TF_MAP.get(timeframe)
        if okx_tf is None:
            raise ValueError(f"unsupported timeframe for okx: {timeframe}")
        channel = f"candle{okx_tf}"
        sub = Subscription(
            key=f"kline:{symbol}:{timeframe}",
            payload={
                "op": "subscribe",
                "args": [{"channel": channel, "instId": inst_id}],
            },
        )
        await self._ws_business.subscribe(sub)

    async def unsubscribe_klines(self, symbol: str, timeframe: str) -> None:
        inst_id = self._symbol_to_inst_id(symbol)
        okx_tf = OKX_TF_MAP.get(timeframe, timeframe)
        await self._ws_business.unsubscribe(
            key=f"kline:{symbol}:{timeframe}",
            unsub_payload={
                "op": "unsubscribe",
                "args": [{"channel": f"candle{okx_tf}", "instId": inst_id}],
            },
        )

    async def subscribe_funding(self, symbol: str) -> None:
        inst_id = self._symbol_to_inst_id(symbol)
        sub = Subscription(
            key=f"funding:{symbol}",
            payload={
                "op": "subscribe",
                "args": [{"channel": "funding-rate", "instId": inst_id}],
            },
        )
        await self._ws_public.subscribe(sub)

    async def unsubscribe_funding(self, symbol: str) -> None:
        inst_id = self._symbol_to_inst_id(symbol)
        await self._ws_public.unsubscribe(
            key=f"funding:{symbol}",
            unsub_payload={
                "op": "unsubscribe",
                "args": [{"channel": "funding-rate", "instId": inst_id}],
            },
        )

    async def subscribe_trades(self, symbol: str) -> None:
        inst_id = self._symbol_to_inst_id(symbol)
        sub = Subscription(
            key=f"trades:{symbol}",
            payload={
                "op": "subscribe",
                "args": [{"channel": "trades", "instId": inst_id}],
            },
        )
        await self._ws_public.subscribe(sub)

    async def unsubscribe_trades(self, symbol: str) -> None:
        inst_id = self._symbol_to_inst_id(symbol)
        await self._ws_public.unsubscribe(
            key=f"trades:{symbol}",
            unsub_payload={
                "op": "unsubscribe",
                "args": [{"channel": "trades", "instId": inst_id}],
            },
        )

    # ─────────────────────── WS message routing ───────────────────────

    async def _handle_public_message(self, msg: dict[str, Any]) -> None:
        # Pong / system messages
        if msg.get("event") == "pong":
            self._ws_public.mark_pong()
            return
        if "event" in msg:
            # subscribe ack, error, etc.
            if msg.get("event") == "error":
                log.warning("okx.public_error", code=msg.get("code"), msg=msg.get("msg"))
            return

        arg = msg.get("arg") or {}
        channel = arg.get("channel", "")
        data = msg.get("data") or []
        if channel.startswith("candle") and data:
            await self._emit_klines(channel, arg, data)
        elif channel == "funding-rate" and data:
            await self._emit_funding(data)
        elif channel == "trades" and data:
            await self._emit_trades(arg, data)

    async def _emit_klines(self, channel: str, arg: dict[str, Any], data: list[list[Any]]) -> None:
        okx_tf = channel.replace("candle", "")
        # reverse-map OKX TF to unified TF
        unified_tf = next((k for k, v in OKX_TF_MAP.items() if v == okx_tf), okx_tf.lower())
        inst_id = arg.get("instId", "")
        symbol = self._inst_id_to_symbol(inst_id)

        for row in data:
            # [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
            try:
                ts = ms_to_datetime(int(row[0]))
                closed = str(row[8]) == "1"
                evt = KlineEvent(
                    exchange=self.name,
                    symbol=symbol,
                    timeframe=unified_tf,
                    ts=ts,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    closed=closed,
                )
                try:
                    self._kline_q.put_nowait(evt)
                except asyncio.QueueFull:
                    log.warning("okx.kline_queue_full", symbol=symbol, tf=unified_tf)
            except (ValueError, IndexError) as e:
                log.warning("okx.kline_parse_error", row=row, error=str(e))

    async def _emit_trades(self, arg: dict[str, Any], data: list[dict[str, Any]]) -> None:
        inst_id = arg.get("instId", "")
        symbol = self._inst_id_to_symbol(inst_id)
        for row in data:
            try:
                side = OrderSide.BUY if row.get("side") == "buy" else OrderSide.SELL
                evt = TradeEvent(
                    exchange=self.name,
                    symbol=symbol,
                    ts=ms_to_datetime(int(row["ts"])),
                    side=side,
                    qty=float(row["sz"]),
                    price=float(row["px"]),
                )
                try:
                    self._trade_q.put_nowait(evt)
                except asyncio.QueueFull:
                    pass  # trades are high-frequency; drop on overflow, CVD will be approximate
            except (KeyError, ValueError):
                pass

    async def _emit_funding(self, data: list[dict[str, Any]]) -> None:
        for row in data:
            try:
                inst_id = row.get("instId", "")
                symbol = self._inst_id_to_symbol(inst_id)
                rate = float(row.get("fundingRate", 0.0))
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
            except (ValueError, asyncio.QueueFull):
                pass

    async def _handle_private_message(self, msg: dict[str, Any]) -> None:
        event = msg.get("event")
        if event == "login":
            code = msg.get("code")
            if code == "0":
                log.info("okx.private_login_ok")
            else:
                log.error("okx.private_login_failed", code=msg.get("code"), msg=msg.get("msg"))
            return
        if event == "pong":
            self._ws_private.mark_pong()
            return
        if event == "error":
            log.warning("okx.private_error", code=msg.get("code"), msg=msg.get("msg"))
            return
        if event is not None:
            return  # subscribe ack, etc.

        arg = msg.get("arg") or {}
        channel = arg.get("channel", "")
        data = msg.get("data") or []
        if channel == "orders" and data:
            await self._emit_orders(data)
        elif channel == "positions" and data:
            await self._emit_positions(data)
        # account channel is useful for balance tracking but is consumed by
        # portfolio tracker via REST reconcile — skip live event emission here.

    async def _emit_orders(self, data: list[dict[str, Any]]) -> None:
        for row in data:
            try:
                inst_id = row.get("instId", "")
                symbol = self._inst_id_to_symbol(inst_id)
                cli_id = row.get("clOrdId") or None
                exch_id = row.get("ordId", "")
                status = _okx_order_status(row.get("state", ""))
                side = OrderSide.BUY if row.get("side") == "buy" else OrderSide.SELL
                qty = float(row.get("sz", 0) or 0)
                filled = float(row.get("accFillSz", 0) or 0)
                avg_px = float(row.get("avgPx", 0) or 0) or None
                ts_ms = int(row.get("uTime", 0) or now_ms())

                update = OrderUpdateEvent(
                    exchange=self.name,
                    symbol=symbol,
                    client_order_id=cli_id,
                    exchange_order_id=exch_id,
                    status=status,
                    side=side,
                    quantity=qty,
                    filled_quantity=filled,
                    average_price=avg_px,
                    ts=ms_to_datetime(ts_ms),
                    raw=row,
                )
                self._order_q.put_nowait(update)

                # If this order produced new fills, emit them too.
                fill_sz = float(row.get("fillSz", 0) or 0)
                if fill_sz > 0:
                    fill = FillEvent(
                        exchange=self.name,
                        symbol=symbol,
                        client_order_id=cli_id,
                        exchange_order_id=exch_id,
                        side=side,
                        quantity=fill_sz,
                        price=float(row.get("fillPx", 0) or 0),
                        fee=abs(float(row.get("fillFee", 0) or 0)),
                        fee_currency=row.get("fillFeeCcy"),
                        realized_pnl=float(row.get("fillPnl", 0) or 0) or None,
                        ts=ms_to_datetime(int(row.get("fillTime", ts_ms))),
                        raw=row,
                    )
                    self._fill_q.put_nowait(fill)
            except asyncio.QueueFull:
                # A dropped fill means the bot has an untracked position.
                # Raise so _run_forever reconnects and the reconciler recovers.
                log.error(
                    "okx.fill_queue_full_dropping_fill",
                    inst_id=row.get("instId"),
                    ord_id=row.get("ordId"),
                )
                raise
            except ValueError:
                log.warning(
                    "okx.order_parse_error",
                    inst_id=row.get("instId"),
                    ord_id=row.get("ordId"),
                    state=row.get("state"),
                )

    async def _emit_positions(self, data: list[dict[str, Any]]) -> None:
        for row in data:
            try:
                inst_id = row.get("instId", "")
                symbol = self._inst_id_to_symbol(inst_id)
                pos_side_str = row.get("posSide", "net")
                qty = float(row.get("pos", 0) or 0)
                if qty == 0:
                    side = PositionSide.FLAT
                elif pos_side_str == "long" or qty > 0:
                    side = PositionSide.LONG
                else:
                    side = PositionSide.SHORT

                evt = PositionUpdateEvent(
                    exchange=self.name,
                    symbol=symbol,
                    side=side,
                    quantity=abs(qty),
                    entry_price=float(row.get("avgPx", 0) or 0) or None,
                    mark_price=float(row.get("markPx", 0) or 0) or None,
                    liquidation_price=float(row.get("liqPx", 0) or 0) or None,
                    unrealized_pnl=float(row.get("upl", 0) or 0) or None,
                    leverage=float(row.get("lever", 0) or 0) or None,
                    ts=now_utc(),
                )
                self._position_q.put_nowait(evt)
            except (ValueError, asyncio.QueueFull):
                pass

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
        tickers = await self._rest.fetch_tickers(params={"instType": "SWAP"})
        out: list[TickerInfo] = []
        for sym, t in tickers.items():
            # Only keep linear USDT perps of the form "XXX/USDT:USDT"
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
        data = await self._rest.fetch_open_interest(symbol)
        # ccxt unified field; fallback to raw info.oi
        oi = float(data.get("openInterestAmount") or data.get("info", {}).get("oi") or 0.0)
        return OIEvent(
            exchange=self.name,
            symbol=symbol,
            ts=now_utc(),
            oi_contracts=oi,
        )

    # ─────────────────────── REST: account ───────────────────────

    async def fetch_balance(self) -> Balance:
        bal = await self._rest.fetch_balance(params={"type": "swap"})
        usdt = bal.get("USDT") or {}
        return Balance(
            total=float(usdt.get("total", 0) or 0),
            free=float(usdt.get("free", 0) or 0),
            used=float(usdt.get("used", 0) or 0),
            currency="USDT",
        )

    async def fetch_positions(self) -> list[PositionUpdateEvent]:
        positions = await self._rest.fetch_positions()
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
            await self._rest.set_leverage(leverage, symbol, params={"mgnMode": "isolated"})
        except Exception as e:
            # OKX returns an error if leverage is already set — swallow it.
            log.debug("okx.set_leverage_noop", symbol=symbol, lev=leverage, error=str(e))

    # ─────────────────────── trading ───────────────────────

    async def place_order(self, req: OrderRequest) -> OrderResult:
        client_order_id = req.client_order_id or self._make_client_id()
        params: dict[str, Any] = {
            "clOrdId": client_order_id.replace("-", "")[:32],  # OKX: alnum, ≤32 chars
            "tdMode": "isolated",
        }
        if req.reduce_only:
            params["reduceOnly"] = True
        if req.leverage is not None:
            await self.set_leverage(req.symbol, req.leverage)

        order_type_str = "market" if req.order_type == OrderType.MARKET else "limit"
        side_str = _side_to_okx(req.side)

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
            status=_okx_order_status(str(response.get("status", "open"))),
            filled_quantity=float(response.get("filled") or 0),
            average_price=float(response.get("average") or 0) or None,
            raw=response,
        )

    async def cancel_order(
        self, symbol: str, *, client_order_id: str | None = None, exchange_order_id: str | None = None
    ) -> None:
        params: dict[str, Any] = {}
        if client_order_id:
            params["clOrdId"] = client_order_id.replace("-", "")[:32]
        try:
            await self._rest.cancel_order(id=exchange_order_id, symbol=symbol, params=params)
        except Exception as e:
            log.warning("okx.cancel_order_failed", symbol=symbol, error=str(e))

    async def cancel_all(self, symbol: str | None = None) -> None:
        try:
            await self._rest.cancel_all_orders(symbol=symbol)
        except Exception as e:
            log.warning("okx.cancel_all_failed", symbol=symbol, error=str(e))

    async def close_position(self, symbol: str) -> OrderResult:
        """Reduce-only market order in the opposite direction of the current position."""
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

    def _symbol_to_inst_id(self, symbol: str) -> str:
        """Unified 'BTC/USDT:USDT' → OKX 'BTC-USDT-SWAP'."""
        mkt = self._rest.markets.get(symbol)
        if mkt and "id" in mkt:
            return str(mkt["id"])
        # Fallback guess
        base, rest = symbol.split("/", 1)
        quote = rest.split(":", 1)[0]
        return f"{base}-{quote}-SWAP"

    def _inst_id_to_symbol(self, inst_id: str) -> str:
        """OKX 'BTC-USDT-SWAP' → unified 'BTC/USDT:USDT'."""
        for sym, mkt in self._rest.markets.items():
            if mkt.get("id") == inst_id:
                return sym
        # Fallback parse
        parts = inst_id.split("-")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}:{parts[1]}"
        return inst_id

    @staticmethod
    def _make_client_id() -> str:
        return f"cpb{uuid.uuid4().hex[:26]}"
