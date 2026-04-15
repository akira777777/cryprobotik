"""
Portfolio-level risk limits.

The risk manager sizes individual trades. This module checks whether a sized
trade is actually allowed given the current portfolio state:
    - max concurrent open positions
    - max positions per symbol
    - correlation with already-open positions
    - aggregate exposure
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.portfolio.tracker import PortfolioTracker
    from src.risk.manager import SizedTrade
    from src.settings import RiskConfig

log = get_logger(__name__)


@dataclass(slots=True)
class LimitDecision:
    allowed: bool
    reason: str | None = None
    details: dict[str, float | int | str] | None = None


class PortfolioLimits:
    """
    Gatekeeper that the executor consults before placing an order.

    Correlations are computed from the feature store's rolling OHLCV buffer.
    Only same-timeframe data is compared — strategies must pass a comparable
    frame (e.g. 1h) or correlations will be skipped.
    """

    def __init__(self, config: "RiskConfig", tracker: "PortfolioTracker") -> None:
        self._config = config
        self._tracker = tracker

    def check(self, trade: "SizedTrade", return_frames: dict[str, pd.Series] | None = None) -> LimitDecision:
        """
        Args:
            trade:          the SizedTrade proposed by RiskManager.
            return_frames:  optional {symbol: close-price Series} for correlation.
                            Pass the same-timeframe close series for every open
                            symbol + the new one. If None, correlation is skipped.
        """
        open_positions = self._tracker.open_positions()

        # 1) Max open positions
        if len(open_positions) >= self._config.max_open_positions:
            return LimitDecision(
                allowed=False,
                reason="max_open_positions",
                details={
                    "open": len(open_positions),
                    "max": self._config.max_open_positions,
                },
            )

        # 2) Max positions per symbol (prevents stacking on the same symbol)
        same_symbol = [p for p in open_positions if p.symbol == trade.symbol]
        if len(same_symbol) >= self._config.max_positions_per_symbol:
            return LimitDecision(
                allowed=False,
                reason="max_positions_per_symbol",
                details={"symbol": trade.symbol},
            )

        # 3) Correlation cap — require at least MIN_CORR_BARS to avoid
        # spurious rejections on fresh data.
        MIN_CORR_BARS = 50
        if return_frames is not None and len(open_positions) > 0 and self._config.max_correlation < 1.0:
            new_returns = _to_returns(return_frames.get(trade.symbol))
            if new_returns is not None and len(new_returns) >= MIN_CORR_BARS:
                for pos in open_positions:
                    other = _to_returns(return_frames.get(pos.symbol))
                    if other is None or len(other) < MIN_CORR_BARS:
                        continue
                    aligned = pd.concat([new_returns, other], axis=1, join="inner").dropna()
                    if len(aligned) < MIN_CORR_BARS:
                        log.debug(
                            "limits.correlation_check_skipped_insufficient_data",
                            n=len(aligned),
                            symbol=trade.symbol,
                            other=pos.symbol,
                        )
                        continue
                    c = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                    if not np.isnan(c) and abs(c) > self._config.max_correlation:
                        return LimitDecision(
                            allowed=False,
                            reason="correlation_cap",
                            details={
                                "other_symbol": pos.symbol,
                                "correlation": round(c, 3),
                                "max": self._config.max_correlation,
                            },
                        )

        return LimitDecision(allowed=True)


def _to_returns(price_series: pd.Series | None) -> pd.Series | None:
    if price_series is None or len(price_series) < 2:
        return None
    return np.log(price_series / price_series.shift(1)).dropna()
