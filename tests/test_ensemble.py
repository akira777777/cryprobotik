"""Ensemble aggregation logic tests with fake strategies."""

from __future__ import annotations

from datetime import UTC, datetime

from src.data.feature_store import FeatureStore
from src.exchanges.base import OrderSide
from src.settings import RegimeConfig
from src.strategies.base import Signal, SignalAction, Strategy
from src.strategies.ensemble import Ensemble
from src.strategies.regime import Regime, RegimeClassifier


class _FixedRegimeClassifier(RegimeClassifier):
    """RegimeClassifier that always returns a fixed regime — no feature store needed."""

    def __init__(self, config: RegimeConfig, regime: Regime) -> None:
        super().__init__(config)
        self._fixed = regime

    def classify(self, symbol: str, store: FeatureStore, exchange: str) -> Regime:
        return self._fixed


class _FakeStrategy(Strategy):
    """Strategy stub that returns a pre-canned list of signals."""

    def __init__(self, name: str, signals: list[Signal]) -> None:
        super().__init__(base_confidence=0.5)
        self.name = name
        self._signals = signals
        self.calls = 0

    def evaluate(
        self,
        symbol: str,
        store: FeatureStore,
        exchange: str,
        ts: datetime,
    ) -> list[Signal]:
        self.calls += 1
        return list(self._signals)


class _ExplodingStrategy(Strategy):
    """Used to verify the ensemble isolates per-strategy exceptions."""

    name = "exploding"

    def evaluate(self, symbol, store, exchange, ts):  # type: ignore[override]
        raise RuntimeError("boom")


def _make_regime_config() -> RegimeConfig:
    return RegimeConfig(
        adx_period=14,
        adx_trend_threshold=25,
        adx_range_threshold=20,
        vol_window_bars=96,
        vol_high_threshold=0.015,
        weights={
            "trend_high_vol": {"momentum": 1.0, "volatility_breakout": 0.8,
                               "mean_reversion": 0.0, "funding_arb": 1.0},
            "trend_low_vol": {"momentum": 0.9, "volatility_breakout": 0.5,
                              "mean_reversion": 0.0, "funding_arb": 1.0},
            "range_high_vol": {"momentum": 0.2, "volatility_breakout": 0.3,
                               "mean_reversion": 0.8, "funding_arb": 1.0},
            "range_low_vol": {"momentum": 0.0, "volatility_breakout": 0.1,
                              "mean_reversion": 1.0, "funding_arb": 1.0},
            "chop": {"momentum": 0.0, "volatility_breakout": 0.0,
                     "mean_reversion": 0.3, "funding_arb": 1.0},
        },
    )


def _sig(strategy: str, side: OrderSide, confidence: float) -> Signal:
    return Signal(
        strategy=strategy,
        symbol="BTC/USDT:USDT",
        action=SignalAction.OPEN,
        side=side,
        confidence=confidence,
        ts=datetime(2026, 4, 9, 12, 0, tzinfo=UTC),
    )


def test_single_momentum_buy_in_trend_regime() -> None:
    cfg = _make_regime_config()
    rc = _FixedRegimeClassifier(cfg, Regime.TREND_HIGH_VOL)
    mom = _FakeStrategy("momentum", [_sig("momentum", OrderSide.BUY, 0.8)])
    ens = Ensemble(strategies=[mom], regime_classifier=rc)

    ts = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)
    consolidated, regime, raw = ens.evaluate_symbol(
        "BTC/USDT:USDT", FeatureStore(), "okx", ts,
    )
    assert consolidated is not None
    assert consolidated.side == OrderSide.BUY
    # momentum weight in trend_high_vol = 1.0, contribution = 0.8
    assert consolidated.confidence == 0.8
    assert regime == Regime.TREND_HIGH_VOL
    assert len(raw) == 1


def test_opposing_signals_cancel_below_threshold() -> None:
    """Buy+sell of equal weighted confidence → net vote 0 → no consolidated signal."""
    cfg = _make_regime_config()
    rc = _FixedRegimeClassifier(cfg, Regime.TREND_HIGH_VOL)
    s_buy = _FakeStrategy("momentum", [_sig("momentum", OrderSide.BUY, 0.6)])
    s_sell = _FakeStrategy(
        "volatility_breakout",
        [_sig("volatility_breakout", OrderSide.SELL, 0.75)],  # 0.75 * 0.8 = 0.6
    )
    ens = Ensemble(strategies=[s_buy, s_sell], regime_classifier=rc)

    ts = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)
    consolidated, _, raw = ens.evaluate_symbol("BTC/USDT:USDT", FeatureStore(), "okx", ts)
    assert consolidated is None
    assert len(raw) == 2  # both raw signals recorded


def test_wrong_regime_zeroes_out_contribution() -> None:
    """
    Mean reversion has 0.0 weight in trend_high_vol regime — so its signals
    must contribute nothing to the net vote.
    """
    cfg = _make_regime_config()
    rc = _FixedRegimeClassifier(cfg, Regime.TREND_HIGH_VOL)
    mr = _FakeStrategy("mean_reversion", [_sig("mean_reversion", OrderSide.BUY, 1.0)])
    ens = Ensemble(strategies=[mr], regime_classifier=rc)

    ts = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)
    consolidated, _, _ = ens.evaluate_symbol("BTC/USDT:USDT", FeatureStore(), "okx", ts)
    assert consolidated is None


def test_funding_arb_is_skipped_by_ensemble() -> None:
    """Funding arb emits via a non-bar path and must not enter the vote."""
    cfg = _make_regime_config()
    rc = _FixedRegimeClassifier(cfg, Regime.TREND_HIGH_VOL)
    fa = _FakeStrategy("funding_arb", [_sig("funding_arb", OrderSide.BUY, 1.0)])
    mom = _FakeStrategy("momentum", [_sig("momentum", OrderSide.BUY, 0.5)])
    ens = Ensemble(strategies=[fa, mom], regime_classifier=rc)

    ts = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)
    consolidated, _, raw = ens.evaluate_symbol("BTC/USDT:USDT", FeatureStore(), "okx", ts)
    assert consolidated is not None
    assert consolidated.confidence == 0.5  # momentum-only
    # funding_arb was not evaluated at all (ensemble shortcuts on name)
    assert fa.calls == 0
    assert mom.calls == 1
    # Raw signals contain only momentum — the ensemble never asked fa for signals
    assert [s.strategy for s in raw] == ["momentum"]


def test_sub_threshold_vote_returns_none() -> None:
    """A net vote below the configured min_net_vote must not emit a signal."""
    cfg = _make_regime_config()
    rc = _FixedRegimeClassifier(cfg, Regime.TREND_HIGH_VOL)
    # Weighted contribution = 0.15 × 1.0 = 0.15 which is below the default
    # min_net_vote of 0.20 in the Ensemble, so nothing should be emitted.
    mom = _FakeStrategy("momentum", [_sig("momentum", OrderSide.BUY, 0.15)])
    ens = Ensemble(strategies=[mom], regime_classifier=rc)

    ts = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)
    consolidated, _, _ = ens.evaluate_symbol("BTC/USDT:USDT", FeatureStore(), "okx", ts)
    assert consolidated is None


def test_exploding_strategy_is_isolated() -> None:
    """One bad strategy must not break the whole ensemble."""
    cfg = _make_regime_config()
    rc = _FixedRegimeClassifier(cfg, Regime.TREND_HIGH_VOL)
    bad = _ExplodingStrategy()
    good = _FakeStrategy("momentum", [_sig("momentum", OrderSide.BUY, 0.6)])
    ens = Ensemble(strategies=[bad, good], regime_classifier=rc)

    ts = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)
    consolidated, _, _ = ens.evaluate_symbol("BTC/USDT:USDT", FeatureStore(), "okx", ts)
    # The good strategy still produced a valid signal.
    assert consolidated is not None
    assert consolidated.side == OrderSide.BUY


def test_confidence_capped_at_one() -> None:
    """
    Multiple strong long signals whose weighted sum > 1.0 get clipped to 1.0
    in the consolidated signal.
    """
    cfg = _make_regime_config()
    rc = _FixedRegimeClassifier(cfg, Regime.TREND_HIGH_VOL)
    # momentum weight 1.0 × conf 1.0 = 1.0
    # volatility_breakout weight 0.8 × conf 1.0 = 0.8
    # Total long = 1.8, no short → net vote = 1.8 → clipped to 1.0
    s1 = _FakeStrategy("momentum", [_sig("momentum", OrderSide.BUY, 1.0)])
    s2 = _FakeStrategy("volatility_breakout",
                       [_sig("volatility_breakout", OrderSide.BUY, 1.0)])
    ens = Ensemble(strategies=[s1, s2], regime_classifier=rc)

    ts = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)
    consolidated, _, _ = ens.evaluate_symbol("BTC/USDT:USDT", FeatureStore(), "okx", ts)
    assert consolidated is not None
    assert consolidated.confidence == 1.0
    assert consolidated.meta["net_vote"] == 1.8
