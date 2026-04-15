"""
ML-based signal filter.

Sits between Ensemble output and RiskManager. Scores each consolidated Signal
with a LightGBM regressor trained to predict trade R-multiples.

The target variable is `realized_pnl / risk_usd` — positive values mean
profitable trades, negative means losers. A signal is accepted when the
predicted R-multiple ≥ MIN_EXPECTED_R (default 0.5).

Using R-multiple regression rather than binary win/loss classification
aligns the model's incentive with actual profitability: a +3R winner is
treated as much better than a +0.1R winner, preventing the model from
optimising for marginal wins.

Lifecycle:
  Cold start  (< MIN_SAMPLES labels): pass-through, predicted R = 0.5 → accept
  Warm        (≥ MIN_SAMPLES):        filter signals with predicted R < MIN_EXPECTED_R
  Retrain     every RETRAIN_EVERY new labels; runs in a thread pool

State persistence: model + training buffer persisted to bot_state DB table as
hex-encoded pickle so restarts don't lose learned knowledge.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac as hmac_lib
import io
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.data.feature_store import CVDStore, FeatureStore, OIStore
    from src.data.storage import Storage
    from src.exchanges.base import OrderSide
    from src.strategies.base import Signal

log = get_logger(__name__)


def _sign_model(secret: bytes, data: bytes) -> str:
    """Return HMAC-SHA256 hex digest of *data* using *secret*."""
    return hmac_lib.new(secret, data, hashlib.sha256).hexdigest()


def _verify_model(secret: bytes, data: bytes, expected: str) -> bool:
    """Constant-time comparison of HMAC digest."""
    actual = _sign_model(secret, data)
    return hmac_lib.compare_digest(actual, expected)


MIN_SAMPLES: int = 50  # labels needed before first training run
RETRAIN_EVERY: int = 50  # retrain after every N new labels (raised from 20 for stability)
MIN_EXPECTED_R: float = 0.5  # predicted R-multiple must be ≥ this to accept a signal
ACCEPT_THRESHOLD: float = MIN_EXPECTED_R  # alias — used in tests and healthcheck
FEE_FLOOR_PCT: float = 0.001  # skip labels within this fraction of entry notional (fee noise)
ML_BUFFER_MAX: int = 2000  # max training examples kept in memory and persisted (raised from 500)
BOT_STATE_KEY = "ml.model_state"


@dataclass
class MLDecision:
    accepted: bool
    ml_score: float
    features: list[float]
    model_version: int
    cold_start: bool


class MLSignalFilter:
    """
    Online-learning signal filter built on scikit-learn GBT.

    Thread-safety: all mutation happens in the asyncio event loop (no threads
    except during the blocking _train call which uses run_in_executor).
    """

    def __init__(
        self,
        storage: "Storage",
        accept_threshold: float = ACCEPT_THRESHOLD,
        cvd_store: "CVDStore | None" = None,
        oi_store: "OIStore | None" = None,
    ) -> None:
        self._storage = storage
        self._threshold = accept_threshold
        self._cvd_store = cvd_store
        self._oi_store = oi_store

        self._model = None  # sklearn Pipeline or None
        self._model_version: int = 0

        # River online model — initialized lazily in load(); None when river
        # is not installed. Learns after every trade (sub-sample adaptation).
        self._online_model = None
        self._online_model_samples: int = 0

        # Training buffer — feature vectors and R-multiple labels (float since v4)
        self._X: list[list[float]] = []
        self._y: list[float] = []  # R-multiple: realized_pnl / risk_usd
        self._new_since_retrain: int = 0

        # Pending features keyed by (symbol, side_value) while trade is open
        self._pending: dict[tuple[str, str], list[float]] = {}
        # Pending risk_usd — stored separately so store_pending() doesn't need
        # to be moved after sizing in the orchestrator.
        self._pending_risk: dict[tuple[str, str], float] = {}

        # SSE subscribers: set so discard() is O(1) and never raises.
        self._sse_queues: set[asyncio.Queue[str]] = set()

        # Background tasks (DB persist + SSE push) — tracked to avoid leaks.
        self._background_tasks: set[asyncio.Task[None]] = set()

        self._lock = asyncio.Lock()

    # ─────────────────────── startup ───────────────────────

    async def load(self) -> None:
        """Restore persisted model and training buffer from DB."""
        state = await self._storage.get_state(BOT_STATE_KEY)
        if not state:
            log.info("ml.cold_start", reason="no persisted state")
            return
        try:
            import os
            from src.ml.features import FEATURE_VERSION as _FEATURE_VERSION
            from src.ml.features import N_FEATURES as _N_FEATURES

            # Check feature version first — version bump means layout changed.
            persisted_fv = state.get("feature_version", 1)
            if persisted_fv != _FEATURE_VERSION:
                log.warning(
                    "ml.feature_version_mismatch",
                    persisted_version=persisted_fv,
                    expected_version=_FEATURE_VERSION,
                    action="discarding persisted model and buffer — cold start",
                )
                return

            model_hex = state.get("model_bytes")
            if model_hex:
                import joblib

                raw_bytes = bytes.fromhex(model_hex)
                # Verify HMAC before deserialization to prevent arbitrary code exec.
                hmac_secret_str = os.getenv("MODEL_HMAC_SECRET", "")
                stored_hmac = state.get("model_hmac", "")
                if hmac_secret_str:
                    if not stored_hmac or not _verify_model(hmac_secret_str.encode(), raw_bytes, stored_hmac):
                        log.critical(
                            "ml.hmac_verification_failed",
                            action="discarding model — cold start",
                        )
                        return
                elif stored_hmac:
                    # Model was signed but we have no secret to verify — refuse to load.
                    log.warning(
                        "ml.hmac_secret_missing",
                        action="model was signed but MODEL_HMAC_SECRET unset — cold start",
                    )
                    return
                self._model = joblib.load(io.BytesIO(raw_bytes))
            X_loaded: list[list[float]] = state.get("X", [])
            y_loaded: list[int] = state.get("y", [])
            # If the feature vector size changed, the persisted model and buffer are
            # incompatible with the current code — discard and cold-start.
            if X_loaded:
                if len(X_loaded[0]) != _N_FEATURES:
                    log.warning(
                        "ml.feature_count_mismatch",
                        persisted=len(X_loaded[0]),
                        expected=_N_FEATURES,
                        action="discarding persisted model and buffer",
                    )
                    X_loaded, y_loaded = [], []
                    self._model = None
            self._X = X_loaded
            self._y = y_loaded
            self._model_version = int(state.get("version", 0))
            log.info(
                "ml.model_loaded",
                version=self._model_version,
                n_samples=len(self._y),
            )
        except Exception as exc:
            log.warning("ml.load_failed", error=str(exc))
            self._model = None

        # Initialize River online model if available. The ARF regressor
        # learns R-multiple after every trade outcome — much faster adaptation
        # than waiting for RETRAIN_EVERY LightGBM retrains.
        try:
            from river.forest import ARFRegressor
            self._online_model = ARFRegressor()
            log.info("ml.online_model_initialized", backend="river.ARFRegressor")
        except ImportError:
            self._online_model = None
            log.debug("ml.online_model_unavailable", reason="river not installed")

    # ─────────────────────── scoring ───────────────────────

    async def evaluate(
        self,
        signal: "Signal",
        store: "FeatureStore",
        exchange: str,
    ) -> MLDecision:
        """Score a consolidated signal. Always returns an MLDecision."""
        from src.ml.features import FEATURE_NAMES, extract_features

        features = extract_features(signal, store, exchange, self._cvd_store, self._oi_store)
        cold = features is None or self._model is None

        if cold:
            # Pass-through during cold start; score = threshold so it's
            # clear in logs that this is the default, not a real prediction.
            score = self._threshold
            accepted = True
        else:
            try:
                # LightGBM regressor predicts R-multiple directly.
                lightgbm_r = float(self._model.predict([features])[0])
            except Exception as exc:
                log.warning("ml.score_error", error=str(exc))
                lightgbm_r = self._threshold

            # Blend with River online ARF regressor when it has enough samples.
            # LightGBM (60%) is the primary; River ARF (40%) adds recency
            # sensitivity without destabilising the cold-start pass-through.
            if self._online_model is not None and self._online_model_samples >= 10:
                try:
                    online_r = self._online_model.predict_one(dict(enumerate(features))) or 0.0
                    score = 0.6 * lightgbm_r + 0.4 * online_r
                except Exception as exc:
                    log.warning("ml.online_score_error", error=str(exc))
                    score = lightgbm_r
            else:
                score = lightgbm_r

            accepted = score >= self._threshold

        decision = MLDecision(
            accepted=accepted,
            ml_score=score,
            features=features or [],
            model_version=self._model_version,
            cold_start=cold,
        )

        log.info(
            "ml.decision",
            symbol=signal.symbol,
            side=signal.side.value,
            score=round(score, 3),
            accepted=accepted,
            cold=cold,
            model_v=self._model_version,
        )

        # Schedule persist + push SSE with leak-safe task tracking.
        task: asyncio.Task[None] = asyncio.create_task(self._log_and_push(signal, decision))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        return decision

    def store_pending(self, symbol: str, side: "OrderSide", features: list[float]) -> None:
        """Remember the features for an open trade so we can label it on close."""
        self._pending[(symbol, side.value)] = features

    def update_pending_risk(self, symbol: str, side: "OrderSide", risk_usd: float) -> None:
        """Store the dollar risk for an open trade (called after RiskManager sizing)."""
        key = (symbol, side.value)
        if key in self._pending:
            self._pending_risk[key] = risk_usd

    # ─────────────────────── outcome recording ───────────────────────

    async def record_outcome(
        self,
        symbol: str,
        side: "OrderSide",
        realized_pnl: float,
        entry_notional: float = 0.0,
    ) -> None:
        """Called when a trade closes. Labels the training example and maybe retrains.

        The label is the R-multiple (realized_pnl / risk_usd).  When risk_usd is
        not available (position opened before ML was active), falls back to ±1.0.
        """
        key = (symbol, side.value)
        features = self._pending.pop(key, None)
        if features is None:
            return  # no matching pending signal (e.g. position from before ML was active)

        risk_usd = self._pending_risk.pop(key, 0.0)

        # Skip near-breakeven labels — round-trip fees create a dead zone where
        # the sign of PnL is determined by noise, not signal quality.
        if entry_notional > 0.0 and abs(realized_pnl) < entry_notional * FEE_FLOOR_PCT:
            log.debug("ml.outcome_skipped_deadzone", symbol=symbol, pnl=round(realized_pnl, 6))
            return

        # R-multiple: positive = winner, negative = loser, magnitude = quality
        if risk_usd > 0.0:
            r_label = realized_pnl / risk_usd
        else:
            r_label = 1.0 if realized_pnl > 0.0 else -1.0  # fallback without risk_usd

        # Clamp to a reasonable range to prevent outliers from destabilising training.
        r_label = max(-5.0, min(10.0, r_label))

        async with self._lock:
            self._X.append(features)
            self._y.append(r_label)
            # Keep buffer bounded — consistent with the persisted window so a restarted
            # session behaves identically to a long-running one.
            if len(self._X) > ML_BUFFER_MAX:
                self._X = self._X[-ML_BUFFER_MAX:]
                self._y = self._y[-ML_BUFFER_MAX:]
            self._new_since_retrain += 1

        # Online learning: River ARF regressor learns immediately after every outcome.
        # This gives sub-sample adaptation without waiting for RETRAIN_EVERY.
        if self._online_model is not None:
            try:
                self._online_model.learn_one(dict(enumerate(features)), r_label)
                self._online_model_samples += 1
            except Exception as exc:
                log.warning("ml.online_learn_error", error=str(exc))

        log.info(
            "ml.outcome",
            symbol=symbol,
            pnl=round(realized_pnl, 4),
            r_label=round(r_label, 3),
            risk_usd=round(risk_usd, 2),
            n_samples=len(self._y),
        )

        if len(self._y) >= MIN_SAMPLES and self._new_since_retrain >= RETRAIN_EVERY:
            await self._retrain()

    # ─────────────────────── training ───────────────────────

    async def _retrain(self) -> None:
        async with self._lock:
            X_snap = list(self._X)
            y_snap = list(self._y)

        if len(y_snap) < MIN_SAMPLES:
            return

        loop = asyncio.get_running_loop()
        try:
            new_model = await loop.run_in_executor(None, _fit_model, X_snap, y_snap)
            async with self._lock:
                self._model = new_model
                self._model_version += 1
                self._new_since_retrain = 0

            from src.monitoring import prom_metrics as _m

            _m.ml_training_samples_gauge.set(len(y_snap))
            _m.ml_model_version_gauge.set(self._model_version)

            avg_r = round(sum(y_snap) / len(y_snap), 3)
            win_rate = round(sum(1 for r in y_snap if r > 0) / len(y_snap), 3)
            log.info(
                "ml.model_retrained",
                version=self._model_version,
                n_samples=len(y_snap),
                avg_r_multiple=avg_r,
                win_rate=win_rate,
            )
            await self._persist()
        except Exception as exc:
            log.error("ml.retrain_failed", error=str(exc), exc_info=True)

    async def _persist(self) -> None:
        try:
            import joblib
            import os
            from src.ml.features import FEATURE_VERSION as _FEATURE_VERSION

            hmac_secret_str = os.getenv("MODEL_HMAC_SECRET", "")
            if not hmac_secret_str:
                log.warning(
                    "ml.persist_skipped",
                    reason="MODEL_HMAC_SECRET not set — running cold-start mode",
                )
                return

            buf = io.BytesIO()
            joblib.dump(self._model, buf)
            raw_bytes = buf.getvalue()
            model_bytes_hex = raw_bytes.hex()
            model_hmac = _sign_model(hmac_secret_str.encode(), raw_bytes)

            X_save = self._X[-ML_BUFFER_MAX:]
            y_save = self._y[-ML_BUFFER_MAX:]
            await self._storage.set_state(
                BOT_STATE_KEY,
                {
                    "model_bytes": model_bytes_hex,
                    "model_hmac": model_hmac,
                    "X": X_save,
                    "y": y_save,
                    "version": self._model_version,
                    "feature_version": _FEATURE_VERSION,
                },
            )
        except Exception as exc:
            log.warning("ml.persist_failed", error=str(exc))

    # ─────────────────────── SSE + DB logging ───────────────────────

    async def _log_and_push(self, signal: "Signal", decision: MLDecision) -> None:
        from src.ml.features import FEATURE_NAMES

        feat_dict = dict(zip(FEATURE_NAMES, decision.features)) if decision.features else {}

        # Write to ml_decisions table
        try:
            async with self._storage.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO ml_decisions
                        (ts, symbol, exchange, side, confidence, ml_score, accepted,
                         cold_start, model_version, regime, net_vote, features)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    """,
                    signal.ts,
                    signal.symbol,
                    signal.preferred_exchange or "ensemble",
                    signal.side.value,
                    signal.confidence,
                    decision.ml_score,
                    decision.accepted,
                    decision.cold_start,
                    decision.model_version,
                    signal.meta.get("regime"),
                    signal.meta.get("net_vote"),
                    json.dumps(feat_dict),
                )
        except Exception as exc:
            log.debug("ml.db_log_failed", error=str(exc))

        # Push to SSE subscribers
        payload = json.dumps(
            {
                "ts": signal.ts.isoformat(),
                "symbol": signal.symbol,
                "side": signal.side.value,
                "confidence": round(signal.confidence, 3),
                "ml_score": round(decision.ml_score, 3),
                "accepted": decision.accepted,
                "cold_start": decision.cold_start,
                "regime": signal.meta.get("regime"),
            }
        )
        dead: set[asyncio.Queue[str]] = set()
        for q in self._sse_queues:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.add(q)
        self._sse_queues -= dead

    def subscribe_sse(self) -> asyncio.Queue[str]:
        """Return a queue that will receive JSON strings for each new decision."""
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
        self._sse_queues.add(q)
        return q

    def unsubscribe_sse(self, q: asyncio.Queue[str]) -> None:
        self._sse_queues.discard(q)

    async def shutdown(self) -> None:
        """Cancel and await all in-flight background tasks. Call on teardown."""
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

    # ─────────────────────── stats ───────────────────────

    def stats(self) -> dict[str, Any]:
        n = len(self._y)
        wins = sum(1 for r in self._y if r > 0)
        avg_r = round(sum(self._y) / n, 3) if n > 0 else None
        imps = self._feature_importances()
        return {
            "model_version": self._model_version,
            "n_samples": n,
            "n_profitable": wins,
            "win_rate": round(wins / n, 3) if n > 0 else None,
            "avg_r_multiple": avg_r,
            "min_expected_r": self._threshold,
            "cold_start": self._model is None,
            "accept_threshold": self._threshold,
            "feature_importances": imps,
            "pending_positions": len(self._pending),
            "online_model_samples": self._online_model_samples,
        }

    def _feature_importances(self) -> dict[str, float] | None:
        from src.ml.features import FEATURE_NAMES

        if self._model is None:
            return None
        try:
            # LightGBM regressor inside sklearn Pipeline — access via named_steps.
            clf = self._model.named_steps["clf"]
            imp = clf.feature_importances_
            return {name: round(float(v), 4) for name, v in zip(FEATURE_NAMES, imp)}
        except (AttributeError, KeyError):
            return None


# ─────────────────────── CPU-bound training ───────────────────────


def _fit_model(X: list[list[float]], y: list[float]):
    """Called in a thread pool. Returns a fitted sklearn Pipeline.

    Trains a LightGBM regressor to predict R-multiple (realized_pnl / risk_usd).
    LightGBM is substantially faster than GradientBoostingClassifier on large
    buffers (2000 examples) and handles sparse/noisy financial data well.
    """
    from lightgbm import LGBMRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LGBMRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    num_leaves=31,
                    min_child_samples=10,
                    random_state=42,
                    verbose=-1,       # suppress LightGBM training output
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe
