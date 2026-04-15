"""
Typed configuration loader.

Loads `config/config.yaml`, overlays environment variables, validates with
pydantic, and enforces HARD CEILINGS on risk parameters. A config that tries
to set e.g. leverage=100 will refuse to load — this is intentional. Capital
preservation is enforced at the config layer, not just at runtime.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ──────────────────────────────────────────────────────────────────────────────
# HARD CEILINGS — do not change these without a deliberate safety review.
# ──────────────────────────────────────────────────────────────────────────────
MAX_LEVERAGE_CEILING: int = 10
MAX_RISK_PER_TRADE_CEILING: float = 0.05  # 5% equity per trade, absolute max
MAX_DAILY_DRAWDOWN_CEILING: float = 0.30  # 30% daily DD kill level, absolute max
MAX_OPEN_POSITIONS_CEILING: int = 20
MAX_MARGIN_UTILIZATION_CEILING: float = 0.90


class RuntimeMode(StrEnum):
    """Bot runtime mode — determines which exchange endpoints are used."""

    TESTNET = "testnet"
    PAPER = "paper"
    LIVE = "live"


# ──────────────────────────────────────────────────────────────────────────────
# Nested config models
# ──────────────────────────────────────────────────────────────────────────────


class ExchangeConfig(BaseModel):
    enabled: bool = True
    rest_rate_limit_per_sec: float = 8.0
    ws_ping_interval_sec: float = 20.0
    ws_reconnect_max_backoff_sec: float = 30.0


class ExchangesConfig(BaseModel):
    okx: ExchangeConfig = Field(default_factory=ExchangeConfig)
    bybit: ExchangeConfig = Field(default_factory=ExchangeConfig)


class UniverseConfig(BaseModel):
    top_n: int = Field(10, ge=1, le=50)
    refresh_interval_hours: float = Field(4.0, ge=0.5, le=24.0)
    quote_currency: str = "USDT"
    instrument_type: str = "swap"
    force_include: list[str] = Field(default_factory=list)
    force_exclude: list[str] = Field(default_factory=list)


class MomentumStrategyConfig(BaseModel):
    enabled: bool = True
    timeframes: list[str] = ["15m", "1h", "4h"]
    ema_fast: int = 9
    ema_mid: int = 21
    ema_slow: int = 55
    rsi_period: int = 14
    rsi_long_threshold: float = 55.0
    rsi_short_threshold: float = 45.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    base_confidence: float = Field(0.6, ge=0.0, le=1.0)
    volume_multiplier: float = Field(1.5, ge=0.0)


class MeanReversionStrategyConfig(BaseModel):
    enabled: bool = True
    timeframe: str = "15m"
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 2
    rsi_long_threshold: float = 10.0
    rsi_short_threshold: float = 90.0
    adx_max: float = 20.0
    base_confidence: float = Field(0.55, ge=0.0, le=1.0)


class FundingArbStrategyConfig(BaseModel):
    enabled: bool = True
    min_rate_delta: float = 0.0005
    min_notional_usd: float = 200.0
    close_before_funding_sec: float = 60.0
    base_confidence: float = Field(0.8, ge=0.0, le=1.0)


class VolatilityBreakoutStrategyConfig(BaseModel):
    enabled: bool = True
    timeframe: str = "1h"
    donchian_period: int = 20
    squeeze_atr_ratio_max: float = 0.8
    squeeze_bars: int = Field(6, ge=2)
    volume_multiple: float = 1.5
    base_confidence: float = Field(0.65, ge=0.0, le=1.0)


class FundingContrarianStrategyConfig(BaseModel):
    enabled: bool = True
    extreme_threshold: float = Field(0.85, ge=0.5, le=1.0)
    low_threshold: float = Field(0.15, ge=0.0, le=0.5)
    base_confidence: float = Field(0.55, ge=0.0, le=1.0)


class VWAPStrategyConfig(BaseModel):
    enabled: bool = True
    timeframe: str = "15m"
    ema_period: int = 50
    vwap_band_pct: float = Field(0.001, gt=0.0)
    base_confidence: float = Field(0.60, ge=0.0, le=1.0)


class LiquidationCascadeStrategyConfig(BaseModel):
    enabled: bool = True
    timeframe: str = "15m"
    oi_roc_threshold: float = Field(-0.03, le=0.0)   # −3% OI drop required
    atr_period: int = Field(14, ge=2)
    atr_multiplier: float = Field(1.5, gt=0.0)
    base_confidence: float = Field(0.60, ge=0.0, le=1.0)


class CVDConfig(BaseModel):
    enabled: bool = True
    max_bars: int = Field(200, ge=10)


class OIConfig(BaseModel):
    enabled: bool = True
    poll_interval_sec: float = Field(300.0, gt=0.0)  # 5 minutes
    max_samples: int = Field(200, ge=10)


class StrategiesConfig(BaseModel):
    momentum: MomentumStrategyConfig = Field(default_factory=MomentumStrategyConfig)
    mean_reversion: MeanReversionStrategyConfig = Field(default_factory=MeanReversionStrategyConfig)
    funding_arb: FundingArbStrategyConfig = Field(default_factory=FundingArbStrategyConfig)
    volatility_breakout: VolatilityBreakoutStrategyConfig = Field(
        default_factory=VolatilityBreakoutStrategyConfig
    )
    funding_contrarian: FundingContrarianStrategyConfig = Field(
        default_factory=FundingContrarianStrategyConfig
    )
    vwap: VWAPStrategyConfig = Field(default_factory=VWAPStrategyConfig)
    liquidation_cascade: LiquidationCascadeStrategyConfig = Field(
        default_factory=LiquidationCascadeStrategyConfig
    )
    cvd: CVDConfig = Field(default_factory=CVDConfig)
    oi: OIConfig = Field(default_factory=OIConfig)


class RegimeConfig(BaseModel):
    adx_period: int = 14
    adx_trend_threshold: float = 25.0
    adx_range_threshold: float = 20.0
    vol_window_bars: int = 96
    vol_high_threshold: float = 0.015
    min_net_vote: float = Field(0.20, ge=0.0, le=1.0)
    regime_hysteresis_bars: int = Field(2, ge=1, le=10)
    weights: dict[str, dict[str, float]] = Field(default_factory=dict)

    @field_validator("weights")
    @classmethod
    def _validate_weights(cls, v: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        for regime_name, strat_weights in v.items():
            for strat, weight in strat_weights.items():
                if not 0.0 <= weight <= 2.0:
                    raise ValueError(f"regime[{regime_name}][{strat}] weight must be in [0, 2], got {weight}")
        return v


class RiskConfig(BaseModel):
    max_daily_drawdown_pct: float = Field(0.30, gt=0.0)
    warning_drawdown_pct: float = Field(0.15, gt=0.0)
    risk_per_trade_pct: float = Field(0.02, gt=0.0)
    max_open_positions: int = Field(4, ge=1)
    max_positions_per_symbol: int = Field(1, ge=1)
    leverage: int = Field(3, ge=1)
    sl_atr_multiplier: float = Field(1.5, gt=0.0)
    min_reward_to_risk: float = Field(1.5, ge=1.0)
    max_correlation: float = Field(0.80, ge=0.0, le=1.0)
    correlation_lookback_bars: int = Field(720, ge=10)
    max_margin_utilization: float = Field(0.70, gt=0.0)
    flatten_on_shutdown: bool = False

    @model_validator(mode="after")
    def _enforce_hard_ceilings(self) -> RiskConfig:
        """Refuse to load if any risk knob exceeds its hard ceiling."""
        if self.max_daily_drawdown_pct > MAX_DAILY_DRAWDOWN_CEILING:
            raise ValueError(
                f"max_daily_drawdown_pct={self.max_daily_drawdown_pct} exceeds hard "
                f"ceiling {MAX_DAILY_DRAWDOWN_CEILING}. Edit settings.py only after "
                f"deliberate safety review."
            )
        if self.risk_per_trade_pct > MAX_RISK_PER_TRADE_CEILING:
            raise ValueError(
                f"risk_per_trade_pct={self.risk_per_trade_pct} exceeds hard ceiling "
                f"{MAX_RISK_PER_TRADE_CEILING}"
            )
        if self.leverage > MAX_LEVERAGE_CEILING:
            raise ValueError(f"leverage={self.leverage} exceeds hard ceiling {MAX_LEVERAGE_CEILING}")
        if self.max_open_positions > MAX_OPEN_POSITIONS_CEILING:
            raise ValueError(
                f"max_open_positions={self.max_open_positions} exceeds hard ceiling "
                f"{MAX_OPEN_POSITIONS_CEILING}"
            )
        if self.max_margin_utilization > MAX_MARGIN_UTILIZATION_CEILING:
            raise ValueError(
                f"max_margin_utilization={self.max_margin_utilization} exceeds hard "
                f"ceiling {MAX_MARGIN_UTILIZATION_CEILING}"
            )
        if self.warning_drawdown_pct >= self.max_daily_drawdown_pct:
            raise ValueError("warning_drawdown_pct must be < max_daily_drawdown_pct")
        return self


class ExecutionConfig(BaseModel):
    max_retries: int = Field(5, ge=0, le=20)
    retry_base_seconds: float = Field(0.5, gt=0.0)
    retry_jitter_seconds: float = Field(0.25, ge=0.0)
    order_timeout_sec: float = Field(10.0, gt=0.0)
    default_order_type: str = "market"
    slippage_tolerance_bps: float = Field(15.0, ge=0.0)


class ExitConfig(BaseModel):
    atr_period: int = Field(14, ge=2)
    atr_trailing_mult: float = Field(1.5, gt=0.0)
    breakeven_trigger_r: float = Field(1.0, gt=0.0)
    partial_tp_trigger_r: float = Field(1.5, gt=0.0)
    partial_tp_fraction: float = Field(0.5, gt=0.0, lt=1.0)
    trailing_trigger_r: float = Field(2.0, gt=0.0)
    max_bars_open: int = Field(48, ge=1)
    time_exit_min_r: float = Field(0.5, ge=0.0)
    check_interval_sec: float = Field(30.0, gt=0.0)

    @model_validator(mode="after")
    def _r_ordering(self) -> "ExitConfig":
        if not (self.breakeven_trigger_r < self.partial_tp_trigger_r):
            raise ValueError(
                f"breakeven_trigger_r ({self.breakeven_trigger_r}) must be < "
                f"partial_tp_trigger_r ({self.partial_tp_trigger_r})"
            )
        if not (self.partial_tp_trigger_r <= self.trailing_trigger_r):
            raise ValueError(
                f"partial_tp_trigger_r ({self.partial_tp_trigger_r}) must be <= "
                f"trailing_trigger_r ({self.trailing_trigger_r})"
            )
        return self


class PaperConfig(BaseModel):
    starting_balance_usd: float = Field(10000.0, gt=0.0)
    slippage_bps: float = Field(5.0, ge=0.0)
    taker_fee_bps: float = Field(6.0, ge=0.0)
    maker_fee_bps: float = Field(2.0, ge=0.0)
    leverage: int = Field(1, ge=1, le=20)


class TelegramConfig(BaseModel):
    enabled: bool = True
    levels: list[str] = ["INFO", "WARN", "CRITICAL"]
    daily_report_hour_utc: int = Field(0, ge=0, le=23)
    enable_commands: bool = True


class NotificationsConfig(BaseModel):
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)


class DatabaseConfig(BaseModel):
    pool_min: int = Field(2, ge=1)
    pool_max: int = Field(10, ge=1)
    statement_cache_size: int = 1024


class MonitoringConfig(BaseModel):
    health_port: int = Field(8080, ge=1, le=65535)
    prometheus_enabled: bool = True
    reconcile_interval_sec: float = Field(30.0, gt=0.0)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    mirror_to_db: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Top-level config composed of nested models + secrets from env
# ──────────────────────────────────────────────────────────────────────────────


class AppConfig(BaseModel):
    """Merged config.yaml — the `config` attribute of Settings."""

    mode: RuntimeMode = RuntimeMode.TESTNET
    exchanges: ExchangesConfig = Field(default_factory=ExchangesConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    exit: ExitConfig = Field(default_factory=ExitConfig)
    paper: PaperConfig = Field(default_factory=PaperConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class Settings(BaseSettings):
    """
    Top-level settings. Secrets come ONLY from env vars (never from YAML).
    Non-secret config comes from `config/config.yaml`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Runtime override ---
    cryprobotik_mode: RuntimeMode | None = None

    # --- Database ---
    # No default — must be supplied via DATABASE_URL environment variable.
    # See .env.example for the required format.
    database_url: SecretStr

    # --- OKX secrets ---
    okx_api_key: SecretStr = SecretStr("")
    okx_api_secret: SecretStr = SecretStr("")
    okx_api_passphrase: SecretStr = SecretStr("")
    okx_testnet: bool = True

    # --- Bybit secrets ---
    bybit_api_key: SecretStr = SecretStr("")
    bybit_api_secret: SecretStr = SecretStr("")
    bybit_testnet: bool = True

    # --- Telegram ---
    telegram_bot_token: SecretStr = SecretStr("")
    telegram_chat_id: str = ""

    # --- Monitoring ---
    health_port: int = 8080

    # --- Logging ---
    log_level: str = "INFO"

    # --- ML model security ---
    # If unset, model persistence is disabled and the bot runs in cold-start
    # mode every restart. Set a strong random secret in .env.
    model_hmac_secret: SecretStr = SecretStr("")

    # --- Loaded YAML config (populated by load_settings) ---
    config: AppConfig = Field(default_factory=AppConfig)

    @property
    def telegram_chat_ids(self) -> list[int]:
        """Parse comma-separated chat IDs into integers."""
        if not self.telegram_chat_id:
            return []
        return [int(cid.strip()) for cid in self.telegram_chat_id.split(",") if cid.strip()]


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(config_path: str | Path = "config/config.yaml") -> Settings:
    """
    Load settings from env + config.yaml.

    Precedence: CLI --mode > env CRYPROBOTIK_MODE > config.yaml `mode`.
    """
    settings = Settings()
    yaml_data = _load_yaml_config(Path(config_path))
    settings.config = AppConfig.model_validate(yaml_data)

    # Env override for mode
    if settings.cryprobotik_mode is not None:
        settings.config.mode = settings.cryprobotik_mode

    return settings
