"""Shared data types for the ICT/SMC detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import pandas as pd


@dataclass
class FVG:
    """A Fair Value Gap (imbalance zone).

    Attributes:
        formed_at:    Timestamp of the bar that created the gap.
        type:         "bullish" or "bearish".
        top:          Upper bound of the gap zone.
        bottom:       Lower bound of the gap zone.
        mitigated_at: Timestamp when price first re-entered the zone, or None.
    """

    formed_at: pd.Timestamp
    type: Literal["bullish", "bearish"]
    top: float
    bottom: float
    mitigated_at: Optional[pd.Timestamp] = field(default=None)

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def is_mitigated(self) -> bool:
        return self.mitigated_at is not None


@dataclass
class BoS:
    """A Break of Structure event.

    Attributes:
        timestamp:   Bar whose close triggered the break.
        direction:   "bullish" or "bearish".
        level:       The swing level that was broken.
        break_price: The closing price that triggered the break.
    """

    timestamp: pd.Timestamp
    direction: Literal["bullish", "bearish"]
    level: float
    break_price: float


@dataclass
class OrderBlock:
    """An Order Block zone (last opposite-direction candle before a BoS).

    Attributes:
        formed_at:    Timestamp of the OB candle.
        confirmed_at: Timestamp of the BoS that validated this OB.
        type:         "bullish" or "bearish".
        top:          High of the OB candle.
        bottom:       Low of the OB candle.
        mitigated_at: Timestamp when price re-entered the OB zone, or None.
    """

    formed_at: pd.Timestamp
    confirmed_at: pd.Timestamp
    type: Literal["bullish", "bearish"]
    top: float
    bottom: float
    mitigated_at: Optional[pd.Timestamp] = field(default=None)

    @property
    def is_mitigated(self) -> bool:
        return self.mitigated_at is not None


_REQUIRED_COLS = {"open", "high", "low", "close"}


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Raise ValueError if *df* is not a valid OHLCV DataFrame."""
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "DataFrame index must be a DatetimeIndex; "
            f"got {type(df.index).__name__}"
        )

    if len(df) > 0 and (df["high"] < df["low"]).any():
        raise ValueError("Invalid OHLC data: high < low on at least one row")
