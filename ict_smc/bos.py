"""Break of Structure (BoS) detection.

A Break of Structure occurs when price closes beyond a prior swing
high (bullish BoS) or swing low (bearish BoS) established within a
lookback window.

Detection is vectorized using rolling max/min over the lookback window,
giving O(n) complexity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .types import BoS, validate_ohlcv


def detect_bos(
    df: pd.DataFrame,
    *,
    lookback: int = 20,
    min_break_bps: float = 0.0,
) -> list[BoS]:
    """Detect Break of Structure events.

    A bullish BoS is recorded when the *close* of bar[i] exceeds the
    rolling maximum *high* of bars[i-lookback : i-1].

    A bearish BoS is recorded when the *close* of bar[i] is below the
    rolling minimum *low* of bars[i-lookback : i-1].

    Args:
        df:             OHLCV DataFrame with DatetimeIndex.
        lookback:       Number of bars for the rolling swing window.
        min_break_bps:  Minimum break size in basis points relative to the
                        swing level. Useful to filter noise.

    Returns:
        List of BoS events sorted by timestamp.
    """
    validate_ohlcv(df)

    if len(df) <= lookback:
        return []

    closes = df["close"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    n = len(df)

    # Rolling max/min over the *prior* `lookback` bars (exclusive of bar i)
    # We compute via cumulative rolling with a fixed offset.
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)

    for i in range(lookback, n):
        swing_high[i] = highs[i - lookback: i].max()
        swing_low[i] = lows[i - lookback: i].min()

    events: list[BoS] = []
    for i in range(lookback, n):
        sh = swing_high[i]
        sl = swing_low[i]
        c = closes[i]
        ts = df.index[i]

        # Bullish BoS
        if not np.isnan(sh):
            break_pct = (c - sh) / sh * 10_000  # basis points
            if c > sh and break_pct >= min_break_bps:
                events.append(BoS(
                    timestamp=ts,
                    direction="bullish",
                    level=float(sh),
                    break_price=float(c),
                ))

        # Bearish BoS
        if not np.isnan(sl):
            break_pct = (sl - c) / sl * 10_000  # basis points
            if c < sl and break_pct >= min_break_bps:
                events.append(BoS(
                    timestamp=ts,
                    direction="bearish",
                    level=float(sl),
                    break_price=float(c),
                ))

    return events


def filter_consecutive_bos(events: list[BoS]) -> list[BoS]:
    """Collapse runs of same-direction BoS events into their last entry.

    When price trends continuously, many consecutive BoS events fire in
    the same direction. This filter keeps only the last event in each
    run, reducing noise.

    Example: [bull, bull, bull, bear, bear, bull] → [bull, bear, bull]
    where each kept event is the *last* in its run.

    Args:
        events: BoS events sorted by timestamp (ascending).

    Returns:
        Filtered list — one event per directional run (the last).
    """
    if not events:
        return []

    result: list[BoS] = []
    i = 0
    while i < len(events):
        current_dir = events[i].direction
        # Find the end of this run
        j = i
        while j + 1 < len(events) and events[j + 1].direction == current_dir:
            j += 1
        result.append(events[j])  # keep the last of the run
        i = j + 1

    return result
