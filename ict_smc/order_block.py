"""Order Block detection.

An Order Block is the last opposite-direction candle before a Break of
Structure. Bullish OB = last bearish candle before a bullish BoS.

The OB candle's high/low define a zone where price often returns and
reacts. Some implementations use the entire range from OB candle to
BoS candle; we use just the OB candle's range, which is tighter and
more conservative.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .bos import detect_bos, filter_consecutive_bos
from .types import BoS, OrderBlock, validate_ohlcv


def detect_order_blocks(
    df: pd.DataFrame,
    *,
    bos_lookback: int = 20,
    min_break_bps: float = 0.0,
    max_search_back: int = 30,
    track_mitigation: bool = True,
    collapse_runs: bool = True,
) -> list[OrderBlock]:
    """Detect Order Blocks.

    For each Break of Structure, walk backwards up to `max_search_back`
    bars to find the last opposite-direction candle. That candle is
    the Order Block.

    Args:
        df: DataFrame with columns [open, high, low, close] and a
            DatetimeIndex.
        bos_lookback: Lookback for BoS detection.
        min_break_bps: Minimum BoS magnitude in bps.
        max_search_back: Max bars to search backwards for the OB candle.
        track_mitigation: Annotate each OB with mitigation timestamp.
        collapse_runs: If True (default), only the last BoS in each
            consecutive same-direction run is used. Recommended.

    Returns:
        List of OrderBlock instances ordered by formation time.
    """
    validate_ohlcv(df)
    bos_events = detect_bos(df, lookback=bos_lookback, min_break_bps=min_break_bps)
    if collapse_runs:
        bos_events = filter_consecutive_bos(bos_events)

    obs: list[OrderBlock] = []
    for bos in bos_events:
        ob = _find_order_block_for_bos(df, bos, max_search_back)
        if ob is not None:
            obs.append(ob)

    obs.sort(key=lambda o: o.formed_at)

    if track_mitigation:
        obs = _mark_mitigation(obs, df)

    return obs


def _find_order_block_for_bos(
    df: pd.DataFrame,
    bos: BoS,
    max_search_back: int,
) -> Optional[OrderBlock]:
    """Walk backwards from a BoS to find the OB candle."""
    bos_idx = df.index.get_loc(bos.timestamp)
    if not isinstance(bos_idx, int):
        # If the timestamp isn't unique, get_loc may return a slice/array
        return None

    start = max(0, bos_idx - max_search_back)
    for j in range(bos_idx - 1, start - 1, -1):
        bar = df.iloc[j]
        is_bearish = bar["close"] < bar["open"]
        is_bullish = bar["close"] > bar["open"]

        if bos.direction == "bullish" and is_bearish:
            return OrderBlock(
                formed_at=df.index[j],
                confirmed_at=bos.timestamp,
                type="bullish",
                top=float(bar["high"]),
                bottom=float(bar["low"]),
            )
        if bos.direction == "bearish" and is_bullish:
            return OrderBlock(
                formed_at=df.index[j],
                confirmed_at=bos.timestamp,
                type="bearish",
                top=float(bar["high"]),
                bottom=float(bar["low"]),
            )

    return None


def _mark_mitigation(obs: list[OrderBlock], df: pd.DataFrame) -> list[OrderBlock]:
    """Annotate each OB with mitigation timestamp.

    OB mitigation happens when price wicks back into the OB zone after
    confirmation (not formation — confirmation is the BoS bar).
    """
    if not obs:
        return []

    out: list[OrderBlock] = []
    for ob in obs:
        # Look at bars strictly after confirmation
        loc = df.index.get_indexer([ob.confirmed_at])[0]
        if loc < 0 or loc >= len(df) - 1:
            out.append(ob)
            continue

        future = df.iloc[loc + 1:]
        if ob.type == "bullish":
            # Mitigated when price wicks down to OB top
            hits = future[future["low"] <= ob.top]
        else:
            # Mitigated when price wicks up to OB bottom
            hits = future[future["high"] >= ob.bottom]

        mitigated_at = hits.index[0] if len(hits) > 0 else None
        out.append(OrderBlock(
            formed_at=ob.formed_at,
            confirmed_at=ob.confirmed_at,
            type=ob.type,
            top=ob.top,
            bottom=ob.bottom,
            mitigated_at=mitigated_at,
        ))

    return out
