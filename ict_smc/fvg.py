"""Fair Value Gap detection.

Detection rule:
  Bullish FVG at bar i:  bar[i].low > bar[i-2].high
  Bearish FVG at bar i:  bar[i].high < bar[i-2].low

Mitigation:
  An FVG is "mitigated" (filled) the first time price re-enters its zone
  after formation. We use the "touch" mitigation model — any wick into
  the zone counts. This is stricter than the "full fill" model and is
  the more common ICT convention.

Only formation detection (the bullish/bearish boolean masks in
detect_fvgs) is vectorized pandas, O(n). Everything else in this module
is a plain per-item Python loop, not vectorized: mitigation tracking
(inside detect_fvgs) walks the bar series once with a working set of
open FVGs, O(n) typical; filter_fvgs is a list comprehension over the
FVGs, O(n).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .types import FVG, validate_ohlcv


def detect_fvgs(
    df: pd.DataFrame,
    *,
    track_mitigation: bool = True,
) -> list[FVG]:
    """Detect all FVGs in an OHLCV DataFrame.

    Args:
        df: DataFrame with columns [open, high, low, close] and a
            DatetimeIndex. Bars must be in chronological order.
        track_mitigation: If True (default), each FVG is annotated with
            the timestamp at which price first re-entered its zone.
            Setting False is faster if you only need formation events.

    Returns:
        List of FVG instances sorted by formation time.

    Examples:
        >>> import pandas as pd
        >>> # Build a tiny series with a clear bullish FVG between bars 0 and 2
        >>> df = pd.DataFrame({
        ...     "open":  [10.0, 11.0, 13.0, 13.5, 12.0],
        ...     "high":  [10.5, 12.0, 14.0, 14.0, 13.0],
        ...     "low":   [ 9.5, 10.5, 12.5, 12.5, 11.5],
        ...     "close": [10.2, 11.8, 13.8, 13.0, 12.5],
        ... }, index=pd.date_range("2024-01-01", periods=5, freq="h"))
        >>> fvgs = detect_fvgs(df)
        >>> len(fvgs) >= 1
        True
        >>> fvgs[0].type
        'bullish'
    """
    validate_ohlcv(df)
    if len(df) < 3:
        return []

    high_2bar = df["high"].shift(2)
    low_2bar = df["low"].shift(2)

    bullish_mask = df["low"] > high_2bar
    bearish_mask = df["high"] < low_2bar

    fvgs: list[FVG] = []

    for ts in df.index[bullish_mask]:
        fvgs.append(FVG(
            formed_at=ts,
            type="bullish",
            top=float(df.at[ts, "low"]),
            bottom=float(high_2bar.at[ts]),
        ))

    for ts in df.index[bearish_mask]:
        fvgs.append(FVG(
            formed_at=ts,
            type="bearish",
            top=float(low_2bar.at[ts]),
            bottom=float(df.at[ts, "high"]),
        ))

    fvgs.sort(key=lambda f: f.formed_at)

    if track_mitigation:
        fvgs = _mark_mitigation(fvgs, df)

    return fvgs


def _mark_mitigation(fvgs: list[FVG], df: pd.DataFrame) -> list[FVG]:
    """Annotate each FVG with the timestamp it was first mitigated.

    Walks the bar series once. Maintains a list of "open" (un-mitigated)
    FVGs and closes them as price wicks into their zone.
    """
    if not fvgs:
        return []

    # Index FVGs by formation timestamp for fast lookup
    by_ts: dict[pd.Timestamp, list[int]] = {}
    for i, f in enumerate(fvgs):
        by_ts.setdefault(f.formed_at, []).append(i)

    open_fvgs: list[int] = []  # indices into the fvgs list
    mitigation_ts: dict[int, pd.Timestamp] = {}

    for ts, bar in df[["high", "low"]].iterrows():
        # Check existing open FVGs against this bar's range
        still_open: list[int] = []
        for idx in open_fvgs:
            f = fvgs[idx]
            if f.type == "bullish":
                # Bullish FVG mitigated when price wicks down to its top
                if bar["low"] <= f.top:
                    mitigation_ts[idx] = ts
                else:
                    still_open.append(idx)
            else:  # bearish
                # Bearish FVG mitigated when price wicks up to its bottom
                if bar["high"] >= f.bottom:
                    mitigation_ts[idx] = ts
                else:
                    still_open.append(idx)
        open_fvgs = still_open

        # Add any FVGs that formed at this bar to the open set
        # (excluding the formation bar itself from mitigation checks)
        for idx in by_ts.get(ts, []):
            open_fvgs.append(idx)

    # Apply mitigation timestamps (FVG is frozen, so build new instances)
    return [
        FVG(
            formed_at=f.formed_at,
            type=f.type,
            top=f.top,
            bottom=f.bottom,
            mitigated_at=mitigation_ts.get(i),
        )
        for i, f in enumerate(fvgs)
    ]


def filter_fvgs(
    fvgs: list[FVG],
    *,
    type: Literal["bullish", "bearish"] | None = None,
    only_unmitigated: bool = False,
    min_height: float | None = None,
    formed_after: pd.Timestamp | None = None,
) -> list[FVG]:
    """Filter a list of FVGs by common criteria.

    Args:
        fvgs: List from detect_fvgs.
        type: Restrict to "bullish" or "bearish" only.
        only_unmitigated: If True, keep only FVGs that haven't been filled.
        min_height: Minimum zone height (top - bottom) in price units.
        formed_after: Keep only FVGs formed at or after this timestamp.
    """
    out = fvgs
    if type is not None:
        out = [f for f in out if f.type == type]
    if only_unmitigated:
        out = [f for f in out if not f.is_mitigated]
    if min_height is not None:
        out = [f for f in out if f.height >= min_height]
    if formed_after is not None:
        out = [f for f in out if f.formed_at >= formed_after]
    return out
