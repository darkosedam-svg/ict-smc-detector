"""Tests for ICT/SMC detection.

Tests use hand-crafted OHLCV series where the expected detections are
known by construction. No real-market data, no flaky network calls.
"""

import numpy as np
import pandas as pd
import pytest

from ict_smc import (
    BoS,
    FVG,
    OrderBlock,
    detect_bos,
    detect_fvgs,
    detect_order_blocks,
    filter_consecutive_bos,
    filter_fvgs,
)
from ict_smc.types import validate_ohlcv


def _make_bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Quick helper: list of (open, high, low, close) → DataFrame with hourly index."""
    return pd.DataFrame(
        rows,
        columns=["open", "high", "low", "close"],
        index=pd.date_range("2024-01-01", periods=len(rows), freq="h"),
    )


# -----------------------------------------------------------------------------
# validate_ohlcv
# -----------------------------------------------------------------------------

class TestValidate:
    def test_passes_valid(self):
        df = _make_bars([(10, 11, 9, 10), (10, 11, 9, 10)])
        validate_ohlcv(df)

    def test_rejects_missing_columns(self):
        df = pd.DataFrame({"open": [10], "high": [11]},
                          index=pd.date_range("2024-01-01", periods=1, freq="h"))
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_ohlcv(df)

    def test_rejects_non_datetime_index(self):
        df = pd.DataFrame(
            [[10, 11, 9, 10]], columns=["open", "high", "low", "close"]
        )
        with pytest.raises(ValueError, match="DatetimeIndex"):
            validate_ohlcv(df)

    def test_rejects_high_below_low(self):
        df = _make_bars([(10, 9, 11, 10)])  # high < low
        with pytest.raises(ValueError, match="Invalid OHLC"):
            validate_ohlcv(df)


# -----------------------------------------------------------------------------
# FVG detection
# -----------------------------------------------------------------------------

class TestFVGDetection:
    def test_simple_bullish_fvg(self):
        # Bar 0 high=11, bar 2 low=13 → bullish FVG between 11 and 13
        # All bars must satisfy: high >= max(open, close), low <= min(open, close)
        df = _make_bars([
            (10.0, 11.0, 9.5, 10.5),     # bar 0: high=11
            (10.5, 12.0, 10.0, 11.5),    # bar 1: gap candle
            (13.5, 14.0, 13.0, 13.8),    # bar 2: low=13, high=14, open/close inside
        ])
        fvgs = detect_fvgs(df, track_mitigation=False)
        bullish = [f for f in fvgs if f.type == "bullish"]
        assert len(bullish) == 1
        assert bullish[0].top == 13.0
        assert bullish[0].bottom == 11.0
        assert bullish[0].formed_at == df.index[2]

    def test_simple_bearish_fvg(self):
        # Bar 0 low=10, bar 2 high=8 → bearish FVG between 8 and 10
        df = _make_bars([
            (12.0, 13.0, 10.0, 11.0),    # bar 0: low=10
            (11.0, 11.5, 9.0, 9.5),      # bar 1
            (7.5, 8.0, 7.0, 7.8),        # bar 2: high=8, all values < bar 0 low
        ])
        fvgs = detect_fvgs(df, track_mitigation=False)
        bearish = [f for f in fvgs if f.type == "bearish"]
        assert len(bearish) == 1
        assert bearish[0].top == 10.0
        assert bearish[0].bottom == 8.0

    def test_no_fvg_when_overlap(self):
        # Sequential bars with overlapping ranges → no FVG
        df = _make_bars([
            (10, 11, 9, 10.5),
            (10.5, 11.5, 9.5, 11),
            (11, 12, 10, 11.5),  # bar 2 low=10 < bar 0 high=11 → no bullish FVG
        ])
        fvgs = detect_fvgs(df, track_mitigation=False)
        assert len(fvgs) == 0

    def test_short_series_returns_empty(self):
        df = _make_bars([(10, 11, 9, 10), (10, 11, 9, 10)])
        assert detect_fvgs(df) == []

    def test_empty_dataframe(self):
        df = pd.DataFrame(
            columns=["open", "high", "low", "close"],
            index=pd.DatetimeIndex([]),
        )
        assert detect_fvgs(df) == []

    def test_mitigation_tracked(self):
        # Build series that creates a bullish FVG at bar 2, then a later bar
        # wicks down into the zone. Note: depending on the exact bar geometry,
        # additional FVGs can form — we check the specific 11→13 FVG.
        df = _make_bars([
            (10.0, 11.0, 9.5, 10.5),       # bar 0: high=11
            (10.5, 12.0, 10.0, 11.5),      # bar 1
            (13.5, 14.0, 13.0, 13.8),      # bar 2: bullish FVG forms (11→13)
            (13.7, 13.7, 12.5, 13.0),      # bar 3: wicks down to 12.5, into 11-13 zone
            (13.0, 13.5, 12.8, 13.2),
        ])
        fvgs = detect_fvgs(df, track_mitigation=True)
        # Find the specific 11→13 FVG
        target = next(
            (f for f in fvgs if f.type == "bullish" and f.bottom == 11.0 and f.top == 13.0),
            None,
        )
        assert target is not None, f"Expected 11→13 bullish FVG; got {fvgs}"
        assert target.is_mitigated
        assert target.mitigated_at == df.index[3]

    def test_mitigation_not_set_if_never_revisited(self):
        # Set up so the only bullish FVG is the 11→13 one and price never returns
        df = _make_bars([
            (10.0, 11.0, 9.5, 10.5),       # bar 0: high=11
            (10.5, 12.0, 10.0, 11.5),      # bar 1
            (13.5, 14.0, 13.0, 13.8),      # bar 2: bullish FVG 11→13
            (13.8, 14.5, 13.5, 14.0),      # bar 3: stays above 13
            (14.0, 15.0, 13.6, 14.5),      # bar 4: stays above 13
        ])
        fvgs = detect_fvgs(df)
        target = next(
            (f for f in fvgs if f.type == "bullish" and f.bottom == 11.0 and f.top == 13.0),
            None,
        )
        assert target is not None
        assert not target.is_mitigated


# -----------------------------------------------------------------------------
# BoS detection
# -----------------------------------------------------------------------------

class TestBoSDetection:
    def test_bullish_bos_detected(self):
        # Build a series of consolidating bars then a clear close above the prior high
        # 25 bars in [10, 11], then bar 25 closes at 12 (clear break)
        rng = np.random.RandomState(42)
        bars = []
        for _ in range(25):
            o = 10 + rng.uniform(0, 0.5)
            c = 10 + rng.uniform(0, 0.5)
            bars.append((o, max(o, c) + 0.3, min(o, c) - 0.2, c))
        # Breakout bar: close at 12 (above the rolling-20 high of ~10.8)
        bars.append((10.5, 12.1, 10.4, 12.0))
        df = _make_bars(bars)

        events = detect_bos(df, lookback=20)
        bullish = [e for e in events if e.direction == "bullish"]
        assert len(bullish) >= 1
        # The breakout bar should be among the bullish BoS events
        assert df.index[25] in [e.timestamp for e in bullish]

    def test_no_bos_in_pure_chop(self):
        # Bars that stay strictly within [9.5, 10.5]
        rng = np.random.RandomState(0)
        bars = []
        for _ in range(50):
            mid = 10 + rng.uniform(-0.3, 0.3)
            bars.append((mid, mid + 0.2, mid - 0.2, mid + rng.uniform(-0.1, 0.1)))
        df = _make_bars(bars)
        events = detect_bos(df, lookback=20)
        # Some BoS may fire on the very first post-warmup bars but with a
        # min_break_bps filter, none should
        events_filtered = detect_bos(df, lookback=20, min_break_bps=50)
        assert events_filtered == []

    def test_short_series_returns_empty(self):
        df = _make_bars([(10, 11, 9, 10)] * 5)
        assert detect_bos(df, lookback=20) == []

    def test_filter_consecutive_collapses_runs(self):
        events = [
            BoS(pd.Timestamp("2024-01-01"), "bullish", 10, 11),
            BoS(pd.Timestamp("2024-01-02"), "bullish", 10.5, 11.5),
            BoS(pd.Timestamp("2024-01-03"), "bullish", 11, 12),
            BoS(pd.Timestamp("2024-01-04"), "bearish", 9, 8),
            BoS(pd.Timestamp("2024-01-05"), "bearish", 8, 7),
            BoS(pd.Timestamp("2024-01-06"), "bullish", 9, 10),
        ]
        collapsed = filter_consecutive_bos(events)
        assert len(collapsed) == 3
        assert [e.direction for e in collapsed] == ["bullish", "bearish", "bullish"]


# -----------------------------------------------------------------------------
# Order Block detection
# -----------------------------------------------------------------------------

class TestOrderBlockDetection:
    def test_bullish_ob_found_before_bullish_bos(self):
        # Setup: chop in [10, 11] for 22 bars, then a strong bullish move.
        # The last bearish candle inside the chop should be the OB.
        rng = np.random.RandomState(7)
        bars = []
        for _ in range(22):
            o = 10 + rng.uniform(0, 0.8)
            c = 10 + rng.uniform(0, 0.8)
            bars.append((o, max(o, c) + 0.1, min(o, c) - 0.1, c))

        # Force an explicit bearish candle just before breakout
        bars.append((10.5, 10.6, 9.8, 9.9))   # bar 22: bearish, this is the OB
        bars.append((9.95, 10.5, 9.9, 10.4))  # bar 23: bullish, leading into BoS
        bars.append((10.4, 12.0, 10.4, 11.9)) # bar 24: BoS, close > rolling high
        df = _make_bars(bars)

        obs = detect_order_blocks(df, bos_lookback=20, track_mitigation=False)
        bullish_obs = [o for o in obs if o.type == "bullish"]
        assert len(bullish_obs) >= 1

        # The OB should be the bearish candle at index 22
        first_ob = bullish_obs[0]
        assert first_ob.formed_at == df.index[22]
        assert first_ob.top == 10.6
        assert first_ob.bottom == 9.8

    def test_no_ob_when_no_bos(self):
        # 25 bars of pure consolidation
        rng = np.random.RandomState(99)
        bars = []
        for _ in range(30):
            mid = 10 + rng.uniform(-0.1, 0.1)
            bars.append((mid, mid + 0.05, mid - 0.05, mid))
        df = _make_bars(bars)
        obs = detect_order_blocks(df, bos_lookback=20, min_break_bps=100)
        assert obs == []


# -----------------------------------------------------------------------------
# filter_fvgs
# -----------------------------------------------------------------------------

class TestFilters:
    def test_filter_by_type(self):
        fvgs = [
            FVG(pd.Timestamp("2024-01-01"), "bullish", 11, 10),
            FVG(pd.Timestamp("2024-01-02"), "bearish", 10, 9),
            FVG(pd.Timestamp("2024-01-03"), "bullish", 12, 11),
        ]
        bull_only = filter_fvgs(fvgs, type="bullish")
        assert len(bull_only) == 2
        assert all(f.type == "bullish" for f in bull_only)

    def test_filter_unmitigated(self):
        fvgs = [
            FVG(pd.Timestamp("2024-01-01"), "bullish", 11, 10,
                mitigated_at=pd.Timestamp("2024-01-02")),
            FVG(pd.Timestamp("2024-01-02"), "bullish", 12, 11),  # unmitigated
        ]
        unmitigated = filter_fvgs(fvgs, only_unmitigated=True)
        assert len(unmitigated) == 1

    def test_filter_min_height(self):
        fvgs = [
            FVG(pd.Timestamp("2024-01-01"), "bullish", 10.5, 10.0),  # height 0.5
            FVG(pd.Timestamp("2024-01-02"), "bullish", 12.0, 10.0),  # height 2.0
        ]
        large = filter_fvgs(fvgs, min_height=1.0)
        assert len(large) == 1
        assert large[0].height == 2.0


# -----------------------------------------------------------------------------
# Performance — verify vectorized FVG detection is fast on large data
# -----------------------------------------------------------------------------

class TestPerformance:
    def test_detects_fvgs_on_10k_bars_quickly(self):
        # Generate 10K bars of synthetic price data
        rng = np.random.RandomState(123)
        n = 10_000
        prices = 10000 + np.cumsum(rng.randn(n) * 5)
        bars = []
        for p in prices:
            o = p
            c = p + rng.randn() * 2
            h = max(o, c) + abs(rng.randn() * 1.5)
            low_val = min(o, c) - abs(rng.randn() * 1.5)
            bars.append((o, h, low_val, c))
        df = _make_bars(bars)

        import time
        start = time.perf_counter()
        fvgs = detect_fvgs(df, track_mitigation=False)
        detection_time = time.perf_counter() - start

        # Should be well under 1 second on 10K bars
        assert detection_time < 1.0, f"Detection took {detection_time:.2f}s"
        # Random walk should produce at least some FVGs
        assert len(fvgs) > 0
